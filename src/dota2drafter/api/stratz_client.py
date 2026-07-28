"""Async STRATZ GraphQL API client for Dota 2 match data."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, cast

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from dota2drafter.config import StratzConfig

logger = logging.getLogger(__name__)

# GraphQL queries
LEAGUES_QUERY = """
    query Leagues($request: LeagueRequestType!) {
        leagues(request: $request) {
            id
            name
            displayName
            tier
            lastMatchDate
        }
    }
"""

MATCHES_BY_LEAGUE_QUERY = """
    query MatchesByLeague($leagueId: Int!, $request: LeagueMatchesRequestType!) {
        league(id: $leagueId) {
            matches(request: $request) {
                id
            }
        }
    }
"""

MATCH_DETAILS_QUERY = """
    query MatchDetails($matchId: Long!) {
        match(id: $matchId) {
            id
            didRadiantWin
            gameMode
            league {
                id
                name
            }
            players {
                isRadiant
                steamAccountId
            }
            pickBans {
                isPick
                heroId
                order
                isRadiant
            }
        }
    }
"""

HEROES_QUERY = """
    query Heroes {
        constants {
            heroes {
                id
                name
                stats {
                    enabled
                }
            }
        }
    }
"""


class StratzClient:
    """Async GraphQL client for the STRATZ API."""

    def __init__(self, config: StratzConfig) -> None:
        self._config = config
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "STRATZ_API",
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a GraphQL query with retry logic."""
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        base_url = self._config.base_url
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        elif base_url.endswith("/v1/"):
            base_url = base_url[:-4]
        url = f"{base_url.rstrip('/')}/graphql"

        async with aiohttp.ClientSession(
            headers=self._headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            async with session.post(
                url,
                json=payload,
            ) as resp:
                if resp.status == 429:
                    logger.warning("Rate limited by STRATZ API, waiting...")
                    await asyncio.sleep(self._config.retry_delay)
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=429,
                    )
                if resp.status == 400:
                    try:
                        data = await resp.json()
                        if "errors" in data:
                            errors = data["errors"]
                            logger.error("GraphQL 400 errors: %s", errors)
                            raise RuntimeError(f"GraphQL 400 errors: {errors}")
                    except Exception as json_err:
                        if isinstance(json_err, RuntimeError):
                            raise
                        logger.error("Failed to parse 400 error body: %s", json_err)
                resp.raise_for_status()
                data = await resp.json()

        if "errors" in data:
            errors = data["errors"]
            logger.error("GraphQL errors: %s", errors)
            raise RuntimeError(f"GraphQL errors: {errors}")

        return data

    async def fetch_leagues(self, tiers: list[int], patch: str) -> list[dict[str, Any]]:
        """Fetch tier 1 and 2 leagues for a given patch."""
        int_tier_to_stratz = {
            1: ["INTERNATIONAL", "MAJOR", "DPC_LEAGUE_FINALS", "DPC_LEAGUE"],
            2: ["PROFESSIONAL", "MINOR", "DPC_QUALIFIER", "DPC_LEAGUE_QUALIFIER"],
            3: ["AMATEUR"],
        }
        stratz_tier_to_int = {
            "INTERNATIONAL": 1,
            "MAJOR": 1,
            "DPC_LEAGUE_FINALS": 1,
            "DPC_LEAGUE": 1,
            "PROFESSIONAL": 2,
            "MINOR": 2,
            "DPC_QUALIFIER": 2,
            "DPC_LEAGUE_QUALIFIER": 2,
            "AMATEUR": 3,
            "UNSET": 0
        }
        
        stratz_tiers = []
        for t in tiers:
            stratz_tiers.extend(int_tier_to_stratz.get(t, []))
            
        request_params = {"take": 1000}
        if stratz_tiers:
            request_params["tiers"] = stratz_tiers
            
        data = await self._graphql(LEAGUES_QUERY, {"request": request_params})
        leagues_data = data.get("data", {}).get("leagues", [])
        
        mapped = []
        now = int(time.time())
        two_weeks_ago = now - 14 * 24 * 3600
        for league in leagues_data:
            t_str = league.get("tier", "UNSET")
            last_match = league.get("lastMatchDate")
            ended = 1 if last_match and last_match < two_weeks_ago else 0
            mapped.append({
                "id": league.get("id"),
                "name": league.get("displayName") or league.get("name") or "",
                "tier": stratz_tier_to_int.get(t_str, 0),
                "ended": ended
            })
        return mapped

    async def fetch_matches_by_league(
        self, league_id: str, patch: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Fetch match IDs for a specific league and patch."""
        league_id_int = int(league_id)
        request_params = {
            "take": min(limit, 100),
            "skip": 0,
            "isParsed": True
        }
        data = await self._graphql(
            MATCHES_BY_LEAGUE_QUERY,
            {"leagueId": league_id_int, "request": request_params},
        )
        matches_data = data.get("data", {}).get("league", {}).get("matches", [])
        return matches_data

    async def fetch_match_details(self, match_id: str) -> dict[str, Any] | None:
        """Fetch detailed match data including draft picks/bans."""
        match_id_long = int(match_id)
        data = await self._graphql(MATCH_DETAILS_QUERY, {"matchId": match_id_long})
        match_data = data.get("data", {}).get("match")
        if match_data is None:
            return None
            
        # Map back to legacy expected structure for compatibility
        mapped_picks_bans = []
        for pb in match_data.get("pickBans", []) or []:
            mapped_picks_bans.append({
                "type": "pick" if pb.get("isPick") else "ban",
                "hero": {"id": pb.get("heroId")} if pb.get("heroId") is not None else None,
                "team": 0 if pb.get("isRadiant") else 1,
                "order": pb.get("order")
            })
            
        mapped_players = []
        for p in match_data.get("players", []) or []:
            mapped_players.append({
                "team": 1 if p.get("isRadiant") else 2,
                "accountid": p.get("steamAccountId")
            })
            
        mapped_match = {
            "id": str(match_data.get("id", "")),
            "radiantWin": match_data.get("didRadiantWin"),
            "draft": {
                "picksBans": mapped_picks_bans
            },
            "players": mapped_players
        }
        
        # Add league details if present
        league = match_data.get("league")
        if league:
            mapped_match["league"] = league
            
        return mapped_match

    async def fetch_heroes(self, patch: str) -> list[dict[str, Any]]:
        """Fetch the hero roster for a given patch."""
        data = await self._graphql(HEROES_QUERY)
        heroes_data = data.get("data", {}).get("constants", {}).get("heroes", [])
        
        # Map fields to match what the HeroIndexer expects (including 'playable')
        mapped_heroes = []
        for hero in heroes_data:
            stats = hero.get("stats") or {}
            mapped_heroes.append({
                "id": hero.get("id"),
                "name": hero.get("name"),
                "playable": stats.get("enabled", False)
            })
        return mapped_heroes

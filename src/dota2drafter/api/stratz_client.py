"""Async STRATZ GraphQL API client for Dota 2 match data."""

from __future__ import annotations

import asyncio
import logging
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
    query Leagues($tier: [Int], $patch: String) {
        leagues(tier: $tier, patch: $patch) {
            data {
                id
                name
                tier
                patch
            }
        }
    }
"""

MATCHES_BY_LEAGUE_QUERY = """
    query MatchesByLeague($leagueId: String, $patch: String, $limit: Int) {
        matches(
            filter: { leagueId: $leagueId, patch: $patch }
            limit: $limit
        ) {
            data {
                id
                pool
                league {
                    id
                }
            }
        }
    }
"""

MATCH_DETAILS_QUERY = """
    query MatchDetails($matchId: String) {
        match(id: $matchId) {
            data {
                id
                pool
                league {
                    id
                    name
                }
                radiantWin
                draft {
                    picksBans {
                        type
                        hero {
                            id
                        }
                        team
                        order
                    }
                }
            }
        }
    }
"""

HEROES_QUERY = """
    query Heroes($patch: String) {
        heroes(patch: $patch) {
            data {
                id
                name
                type
                playable
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

        async with aiohttp.ClientSession(
            headers=self._headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            async with session.post(
                f"{self._config.base_url}/graphql",
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
                resp.raise_for_status()
                data = await resp.json()

        if "errors" in data:
            errors = data["errors"]
            logger.error("GraphQL errors: %s", errors)
            raise RuntimeError(f"GraphQL errors: {errors}")

        return data

    async def fetch_leagues(self, tiers: list[int], patch: str) -> list[dict[str, Any]]:
        """Fetch tier 1 and 2 leagues for a given patch."""
        data = await self._graphql(LEAGUES_QUERY, {"tier": tiers, "patch": patch})
        leagues_data = data.get("data", {}).get("leagues", {}).get("data", [])
        return leagues_data

    async def fetch_matches_by_league(
        self, league_id: str, patch: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Fetch match IDs for a specific league and patch."""
        data = await self._graphql(
            MATCHES_BY_LEAGUE_QUERY,
            {"leagueId": league_id, "patch": patch, "limit": limit},
        )
        matches_data = data.get("data", {}).get("matches", {}).get("data", [])
        return matches_data

    async def fetch_match_details(self, match_id: str) -> dict[str, Any] | None:
        """Fetch detailed match data including draft picks/bans."""
        data = await self._graphql(MATCH_DETAILS_QUERY, {"matchId": match_id})
        match_data = data.get("data", {}).get("match", {}).get("data")
        return match_data

    async def fetch_heroes(self, patch: str) -> list[dict[str, Any]]:
        """Fetch the hero roster for a given patch."""
        data = await self._graphql(HEROES_QUERY, {"patch": patch})
        heroes_data = data.get("data", {}).get("heroes", {}).get("data", [])
        return heroes_data

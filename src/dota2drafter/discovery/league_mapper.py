"""League discovery engine - resolves tier 1 and 2 tournament league IDs."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from dota2drafter.api.opendota_client import OpenDotaClient
from dota2drafter.api.stratz_client import StratzClient
from dota2drafter.config import ConcurrencyConfig, PipelineConfig
from dota2drafter.state import StateDatabase

logger = logging.getLogger(__name__)

# Liquipedia tier 1 and tier 2 tournament names to map against
# These are used as a reference; STRATZ league data is the authoritative source
LIQUIPEDIA_TIER_1_NAMES = [
    "The International",
    "DPC Western Europe",
    "DPC Eastern Europe",
    "DPC China",
    "DPC South East Asia",
    "DPC North America",
    "DPC South America",
    "ESL One",
    "DreamLeague",
    "BLAST Bounty Hunt",
    "BetBoom Dacha",
    "Riyadh Masters",
    "CCT Season Cup",
    "WePlay! Tournaments",
    "WTA Esports Games",
    "DPC League",
    "Esports World Cup",
    "EWC",
]

LIQUIPEDIA_TIER_2_NAMES = [
    "DPC Development League",
    "ESL Open Alpha",
    "DreamLeague Season",
    "BetBoom Team Bullfight",
    "CET League",
    "NEPAL ISLAND.GG League",
    "SberLeague",
    "FALLFALL",
    "DPC Tournament",
    "European Pro League",
    "EPL",
]


class LeagueMapper:
    """Discovers and registers tier 1 and 2 leagues with matches on or after cutoff_date."""

    def __init__(
        self,
        stratz_client: StratzClient,
        opendota_client: OpenDotaClient,
        state_db: StateDatabase,
        config: PipelineConfig,
        concurrency: ConcurrencyConfig,
    ) -> None:
        self._stratz = stratz_client
        self._opendota = opendota_client
        self._state_db = state_db
        self._config = config
        self._semaphore = asyncio.Semaphore(concurrency.max_connections_per_host)

    def _is_tier_match(self, league_name: str, tiers: list[int]) -> bool:
        """Check if a league name matches known tier 1 or tier 2 tournaments."""
        name_lower = league_name.lower()
        for tier1_name in LIQUIPEDIA_TIER_1_NAMES:
            if tier1_name.lower() in name_lower or name_lower in tier1_name.lower():
                return True
        for tier2_name in LIQUIPEDIA_TIER_2_NAMES:
            if tier2_name.lower() in name_lower or name_lower in tier2_name.lower():
                return True
        return False

    async def discover_leagues(self) -> list[dict[str, Any]]:
        """Discover all tier 1 and 2 leagues with matches on or after cutoff_date using OpenDota exclusively."""
        logger.info(
            "Discovering tier %s leagues with cutoff_date %s",
            self._config.tiers,
            self._config.cutoff_date,
        )

        matched = []

        # Discover from OpenDota (exclusively)
        logger.info("Discovering leagues from OpenDota pro matches...")
        opendota_leagues = {}
        try:
            raw_leagues = await self._opendota.fetch_leagues()
            for l in raw_leagues:
                l_id = str(l.get("leagueid", ""))
                if l_id:
                    opendota_leagues[l_id] = l.get("tier")
        except Exception as e:
            logger.warning("Failed to fetch global leagues from OpenDota: %s", e)

        opendota_discovered = {}
        less_than_match_id = None
        cutoff_timestamp = int(datetime.fromisoformat(self._config.cutoff_date).timestamp())

        try:
            while True:
                matches = await self._opendota.fetch_recent_pro_matches(less_than_match_id)
                if not matches:
                    break

                last_match_time = None
                for m in matches:
                    start_time = m.get("start_time")
                    if start_time is not None:
                        last_match_time = start_time
                        if start_time >= cutoff_timestamp:
                            league_id = str(m.get("leagueid", ""))
                            league_name = m.get("league_name") or ""
                            if league_id:
                                opendota_discovered[league_id] = league_name

                if last_match_time is None or last_match_time < cutoff_timestamp:
                    break

                less_than_match_id = matches[-1].get("match_id")
                await asyncio.sleep(0.5)  # rate limiting polite sleep
        except Exception as e:
            logger.warning("Failed to discover leagues from OpenDota pro matches: %s", e)

        opendota_tier_map = {
            "premium": 1,
            "professional": 2,
            "amateur": 3,
        }

        for league_id, league_name in opendota_discovered.items():
            # Check if already discovered via STRATZ
            if any(str(l.get("id")) == league_id for l in matched):
                continue

            # Look up tier, defaulting to professional (2)
            od_tier_str = opendota_leagues.get(league_id, "professional")
            league_tier = opendota_tier_map.get(od_tier_str, 2)

            # Name-based overrides to correct tier misclassifications from the APIs
            name_lower = league_name.lower()
            is_tier1 = any(t1.lower() in name_lower for t1 in LIQUIPEDIA_TIER_1_NAMES)
            is_tier2 = any(t2.lower() in name_lower for t2 in LIQUIPEDIA_TIER_2_NAMES)
            
            if is_tier1:
                league_tier = 1
            elif is_tier2:
                league_tier = 2

            if league_tier in self._config.tiers:
                self._state_db.insert_league(league_id, league_name, league_tier)
                matched.append({
                    "id": int(league_id),
                    "name": league_name,
                    "tier": league_tier
                })
                logger.info("Found league from OpenDota: %s (ID: %s, tier: %s)", league_name, league_id, league_tier)

        logger.info("Discovered %d tier 1/2 leagues with cutoff_date %s", len(matched), self._config.cutoff_date)
        return matched

    async def discover_leagues_concurrent(self) -> list[dict[str, Any]]:
        """Discover leagues with concurrency control."""
        async with self._semaphore:
            return await self.discover_leagues()

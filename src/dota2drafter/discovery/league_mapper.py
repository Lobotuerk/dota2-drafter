"""League discovery engine - resolves tier 1 and 2 tournament league IDs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

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
]


class LeagueMapper:
    """Discovers and registers tier 1 and 2 leagues with matches on or after cutoff_date."""

    def __init__(
        self,
        stratz_client: StratzClient,
        state_db: StateDatabase,
        config: PipelineConfig,
        concurrency: ConcurrencyConfig,
    ) -> None:
        self._stratz = stratz_client
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
        """Discover all tier 1 and 2 leagues with matches on or after cutoff_date."""
        logger.info(
            "Discovering tier %s leagues with cutoff_date %s",
            self._config.tiers,
            self._config.cutoff_date,
        )

        leagues = await self._stratz.fetch_leagues(self._config.tiers, self._config.cutoff_date)
        matched = []

        for league in leagues:
            league_id = league.get("id", "")
            league_name = league.get("name", "")
            league_tier = league.get("tier", 0)

            if not league_id:
                continue

            # STRATZ already filters by tier, but also check Liquipedia names
            is_match = league_tier in self._config.tiers or self._is_tier_match(
                league_name, self._config.tiers
            )

            if is_match:
                self._state_db.insert_league(league_id, league_name, league_tier)
                matched.append(league)
                logger.debug("Found league: %s (ID: %s, tier: %s)", league_name, league_id, league_tier)

        logger.info("Discovered %d tier 1/2 leagues with cutoff_date %s", len(matched), self._config.cutoff_date)
        return matched

    async def discover_leagues_concurrent(self) -> list[dict[str, Any]]:
        """Discover leagues with concurrency control."""
        async with self._semaphore:
            return await self.discover_leagues()

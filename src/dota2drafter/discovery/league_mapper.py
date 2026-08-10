"""League discovery engine - lists every league with matches on/after the cutoff."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dota2drafter.api.opendota_client import OpenDotaClient
from dota2drafter.api.stratz_client import StratzClient
from dota2drafter.config import ConcurrencyConfig, PipelineConfig

logger = logging.getLogger(__name__)

_OPENDOTA_TIER_MAP = {
    "premium": 1,
    "professional": 2,
    "amateur": 3,
}


@dataclass
class _LeagueObservation:
    """Accumulates the name and match boundary timestamps seen for one league."""

    name: str = ""
    earliest_match_time: int | None = None
    latest_match_time: int | None = None

    def observe(self, name: str, start_time: int) -> None:
        self.name = name or self.name
        if self.earliest_match_time is None:
            self.earliest_match_time = start_time
        else:
            self.earliest_match_time = min(self.earliest_match_time, start_time)
        if self.latest_match_time is None:
            self.latest_match_time = start_time
        else:
            self.latest_match_time = max(self.latest_match_time, start_time)


def _format_date(timestamp: int) -> str:
    """Format a Unix timestamp as an ISO-8601 date string."""
    return datetime.fromtimestamp(timestamp).date().isoformat()


class LeagueMapper:
    """Discovers every league with matches on or after cutoff_date.

    No tier inference or name-based filtering is applied - the caller decides
    which leagues to keep after a human review pass.
    """

    def __init__(
        self,
        opendota_client: OpenDotaClient,
        config: PipelineConfig,
        concurrency: ConcurrencyConfig,
        stratz_client: StratzClient | None = None,
    ) -> None:
        self._opendota = opendota_client
        self._stratz = stratz_client
        self._config = config
        self._semaphore = asyncio.Semaphore(concurrency.max_connections_per_host)

    async def discover_leagues(self) -> list[dict[str, Any]]:
        """Discover all leagues having a match on/after cutoff_date.

        Each returned dict carries ``id``, ``name``, ``tier``, ``start_date``
        and ``end_date`` inferred from the boundaries of the observed matches.
        """
        logger.info(
            "Discovering leagues with matches on/after cutoff_date %s",
            self._config.cutoff_date,
        )
        cutoff_timestamp = int(datetime.fromisoformat(self._config.cutoff_date).timestamp())

        observations: dict[str, _LeagueObservation] = {}
        less_than_match_id = None

        try:
            while True:
                matches = await self._opendota.fetch_recent_pro_matches(less_than_match_id)
                if not matches:
                    break

                last_match_time: int | None = None
                for match in matches:
                    start_time = match.get("start_time")
                    if start_time is None:
                        continue
                    last_match_time = start_time
                    if start_time >= cutoff_timestamp:
                        league_id = str(match.get("leagueid", ""))
                        if league_id:
                            observations.setdefault(league_id, _LeagueObservation()).observe(
                                match.get("league_name") or "", start_time
                            )

                if last_match_time is None or last_match_time < cutoff_timestamp:
                    break

                less_than_match_id = matches[-1].get("match_id")
                await asyncio.sleep(0.5)  # rate limiting polite sleep
        except Exception as e:
            logger.warning("Failed to discover leagues from OpenDota pro matches: %s", e)

        api_tiers = await self._fetch_api_tiers()

        # Promote major tournaments to Tier 1 manually due to Valve/API tiering flaws post-DPC
        tier1_keywords = ["esports world cup", "riyadh masters", "the international", "esl one", "dreamleague", "pgl wallachia", "blast slam"]
        
        leagues = []
        for league_id, observation in observations.items():
            name = observation.name
            tier = api_tiers.get(league_id, 2)
            
            # Keyword promotion
            name_lower = name.lower()
            if any(kw in name_lower for kw in tier1_keywords):
                tier = 1
                
            leagues.append({
                "id": int(league_id),
                "name": name,
                "tier": tier,
                "start_date": self._boundary_date(observation.earliest_match_time),
                "end_date": self._boundary_date(observation.latest_match_time),
            })

        logger.info(
            "Discovered %d leagues with matches on/after cutoff_date %s",
            len(leagues),
            self._config.cutoff_date,
        )
        return leagues

    async def _fetch_api_tiers(self) -> dict[str, int]:
        """Map league ids to their tier values using Stratz if available, else OpenDota."""
        tiers: dict[str, int] = {}
        
        if self._stratz:
            try:
                # Stratz gives excellent tier classification (1, 2, 3)
                raw_leagues = await self._stratz.fetch_leagues(tiers=[1, 2, 3], cutoff_date=self._config.cutoff_date)
                for league in raw_leagues:
                    league_id = str(league.get("id", ""))
                    if league_id:
                        tiers[league_id] = league.get("tier", 2)
                if tiers:
                    return tiers
            except Exception as e:
                logger.warning("Failed to fetch leagues from Stratz: %s", e)

        try:
            raw_leagues = await self._opendota.fetch_leagues()
        except Exception as e:
            logger.warning("Failed to fetch global leagues from OpenDota: %s", e)
            return tiers

        for league in raw_leagues:
            league_id = str(league.get("leagueid", ""))
            if league_id:
                tiers[league_id] = _OPENDOTA_TIER_MAP.get(league.get("tier"), 2)
        return tiers

    @staticmethod
    def _boundary_date(timestamp: int | None) -> str | None:
        """Return the boundary date, or None when no match was observed."""
        if timestamp is None:
            return None
        return _format_date(timestamp)

    async def discover_leagues_concurrent(self) -> list[dict[str, Any]]:
        """Discover leagues with concurrency control."""
        async with self._semaphore:
            return await self.discover_leagues()

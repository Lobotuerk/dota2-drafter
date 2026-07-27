"""Match discovery engine - fetches match IDs from discovered leagues."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from dota2drafter.api.stratz_client import StratzClient
from dota2drafter.config import ConcurrencyConfig
from dota2drafter.state import StateDatabase

logger = logging.getLogger(__name__)


class MatchFinder:
    """Collects match IDs from discovered leagues and registers them in the state DB."""

    def __init__(
        self,
        stratz_client: StratzClient,
        state_db: StateDatabase,
        config: Any,
        concurrency: ConcurrencyConfig,
        batch_size: int = 1000,
    ) -> None:
        self._stratz = stratz_client
        self._state_db = state_db
        self._config = config
        self._semaphore = asyncio.Semaphore(concurrency.max_connections_per_host)
        self._batch_size = batch_size

    async def find_matches_for_league(self, league_id: str) -> list[tuple[str, str]]:
        """Fetch match IDs for a single league and register them in the state DB."""
        matches_data = await self._stratz.fetch_matches_by_league(league_id, self._config.patch, self._batch_size)
        new_matches = []

        for match_entry in matches_data:
            match_id = match_entry.get("id", "")
            if not match_id:
                continue

            # Only insert if not already present (avoid duplicates)
            status = "pending"
            new_matches.append((match_id, status, league_id))

        if new_matches:
            self._state_db.upsert_matches(new_matches)
            logger.info("Registered %d matches for league %s", len(new_matches), league_id)

       return new_matches

    async def find_all_matches(self, leagues: list[dict[str, Any]]) -> int:
        """Discover matches from all leagues concurrently."""
        tasks = [self.find_matches_for_league(league["id"]) for league in leagues]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error("Failed to discover matches: %s", result)
            elif isinstance(result, list):
                total += len(result)

        logger.info("Total new matches registered: %d", total)
        return total

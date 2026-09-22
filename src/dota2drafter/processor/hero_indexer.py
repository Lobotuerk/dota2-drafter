"""Hero indexer - maps API hero IDs to contiguous indices."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HeroIndexer:
    """Builds a bidirectional mapping between API hero IDs and contiguous indices."""

    def __init__(self) -> None:
        self._api_to_index: dict[int, int] = {}
        self._index_to_api: dict[int, int] = {}
        self._hero_count: int = 0

    def build_mapping(self, heroes: list[dict[str, Any]]) -> None:
        """Build the hero ID mapping from a list of hero data.

        Only playable heroes are included.
        Indices are 1-based: h in {1, ..., K}.
        """
        playable = [h for h in heroes if h.get("playable", False)]
        playable.sort(key=lambda h: int(h.get("id", 0)))

        self._api_to_index.clear()
        self._index_to_api.clear()

        for idx, hero in enumerate(playable, start=1):
            hero_id = hero.get("id")
            if hero_id is not None:
                hero_id_int = int(hero_id)
                self._api_to_index[hero_id_int] = idx
                self._index_to_api[idx] = hero_id_int

        self._hero_count = len(playable)
        logger.info("Built hero mapping: %d heroes (K=%d)", self._hero_count, self._hero_count)

    def map_hero_id(self, api_hero_id: int | str) -> int | None:
        """Map an API hero ID to a contiguous index.

        Returns None if the hero ID is not in the mapping.
        Raises ValueError or TypeError if api_hero_id cannot be converted to int.
        """
        return self._api_to_index.get(int(api_hero_id))

    def get_contiguous_count(self) -> int:
        """Return K, the number of playable heroes."""
        return self._hero_count

    def get_mapping(self) -> dict[int, int]:
        """Return the full API-to-index mapping."""
        return dict(self._api_to_index)

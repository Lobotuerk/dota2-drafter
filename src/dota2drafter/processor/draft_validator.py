"""Draft validator - filters matches by game mode and draft completeness."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# OpenDota game mode constants
GAME_MODE_CAPTAINS_MODE = 2
# STRATZ pool constants (if needed)
POOL_CAPTAINS_MODE = "captains"


class DraftValidator:
    """Validates match payloads for Captains Mode compliance."""

    def __init__(self, required_steps: int = 24) -> None:
        self._required_steps = required_steps

    def validate_opendota(self, match_data: dict[str, Any]) -> bool:
        """Validate an OpenDota match payload.

        Checks:
        - Game mode is Captains Mode (2).
        - picks_bans has exactly 24 steps.
        """
        if match_data.get("radiant_win") is None:
            logger.debug("Missing radiant_win in match data")
            return False

        game_mode = match_data.get("game_mode")
        if game_mode != GAME_MODE_CAPTAINS_MODE:
            logger.debug(
                "Not Captains Mode (game_mode=%s) for match %s",
                game_mode,
                match_data.get("match_id"),
            )
            return False

        picks_bans = match_data.get("picks_bans", [])
        if len(picks_bans) != self._required_steps:
            logger.debug(
                "Invalid draft length (%d steps, expected %d) for match %s",
                len(picks_bans),
                self._required_steps,
                match_data.get("match_id"),
            )
            return False

        return True

    def validate_stratz(self, match_data: dict[str, Any]) -> bool:
        """Validate a STRATZ match payload.

        Checks:
        - Draft pool is Captains Mode.
        - picksBans has exactly 24 steps.
        """
        draft = match_data.get("draft", {})
        picks_bans = draft.get("picksBans", [])

        if len(picks_bans) != self._required_steps:
            logger.debug(
                "Invalid STRATZ draft length (%d steps, expected %d) for match %s",
                len(picks_bans),
                self._required_steps,
                match_data.get("id"),
            )
            return False

        return True

    def validate(self, match_data: dict[str, Any], source: str = "stratz") -> bool:
        """Validate a match payload from the specified source."""
        if source == "stratz":
            return self.validate_stratz(match_data)
        return self.validate_opendota(match_data)

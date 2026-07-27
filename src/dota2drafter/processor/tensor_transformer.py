"""Tensor transformer - converts draft sequences to PyTorch tensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from dota2drafter.processor.draft_validator import DraftValidator
from dota2drafter.processor.hero_indexer import HeroIndexer

logger = logging.getLogger(__name__)


@dataclass
class ProcessedMatch:
    """A single processed match ready for tensor conversion."""

    x_tensor: torch.Tensor  # (24, 3) - draft sequence
    y_tensor: torch.Tensor  # (1,) - radiant_win label
    match_id: str


class TensorTransformer:
    """Converts raw draft data into PyTorch tensors."""

    def __init__(self, hero_indexer: HeroIndexer, validator: DraftValidator) -> None:
        self._hero_indexer = hero_indexer
        self._validator = validator

    def transform_stratz(self, match_data: dict[str, Any]) -> ProcessedMatch | None:
        """Transform a STRATZ match payload into tensors.

        Returns None if validation fails.
        """
        match_id = match_data.get("id", "")

        if not self._validator.validate_stratz(match_data):
            logger.debug("STRATZ match %s failed validation", match_id)
            return None

        radiant_win = match_data.get("radiantWin", False)
        draft = match_data.get("draft", {})
        picks_bans = draft.get("picksBans", [])

        steps: list[list[float]] = []
        for pb in picks_bans:
            is_pick = 1.0 if pb.get("type") == "pick" else 0.0
            team = float(pb.get("team", 0))
            hero_id = pb.get("hero", {}).get("id") if isinstance(pb.get("hero"), dict) else pb.get("hero")

            if hero_id is not None:
                hero_idx = self._hero_indexer.map_hero_id(int(hero_id))
                hero_val = float(hero_idx) if hero_idx is not None else -1.0
            else:
                hero_val = -1.0

            steps.append([is_pick, team, hero_val])

        x_tensor = torch.tensor(steps, dtype=torch.float32)  # (24, 3)
        y_tensor = torch.tensor([1.0 if radiant_win else 0.0], dtype=torch.float32)  # (1,)

        return ProcessedMatch(x_tensor=x_tensor, y_tensor=y_tensor, match_id=match_id)

    def transform_opendota(self, match_data: dict[str, Any]) -> ProcessedMatch | None:
        """Transform an OpenDota match payload into tensors.

        Returns None if validation fails.
        """
        match_id = str(match_data.get("match_id", ""))

        if not self._validator.validate_opendota(match_data):
            logger.debug("OpenDota match %s failed validation", match_id)
            return None

        radiant_win = match_data.get("radiant_win", False)
        picks_bans = match_data.get("picks_bans", [])

        steps: list[list[float]] = []
        for pb in picks_bans:
            is_pick = 1.0 if pb.get("is_pick", False) else 0.0
            team = float(pb.get("team", 0))
            hero_id = pb.get("hero_id")

            if hero_id is not None:
                hero_idx = self._hero_indexer.map_hero_id(int(hero_id))
                hero_val = float(hero_idx) if hero_idx is not None else -1.0
            else:
                hero_val = -1.0

            steps.append([is_pick, team, hero_val])

        x_tensor = torch.tensor(steps, dtype=torch.float32)  # (24, 3)
        y_tensor = torch.tensor([1.0 if radiant_win else 0.0], dtype=torch.float32)  # (1,)

        return ProcessedMatch(x_tensor=x_tensor, y_tensor=y_tensor, match_id=match_id)

    def transform(self, match_data: dict[str, Any], source: str = "stratz") -> ProcessedMatch | None:
        """Transform a match payload from the specified source."""
        if source == "stratz":
            return self.transform_stratz(match_data)
        return self.transform_opendota(match_data)

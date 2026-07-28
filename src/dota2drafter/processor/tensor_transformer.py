"""Tensor transformer - converts draft sequences to PyTorch tensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from dota2drafter.processor.draft_validator import DraftValidator
from dota2drafter.processor.hero_indexer import HeroIndexer

logger = logging.getLogger(__name__)


@dataclass
class ProcessedMatch:
    """A single processed match ready for tensor conversion."""

    x_tensor: torch.Tensor  # (24, 4) - draft sequence: [hero_val, is_pick, team, step_index]
    y_tensor: torch.Tensor  # (1,) - radiant_win label
    match_id: str
    radiant_players: list[int] = field(default_factory=list)
    dire_players: list[int] = field(default_factory=list)
    radiant_heroes: list[int] = field(default_factory=list)
    dire_heroes: list[int] = field(default_factory=list)
    player_comfort: torch.Tensor | None = None  # (B, 10, C) - optional comfort tensor


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

        # Extract player account IDs and hero IDs from STRATZ match data
        radiant_players, radiant_heroes = self._extract_stratz_players(match_data, team=0)
        dire_players, dire_heroes = self._extract_stratz_players(match_data, team=1)

        steps: list[list[float]] = []
        for step_idx, pb in enumerate(picks_bans):
            is_pick = 1.0 if pb.get("type") == "pick" else 0.0
            team = float(pb.get("team", 0))
            hero_id = pb.get("hero", {}).get("id") if isinstance(pb.get("hero"), dict) else pb.get("hero")

            if hero_id is not None:
                hero_idx = self._hero_indexer.map_hero_id(int(hero_id))
                hero_val = float(hero_idx) if hero_idx is not None else -1.0
            else:
                hero_val = -1.0

            # (24, 4): [hero_val, is_pick, team, step_index]
            steps.append([is_pick, team, hero_val, float(step_idx)])

        x_tensor = torch.tensor(steps, dtype=torch.float32)  # (24, 4)
        y_tensor = torch.tensor([1.0 if radiant_win else 0.0], dtype=torch.float32)  # (1,)

        return ProcessedMatch(
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            match_id=match_id,
            radiant_players=radiant_players,
            dire_players=dire_players,
            radiant_heroes=radiant_heroes,
            dire_heroes=dire_heroes,
        )

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

        # Extract player account IDs and hero IDs from OpenDota match data
        radiant_players, radiant_heroes = self._extract_opendota_players(match_data, team=0)
        dire_players, dire_heroes = self._extract_opendota_players(match_data, team=1)

        steps: list[list[float]] = []
        for step_idx, pb in enumerate(picks_bans):
            is_pick = 1.0 if pb.get("is_pick", False) else 0.0
            team = float(pb.get("team", 0))
            hero_id = pb.get("hero_id")

            if hero_id is not None:
                hero_idx = self._hero_indexer.map_hero_id(int(hero_id))
                hero_val = float(hero_idx) if hero_idx is not None else -1.0
            else:
                hero_val = -1.0

            # (24, 4): [hero_val, is_pick, team, step_index]
            steps.append([is_pick, team, hero_val, float(step_idx)])

        x_tensor = torch.tensor(steps, dtype=torch.float32)  # (24, 4)
        y_tensor = torch.tensor([1.0 if radiant_win else 0.0], dtype=torch.float32)  # (1,)

        return ProcessedMatch(
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            match_id=match_id,
            radiant_players=radiant_players,
            dire_players=dire_players,
            radiant_heroes=radiant_heroes,
            dire_heroes=dire_heroes,
        )

    def transform(self, match_data: dict[str, Any], source: str = "stratz") -> ProcessedMatch | None:
        """Transform a match payload from the specified source."""
        if source == "stratz":
            return self.transform_stratz(match_data)
        return self.transform_opendota(match_data)

    def _extract_stratz_players(self, match_data: dict[str, Any], team: int) -> tuple[list[int], list[int]]:
        """Extract account IDs and hero IDs for a team from STRATZ match data.

        Args:
            match_data: STRATZ match payload.
            team: Team identifier (0 = Radiant, 1 = Dire).

        Returns:
            Tuple of (account_ids, hero_ids) for the specified team.
        """
        players_key = "players"
        players = match_data.get(players_key, [])

        account_ids: list[int] = []
        hero_ids: list[int] = []
        for player in players:
            player_team = player.get("team", 0)
            if player_team == team + 1:  # STRATZ uses 1-based team (1=Radiant, 2=Dire)
                account_id = player.get("accountid")
                hero_id = player.get("hero_id")
                account_ids.append(int(account_id) if account_id is not None else 0)
                if hero_id is not None:
                    mapped = self._hero_indexer.map_hero_id(int(hero_id))
                    hero_ids.append(mapped if mapped is not None else -1)
                else:
                    hero_ids.append(-1)

        # Zero-pad to exactly 5 elements
        while len(account_ids) < 5:
            account_ids.append(0)
            hero_ids.append(-1)

        return account_ids, hero_ids

    def _extract_opendota_players(self, match_data: dict[str, Any], team: int) -> tuple[list[int], list[int]]:
        """Extract account IDs and hero IDs for a team from OpenDota match data.

        Args:
            match_data: OpenDota match payload.
            team: Team identifier (0 = Radiant, 1 = Dire).

        Returns:
            Tuple of (account_ids, hero_ids) for the specified team.
        """
        players = match_data.get("players", [])

        account_ids: list[int] = []
        hero_ids: list[int] = []
        for player in players:
            player_team = player.get("player_slot", 0)
            is_radiant = player_team < 128
            player_team_idx = 0 if is_radiant else 1

            if player_team_idx == team:
                account_id = player.get("account_id")
                hero_id = player.get("hero_id")
                account_ids.append(int(account_id) if account_id is not None else 0)
                if hero_id is not None:
                    mapped = self._hero_indexer.map_hero_id(int(hero_id))
                    hero_ids.append(mapped if mapped is not None else -1)
                else:
                    hero_ids.append(-1)

        # Zero-pad to exactly 5 elements
        while len(account_ids) < 5:
            account_ids.append(0)
            hero_ids.append(-1)

        return account_ids, hero_ids

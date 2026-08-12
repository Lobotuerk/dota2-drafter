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
    patch_id: int  # Added: ID representing the game patch version
    radiant_players: list[int] = field(default_factory=list)
    dire_players: list[int] = field(default_factory=list)
    radiant_heroes: list[int] = field(default_factory=list)
    dire_heroes: list[int] = field(default_factory=list)
    player_comfort: torch.Tensor | None = None  # (B, 10, C) - optional comfort tensor



PATCH_DATES = [
    (1747872000, 0),  # 7.39 - 2025-05-21
    (1748563200, 1),  # 7.39b - 2025-05-29
    (1750809600, 2),  # 7.39c - 2025-06-24
    (1754438400, 3),  # 7.39d - 2025-08-05
    (1759363200, 4),  # 7.39e - 2025-10-02
    (1765756800, 5),  # 7.40 - 2025-12-15
    (1766448000, 6),  # 7.40b - 2025-12-23
    (1768953600, 7),  # 7.40c - 2026-01-21
    (1774310400, 8),  # 7.41 - 2026-03-24
    (1774656000, 9),  # 7.41a - 2026-03-28
    (1775520000, 10), # 7.41b - 2026-04-07
    (1778025600, 11), # 7.41c - 2026-05-06
    (1780531200, 12), # 7.41d - 2026-06-04
    (1785369600, 13), # 7.41e - 2026-07-30
]

def get_patch_id(timestamp: int | None) -> int:
    """Return the patch ID based on the Unix timestamp. Default to latest if unknown."""
    if not timestamp:
        return PATCH_DATES[-1][1]
    
    # Iterate backwards to find the first patch date before the timestamp
    for ts, pid in reversed(PATCH_DATES):
        if timestamp >= ts:
            return pid
    return PATCH_DATES[0][1]

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
            is_pick = 1.0 if pb.get("isPick") else 0.0
            team = float(0) if pb.get("isRadiant") else float(1)
            hero_id = pb.get("heroId")

            if hero_id is not None:
                hero_idx = self._hero_indexer.map_hero_id(int(hero_id))
                hero_val = float(hero_idx) if hero_idx is not None else -1.0
            else:
                hero_val = -1.0

            # (24, 4): [hero_val, is_pick, team, step_index]
            steps.append([is_pick, team, hero_val, float(step_idx)])

        x_tensor = torch.tensor(steps, dtype=torch.float32)  # (24, 4)
        y_tensor = torch.tensor([1.0 if radiant_win else 0.0], dtype=torch.float32)  # (1,)

        # Extract timestamp, Stratz doesn't typically provide a clean start_time in the basic query
        # so we fallback to latest patch. (In a full implementation, you'd add startDateTime to the GraphQL query)
        patch_id = get_patch_id(match_data.get("startDateTime"))

        return ProcessedMatch(
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            match_id=match_id,
            patch_id=patch_id,
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

        patch_id = get_patch_id(match_data.get("start_time"))

        return ProcessedMatch(
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            match_id=match_id,
            patch_id=patch_id,
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
                hero_id = player.get("heroId")
                account_ids.append(int(account_id) if account_id is not None else 0)
                if hero_id is not None:
                    # Stratz hero IDs might come in as strings or ints, and they map differently sometimes
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

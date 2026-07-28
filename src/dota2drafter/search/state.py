"""Draft state and move for MCTS-based draft decision support.

Provides DraftMove and DraftState classes that represent individual
draft actions and the evolving partial-draft tree state. The DraftState
implements the MCTS interface: it generates valid actions, supports
state expansion, evaluates rollouts via MatchNetwork, and computes
comfort-scaled prior probabilities for PUCT selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from dota2drafter.processor.hero_indexer import HeroIndexer

logger = logging.getLogger(__name__)

# Standard Dota 2 Captains Mode draft schedule (24 steps).
# Each tuple: (action_type, team) where action_type is 'ban' or 'pick',
# and team is 0 (Radiant) or 1 (Dire).
DRAFT_SCHEDULE: list[tuple[str, int]] = [
    ("ban", 1),   # 0
    ("ban", 0),   # 1
    ("pick", 0),  # 2
    ("pick", 1),  # 3
    ("ban", 1),   # 4
    ("ban", 0),   # 5
    ("pick", 1),  # 6
    ("pick", 0),  # 7
    ("ban", 0),   # 8
    ("ban", 1),   # 9
    ("pick", 1),  # 10
    ("pick", 0),  # 11
    ("ban", 1),   # 12
    ("ban", 0),   # 13
    ("pick", 0),  # 14
    ("pick", 1),  # 15
    ("ban", 0),   # 16
    ("ban", 1),   # 17
    ("pick", 0),  # 18
    ("pick", 1),  # 19
    ("pick", 1),  # 20
    ("pick", 1),  # 21
    ("pick", 0),  # 22
    ("pick", 0),  # 23
]

# Zeroed dummy step for padding rollouts to 24 steps.
_ZERO_STEP = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)


@dataclass
class DraftMove:
    """A single draft action (pick or ban) in the MCTS tree.

    Attributes:
        hero_id: Hero index (1-based, from HeroIndexer). 0 for dummy/padding.
        is_pick: True for a pick, False for a ban.
        team: Team making the move (0 = Radiant, 1 = Dire).
        step_index: Position in the 24-step draft schedule (0-23).
    """

    hero_id: int
    is_pick: bool
    team: int
    step_index: int

    def __eq__(self, other: object) -> bool:
        """Check equality by hero_id, action type, and team at same step."""
        if not isinstance(other, DraftMove):
            return False
        return (
            self.hero_id == other.hero_id
            and self.is_pick == other.is_pick
            and self.team == other.team
            and self.step_index == other.step_index
        )

    def __hash__(self) -> int:
        """Hash by hero_id, is_pick, team, and step_index."""
        return hash((self.hero_id, self.is_pick, self.team, self.step_index))

    def __str__(self) -> str:
        """Human-readable representation."""
        action = "pick" if self.is_pick else "ban"
        team_str = "Radiant" if self.team == 0 else "Dire"
        return f"{team_str} {action} hero {self.hero_id} at step {self.step_index}"


def _build_tensor_from_moves(actions: list[DraftMove], num_heroes: int) -> torch.Tensor:
    """Convert a list of DraftMove objects to a (24, 4) tensor.

    Steps not yet taken are zeroed. The tensor format is:
    [is_pick, team, hero_index, step_index]

    Args:
        actions: List of DraftMove objects (may be fewer than 24).
        num_heroes: Number of heroes for validation.

    Returns:
        Tensor of shape (24, 4).
    """
    steps: list[torch.Tensor] = [_ZERO_STEP.clone() for _ in range(24)]
    for move in actions:
        is_pick_val = 1.0 if move.is_pick else 0.0
        hero_val = float(move.hero_id) if move.hero_id > 0 else -1.0
        steps[move.step_index] = torch.tensor(
            [is_pick_val, float(move.team), hero_val, float(move.step_index)],
            dtype=torch.float32,
        )
    return torch.stack(steps)


class DraftState:
    """MCTS state representing a partial draft tree node.

    Wraps a sequence of draft moves and provides the MCTS interface:
    generating valid child actions, expanding the state, evaluating
    rollouts via MatchNetwork, and computing comfort-scaled priors.

    Attributes:
        actions: List of DraftMove objects applied so far.
        model: MatchNetwork for win-probability evaluation.
        comfort_matrix: Player comfort tensor of shape (10, C).
        active_team: Team id (0 or 1) being optimized at the root.
        hero_indexer: HeroIndexer for hero ID management.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        comfort_matrix: torch.Tensor,
        active_team: int = 0,
        hero_indexer: HeroIndexer | None = None,
        initial_actions: list[DraftMove] | None = None,
    ) -> None:
        """Initialize a draft state for MCTS search.

        Args:
            model: MatchNetwork (or HierarchicalTransformer) for evaluation.
            comfort_matrix: Tensor of shape (10, C) with player comfort metrics.
            active_team: Team being optimized (0 = Radiant, 1 = Dire).
            hero_indexer: HeroIndexer for hero ID management.
            initial_actions: Pre-applied moves (for continuing from partial state).
        """
        self.model = model
        self.comfort_matrix = comfort_matrix
        self.active_team = active_team
        self.hero_indexer = hero_indexer
        self.actions: list[DraftMove] = initial_actions if initial_actions is not None else []
        self._num_heroes = hero_indexer.get_contiguous_count() if hero_indexer else 120

    # ------------------------------------------------------------------
    # MCTS interface methods
    # ------------------------------------------------------------------

    def actions_to_try(self) -> list[DraftMove]:
        """Yield all valid DraftMove candidates for the current step.

        Filters out heroes that have already been picked or banned.
        Only returns moves matching the current schedule entry.

        Returns:
            List of valid DraftMove objects.
        """
        step_idx = len(self.actions)
        if step_idx >= 24:
            return []

        schedule_action, schedule_team = DRAFT_SCHEDULE[step_idx]

        # Collect already-used heroes
        used_heroes: set[int] = set()
        for move in self.actions:
            if move.hero_id > 0:
                used_heroes.add(move.hero_id)

        valid_moves: list[DraftMove] = []
        if schedule_action == "ban":
            # For bans, suggest all unpicked heroes (up to a reasonable limit)
            for hero_idx in range(1, self._num_heroes + 1):
                if hero_idx not in used_heroes:
                    valid_moves.append(
                        DraftMove(
                            hero_id=hero_idx,
                            is_pick=False,
                            team=schedule_team,
                            step_index=step_idx,
                        )
                    )
            # Limit to top 20 unpicked heroes for performance
            if len(valid_moves) > 20:
                valid_moves = valid_moves[:20]
        else:
            # For picks, suggest all unpicked heroes (up to a reasonable limit)
            for hero_idx in range(1, self._num_heroes + 1):
                if hero_idx not in used_heroes:
                    valid_moves.append(
                        DraftMove(
                            hero_id=hero_idx,
                            is_pick=True,
                            team=schedule_team,
                            step_index=step_idx,
                        )
                    )
            # Limit to top 20 unpicked heroes for performance
            if len(valid_moves) > 20:
                valid_moves = valid_moves[:20]

        return valid_moves

    def next_state(self, move: DraftMove) -> DraftState:
        """Create a new DraftState with the move applied.

        Args:
            move: The DraftMove to apply.

        Returns:
            A new DraftState with the move appended to actions.
        """
        new_actions = self.actions + [move]
        return DraftState(
            model=self.model,
            comfort_matrix=self.comfort_matrix,
            active_team=self.active_team,
            hero_indexer=self.hero_indexer,
            initial_actions=new_actions,
        )

    def is_terminal(self) -> bool:
        """Check if the draft is complete (24 steps)."""
        return len(self.actions) >= 24

    def is_self_side_turn(self) -> bool:
        """Check if the current step belongs to the active team.

        Returns:
            True if the team optimizing at the root is taking the next turn.
        """
        step_idx = len(self.actions)
        if step_idx >= 24:
            return False
        _, team = DRAFT_SCHEDULE[step_idx]
        return team == self.active_team

    def rollout(self) -> float:
        """Evaluate the current partial draft via rollout.

        Pads the sequence to 24 steps with zeroed dummy moves, then
        evaluates via MatchNetwork.predict_proba(). Returns the win
        probability for the active team.

        Returns:
            Float win probability in [0, 1].
        """
        tensor = _build_tensor_from_moves(self.actions, self._num_heroes)

        # Pad to 24 steps if needed
        if tensor.shape[0] < 24:
            padding = torch.stack([_ZERO_STEP.clone() for _ in range(24 - tensor.shape[0])])
            tensor = torch.cat([tensor, padding], dim=0)

        # Expand to batch of 1
        tensor = tensor.unsqueeze(0)  # (1, 24, 4)

        # Expand comfort matrix to batch of 1
        comfort = self.comfort_matrix.unsqueeze(0)  # (1, 10, C)

        self.model.eval()
        with torch.no_grad():
            win_prob = self.model.predict_proba(tensor, comfort)

        radiant_win_prob = win_prob.item()

        # Return win probability for the active team
        if self.active_team == 0:
            return radiant_win_prob
        else:
            return 1.0 - radiant_win_prob

    def get_action_probabilities(self) -> dict[int, float]:
        """Compute comfort-scaled prior probabilities for all valid actions.

        For each valid action a, constructs the extended draft sequence,
        evaluates P(Win | S + a) via MatchNetwork in a single GPU batch,
        then scales by the player comfort matrix:
            P'(a|S) = Softmax(Logits(P) * W_comfort)

        Returns:
            Dict mapping hero_id to scaled prior probability.
        """
        valid_moves = self.actions_to_try()
        if not valid_moves:
            return {}

        # Build extended sequences for all valid actions
        sequences: list[torch.Tensor] = []
        hero_ids: list[int] = []
        for move in valid_moves:
            extended = self.actions + [move]
            tensor = _build_tensor_from_moves(extended, self._num_heroes)
            sequences.append(tensor)
            hero_ids.append(move.hero_id)

        # Stack into batch
        batch = torch.stack(sequences)  # (N, 24, 4)
        comfort = self.comfort_matrix.unsqueeze(0)  # (1, 10, C)

        self.model.eval()
        with torch.no_grad():
            win_probs = self.model.predict_proba(batch, comfort)  # (N,)

        # Convert to radiant win probabilities
        radiant_probs = win_probs.cpu().numpy()

        # Adjust for active team
        if self.active_team == 0:
            team_probs = radiant_probs
        else:
            team_probs = 1.0 - radiant_probs

        # Comfort scaling: scale logits by comfort matrix
        comfort_scaled = self._apply_comfort_scaling(team_probs, valid_moves)

        # Softmax to get prior probabilities
        log_probs = torch.tensor(comfort_scaled, dtype=torch.float32)
        log_probs = log_probs - log_probs.max()  # numerical stability
        priors = torch.exp(log_probs)
        priors = priors / priors.sum()

        result: dict[int, float] = {}
        for hero_id, prior in zip(hero_ids, priors.tolist()):
            result[hero_id] = prior

        return result

    def _apply_comfort_scaling(
        self, probs: np.ndarray, moves: list[DraftMove]
    ) -> list[float]:
        """Apply comfort matrix scaling to action probabilities.

        Scales P(Win | S + a) by the comfort matrix W_comfort for the
        team making the pick/ban.

        Args:
            probs: Raw win probabilities per action.
            moves: Corresponding draft moves.

        Returns:
            Scaled log-probabilities.
        """

        scaled: list[float] = []

        for i, move in enumerate(moves):
            if not move.is_pick:
                # For bans, use a uniform scaling (no comfort penalty)
                scaled.append(float(probs[i]))
                continue

            # For picks, scale by the comfort of the picking player
            # The player index within the team depends on the step
            player_idx = self._get_player_index_for_step(move.step_index, move.team)
            if player_idx is not None and player_idx < self.comfort_matrix.shape[0]:
                comfort_weight = self.comfort_matrix[player_idx].mean().item()
                raw_prob = float(probs[i])
                # Blend: weighted combination of raw prob and comfort
                scaled_val = raw_prob * (0.5 + 0.5 * comfort_weight)
                scaled.append(scaled_val)
            else:
                scaled.append(float(probs[i]))

        return scaled

    def _get_player_index_for_step(self, step_index: int, team: int) -> int | None:
        """Determine which player index on the team is picking at this step.

        Args:
            step_index: Current step in the draft schedule (0-23).
            team: Team making the pick (0 or 1).

        Returns:
            Player index (0-4) or None if not a pick step.
        """
        if step_index >= 24:
            return None

        # Count how many picks this team has made before this step
        pick_count = 0
        for s in range(step_index):
            action_type, s_team = DRAFT_SCHEDULE[s]
            if action_type == "pick" and s_team == team:
                pick_count += 1

        return pick_count

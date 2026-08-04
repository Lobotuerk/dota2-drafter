"""Draft state and move for MCTS-based draft decision support.

Provides DraftMove and DraftState classes that inherit from pymcts
C++ bindings (MCTS_move and MCTS_state) to enable high-performance
adversarial minimax lookahead via the MonteCarloTreeSearch library.

The DraftState implements the MCTS interface: it generates valid actions,
supports state expansion, evaluates rollouts via MatchNetwork, and
computes comfort-scaled prior probabilities for PUCT selection.
"""

from __future__ import annotations

import logging

import pymcts
import torch

from dota2drafter.processor.hero_indexer import HeroIndexer

logger = logging.getLogger(__name__)

# Standard Dota 2 Captains Mode draft schedule (24 steps).
# Each tuple: (action_type, team) where action_type is 'ban' or 'pick',
# and team is 0 (Radiant) or 1 (Dire).
DRAFT_SCHEDULE: list[tuple[str, int]] = [
    ("ban", 1),   # 0
    ("ban", 1),   # 1
    ("ban", 0),  # 2
    ("ban", 0),  # 3
    ("ban", 1),   # 4
    ("ban", 0),   # 5
    ("ban", 0),  # 6
    ("pick", 1),  # 7
    ("pick", 0),   # 8
    ("ban", 1),   # 9
    ("ban", 1),  # 10
    ("ban", 0),  # 11
    ("pick", 0),   # 12
    ("pick", 1),   # 13
    ("pick", 1),  # 14
    ("pick", 0),  # 15
    ("pick", 0),   # 16
    ("pick", 1),   # 17
    ("ban", 1),  # 18
    ("ban", 0),  # 19
    ("ban", 1),  # 20
    ("ban", 0),  # 21
    ("pick", 1),  # 22
    ("pick", 0),  # 23
]

# Zeroed dummy step for padding rollouts to 24 steps.
_ZERO_STEP = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)


class DraftMove(pymcts.MCTS_move):
    """A single draft action (pick or ban) in the MCTS tree.

    Inherits from pymcts.MCTS_move to integrate with the C++ MCTS engine.

    Attributes:
        hero_id: Hero index (1-based, from HeroIndexer). 0 for dummy/padding.
        is_pick: True for a pick, False for a ban.
        team: Team making the move (0 = Radiant, 1 = Dire).
        step_index: Position in the 24-step draft schedule (0-23).
    """

    def __init__(
        self,
        hero_id: int,
        is_pick: bool,
        team: int,
        step_index: int,
    ) -> None:
        """Initialize a draft move.

        Args:
            hero_id: Hero index (1-based).
            is_pick: True for pick, False for ban.
            team: Team making the move (0 or 1).
            step_index: Position in the draft schedule (0-23).
        """
        super().__init__()
        self.hero_id = hero_id
        self.is_pick = is_pick
        self.team = team
        self.step_index = step_index

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

    def sprint(self) -> str:
        """Human-readable string representation for C++ MCTS.

        Returns:
            Formatted string describing this move.
        """
        action = "pick" if self.is_pick else "ban"
        team_str = "Radiant" if self.team == 0 else "Dire"
        return f"{team_str} {action} hero {self.hero_id} at step {self.step_index}"

    def __hash__(self) -> int:
        """Hash by hero_id, is_pick, team, and step_index."""
        return hash((self.hero_id, self.is_pick, self.team, self.step_index))

    def to_numpy(self) -> list[float]:
        """Convert move to numpy-compatible array.

        Returns:
            List of floats: [hero_id, is_pick, team, step_index].
        """
        return [
            float(self.hero_id),
            1.0 if self.is_pick else 0.0,
            float(self.team),
            float(self.step_index),
        ]

    def to_env_action(self) -> list[int]:
        """Convert move to environment action format.

        Returns:
            List of ints: [hero_id, is_pick, team, step_index].
        """
        return [
            self.hero_id,
            1 if self.is_pick else 0,
            self.team,
            self.step_index,
        ]


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


class DraftState(pymcts.MCTS_state):
    """MCTS state representing a partial draft tree node.

    Inherits from pymcts.MCTS_state to integrate with the C++ MCTS engine.
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
        super().__init__()
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
        """Return all valid DraftMove candidates for the current step.

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

    def clone(self) -> DraftState:
        """Create a deep copy of this state for MCTS tree expansion.

        Returns:
            A new DraftState with identical attributes.
        """
        return DraftState(
            model=self.model,
            comfort_matrix=self.comfort_matrix,
            active_team=self.active_team,
            hero_indexer=self.hero_indexer,
            initial_actions=list(self.actions),
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

    def print(self) -> None:
        """Print the current draft state for debugging."""
        print(f"DraftState: {len(self.actions)}/24 steps, active_team={self.active_team}")

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

        device = torch.device("cpu")
        if hasattr(self.model, "parameters") and "Mock" not in type(self.model).__name__:
            try:
                model_device = next(self.model.parameters()).device
                if isinstance(model_device, (torch.device, str)) and "Mock" not in type(model_device).__name__:
                    device = model_device
            except (StopIteration, AttributeError):
                pass
        tensor = tensor.to(device)
        comfort = comfort.to(device)

        self.model.eval()
        with torch.no_grad():
            win_prob = self.model.predict_proba(tensor, comfort)

        radiant_win_prob = win_prob.item()

        # Return win probability for the active team
        if self.active_team == 0:
            return radiant_win_prob
        else:
            return 1.0 - radiant_win_prob

    def get_action_probabilities(self) -> list[float]:
        """Compute comfort-scaled prior probabilities for all valid actions.

        For each valid action a, constructs the extended draft sequence,
        evaluates raw logits via MatchNetwork.forward() in a single GPU batch,
        then scales the logits by the player comfort matrix:
            P'(a|S) = Softmax(Logits(P(a|S)) * W_comfort)

        Returns:
            List of prior probabilities (one per valid action), sorted
            in the same order as actions_to_try().
        """
        valid_moves = self.actions_to_try()
        if not valid_moves:
            return []

        # Build extended sequences for all valid actions
        sequences: list[torch.Tensor] = []
        for move in valid_moves:
            extended = self.actions + [move]
            tensor = _build_tensor_from_moves(extended, self._num_heroes)
            sequences.append(tensor)

        # Stack into batch
        batch = torch.stack(sequences)  # (N, 24, 4)
        N = batch.size(0)
        comfort = self.comfort_matrix.unsqueeze(0).expand(N, -1, -1)  # (N, 10, C)

        device = torch.device("cpu")
        if hasattr(self.model, "parameters") and "Mock" not in type(self.model).__name__:
            try:
                model_device = next(self.model.parameters()).device
                if isinstance(model_device, (torch.device, str)) and "Mock" not in type(model_device).__name__:
                    device = model_device
            except (StopIteration, AttributeError):
                pass
        batch = batch.to(device)
        comfort = comfort.to(device)

        self.model.eval()
        with torch.no_grad():
            # Use forward() to get raw logits (no sigmoid)
            logits = self.model.forward(batch, comfort)  # (N,)

        # Adjust logits for active team
        if self.active_team == 1:
            # For Dire, flip the logits (Win for Dire = 1 - Win for Radiant)
            logits = -logits

        # Apply comfort scaling to logits
        comfort_scaled = self._apply_comfort_scaling_to_logits(logits, valid_moves)

        # Softmax to get prior probabilities
        log_probs = comfort_scaled - comfort_scaled.max()  # numerical stability
        priors = torch.exp(log_probs)
        priors = priors / priors.sum()

        return priors.tolist()

    def _apply_comfort_scaling_to_logits(
        self, logits: torch.Tensor, moves: list[DraftMove]
    ) -> torch.Tensor:
        """Apply comfort matrix scaling to action logits.

        Implements P'(a|S) = Softmax(Logits(P(a|S)) * W_comfort).
        For picks, scales logits by the comfort weight of the picking player.
        For bans, leaves logits unchanged.

        Args:
            logits: Raw logits from MatchNetwork.forward().
            moves: Corresponding draft moves.

        Returns:
            Scaled logits ready for softmax.
        """
        scaled = logits.clone()

        for i, move in enumerate(moves):
            if not move.is_pick:
                # For bans, no comfort scaling
                continue

            # For picks, scale by the comfort of the picking player
            player_idx = self._get_player_index_for_step(move.step_index, move.team)
            if player_idx is not None and player_idx < self.comfort_matrix.shape[0]:
                comfort_weight = self.comfort_matrix[player_idx].mean().item()
                # Scale logits by comfort weight
                scaled[i] = logits[i] * comfort_weight

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

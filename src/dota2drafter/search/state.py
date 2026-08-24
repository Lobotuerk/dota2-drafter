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
        super().__init__()
        self.hero_id = hero_id
        self.is_pick = is_pick
        self.team = team
        self.step_index = step_index
        action = "pick" if self.is_pick else "ban"
        team_str = "Radiant" if self.team == 0 else "Dire"
        self._sprint_cache = f"{team_str} {action} hero {self.hero_id} at step {self.step_index}"
        self._hash_cache = hash((self.hero_id, self.is_pick, self.team, self.step_index))

    def sprint(self) -> str:
        return self._sprint_cache

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
    tensor = torch.zeros((24, 4), dtype=torch.float32)
    
    # CRITICAL: Empty/padding steps must have hero_id = -1.0 so the network routes it 
    # to the dedicated padding embedding (index 127), exactly as it was trained!
    tensor[:, 2] = -1.0 
    
    for move in actions:
        tensor[move.step_index, 0] = 1.0 if move.is_pick else 0.0
        tensor[move.step_index, 1] = float(move.team)
        tensor[move.step_index, 2] = float(move.hero_id) if move.hero_id > 0 else -1.0
        tensor[move.step_index, 3] = float(move.step_index)
    return tensor


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
        max_candidates: int = 20,
    ) -> None:
        """Initialize a draft state for MCTS search.

        Args:
            model: MatchNetwork (or HierarchicalTransformer) for evaluation.
            comfort_matrix: Tensor of shape (10, C) with player comfort metrics.
            active_team: Team being optimized (0 = Radiant, 1 = Dire).
            hero_indexer: HeroIndexer for hero ID management.
            initial_actions: Pre-applied moves (for continuing from partial state).
            max_candidates: Maximum number of candidate moves to evaluate and return.
        """
        super().__init__()
        self.model = model
        self.comfort_matrix = comfort_matrix
        self.active_team = active_team
        self.hero_indexer = hero_indexer
        self.actions: list[DraftMove] = initial_actions if initial_actions is not None else []
        self._num_heroes = hero_indexer.get_contiguous_count() if hero_indexer else 120
        self.max_candidates = max_candidates
        self._cached_valid_moves: list[DraftMove] | None = None
        self._cached_priors: list[float] | None = None

    # ------------------------------------------------------------------
    # MCTS interface methods
    # ------------------------------------------------------------------

    def _evaluate_and_prune_moves(
        self,
    ) -> tuple[list[DraftMove], list[float]]:
        """Evaluate all valid moves via a single batched forward pass and prune to top candidates.

        Generates all unpicked/unbanned candidate moves, runs a single
        model.forward(batch, comfort) to get raw logits, applies comfort
        scaling, selects the top ``max_candidates`` based on the schedule
        team's perspective, and computes PUCT priors for the active team.

        Returns:
            Tuple of (top moves, prior probabilities).
        """
        step_idx = len(self.actions)
        if step_idx >= 24:
            return ([], [])

        schedule_action, schedule_team = DRAFT_SCHEDULE[step_idx]

        # Collect already-used heroes
        used_heroes: set[int] = set()
        for move in self.actions:
            if move.hero_id > 0:
                used_heroes.add(move.hero_id)

        # Generate all valid candidate moves
        valid_moves: list[DraftMove] = []
        for hero_idx in range(1, self._num_heroes + 1):
            if hero_idx not in used_heroes:
                valid_moves.append(
                    DraftMove(
                        hero_id=hero_idx,
                        is_pick=(schedule_action == "pick"),
                        team=schedule_team,
                        step_index=step_idx,
                    )
                )

        if not valid_moves:
            return ([], [])

        # Build batch tensor for all candidates (Vectorized)
        base_tensor = _build_tensor_from_moves(self.actions, self._num_heroes)
        num_seqs = len(valid_moves)

        batch = base_tensor.unsqueeze(0).expand(num_seqs, 24, 4).clone()
        for i, move in enumerate(valid_moves):
            batch[i, step_idx, 0] = 1.0 if move.is_pick else 0.0
            batch[i, step_idx, 1] = float(move.team)
            batch[i, step_idx, 2] = float(move.hero_id)
            batch[i, step_idx, 3] = float(step_idx)

        comfort = self.comfort_matrix.unsqueeze(0).expand(num_seqs, -1, -1)

        device = torch.device("cpu")
        if hasattr(self.model, "parameters"):
            try:
                model_device = next(self.model.parameters()).device
                if isinstance(model_device, (torch.device, str)):
                    device = model_device
            except (StopIteration, AttributeError):
                pass
        batch = batch.to(device)
        comfort = comfort.to(device)

        self.model.eval()
        with torch.no_grad():
            raw_logits = self.model.forward(batch, comfort)

        # Determine perspective from schedule team
        if schedule_team == 1:
            sort_logits = -raw_logits
        else:
            sort_logits = raw_logits.clone()

        # Apply comfort scaling and sort descending
        sort_scores = self._apply_comfort_scaling_to_logits(sort_logits, valid_moves)
        _, top_indices = torch.sort(sort_scores, descending=True)
        top_indices = top_indices[: self.max_candidates]

        top_moves = [valid_moves[i] for i in top_indices]
        top_raw_logits = raw_logits[top_indices]

        # Compute PUCT priors from top_raw_logits for the active team
        if self.active_team == 1:
            active_logits = -top_raw_logits
        else:
            active_logits = top_raw_logits.clone()

        active_scores = self._apply_comfort_scaling_to_logits(active_logits, top_moves)
        log_probs = active_scores - active_scores.max()
        priors = torch.exp(log_probs) / torch.exp(log_probs).sum()

        return (top_moves, priors.tolist())

    def actions_to_try(self) -> list[DraftMove]:
        if self._cached_valid_moves is not None:
            return self._cached_valid_moves
        
        step_idx = len(self.actions)
        if step_idx >= 24:
            self._cached_valid_moves = []
            return []
            
        schedule_action, schedule_team = DRAFT_SCHEDULE[step_idx]
        used_heroes = {m.hero_id for m in self.actions if m.hero_id > 0}
        
        valid_moves = []
        for hero_idx in range(1, self._num_heroes + 1):
            if hero_idx not in used_heroes:
                valid_moves.append(
                    DraftMove(
                        hero_id=hero_idx,
                        is_pick=(schedule_action == "pick"),
                        team=schedule_team,
                        step_index=step_idx,
                    )
                )
        self._cached_valid_moves = valid_moves
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
            max_candidates=self.max_candidates,
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
            max_candidates=self.max_candidates,
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
                is_device_str = isinstance(model_device, (torch.device, str))
                if is_device_str and "Mock" not in type(model_device).__name__:
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
        if self._cached_priors is None:
            self.evaluate_batch([self])
        return self._cached_priors

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
                scaled[i] = logits[i] + comfort_weight

        return scaled

    def _resolve_model_device(self) -> torch.device:
        """Resolve the device of the model for tensor placement.

        Returns:
            torch.device for model inference.
        """
        device = torch.device("cpu")
        if hasattr(self.model, "parameters"):
            try:
                model_device = next(self.model.parameters()).device
                if isinstance(model_device, (torch.device, str)):
                    device = model_device
            except (StopIteration, AttributeError, TypeError):
                pass
        return device

    def evaluate_batch(self, states: list[DraftState]) -> list[tuple[float, list[float]]]:
        if not states:
            return []

        M = len(states)
        K = self._num_heroes
        device = self._resolve_model_device()
        self.model.eval()

        # 1. Base batch
        draft_tensors = [_build_tensor_from_moves(s.actions, K) for s in states]
        base_batch = torch.stack(draft_tensors, dim=0).to(device)
        comfort_base = self.comfort_matrix.unsqueeze(0).expand(M, -1, -1).to(device)

        with torch.no_grad():
            win_probs = self.model.predict_proba(base_batch, comfort_base)

        # 2. Vectorized child setup
        valid_mask = torch.ones((M, K), dtype=torch.bool, device=device)
        
        step_indices = [len(s.actions) for s in states]
        
        for i, s in enumerate(states):
            step_idx = step_indices[i]
            if step_idx >= 24:
                valid_mask[i, :] = False
                continue
                
            used_heroes = {m.hero_id for m in s.actions if m.hero_id > 0}
            for hero_id in used_heroes:
                valid_mask[i, hero_id - 1] = False


        
        # 3. AlphaZero-Style MCTS Evaluation
        # Instead of doing 127 forward passes of win-probability to get the priors (Value),
        # we evaluate the MLM head (Policy) on the single base_batch to get the true priors instantly!
        with torch.no_grad():
            # Run the base states through the MLM head to get Policy Logits for the next step
            # mlm_logits shape: (M, 24, num_heroes + 1)
            # The base_batch at step_idx contains our padding token (-1.0), so the MLM 
            # will explicitly try to predict which hero belongs in that empty slot!
            mlm_logits = self.model(base_batch, comfort_base, mlm_mode=True)
            
            # Extract the specific logits for the exact step we are trying to predict
            policy_logits = torch.zeros(M, K, device=device)
            for i, s in enumerate(states):
                step_idx = step_indices[i]
                if step_idx < 24:
                    # We slice 1:K+1 because index 0 is reserved/padding in the MLM vocabulary
                    policy_logits[i] = mlm_logits[i, step_idx, 1:K+1]

        # The Policy Logits represent P(a|s).
        # Because we passed comfort_base into the model above, the network's internal cross-attention
        # ALREADY dynamically adjusts policy_logits based on the Affinity and Wilson Scores of the players!
        sort_scores = policy_logits.clone()
        sort_scores = sort_scores.masked_fill(~valid_mask, float('-inf'))
        
        # Since we use the Policy for the Prior directly, we don't need perspective_mult 
        # (the MLM naturally predicts the correct hero for the team whose turn it is).
        active_scores = sort_scores.clone()

        # To slice to max_candidates, we set all non-topk to -inf
        max_c = min(s.max_candidates for s in states) if states else 20
        max_c = min(max_c, K)
        
        _, top_indices = torch.topk(sort_scores, k=max_c, dim=1)
        topk_mask = torch.zeros((M, K), dtype=torch.bool, device=device)
        topk_mask.scatter_(1, top_indices, True)
        
        active_scores = active_scores.masked_fill(~topk_mask, float('-inf'))
        
        log_probs = active_scores - active_scores.max(dim=1, keepdim=True).values
        is_invalid = active_scores == float('-inf')
        log_probs = log_probs.masked_fill(is_invalid, float('-inf'))
        
        priors_tensor = torch.exp(log_probs)
        priors_sum = priors_tensor.sum(dim=1, keepdim=True)
        priors_tensor = priors_tensor / priors_sum.clamp(min=1e-9)
        
        # CRITICAL FIX: PyMCTS explores nodes with P=0.0 because of the PUCT formula.
        # We MUST assign a massive negative prior to completely block exploration of non-top-k branches!
        priors_tensor = priors_tensor.masked_fill(is_invalid, -1e9)
        
        # 4. Write back to states and return
        results = []
        win_probs_cpu = win_probs.cpu().tolist()
        priors_cpu = priors_tensor.cpu().tolist()
        
        for i, s in enumerate(states):
            radiant = win_probs_cpu[i]
            value = radiant if s.active_team == 0 else 1.0 - radiant
            
            step_idx = step_indices[i]
            if step_idx >= 24:
                s._cached_priors = []
                results.append((value, []))
                continue
                
            valid_priors = []
            valid_moves = s.actions_to_try()
            priors_list = priors_cpu[i]
            for m in valid_moves:
                hero_idx = m.hero_id - 1
                valid_priors.append(priors_list[hero_idx])
                
            s._cached_priors = valid_priors
            results.append((value, valid_priors))
            
        return results

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

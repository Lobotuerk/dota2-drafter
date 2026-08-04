# Technical Specification: AUT-20 - MCTS valid moves

## 1. Context and Goals
Currently, the `DraftState.actions_to_try()` method naïvely truncates the list of valid draft moves (bans and picks) by their hero index (e.g., `valid_moves[:20]`) to restrict the branching factor of the Monte Carlo Tree Search. This static truncation ignores the neural network's evaluation of the moves, leading to suboptimal exploration.

The goal is to dynamically select the `max_candidates` (default 20) best moves using a neural network forward pass. To do this efficiently, we must:
1. Generate all unpicked/unbanned candidates.
2. Evaluate them in a single batched forward pass (`model.forward`).
3. Apply comfort scaling to the raw logits.
4. Select the top `max_candidates` based on the current `schedule_team`'s perspective (maximize win probability for the schedule team).
5. Cache the resulting top moves and their calculated prior probabilities so `get_action_probabilities()` can reuse them without an additional forward pass.

## 2. Design Philosophy
- **Modular Depth / Clean Interfaces**: Introduce a shared private helper `_evaluate_and_prune_moves()` inside `DraftState` to encapsulate batch construction, forward pass, perspective adjustment, and caching. The public interfaces `actions_to_try()` and `get_action_probabilities()` become thin accessors of this cached state.
- **Fail Fast**: Do not implement silent fallbacks if the tensor shapes mismatch. Mock models in tests must return shapes corresponding to the dynamically generated batch sizes.

## 3. Implementation Details

### 3.1. `DraftState.__init__` Updates
- **New Parameter**: Add `max_candidates: int = 20` to the constructor and save it as `self.max_candidates`.
- **Cache Initialization**: Initialize `self._cached_valid_moves: list[DraftMove] | None = None` and `self._cached_priors: list[float] | None = None`. (These should be reset to `None` for every new `DraftState` instance).

### 3.2. New Helper: `_evaluate_and_prune_moves`
Add `def _evaluate_and_prune_moves(self) -> tuple[list[DraftMove], list[float]]:` to `DraftState`:
1. **Generate All Candidates**: Construct all valid `DraftMove` candidates for the current `step_idx` (up to `~120`). If the list is empty, return `([], [])`.
2. **Batch Forward Pass**:
   - Construct sequences for each candidate using `_build_tensor_from_moves`.
   - Stack into a `batch` tensor and expand the `comfort_matrix`.
   - Run `raw_logits = self.model.forward(batch, comfort)` inside `torch.no_grad()`.
3. **Determine Perspective**:
   - Identify the `schedule_team` from `DRAFT_SCHEDULE[step_idx]`.
   - Calculate sorting logits: if `schedule_team == 1` (Dire), `sort_logits = -raw_logits`, else `sort_logits = raw_logits.clone()`.
4. **Apply Comfort Scaling \u0026 Sort**:
   - Apply comfort scaling: `sort_scores = self._apply_comfort_scaling_to_logits(sort_logits, valid_moves)`.
   - Sort `sort_scores` descending to find the top `self.max_candidates` indices.
   - Extract `top_moves` and their corresponding `top_raw_logits` using these indices.
5. **Compute PUCT Priors**:
   - Calculate priors from `top_raw_logits` for the `active_team` (root's perspective):
     - If `self.active_team == 1`, `active_logits = -top_raw_logits`.
     - Else `active_logits = top_raw_logits.clone()`.
   - Apply comfort scaling: `active_scores = self._apply_comfort_scaling_to_logits(active_logits, top_moves)`.
   - Apply Softmax: `log_probs = active_scores - active_scores.max()`, `priors = torch.exp(log_probs) / torch.exp(log_probs).sum()`.
6. **Return**: `(top_moves, priors.tolist())`.

### 3.3. `actions_to_try` Updates
- Replace the existing naive slicing logic.
- Check `if self._cached_valid_moves is None:`
  - Call `self._cached_valid_moves, self._cached_priors = self._evaluate_and_prune_moves()`.
- Return `self._cached_valid_moves`.

### 3.4. `get_action_probabilities` Updates
- Replace the existing batch construction and forward pass logic.
- Check `if self._cached_priors is None:`
  - Call `self.actions_to_try()` to trigger the evaluation and cache population.
- Return `self._cached_priors`.

### 3.5. `clone` \u0026 `next_state` Methods
- Ensure `clone()` and `next_state()` pass `self.max_candidates` to the new `DraftState` instance. The caches (`_cached_valid_moves`, `_cached_priors`) should not be copied; they will lazily initialize in the new state.

### 3.6. Test Updates (`tests/test_search.py`)
- Update `test_draft_state_get_action_probabilities_uses_logits` and other MCTS search tests:
  - `mock_model.forward` must now return tensors of dynamic size based on the batch input (`return torch.zeros(batch.shape[0])` or similar) instead of a fixed size `[0.0, 0.5, 1.0, 1.5, 2.0]`.
  - Validate that `actions_to_try()` correctly returns up to 20 elements based on the model's highest returned values.

# Technical Specification - AUT-31: Ban State MCTS Reward Distortion

## 1. Problem Statement
The current implementation of `DraftState.rollout()` and `DraftState.evaluate_batch()` in `src/dota2drafter/search/state.py` overrides the evaluation of ban states, returning a win probability of `0.0`. In a binary win probability space of `[0.0, 1.0]`, `0.5` represents a neutral game state, while `0.0` represents a guaranteed instant loss. This causes MCTS tree search to severely penalize and avoid branch paths leading to ban states, distorting draft choices during ban phases.

Furthermore, ban states are currently assigned uniform prior probabilities ($1/N$). To direct MCTS search toward high-priority tactical bans and avoid wasting search iterations, we must leverage the auto-regressively trained policy head (`mlm_logits`) to calculate informed priors for bans, masked with the invalid/used hero mask and normalized using softmax.

Additionally, the test suite currently fails due to a pre-existing `NameError` in `src/dota2drafter/models/match_network.py` (line 641) where the variable `batch_size` is undefined. We include a one-line fix for this bug to ensure the entire test suite compiles and runs cleanly.

---

## 2. File Structure Changes

| File Path | Action | Description |
| :--- | :--- | :--- |
| `src/dota2drafter/search/state.py` | Modify | Update `rollout()` to evaluate ban states through the network instead of returning static `0.0`. Update `evaluate_batch()` to process all states (picks and bans) uniformly in a unified batch, applying informed priors using `mlm_logits` and masking used heroes. |
| `src/dota2drafter/models/match_network.py` | Modify | Fix pre-existing `NameError` by defining `B = x_draft.size(0)` or using `x_draft.size(0)` instead of `batch_size`. |
| `tests/test_search.py` | Modify | Update MCTS search tests to assert correct batching, neural network evaluation, and `mlm_logits`-based priors for ban states. |

---

## 3. Interfaces & Signatures

No new public interfaces are introduced. Existing signatures remain fully compatible:

### `DraftState.rollout()`
* **Signature:** `def rollout(self) -> float`
* **Changes:** Remove the check `if len(self.actions) > 0 and not self.actions[-1].is_pick: return 0.0`. Always invoke `self.model.predict_proba()` to evaluate partial draft win probability.

### `DraftState.evaluate_batch()`
* **Signature:** `def evaluate_batch(self, states: list[DraftState]) -> list[tuple[float, list[float]]]`
* **Changes:**
  * Eliminate the state partitioning into `pick_indices` and `ban_indices`.
  * Construct a single, unified batch of all input states and pass it to `self.model(base_batch, comfort_base)`.
  * For all non-terminal states (including ban states), extract policy logits at `step_idx` from `mlm_logits[i, step_idx, 1:K+1]`.
  * Apply `valid_mask` (masking out used/invalid heroes) and `max_candidates` filtering.
  * Compute softmax probabilities over the valid candidates.
  * Map priors to moves returned by `s.actions_to_try()`.
  * For terminal states (`step_idx >= 24`), return win probability and empty priors `[]`.

### `HierarchicalTransformer.forward()` (in `src/dota2drafter/models/match_network.py`)
* **Signature:** `def forward(self, x_draft: torch.Tensor, player_pref_vectors: torch.Tensor, patch_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]`
* **Changes:** In subtractive role-inhibition penalty generation (around line 641), change the undefined variable `batch_size` to `x_draft.size(0)`.

---

## 4. Edge Cases and Defense-in-Depth
* **Terminal States (`step_idx >= 24`):** If a state is terminal, the step index is out of bounds for the `mlm_logits` lookup. The logic must catch this check, return the computed win probability, and assign `[]` as priors.
* **Device Mapping:** Tensors generated during batch evaluation must be dynamically mapped to the correct hardware device (CPU/GPU) resolved from the model's active parameters.
* **Masking Robustness:** Used heroes mask must correctly handle all picked and banned heroes across both Radiant and Dire teams to prevent selecting previously drafted heroes.

---

## 5. Testing Strategy

### 5.1 Pre-existing Bug Verification
Verify that `PYTHONPATH=src pytest tests/test_models/test_match_network.py` and `tests/test_search.py` pass 100% cleanly after applying the `batch_size` fix in `match_network.py`.

### 5.2 Search State Unit Tests (to be updated/written in `tests/test_search.py`)
1. **`test_draft_state_rollout_ban_vs_pick`**:
   - Update this test to assert that `ban_state.rollout()` **does** call `mock_model.predict_proba` and returns the actual model probability (e.g. `0.75` for Radiant active, `0.25` for Dire active) instead of a hardcoded `0.0`.
2. **`test_draft_state_evaluate_batch_unified`** (formerly `test_draft_state_evaluate_batch_filtering`):
   - Update this test to assert that the mock model is called exactly **once** with a batch size of **3** (since the pick state, ban state, and root state are now processed together in a single batch).
   - Assert that `ban_state` returns the active-team win probability derived from the neural network logits (sigmoid value) instead of `0.0`.
   - Assert that `ban_state` priors are successfully retrieved from `mlm_logits` instead of being uniform.
3. **`test_draft_state_evaluate_batch_terminal`**:
   - Verify that when a terminal state is included in the batch, `evaluate_batch` processes it safely, returns correct win probability, and outputs an empty priors list `[]`.

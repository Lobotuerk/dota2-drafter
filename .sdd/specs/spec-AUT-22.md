# 📋 Technical Specification: AUT-22 (pymcts upgrade and batching)

## 1. Overview
Upgrading `pymcts` integration to utilize the new batched MCTS support. The C++ `pymcts.MCTS_agent` now exposes `exploration_constant`, `batch_size`, and `num_search_threads` and activates a batched `grow_tree` path when `evaluate_batch(states)` is implemented on the wrapped Python state.

## 2. Changes in `src/dota2drafter/search/state.py`

### 2.1. Implement `DraftState.evaluate_batch`
Add the `evaluate_batch` method to the `DraftState` class.

**Signature:**
```python
def evaluate_batch(self, states: list[DraftState]) -> list[tuple[float, list[float]]]:
```

**Implementation Details:**
1. If `states` is empty, return `[]`.
2. Iterate through `states` to build a batch tensor. Use `_build_tensor_from_moves(state.actions, self._num_heroes)` for each state.
3. If a state's tensor is less than 24 steps long, pad it to 24 steps with `_ZERO_STEP`.
4. Stack the padded tensors into a single batch of shape `(B, 24, 4)`.
5. Expand `self.comfort_matrix` to `(B, 10, C)`.
6. Extract the correct `device` from the model (fallback to `"cpu"`).
7. Execute a single batched forward pass: `win_probs = self.model.predict_proba(batch, comfort)`.
8. Iterate over `states` and the corresponding `win_probs[i]`:
   - Compute the active team's win probability (`radiant_win_prob = win_probs[i].item()`; `value = radiant_win_prob if self.active_team == 0 else 1.0 - radiant_win_prob`).
   - If `state.is_terminal()`, priors is an empty list `[]`.
   - Else, priors is the cached result of `state.get_action_probabilities()`.
   - Append `(value, priors)` to the results list.
9. Return the results list.

## 3. Changes in `src/dota2drafter/search/mcts_agent.py`

### 3.1. Update `Dota2DraftAgent.__init__`
Modify the constructor to accept the new arguments: `batch_size: int = 64` and `num_search_threads: int = 4`.

Pass the parameters correctly into `pymcts.MCTS_agent` (using the 4th positional argument for `exploration_constant`):
```python
self.agent = pymcts.MCTS_agent(
    wrapped_state,
    max_iter=int(max_iterations),
    max_seconds=int(max_seconds),
    exploration_constant=float(c_puct),
)
self.agent.batch_size = int(batch_size)
self.agent.num_search_threads = int(num_search_threads)
```
*(Ensure to retain `c_puct` in the constructor or expose `batch_size` and `num_search_threads` as properties if necessary to reconstruct during state updates)*

### 3.2. Update `Dota2DraftAgent.update_state`
In the fallback branch where `pymcts.MCTS_agent` is recreated:
```python
new_agent = pymcts.MCTS_agent(
    wrapped_state,
    max_iter=self.agent.max_iter,
    max_seconds=self.agent.max_seconds,
    exploration_constant=self.agent.exploration_constant,
)
new_agent.batch_size = self.agent.batch_size
new_agent.num_search_threads = self.agent.num_search_threads
self.agent = new_agent
```

## 4. Changes in `scripts/interactive_draft.py`

### 4.1. Add CLI Arguments
In `parse_args()`, add the new parameters:
```python
parser.add_argument("--c_puct", type=float, default=1.414, help="PUCT exploration constant (default: 1.414)")
parser.add_argument("--batch_size", type=int, default=64, help="MCTS batch size (default: 64)")
parser.add_argument("--num_search_threads", type=int, default=4, help="MCTS search threads (default: 4)")
```

### 4.2. Forward Arguments to `Dota2DraftAgent`
In `main()`, pass these parsed arguments into the agent instantiation:
```python
agent = Dota2DraftAgent(
    # ... existing args ...
    c_puct=args.c_puct,
    batch_size=args.batch_size,
    num_search_threads=args.num_search_threads,
)
```

## 5. Testing Scope (`tests/test_search.py`)

1. **`test_mcts_agent_params_propagation`**: Instantiate `Dota2DraftAgent` with custom `c_puct`, `batch_size`, and `num_search_threads`. Validate they are correctly assigned to `self.agent`. Trigger an unexplored move to hit `update_state`'s cold-start path and assert the new agent instance successfully retained the custom settings.
2. **`test_draft_state_evaluate_batch`**: Use a mock `MatchNetwork` asserting that `predict_proba` is called exactly once with a batch tensor of shape `(B, 24, 4)`. Provide a mix of terminal and non-terminal states and assert the returned list correctly aligns the values and matches the priors logic.

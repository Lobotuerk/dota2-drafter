# Technical Specification: AUT-27 - MCTS Usage of New Architecture (Excluding Ban Nodes from Value Evaluation)

## Overview
Currently, win probability (value) is only calculated on heroes picked, meaning that the neural network value head ignores bans entirely. However, the Monte Carlo Tree Search (MCTS) engine currently evaluates the neural network value head on **every** node expansion (including ban nodes). This is a waste of processing and introduces meaningless value estimates for ban states (which are simply the win probability of the picks made so far).

This specification details the design for:
1. Skipping neural network value evaluation for ban nodes, returning a neutral value of `0.0`.
2. Excluding ban states from the neural network batch during batched evaluation (`evaluate_batch`), setting their value to `0.0` and assigning uniform policy priors.
3. Propagating values naturally up through ancestors so that ban nodes compute an average of their descendants' pick values.

All algorithmic changes will be localized within the `dota2-drafter` search module (`src/dota2drafter/search/state.py`). No C++ modifications are required.

---

## 1. Architectural Strategy & Design

### Value Propagation & Backpropagation Semantics
In the Captains Mode draft, a ban node represents a state where the incoming action was a ban. Its value should represent the value of the picks it allows or counteracts.
- A ban node does not have a native pick-based win probability from the neural network.
- When first expanded, a ban node will return a rollout/evaluation value of `0.0`.
- The C++ MCTS backpropagation naturally accumulates `score += w` and `number_of_simulations += n` along the root-to-leaf path. Thus, as descendant pick nodes are expanded and evaluated with true neural network win probabilities, those real values propagate up to parent ban nodes. The winrate formula `score / number_of_simulations` naturally computes the running average of descendant pick values.

### Non-Evaluation of Ban States in `evaluate_batch`
The `evaluate_batch` method in `DraftState` will partition the input states into pick states and ban states:
- **Pick States**: A state `s` is a pick state if `len(s.actions) == 0` (the root node) or `s.actions[-1].is_pick == True`.
- **Ban States**: A state `s` is a ban state if `len(s.actions) > 0` and `s.actions[-1].is_pick == False`.

To prevent wasted processing:
- Only pick states are batched and passed to the neural network forward pass (`self.model(base_batch, comfort_base)`).
- Ban states are excluded from the neural network entirely.
- For pick states, the value is retrieved from the value head and priors are calculated from the MLM policy logits as before.
- For ban states, the value is set to `0.0`. Their policy priors are assigned a normalized uniform distribution (`1.0 / len(valid_moves)`) over all valid moves, which is mathematically robust and efficient.

---

## 2. Code Changes

### File: `dota2-drafter/src/dota2drafter/search/state.py`

#### A. Modify `rollout(self) -> float`
We check if the state is a ban state (meaning the last action was a ban). If so, we immediately return `0.0` without executing model inference.

```python
    def rollout(self) -> float:
        """Evaluate the current partial draft via rollout.

        Pads the sequence to 24 steps with zeroed dummy moves, then
        evaluates via MatchNetwork.predict_proba(). Returns the win
        probability for the active team.

        Returns:
            Float win probability in [0, 1].
        """
        # Exclude ban states from neural network evaluation (return neutral 0.0)
        if len(self.actions) > 0 and not self.actions[-1].is_pick:
            return 0.0

        tensor = _build_tensor_from_moves(self.actions, self._num_heroes)
        ...
```

#### B. Refactor `evaluate_batch(self, states: list[DraftState]) -> list[tuple[float, list[float]]]`
We partition the states into `pick_indices` and `ban_indices`. The neural network is only invoked if `pick_indices` is non-empty. We then populate the results correctly for all states.

```python
    def evaluate_batch(self, states: list[DraftState]) -> list[tuple[float, list[float]]]:
        if not states:
            return []

        M = len(states)
        K = self._num_heroes
        device = self._resolve_model_device()
        self.model.eval()

        # Partition states into pick and ban states
        pick_indices = []
        ban_indices = []
        for i, s in enumerate(states):
            if len(s.actions) > 0 and not s.actions[-1].is_pick:
                ban_indices.append(i)
            else:
                pick_indices.append(i)

        # Initialize results container
        results = [None] * M

        # 1. Evaluate Pick States using the Neural Network
        if pick_indices:
            pick_states = [states[idx] for idx in pick_indices]
            num_picks = len(pick_states)

            # Base batch
            draft_tensors = [_build_tensor_from_moves(s.actions, K) for s in pick_states]
            base_batch = torch.stack(draft_tensors, dim=0).to(device)
            comfort_base = self.comfort_matrix.unsqueeze(0).expand(num_picks, -1, -1).to(device)

            # Vectorized child setup
            valid_mask = torch.ones((num_picks, K), dtype=torch.bool, device=device)
            step_indices = [len(s.actions) for s in pick_states]

            for i, s in enumerate(pick_states):
                step_idx = step_indices[i]
                if step_idx >= 24:
                    valid_mask[i, :] = False
                    continue

                used_heroes = {m.hero_id for m in s.actions if m.hero_id > 0}
                for hero_id in used_heroes:
                    valid_mask[i, hero_id - 1] = False

            # Forward pass
            with torch.no_grad():
                logits, mlm_logits = self.model(base_batch, comfort_base)
                win_probs = torch.sigmoid(logits)

                # Extract policy logits for the step we want to predict
                policy_logits = torch.zeros(num_picks, K, device=device)
                for i, s in enumerate(pick_states):
                    step_idx = step_indices[i]
                    if step_idx < 24:
                        policy_logits[i] = mlm_logits[i, step_idx, 1:K+1]

            # Compute normalized priors with PUCT constraints
            sort_scores = policy_logits.clone()
            sort_scores = sort_scores.masked_fill(~valid_mask, float('-inf'))
            active_scores = sort_scores.clone()

            max_c = min(s.max_candidates for s in pick_states) if pick_states else 20
            max_c = min(max_c, K)

            _, top_indices = torch.topk(sort_scores, k=max_c, dim=1)
            topk_mask = torch.zeros((num_picks, K), dtype=torch.bool, device=device)
            topk_mask.scatter_(1, top_indices, True)

            active_scores = active_scores.masked_fill(~topk_mask, float('-inf'))

            log_probs = active_scores - active_scores.max(dim=1, keepdim=True).values
            is_invalid = active_scores == float('-inf')
            log_probs = log_probs.masked_fill(is_invalid, float('-inf'))

            priors_tensor = torch.exp(log_probs)
            priors_sum = priors_tensor.sum(dim=1, keepdim=True)
            priors_tensor = priors_tensor / priors_sum.clamp(min=1e-9)
            priors_tensor = priors_tensor.masked_fill(is_invalid, -1e9)

            win_probs_cpu = win_probs.cpu().tolist()
            priors_cpu = priors_tensor.cpu().tolist()

            # Populate results for Pick States
            for i, idx in enumerate(pick_indices):
                s = states[idx]
                radiant = win_probs_cpu[i]
                value = radiant if s.active_team == 0 else 1.0 - radiant

                step_idx = step_indices[i]
                if step_idx >= 24:
                    s._cached_priors = []
                    results[idx] = (value, [])
                    continue

                valid_priors = []
                valid_moves = s.actions_to_try()
                priors_list = priors_cpu[i]
                for m in valid_moves:
                    hero_idx = m.hero_id - 1
                    valid_priors.append(priors_list[hero_idx])

                s._cached_priors = valid_priors
                results[idx] = (value, valid_priors)

        # 2. Assign Neutral Values and Uniform Priors to Ban States
        for idx in ban_indices:
            s = states[idx]
            value = 0.0

            step_idx = len(s.actions)
            if step_idx >= 24:
                s._cached_priors = []
                results[idx] = (value, [])
                continue

            valid_moves = s.actions_to_try()
            if not valid_moves:
                valid_priors = []
            else:
                p = 1.0 / len(valid_moves)
                valid_priors = [p] * len(valid_moves)

            s._cached_priors = valid_priors
            results[idx] = (value, valid_priors)

        return results
```

---

## 3. Testing & Verification Plan

The implementer must update the test suite to verify this new logic thoroughly.

### Unit Tests to Add in `dota2-drafter/tests/test_search.py`

1. **`test_draft_state_rollout_ban_vs_pick`**:
   - Create a mock model.
   - Initialize a `DraftState` where the last action is a ban (`is_pick=False`). Verify `rollout()` immediately returns `0.0` and that the model is never called.
   - Initialize a `DraftState` where the last action is a pick (`is_pick=True`). Verify `rollout()` returns the model's win probability and that the model is called.

2. **`test_draft_state_evaluate_batch_filtering`**:
   - Create a recording mock model.
   - Build a list of states comprising:
     - 1 pick state (last action `is_pick=True`)
     - 1 ban state (last action `is_pick=False`)
     - 1 root state (0 actions)
   - Call `evaluate_batch(states)`.
   - Assert:
     - The neural network's `forward` is called with a batch size of exactly `2` (the pick state and the root state).
     - The ban state is not in the tensor batch passed to the model.
     - The returned results has size `3` with correct ordering preserved.
     - The ban state returns value `0.0`.
     - The ban state returns a uniform, normalized list of priors of the same size as `actions_to_try()`.

### Verification Command
Run the test suite using pytest to ensure all tests pass:
```bash
pytest tests/test_search.py
```

# 📋 Technical Specification: AUT-22 — pymcts was upgraded

> **Issue:** AUT-22 · **Branch:** `sdd/feature-AUT-22` · **Status:** Drafted for implementation
> **Repos:** `dota2-drafter` (this change), `MonteCarloTreeSearch` (upstream `pymcts`, already upgraded via TFT-232)

## 1. Overview

The `pymcts` engine (MonteCarloTreeSearch, commit `41d22d5` / TFT-232) now ships a **batched MCTS**
search path plus tunable search parameters:

- `MCTS_agent(..., exploration_constant=...)` — 4th positional/keyword constructor arg (PUCT `c`).
- `agent.batch_size` (default 64) and `agent.num_search_threads` (default 4) — settable properties.
- A batched `grow_tree()` state machine (`SearchThreadPool` + `expand_with_priors` + virtual loss)
  that activates **only when** the wrapped Python state implements `evaluate_batch(states)`.
  Otherwise it silently falls back to the legacy sequential loop.

This issue consumes the upgraded API from `dota2-drafter`:

1. Add `DraftState.evaluate_batch(states)` so the batched path actually engages.
2. Plumb `c_puct`, `batch_size`, `num_search_threads` from `scripts/interactive_draft.py` through
   `Dota2DraftAgent` into the `pymcts.MCTS_agent`.
3. Add unit tests in `tests/test_search.py` for param propagation and `evaluate_batch` correctness.

There is no visual/UI surface in this change (terminal CLI + search backend), so no Visual
Playtesting Playbook applies.

---

## 2. The pymcts Batched Contract (ground truth)

Verified against `MonteCarloTreeSearch/pybind/mcts_python.cpp`, `py_wrappers.cpp` and
`pymcts.cpp` at commit `41d22d5`. The implementer must satisfy this contract exactly.

### 2.1 How the batched path engages

- `MCTS_tree::grow_tree()` checks `py::hasattr(root_state, "evaluate_batch")`. Presence of the
  method (not the return value) switches to the batched state machine.
- Search threads walk the tree, apply virtual loss to unexplored non-terminal leaves, and enqueue
  them. The main thread drains the queue and calls the **root** `DraftState.evaluate_batch` with the
  list of **leaf** Python `DraftState` objects.

### 2.2 `evaluate_batch` signature and return contract

```python
def evaluate_batch(self, states: list[DraftState]) -> list[tuple[float, list[float]]]:
```

- Called **on the root state**, receives the leaf states as arguments. Root and leaves share the
  same `model`, `comfort_matrix`, `active_team`, `hero_indexer`, and `_num_heroes`.
- **Must return exactly `len(states)` entries, in the same order** — the C++ side pairs results with
  nodes by index (`min(nodes.size(), results.size())`). A short/misaligned list leaves nodes with
  virtual loss applied and never backpropagated, corrupting the tree.
- Each entry is a `(value, priors)` tuple:
  - `value: float` — the leaf's evaluation, backpropagated as node score. Semantics must match
    `rollout()`: **win probability of the active team** (the team fixed at the root).
  - `priors: list[float]` — AlphaZero-style child priors, **aligned 1:1 with the leaf's
    `actions_to_try()` order** (the `untried_actions` queue order). Consumed by C++
    `expand_with_priors()`; entries past the end default to `1.0`.

### 2.3 Terminal leaves

Terminal leaves are short-circuited **inline in C++** (`node->state->rollout()` +
`backpropagate`) and are **never** enqueued for `evaluate_batch`. The Python method still handles a
terminal/`no-valid-moves` state defensively (value only, `priors = []`).

### 2.4 How the parameters surface in Python

Verified on the installed `pymcts` module (`pymcts.cpython-313-x86_64-linux-gnu.so`):

- `MCTS_agent(wrapped_state, max_iter=..., max_seconds=..., exploration_constant=1.41)` — keyword
  `exploration_constant` works.
- `agent.exploration_constant`, `agent.batch_size`, `agent.num_search_threads` — readable/writable
  properties. `agent.max_iter`, `agent.max_seconds` — readable/writable attributes.
- `agent.tree` — read-only handle to `MCTS_tree`.
- `MCTS_tree` binding exposes **no** `batch_size`/`num_search_threads` setters (see §6 limitation).

---

## 3. Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Value = active-team win probability from a single batched `predict_proba`.** | `MatchNetwork.predict_proba` takes `(B, 24, 4)` + `(B, 10, C)` and returns `(B,)` sigmoid radiant-win probs — natively batched. Same semantics as `rollout()` (which calls `predict_proba`), so batched search quality matches the legacy path. `value = win_prob[i]` if `active_team == 0` else `1.0 - win_prob[i]`. |
| D2 | **Priors = the leaf's cached comfort-scaled PUCT priors (`get_action_probabilities()`), empty for terminal states.** | `_evaluate_and_prune_moves` already computes top-`max_candidates` priors in `actions_to_try()` order and caches them; C++ already called `actions_to_try()` at node construction, so the cache is populated and **no extra forward pass** is needed. Order matches the `untried_actions` queue 1:1. |
| D3 | **Terminal/no-valid-moves states → `(value, [])`.** | C++ never enqueues terminals, but defensive handling keeps the method total and test-friendly. An empty priors list still backpropagates the value. |
| D4 | **`c_puct` forwarded as `exploration_constant` (constructor arg) AND passed to `tree.grow_tree(..., c=...)` in `search()`.** | Constructor sets the agent's value; but `search()` grows the tree directly (`agent.tree.grow_tree(...)`), which currently uses the binding default `c=1.41` — so `c_puct` is **currently inert**. Passing `c` makes it effective on both paths. |
| D5 | **`batch_size` / `num_search_threads` set on the `pymcts.MCTS_agent` properties; preserved on the `update_state()` cold-start recreation.** | This is the literal API the issue asks to use. Values take effect wherever the agent-level config is honored (see §6 limitation for the direct `grow_tree` path). |
| D6 | **Single device-resolution helper used by `evaluate_batch`.** | The device block is currently duplicated in `rollout()` and `_evaluate_and_prune_moves()`; add one private helper for the new method to avoid a third copy (existing methods stay untouched). |

---

## 4. Changes — `src/dota2drafter/search/state.py`

### 4.1 Add private helper `_resolve_model_device(self) -> torch.device`

Mirror the pattern already used in `_evaluate_and_prune_moves` (robust against mocks):

```python
def _resolve_model_device(self) -> torch.device:
    device = torch.device("cpu")
    if hasattr(self.model, "parameters"):
        try:
            model_device = next(self.model.parameters()).device
            if isinstance(model_device, (torch.device, str)):
                device = model_device
        except (StopIteration, AttributeError, TypeError):
            pass
    return device
```

### 4.2 Add `DraftState.evaluate_batch(states)`

```python
def evaluate_batch(self, states: list[DraftState]) -> list[tuple[float, list[float]]]:
```

Algorithm:

1. If `states` is empty → return `[]`.
2. Build the draft batch: for each `s in states`, `_build_tensor_from_moves(s.actions, s._num_heroes)`
   (always returns `(24, 4)` — already zero-padded; **no manual `_ZERO_STEP` padding needed**).
   `torch.stack(tensors, dim=0)` → `(B, 24, 4)`.
3. Build the comfort batch: `self.comfort_matrix.unsqueeze(0).expand(B, -1, -1)` → `(B, 10, C)`.
4. `device = self._resolve_model_device()`; move both tensors to `device`.
5. `self.model.eval()`; with `torch.no_grad()`: `win_probs = self.model.predict_proba(batch, comfort)`
   → `(B,)` radiant win probabilities. **Exactly one batched forward pass.**
6. For each index `i`:
   - `radiant = win_probs[i].item()`
   - `value = radiant if s.active_team == 0 else 1.0 - radiant` (D1)
   - `priors = [] if s.is_terminal() else s.get_action_probabilities()` (D2, D3)
   - append `(value, priors)`
7. Return the list — **length and order must equal `states`**.

Notes for the implementer:
- Reuse the `torch.no_grad()` / `model.eval()` discipline from `rollout()`.
- Do **not** call `model.forward` directly for the value — `predict_proba` owns the sigmoid, matching
  `rollout()` exactly.
- `get_action_probabilities()` returns the cached `_cached_priors` (computed during `actions_to_try()`
  at node construction); it lazily computes them if the cache is empty, so it never crashes.

---

## 5. Changes — `src/dota2drafter/search/mcts_agent.py`

### 5.1 `Dota2DraftAgent.__init__` — accept and forward the new params

Add constructor params `batch_size: int = 64`, `num_search_threads: int = 4`. Keep `c_puct`
(default 1.414). Replace the agent construction (currently `mcts_agent.py:90`):

```python
self.c_puct = float(c_puct)
self.batch_size = int(batch_size)
self.num_search_threads = int(num_search_threads)

self.agent = pymcts.MCTS_agent(
    wrapped_state,
    max_iter=int(max_iterations),
    max_seconds=int(max_seconds),
    exploration_constant=self.c_puct,
)
self.agent.batch_size = self.batch_size
self.agent.num_search_threads = self.num_search_threads
```

Store the three values on `self` so the cold-start path can re-apply them.

### 5.2 `search()` — make `c_puct` effective (D4)

Change the grow call (`mcts_agent.py:109`):

```python
self.agent.tree.grow_tree(self.agent.max_iter, self.agent.max_seconds, c=self.c_puct)
```

### 5.3 `update_state()` cold-start — preserve all three params

In the fallback branch (`mcts_agent.py:295`), recreate the agent carrying over the tuned values:

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

(`self.agent.max_iter`/`max_seconds`/`exploration_constant`/`batch_size`/`num_search_threads` are all
readable on the pymcts binding.)

---

## 6. Changes — `scripts/interactive_draft.py`

### 6.1 Add CLI flags in `parse_args()`

```python
parser.add_argument("--c_puct", type=float, default=1.414,
                    help="PUCT exploration constant (default: 1.414)")
parser.add_argument("--batch_size", type=int, default=64,
                    help="MCTS batch size for batched leaf evaluation (default: 64)")
parser.add_argument("--num_search_threads", type=int, default=4,
                    help="MCTS parallel search threads (default: 4)")
```

### 6.2 Forward into `Dota2DraftAgent` in `main()`

```python
agent = Dota2DraftAgent(
    model=model,
    comfort_matrix=comfort_tensor,
    active_team=active_team,
    max_iterations=args.max_iterations,
    max_seconds=args.max_seconds,
    c_puct=args.c_puct,
    batch_size=args.batch_size,
    num_search_threads=args.num_search_threads,
    top_n=args.top_n,
    hero_indexer=hero_indexer,
    max_candidates=args.max_candidates,
)
```

No other script/entry point exposes these (per approved scope).

---

## 7. Known limitation & follow-up (documented, not blocking)

`pymcts.MCTS_tree` does **not** expose `batch_size` / `num_search_threads` setters in the Python
binding. The agent-level properties are synced into the tree **only** by `pymcts.MCTS_agent.genmove()`.
`Dota2DraftAgent.search()` grows the tree directly via `agent.tree.grow_tree(...)`, so on that path
the tree applies its internal defaults (64 / 4 — identical to the new CLI defaults).

Consequences of the chosen design:
- `c_puct` is effective on **both** paths (D4).
- `batch_size` / `num_search_threads` are passed to and stored on the `pymcts.MCTS_agent` (the issue's
  literal requirement) and are honored on the `genmove()` path; on the `search()` path the batched
  path still engages (via `evaluate_batch`) using the tree defaults 64/4.
- Recommended follow-up (separate issue, in `MonteCarloTreeSearch`): expose
  `def_property("batch_size"...)` / `def_property("num_search_threads"...)` on the `MCTS_tree`
  binding, then `Dota2DraftAgent` can sync them in `__init__`/`update_state` for full tuning on the
  `search()` path.

The implementer must **not** rebuild/modify the `pymcts` package in this issue.

---

## 8. Testing — `tests/test_search.py`

### 8.1 `test_mcts_agent_params_propagation`

- Build `Dota2DraftAgent` with `c_puct=2.0`, `batch_size=16`, `num_search_threads=2` (mock model,
  `comfort_matrix=torch.zeros(10, 64)`).
- Assert `agent.agent.exploration_constant == 2.0`, `agent.agent.batch_size == 16`,
  `agent.agent.num_search_threads == 2`.
- Trigger the cold-start path with an unexplored move (e.g. `DraftMove(hero_id=99, is_pick=True,
  team=0, step_index=0)`) and assert `id(agent.agent)` changed **and** the new agent retained all
  three values (`exploration_constant == 2.0`, `batch_size == 16`, `num_search_threads == 2`).
- Follow the mock-model conventions already in the file
  (`predict_proba.side_effect = lambda x, c: torch.full((x.shape[0],), 0.55)`).

### 8.2 `test_draft_state_evaluate_batch`

- Use a recording mock whose `predict_proba(batch, comfort)` appends `(batch.shape, comfort.shape)`
  and returns a `(B,)` tensor (e.g. `torch.full((batch.shape[0],), 0.7)`).
- Build 2–3 non-terminal `DraftState`s (vary `active_team` 0 and 1) plus one terminal state
  (24 actions).
- Call `root_state.evaluate_batch(states)`.
- Assert:
  - `predict_proba` called **exactly once**, with shapes `(B, 24, 4)` and `(B, 10, C)`.
  - Return length `== len(states)` and order matches input.
  - Values: radiant-win 0.7 → `active_team==0` gives `0.7`, `active_team==1` gives `0.3`.
  - Non-terminal states: `priors` non-empty, `len == max_candidates`, sum ≈ 1.0.
  - Terminal state: `priors == []` and value present.
- Keep mock `forward()` defined too (so `actions_to_try()` works), and `eval()` returning `self` —
  mirror `_ScorePredictorMock`/`_DynamicLogitsMock` in the existing suite.

### 8.3 Regression

All existing `tests/test_search.py` tests must continue to pass unchanged (legacy sequential path and
`Dota2DraftAgent` construction are backward-compatible). Run `python -m pytest tests/test_search.py -q`.

---

## 9. Files touched

| File | Change |
|------|--------|
| `src/dota2drafter/search/state.py` | `+evaluate_batch`, `+_resolve_model_device` |
| `src/dota2drafter/search/mcts_agent.py` | `+batch_size`, `+num_search_threads` params; forward `c_puct`; pass `c` in `search()`; preserve params in `update_state()` cold start |
| `scripts/interactive_draft.py` | `+--c_puct`, `+--batch_size`, `+--num_search_threads`; forward to agent |
| `tests/test_search.py` | `+test_mcts_agent_params_propagation`, `+test_draft_state_evaluate_batch` |

## 10. Out of scope

- Modifying/rebuilding the `pymcts` package or `MonteCarloTreeSearch` (follow-up per §7).
- Exposing these flags in any other CLI/entry point.
- Changing `_evaluate_and_prune_moves` / `rollout` device logic (they may adopt `_resolve_model_device`
  in a later refactor).

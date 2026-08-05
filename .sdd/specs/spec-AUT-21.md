# Technical Specification: MCTS Tree Reuse for Dota 2 Drafter

## 1. Overview
Currently, the `Dota2DraftAgent.update_state()` method completely destroys the existing Monte Carlo Tree Search (MCTS) tree and recreates the agent from scratch whenever a move is made by the user or the opponent. This "cold start" discards all lookahead statistics, visit counts, and win probabilities accumulated during previous search iterations.

This specification details a purely Python-based implementation to leverage `pymcts.MCTS_tree.advance_tree(move)`. By reusing the subtree corresponding to the selected move, the agent will compound its lookahead depth across turns and avoid redundant rollout computations.

## 2. Technical Design

### 2.1 File to Modify
- `src/dota2drafter/search/mcts_agent.py`

### 2.2 Mechanism for Tree Advancement
The C++ backend exposes `self.agent.tree.advance_tree(move)`, but move matching in C++ relies on pointer/wrapper equality (`operator==` for `PythonMoveWrapper`). Passing a newly instantiated Python `DraftMove` into the C++ `advance_tree` will silently fail and cause a fallback to cold start.

To robustly match the move:
1. We iterate over `self.agent.tree.root.get_children()`.
2. Extract the C++ move wrapper via `child.get_move()`.
3. Convert it to a Python `DraftMove` using the existing `self._extract_python_move(cpp_move)` method.
4. Compare it against the applied `move` using standard Python equality.
5. If a match is found, invoke `self.agent.tree.advance_tree(matching_cpp_move)`.

### 2.3 Fallback Behavior
If the applied move is not found among the root's children (e.g., an opponent played a move completely unvisited by our MCTS lookahead):
- Log a warning using the `logging` module indicating the move was not found.
- Fall back to the existing "cold start" logic by completely recreating `self.agent = pymcts.MCTS_agent(...)`.

### 2.4 Modifications to `update_state(self, move: DraftMove) -> None`
The implementation will be refactored as follows:

```python
    def update_state(self, move: DraftMove) -> None:
        \"\"\"Update the internal state after a move.

        Advances the draft state by applying the action.
        Attempts to reuse the existing MCTS tree if the move is found
        among the root's children, preserving lookahead statistics.
        Falls back to a cold-start if the move is unexplored.

        Args:
            move: The DraftMove to apply.
        \"\"\"
        self.state = self.state.next_state(move)

        matching_cpp_move = None
        if hasattr(self.agent, "tree") and self.agent.tree is not None:
            root = self.agent.tree.root
            if root is not None:
                for child in root.get_children():
                    cpp_move = child.get_move()
                    if cpp_move is None:
                        continue
                    py_move = self._extract_python_move(cpp_move)
                    if py_move == move:
                        matching_cpp_move = cpp_move
                        break

        if matching_cpp_move is not None:
            self.agent.tree.advance_tree(matching_cpp_move)
        else:
            logger.warning("Move %s not found in MCTS tree. Falling back to cold start.", move)
            # Re-wrap the new state for the C++ agent
            wrapped_state = pymcts.SerializedPythonState(self.state)
            # Create a new agent with the updated state
            self.agent = pymcts.MCTS_agent(
                wrapped_state,
                max_iter=self.agent.max_iter,
                max_seconds=self.agent.max_seconds,
            )
```

## 3. Assumptions and Validation
- **`genmove()` Usage**: The `genmove()` method already delegates to the C++ `self.agent.genmove()`, which internally grows, picks the best move, and advances the tree. Since `genmove()` is unchanged, it will naturally take advantage of the subtree reuse implemented here when called sequentially.
- **`max_iter` and `max_seconds` Budget**: The turn budget logic remains unchanged. Over consecutive turns, the cumulative rollouts under the reused root will exceed `max_iter`, giving the agent a significantly deeper lookahead.
- **State Integrity**: The Python `self.state` keeps its own exact track via `self.state.next_state(move)`, maintaining caching invariants independently of the C++ internal node state.

## 4. Acceptance Criteria
- `update_state` attempts to reuse the MCTS tree when the matching child move exists.
- The C++ subtree stats (visits, scores) are preserved when the tree advances.
- The agent properly falls back to recreating the MCTS engine when the requested move has 0 visits in the previous tree.
- Logging properly emits a warning on cold starts.

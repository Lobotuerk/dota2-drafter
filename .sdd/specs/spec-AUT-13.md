# Technical Specification: Adversarial Decision Search

## 1. Architectural Overview
This module integrates an adversarial Monte Carlo Tree Search (MCTS) into the `dota2-drafter` pipeline to transform the existing static `MatchNetwork` into an active recommendation agent. We wrap the `pymcts` C++ bindings to create a Python `DraftState` and use the `MatchNetwork` to batch-evaluate node priors and state rollouts. This allows us to perform minimax lookahead across the partial draft tree, incorporating player comfort matrices to penalize recommendations that real players cannot execute.

## 2. Core Dependencies
- `pymcts` (via `MonteCarloTreeSearch` repository bindings).
- `dota2drafter.models.match_network.MatchNetwork` for predicting partial-state win probabilities.
- `dota2drafter.models.player_network.PlayerComfortNetwork` for player vectors.
- `rich` for the interactive terminal UI.

## 3. Data Structures & Classes

### 3.1 `dota2drafter.search.state.DraftMove`
Inherits from `pymcts.MCTS_move`.
- **Fields**: `hero_id` (int), `is_pick` (bool), `team` (int), `step_index` (int).
- **Methods**: `__eq__`, `__str__`.

### 3.2 `dota2drafter.search.state.DraftState`
Inherits from `pymcts.MCTS_state` (or uses `SerializedPythonState` via bindings).
- **Attributes**: 
  - `actions`: List of applied `DraftMove` objects.
  - `model`: Reference to the `MatchNetwork` for evaluation.
  - `comfort_matrix`: Tensor `(10, C)` containing player metrics.
  - `active_team`: The team id (0 or 1) that is currently optimizing its win probability (the root node's team).
  - `draft_schedule`: A predefined list of 24 tuples `(is_pick, team)` dictating the sequence of turns.
- **MCTS Interface Methods**:
  - `actions_to_try()`: Yields valid `DraftMove`s based on `self.draft_schedule[len(self.actions)]` and the remaining unpicked/unbanned heroes.
  - `next_state(move)`: Returns a new `DraftState` instance with `self.actions + [move]`.
  - `is_terminal()`: `len(self.actions) == 24`.
  - `is_self_side_turn()`: Returns `True` if `self.draft_schedule[len(self.actions)][1] == self.active_team`.
  - `rollout()`: Pads the current `actions` sequence to length 24 using zeroed dummy steps. Batches via `self.model.predict_proba()` to compute the expected win rate. Returns the win probability for `self.active_team` (i.e. if `active_team == 0`, return Radiant win prob, else `1.0 - Radiant win prob`).
  - `get_action_probabilities()`: Constructs child sequences for all valid actions, pads them to 24, and evaluates them via `MatchNetwork` in a single GPU batch. Scales the predicted probability `P(Win | S + a)` by the comfort matrix: `P'(a|S) = Softmax(Logits(P) * W_comfort)`. Returns these scaled probabilities as the prior `P(a|S)` for the PUCT formula.

### 3.3 `dota2drafter.search.mcts_agent.Dota2DraftAgent`
A wrapper around `pymcts.MCTS_agent`.
- **Initialization**: `agent = pymcts.MCTS_agent(DraftState(...), max_iter, max_seconds)`.
- **`get_recommendations()`**: Retrieves the root node's children, sorts by visit count, and returns the top N recommended actions.
- **`get_principal_variation()`**: Walks down the most visited edges from the root to extract the expected sequence of moves.

## 4. Interactive Script (`scripts/interactive_draft.py`)
Provides an active draft support terminal UI.
- Prompts user to select team and provide a comfort matrix.
- Enters a 24-step loop following `draft_schedule`.
- When it's the opponent's turn: user inputs the opponent's action.
- When it's the active team's turn: invokes `Dota2DraftAgent.genmove()` allowing up to 30 seconds.
- Displays a `rich` table containing recommendations (Action, Hero, Win Prob, Prior Prob, Visit Count).
- Displays the principal variation sequence.

## 5. Visual Playtesting Playbook
- **Environment**: Open terminal, run `python scripts/interactive_draft.py` (ensure `pymcts` is installed via `pip install -e /path/to/MonteCarloTreeSearch`).
- **Setup**: Choose Radiant as the active team.
- **Execution**: The interface will present standard `rich` tables. Provide an initial opponent ban if required. Let the tool run MCTS for its 30s limit.
- **Verification**: The terminal MUST print a cleanly formatted table ranking the recommended Heroes to pick/ban. Check that a "Principal Variation" (Expected Draft Plan) is printed out sequentially beneath the table.

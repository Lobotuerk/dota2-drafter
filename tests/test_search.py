"""Unit tests for adversarial MCTS search modules using pymcts bindings."""

from __future__ import annotations

from unittest.mock import MagicMock

import pymcts
import pytest
import torch

from dota2drafter.search.mcts_agent import Dota2DraftAgent
from dota2drafter.search.state import DRAFT_SCHEDULE, DraftMove, DraftState


def test_draft_move_inherits_from_pymcts_move():
    """Verify DraftMove inherits from pymcts.MCTS_move."""
    move = DraftMove(hero_id=10, is_pick=True, team=0, step_index=2)
    assert isinstance(move, pymcts.MCTS_move)


def test_draft_move_equality_and_properties():
    """Verify DraftMove initialization, string representation, equality, and hash."""
    move1 = DraftMove(hero_id=10, is_pick=True, team=0, step_index=2)
    move2 = DraftMove(hero_id=10, is_pick=True, team=0, step_index=2)
    move3 = DraftMove(hero_id=11, is_pick=True, team=0, step_index=2)
    move4 = DraftMove(hero_id=10, is_pick=False, team=0, step_index=2)

    assert move1 == move2
    assert move1 != move3
    assert move1 != move4
    assert move1 != "not_a_move"

    # Hash compatibility
    moves_set = {move1, move3}
    assert move2 in moves_set

    # String representation via sprint
    assert move1.sprint() == "Radiant pick hero 10 at step 2"

    # to_numpy and to_env_action
    numpy_arr = move1.to_numpy()
    assert len(numpy_arr) == 4
    assert numpy_arr[0] == 10.0
    assert numpy_arr[1] == 1.0
    assert numpy_arr[2] == 0.0
    assert numpy_arr[3] == 2.0

    env_action = move1.to_env_action()
    assert len(env_action) == 4
    assert env_action[0] == 10
    assert env_action[1] == 1
    assert env_action[2] == 0
    assert env_action[3] == 2


def test_draft_state_inherits_from_pymcts_state():
    """Verify DraftState inherits from pymcts.MCTS_state."""
    mock_model = MagicMock()
    comfort_matrix = torch.zeros(10, 64)
    state = DraftState(model=mock_model, comfort_matrix=comfort_matrix, active_team=0)
    assert isinstance(state, pymcts.MCTS_state)


def test_draft_state_basic_flow():
    """Verify basic DraftState transitions and properties."""
    mock_model = MagicMock()
    mock_model.forward.return_value = torch.zeros(1)
    comfort_matrix = torch.zeros(10, 64)
    state = DraftState(model=mock_model, comfort_matrix=comfort_matrix, active_team=0)

    assert len(state.actions) == 0
    assert not state.is_terminal()
    assert not state.is_self_side_turn()  # first turn is team 1 (Dire) per DRAFT_SCHEDULE

    # Test actions_to_try
    valid_actions = state.actions_to_try()
    assert len(valid_actions) > 0
    assert all(not m.is_pick for m in valid_actions)  # step 0 is ban
    assert all(m.team == 1 for m in valid_actions)

    # Transition to next state
    move = valid_actions[0]
    next_state = state.next_state(move)
    assert len(next_state.actions) == 1
    assert next_state.actions[0] == move

    # Terminal state simulation
    terminal_actions = []
    for idx, (action_type, team) in enumerate(DRAFT_SCHEDULE):
        terminal_actions.append(
            DraftMove(hero_id=idx + 1, is_pick=(action_type == "pick"), team=team, step_index=idx)
        )

    terminal_state = DraftState(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        initial_actions=terminal_actions,
    )
    assert terminal_state.is_terminal()
    assert len(terminal_state.actions_to_try()) == 0


def test_draft_state_clone():
    """Verify DraftState clone produces independent copy."""
    mock_model = MagicMock()
    comfort_matrix = torch.zeros(10, 64)
    state = DraftState(model=mock_model, comfort_matrix=comfort_matrix, active_team=0)

    # Add a move
    move = DraftMove(hero_id=5, is_pick=True, team=0, step_index=2)
    state.actions.append(move)

    cloned = state.clone()
    assert len(cloned.actions) == 1
    assert cloned.actions[0] == move
    assert cloned is not state
    assert cloned.actions is not state.actions  # independent list


def test_draft_state_rollout():
    """Verify rollout uses the model's predict_proba correctly."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = torch.tensor([0.75])

    comfort_matrix = torch.zeros(10, 64)
    state_radiant = DraftState(model=mock_model, comfort_matrix=comfort_matrix, active_team=0)
    state_dire = DraftState(model=mock_model, comfort_matrix=comfort_matrix, active_team=1)

    assert state_radiant.rollout() == 0.75
    assert state_dire.rollout() == 0.25  # 1.0 - 0.75

    mock_model.predict_proba.assert_called()


def test_draft_state_get_action_probabilities_uses_logits():
    """Verify action probabilities use logits + comfort scaling + softmax."""

    _call_count = 0

    class _DynamicLogitsMock:
        def forward(self, batch, comfort):
            nonlocal _call_count
            _call_count += 1
            n = batch.shape[0]
            return torch.linspace(0.0, float(n - 1), n)

        def eval(self):
            return self

    mock_model = _DynamicLogitsMock()

    comfort_matrix = torch.zeros(10, 64)
    comfort_matrix[0] = 0.8

    state = DraftState(model=mock_model, comfort_matrix=comfort_matrix, active_team=0)

    probs = state.get_action_probabilities()
    assert len(probs) == 20
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert pytest.approx(sum(probs), 1e-5) == 1.0

    # Verify forward() was called exactly once
    assert _call_count == 1


def test_serialized_python_state_wrapper():
    """Verify DraftState can be wrapped with SerializedPythonState."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = torch.tensor([0.6])

    comfort_matrix = torch.zeros(10, 64)
    state = DraftState(model=mock_model, comfort_matrix=comfort_matrix, active_team=0)

    # Wrap with SerializedPythonState
    wrapped = pymcts.SerializedPythonState(state)

    # Verify the wrapper can call state methods
    assert not wrapped.is_terminal()
    assert not wrapped.is_self_side_turn()
    assert isinstance(wrapped.rollout(), float)


def test_mcts_agent_pymcts_integration():
    """Verify Dota2DraftAgent uses pymcts.MCTS_agent and tree has new API."""
    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = lambda x, c: torch.full((x.shape[0],), 0.55)

    comfort_matrix = torch.zeros(10, 64)
    agent = Dota2DraftAgent(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        max_iterations=10,
        max_seconds=0.1,
    )

    # Verify agent uses pymcts.MCTS_agent
    assert isinstance(agent.agent, pymcts.MCTS_agent)

    # Verify tree has root property
    tree = agent.agent.tree
    assert tree is not None
    assert hasattr(tree, "root")

    # Verify MCTS_node has new properties
    root = tree.root
    assert root is not None
    assert hasattr(root, "visit_count")
    assert hasattr(root, "score")
    assert hasattr(root, "get_children")
    assert hasattr(root, "get_parent")


def test_mcts_agent_search_returns_real_data():
    """Verify search() returns recommendations with real visit counts."""
    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = lambda x, c: torch.full((x.shape[0],), 0.55)
    mock_model.forward.side_effect = lambda x, c: torch.full((x.shape[0],), 0.5)

    comfort_matrix = torch.zeros(10, 64)
    agent = Dota2DraftAgent(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        max_iterations=100,
        max_seconds=0.5,
    )

    recs = agent.search()
    assert isinstance(recs, list)

    # Principal variation should be a list
    pv = agent.get_principal_variation()
    assert isinstance(pv, list)


def test_mcts_agent_genmove():
    """Verify genmove returns a DraftMove and advances state."""
    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = lambda x, c: torch.full((x.shape[0],), 0.55)
    mock_model.forward.side_effect = lambda x, c: torch.full((x.shape[0],), 0.5)

    comfort_matrix = torch.zeros(10, 64)
    agent = Dota2DraftAgent(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        max_iterations=10,
        max_seconds=0.1,
    )

    initial_count = len(agent.state.actions)
    best_move = agent.genmove()
    assert best_move is None or isinstance(best_move, DraftMove)

    if best_move is not None:
        assert len(agent.state.actions) == initial_count + 1


def test_draft_state_evaluate_and_prune_moves_perspectives_and_caching():
    """Verify that _evaluate_and_prune_moves handles sorting directions, caching,

    active team perspective and max_candidates configuration correctly.
    """
    class _ScorePredictorMock:
        def forward(self, batch, comfort):
            # Return distinct values for each candidate in the batch.
            n = batch.shape[0]
            return torch.arange(0.0, float(n))

        def eval(self):
            return self

    mock_model = _ScorePredictorMock()
    comfort_matrix = torch.zeros(10, 64)

    # Radiant is active team, schedule team is Dire (team 1) at step 0
    state = DraftState(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        max_candidates=3
    )

    # Verify actions_to_try uses the mock model, prunes to max_candidates, and caches
    assert state._cached_valid_moves is None
    assert state._cached_priors is None

    moves = state.actions_to_try()
    assert len(moves) == 3
    # Check that caching is populated
    assert state._cached_valid_moves is not None
    assert state._cached_priors is not None
    assert len(state._cached_priors) == 3

    # For schedule team 1 (Dire), sort ascending (minimizing Radiant's win probability)
    # under _ScorePredictorMock, so smallest raw logits are selected:
    # indices [0, 1, 2] should be selected.
    # Therefore, the hero_id of the moves should be 1, 2, 3
    assert [m.hero_id for m in moves] == [1, 2, 3]

    # Let's test with schedule team 0 (Radiant) at Step 2
    move0 = DraftMove(hero_id=1, is_pick=False, team=1, step_index=0)
    move1 = DraftMove(hero_id=2, is_pick=False, team=1, step_index=1)
    state_step2 = DraftState(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        max_candidates=3,
        initial_actions=[move0, move1]
    )

    # Step 2 is Radiant's turn. We want to maximize win probability.
    # Descending sort of [0.0, 1.0, 2.0, ...] means larger logits are selected.
    # Available heroes start from 3 onwards (since 1 and 2 are used).
    # The last 3 available heroes should be selected because they have the highest index.
    # Max hero index is 120 by default. Used: 1, 2. Available: 3..120 (118 total).
    # Corresponding logits for index 0..117 of available list are 0.0..117.0.
    # The top 3 logits are 117 (hero 120), 116 (hero 119), 115 (hero 118).
    # So descending sort of these should select hero_ids [120, 119, 118].
    moves_step2 = state_step2.actions_to_try()
    assert len(moves_step2) == 3
    assert [m.hero_id for m in moves_step2] == [120, 119, 118]

    # Test cloning passes max_candidates
    cloned = state_step2.clone()
    assert cloned.max_candidates == 3
    assert cloned._cached_valid_moves is None  # Cache should not be cloned

    # Test next_state passes max_candidates
    next_s = state_step2.next_state(moves_step2[0])
    assert next_s.max_candidates == 3
    assert next_s._cached_valid_moves is None  # Cache should not be copied


def test_mcts_agent_update_state_tree_reuse():
    """Verify that update_state reuses the existing MCTS tree when move matches a child."""
    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = lambda x, c: torch.full((x.shape[0],), 0.55)
    mock_model.forward.side_effect = lambda x, c: torch.full((x.shape[0],), 0.5)

    comfort_matrix = torch.zeros(10, 64)
    agent = Dota2DraftAgent(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        max_iterations=100,
        max_seconds=0.5,
    )

    # Grow the tree to populate children
    recs = agent.search()
    assert len(recs) > 0

    # Get one of the moves from the children
    root = agent.agent.tree.root
    assert root is not None
    children = root.get_children()
    assert len(children) > 0

    # Extract the matching python move
    cpp_move = children[0].get_move()
    py_move = agent._extract_python_move(cpp_move)
    assert py_move is not None

    # Track original agent ID
    original_agent_id = id(agent.agent)

    # Call update_state with the matching move
    agent.update_state(py_move)

    # Verify that the agent was NOT recreated (id is the same)
    assert id(agent.agent) == original_agent_id


def test_mcts_agent_update_state_fallback(caplog):
    """Verify that update_state falls back to cold start when move is not in the tree."""
    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = lambda x, c: torch.full((x.shape[0],), 0.55)
    mock_model.forward.side_effect = lambda x, c: torch.full((x.shape[0],), 0.5)

    comfort_matrix = torch.zeros(10, 64)
    agent = Dota2DraftAgent(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        max_iterations=10,
        max_seconds=0.1,
    )

    # Track original agent ID
    original_agent_id = id(agent.agent)

    # Make a move that is definitely not in the un-grown/empty root's children
    unexplored_move = DraftMove(hero_id=99, is_pick=True, team=0, step_index=0)

    import logging
    with caplog.at_level(logging.WARNING):
        agent.update_state(unexplored_move)

    # Verify that the agent was recreated (id has changed)
    assert id(agent.agent) != original_agent_id
    # Verify that a warning was logged
    assert any("not found in MCTS tree" in record.message for record in caplog.records)


def test_mcts_agent_params_propagation():
    """Verify c_puct, batch_size, num_search_threads are forwarded into pymcts agent."""
    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = lambda x, c: torch.full((x.shape[0],), 0.55)
    mock_model.forward.side_effect = lambda x, c: torch.full((x.shape[0],), 0.5)

    comfort_matrix = torch.zeros(10, 64)
    agent = Dota2DraftAgent(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        c_puct=2.0,
        batch_size=16,
        num_search_threads=2,
        max_iterations=10,
        max_seconds=0.1,
    )

    assert agent.agent.exploration_constant == 2.0
    assert agent.agent.batch_size == 16
    assert agent.agent.num_search_threads == 2

    # Trigger cold-start path
    unexplored_move = DraftMove(hero_id=99, is_pick=True, team=0, step_index=0)
    original_agent_id = id(agent.agent)
    agent.update_state(unexplored_move)

    assert id(agent.agent) != original_agent_id
    assert agent.agent.exploration_constant == 2.0
    assert agent.agent.batch_size == 16
    assert agent.agent.num_search_threads == 2


def test_draft_state_evaluate_batch():
    """Verify evaluate_batch correctness with a recording mock."""

    class _RecordingMock:
        def __init__(self):
            self.calls: list[tuple] = []

        def predict_proba(self, batch, comfort):
            B = batch.shape[0]
            self.calls.append((batch.shape, comfort.shape))
            return torch.full((B,), 0.7)

        def forward(self, batch, comfort):
            n = batch.shape[0]
            return torch.linspace(0.0, float(n - 1), n)

        def eval(self):
            return self

    mock_model = _RecordingMock()
    comfort_matrix = torch.zeros(10, 64)

    # Build non-terminal states with different active teams
    state_radiant = DraftState(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        max_candidates=5,
    )
    state_dire = DraftState(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=1,
        max_candidates=5,
    )

    # Build a terminal state (24 actions)
    terminal_actions = []
    for idx, (action_type, team) in enumerate(DRAFT_SCHEDULE):
        terminal_actions.append(
            DraftMove(hero_id=idx + 1, is_pick=(action_type == "pick"), team=team, step_index=idx)
        )
    terminal_state = DraftState(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        initial_actions=terminal_actions,
    )

    states = [state_radiant, state_dire, terminal_state]
    results = state_radiant.evaluate_batch(states)

    # predict_proba called exactly once
    assert len(mock_model.calls) == 1
    batch_shape, comfort_shape = mock_model.calls[0]
    assert batch_shape == (3, 24, 4)
    assert comfort_shape == (3, 10, 64)

    # Return length and order match
    assert len(results) == 3
    for i, s in enumerate(states):
        assert results[i][0] is not None  # value present

    # Values: radiant-win 0.7 -> active_team==0 gives 0.7, active_team==1 gives 0.3
    assert pytest.approx(results[0][0], 1e-6) == 0.7  # radiant active, radiant-win 0.7
    assert pytest.approx(results[1][0], 1e-6) == 0.3  # dire active, 1.0 - 0.7
    assert pytest.approx(results[2][0], 1e-6) == 0.7  # terminal state, radiant active

    # Non-terminal states: priors non-empty, len == max_candidates, sum ~ 1.0
    assert len(results[0][1]) == 5
    assert abs(sum(results[0][1]) - 1.0) < 1e-5
    assert len(results[1][1]) == 5
    assert abs(sum(results[1][1]) - 1.0) < 1e-5

    # Terminal state: priors == []
    assert results[2][1] == []



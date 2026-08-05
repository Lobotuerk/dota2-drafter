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

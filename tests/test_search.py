"""Unit tests for adversarial MCTS search modules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from dota2drafter.search.mcts_agent import Dota2DraftAgent
from dota2drafter.search.state import DRAFT_SCHEDULE, DraftMove, DraftState


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

    # String representation
    assert str(move1) == "Radiant pick hero 10 at step 2"


def test_draft_state_basic_flow():
    """Verify basic DraftState transitions and properties."""
    mock_model = MagicMock()
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


def test_draft_state_get_action_probabilities():
    """Verify action probabilities are computed with comfort scaling and softmax."""
    mock_model = MagicMock()
    # Mock prediction for a small number of valid actions
    # Suppose there are 5 valid actions
    mock_model.predict_proba.return_value = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8])

    comfort_matrix = torch.zeros(10, 64)
    # Give a player some comfort score values
    comfort_matrix[0] = 0.8

    state = DraftState(model=mock_model, comfort_matrix=comfort_matrix, active_team=0)
    # Stub actions_to_try to return only 5 moves
    moves = [
        DraftMove(hero_id=h, is_pick=True, team=0, step_index=2)
        for h in range(1, 6)
    ]
    state.actions_to_try = MagicMock(return_value=moves)

    priors = state.get_action_probabilities()
    assert len(priors) == 5
    assert all(0.0 <= p <= 1.0 for p in priors.values())
    assert pytest.approx(sum(priors.values()), 1e-5) == 1.0


def test_mcts_agent_basic():
    """Verify Dota2DraftAgent initializes and performs a search cycle."""
    mock_model = MagicMock()
    # Return a tensor of shape (B,) matching the batch size of input draft tensor
    mock_model.predict_proba.side_effect = lambda x, c: torch.full((x.shape[0],), 0.55)

    comfort_matrix = torch.zeros(10, 64)
    agent = Dota2DraftAgent(
        model=mock_model,
        comfort_matrix=comfort_matrix,
        active_team=0,
        max_iterations=10,
        max_seconds=0.1,
    )

    recs = agent.search()
    assert len(recs) > 0
    assert recs[0].visit_count >= 0
    assert 0.0 <= recs[0].win_probability <= 1.0

    pv = agent.get_principal_variation()
    assert isinstance(pv, list)

    best_move = agent.genmove()
    assert best_move is None or isinstance(best_move, DraftMove)

    if best_move:
        agent.update_state(best_move)
        assert len(agent.state.actions) == 1

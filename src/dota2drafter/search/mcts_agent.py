"""Dota 2 draft MCTS agent wrapper.

Provides the Dota2DraftAgent class that wraps the MCTS engine and
exposes a high-level interface for draft recommendations and
principal variation extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from dota2drafter.search.mcts_engine import (
    MCTSConfig,
    MCTSEngine,
    get_principal_variation,
    get_recommendations,
)
from dota2drafter.search.state import DraftMove, DraftState

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single draft recommendation from MCTS search.

    Attributes:
        move: The recommended draft action.
        visit_count: Number of MCTS visits for this action.
        win_probability: Estimated win probability from rollouts.
        prior_probability: Prior probability from the model.
    """

    move: DraftMove
    visit_count: int
    win_probability: float
    prior_probability: float


class Dota2DraftAgent:
    """MCTS-based draft recommendation agent.

    Wraps the MCTS engine to perform adversarial lookahead search
    over the draft tree. Provides methods for getting hero recommendations
    and the expected draft plan (principal variation).

    Attributes:
        state: The current DraftState for search.
        engine: The MCTS engine performing search.
        config: MCTS configuration.
        top_n: Number of recommendations to return.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        comfort_matrix: torch.Tensor,
        active_team: int = 0,
        max_iterations: int = 1000,
        max_seconds: float = 30.0,
        c_puct: float = 1.414,
        top_n: int = 5,
        hero_indexer: Any | None = None,
    ) -> None:
        """Initialize the draft agent.

        Args:
            model: MatchNetwork for win-probability evaluation.
            comfort_matrix: Player comfort tensor of shape (10, C).
            active_team: Team being optimized (0 = Radiant, 1 = Dire).
            max_iterations: Maximum MCTS simulations.
            max_seconds: Maximum search time in seconds.
            c_puct: PUCT exploration constant.
            top_n: Number of top recommendations to return.
            hero_indexer: HeroIndexer for hero ID management.
        """
        self.top_n = top_n
        self.config = MCTSConfig(
            max_iterations=max_iterations,
            max_seconds=max_seconds,
            c_puct=c_puct,
        )

        root_state = DraftState(
            model=model,
            comfort_matrix=comfort_matrix,
            active_team=active_team,
            hero_indexer=hero_indexer,
        )

        self.state = root_state
        self.engine = MCTSEngine(root_state, self.config)

    def search(self) -> list[Recommendation]:
        """Run MCTS search and return recommendations.

        Executes the full MCTS search cycle and extracts the top-N
        recommended actions sorted by visit count.

        Returns:
            List of Recommendation objects.
        """
        self.engine.run()
        recs = get_recommendations(self.engine.root, top_n=self.top_n)

        # Get priors for each recommendation
        priors = self.state.get_action_probabilities()

        recommendations: list[Recommendation] = []
        for move, visit_count, win_prob in recs:
            prior = priors.get(move.hero_id, 0.0)
            recommendations.append(
                Recommendation(
                    move=move,
                    visit_count=visit_count,
                    win_probability=win_prob,
                    prior_probability=prior,
                )
            )

        return recommendations

    def get_principal_variation(self) -> list[DraftMove]:
        """Get the principal variation (expected draft plan).

        Walks down the most visited edges from the root to extract
        the expected sequence of moves.

        Returns:
            List of DraftMove objects representing the principal variation.
        """
        return get_principal_variation(self.engine.root)

    def genmove(self) -> DraftMove | None:
        """Generate the best move for the current draft step.

        Runs MCTS search and returns the top recommended move.
        Returns None if no valid moves are available.

        Returns:
            The best DraftMove, or None if no valid moves exist.
        """
        recommendations = self.search()
        if not recommendations:
            return None
        return recommendations[0].move

    def update_state(self, move: DraftMove) -> None:
        """Update the internal state after an opponent's move.

        Advances the draft state by applying the opponent's action.

        Args:
            move: The DraftMove to apply.
        """
        self.state = self.engine.root.state.next_state(move)
        # Rebuild engine with new state
        self.engine = MCTSEngine(self.state, self.config)

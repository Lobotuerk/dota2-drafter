"""Dota 2 draft MCTS agent wrapper.

Provides the Dota2DraftAgent class that wraps pymcts.MCTS_agent with
SerializedPythonState to perform high-performance adversarial lookahead
search over the draft tree using the C++ MCTS engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pymcts
import torch

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
    """MCTS-based draft recommendation agent using pymcts C++ engine.

    Wraps pymcts.MCTS_agent to perform adversarial lookahead search
    over the draft tree. Uses SerializedPythonState to bridge Python
    DraftState with the C++ MCTS engine.

    Attributes:
        state: The current DraftState for search.
        agent: The pymcts.MCTS_agent performing search.
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
        max_candidates: int = 20,
    ) -> None:
        """Initialize the draft agent.

        Args:
            model: MatchNetwork for win-probability evaluation.
            comfort_matrix: Player comfort tensor of shape (10, C).
            active_team: Team being optimized (0 = Radiant, 1 = Dire).
            max_iterations: Maximum MCTS simulations.
            max_seconds: Maximum search time in seconds.
            c_puct: PUCT exploration constant (unused by pymcts, kept for API compat).
            top_n: Number of top recommendations to return.
            hero_indexer: HeroIndexer for hero ID management.
            max_candidates: Number of candidates to consider per node.
        """
        self.top_n = top_n

        root_state = DraftState(
            model=model,
            comfort_matrix=comfort_matrix,
            active_team=active_team,
            hero_indexer=hero_indexer,
            max_candidates=max_candidates,
        )

        # Wrap Python state with SerializedPythonState for C++ MCTS engine
        wrapped_state = pymcts.SerializedPythonState(root_state)

        self.agent = pymcts.MCTS_agent(
            wrapped_state,
            max_iter=int(max_iterations),
            max_seconds=int(max_seconds),
        )

        self.state = root_state

    def search(self) -> list[Recommendation]:
        """Run MCTS search and return recommendations.

        Executes the full MCTS search cycle via the C++ engine and
        extracts the top-N recommended actions from the tree root's
        children, sorted by visit count.

        Returns:
            List of Recommendation objects.
        """
        # Grow the tree from the current root (without advancing it!)
        self.agent.tree.grow_tree(self.agent.max_iter, self.agent.max_seconds)

        # Extract recommendations from the C++ tree
        recommendations = self._extract_recommendations()

        # Get priors for each recommendation
        priors = self.state.get_action_probabilities()
        prior_map = {move.hero_id: p for move, p in zip(self.state.actions_to_try(), priors)}

        result: list[Recommendation] = []
        for move, visit_count, win_prob in recommendations:
            prior = prior_map.get(move.hero_id, 0.0)
            result.append(
                Recommendation(
                    move=move,
                    visit_count=visit_count,
                    win_probability=win_prob,
                    prior_probability=prior,
                )
            )

        return result

    def _extract_recommendations(self) -> list[tuple[DraftMove, int, float]]:
        """Extract recommendations from the C++ MCTS tree root.

        Reads the root node's children and extracts visit counts and
        scores to produce real MCTS recommendations.

        Returns:
            List of (move, visit_count, avg_reward) tuples, sorted by visit count.
        """
        tree = self.agent.tree
        if tree is None:
            return []

        root = tree.root
        if root is None:
            return []

        children = root.get_children()
        if not children:
            return []

        # Build list of (move, visit_count, avg_reward) from children
        recs: list[tuple[DraftMove, int, float]] = []
        for child in children:
            cpp_move = child.get_move()
            if cpp_move is None:
                continue

            py_move = self._extract_python_move(cpp_move)
            if py_move is None:
                continue

            visit_count = child.visit_count
            avg_reward = child.score / max(visit_count, 1)

            recs.append((py_move, visit_count, avg_reward))

        # Sort by visit count descending, then by avg reward
        recs.sort(key=lambda x: (x[1], x[2]), reverse=True)

        return recs

    def get_principal_variation(self) -> list[DraftMove]:
        """Get the principal variation (expected draft plan).

        Walks down the most visited edges from the root to extract
        the expected sequence of moves.

        Returns:
            List of DraftMove objects representing the principal variation.
        """
        tree = self.agent.tree
        if tree is None:
            return []

        root = tree.root
        if root is None:
            return []

        variation: list[DraftMove] = []
        node = root

        while not node.is_terminal():
            children = node.get_children()
            if not children:
                break

            # Select child with highest visit count
            best_child = max(children, key=lambda c: c.visit_count)
            cpp_move = best_child.get_move()
            if cpp_move is None:
                break

            py_move = self._extract_python_move(cpp_move)
            if py_move is None:
                break

            variation.append(py_move)
            node = best_child

        return variation

    def genmove(self) -> DraftMove | None:
        """Generate the best move for the current draft step.

        Runs MCTS search via the C++ engine and returns the top
        recommended move. Returns None if no valid moves are available.

        Returns:
            The best DraftMove, or None if no valid moves exist.
        """
        move = self.agent.genmove()
        if move is None:
            return None

        # Extract Python move from C++ move
        py_move = self._extract_python_move(move)
        if py_move is not None:
            self.state = self.state.next_state(py_move)
        return py_move

    def _extract_python_move(self, cpp_move: Any) -> DraftMove | None:
        """Extract Python DraftMove from C++ MCTS move.

        Args:
            cpp_move: C++ MCTS_move pointer (wrapped by pymcts).

        Returns:
            DraftMove or None.
        """
        move_str = cpp_move.sprint() if hasattr(cpp_move, "sprint") else str(cpp_move)

        # Parse the sprint string to reconstruct the move
        # Format: "Radiant pick hero 10 at step 2" or "Dire ban hero 5 at step 0"
        parts = move_str.split()
        if len(parts) >= 7:
            team_str = parts[0]
            action = parts[1]
            hero_id = int(parts[3])
            step_index = int(parts[6])

            return DraftMove(
                hero_id=hero_id,
                is_pick=(action == "pick"),
                team=0 if team_str == "Radiant" else 1,
                step_index=step_index,
            )

        return None

    def update_state(self, move: DraftMove) -> None:
        """Update the internal state after a move.

        Advances the draft state by applying the action.
        Attempts to reuse the existing MCTS tree if the move is found
        among the root's children, preserving lookahead statistics.
        Falls back to a cold-start if the move is unexplored.

        Args:
            move: The DraftMove to apply.
        """
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

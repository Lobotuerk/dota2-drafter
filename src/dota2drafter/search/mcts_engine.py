"""Pure-Python MCTS engine with UCT selection.

Implements Monte Carlo Tree Search with Upper Confidence bounds for
Trees (UCT) selection, supporting adversarial minimax lookahead for
the Dota 2 draft recommendation system.

This module provides a self-contained MCTS implementation that wraps
the DraftState interface to perform tree search over the draft space.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from dota2drafter.search.state import DraftMove, DraftState


@dataclass
class MCTSNode:
    """A node in the MCTS search tree.

    Attributes:
        state: The DraftState at this node.
        parent: Parent node, or None for root.
        children: Dict mapping DraftMove to child MCTSNode.
        visit_count: Number of times this node has been visited.
        total_reward: Sum of rewards from all simulations through this node.
        move: The move taken to reach this node from parent.
    """

    state: DraftState
    parent: MCTSNode | None = None
    children: dict[DraftMove, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_reward: float = 0.0
    move: DraftMove | None = None

    def is_fully_expanded(self) -> bool:
        """Check if all valid actions have been expanded as children."""
        if self.state.is_terminal():
            return True
        return len(self.children) >= len(self.state.actions_to_try())

    def uct_value(self, c_puct: float) -> float:
        """Compute the UCT value for this node.

        UCT(S, a) = Q(S, a) + c_puct * P(a|S) * sqrt(N(S)) / (1 + N(S, a))

        Args:
            c_puct: Exploration constant for the PUCT formula.

        Returns:
            UCT value, or -inf if not visited.
        """
        if self.visit_count == 0:
            return float("inf")

        # Exploitation: average reward
        q_value = self.total_reward / self.visit_count

        # Exploration: UCT term
        if self.parent is not None:
            parent_visits = self.parent.visit_count
            exploration = c_puct * math.sqrt(parent_visits) / (1 + self.visit_count)
            return q_value + exploration

        return q_value


class MCTSConfig:
    """Configuration for the MCTS engine.

    Attributes:
        max_iterations: Maximum number of MCTS simulations.
        max_seconds: Maximum wall-clock time in seconds.
        c_puct: Exploration constant for the PUCT formula.
        rollout_depth: Depth of random rollout (0 = use model directly).
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        max_seconds: float = 30.0,
        c_puct: float = 1.414,
        rollout_depth: int = 0,
    ) -> None:
        """Initialize MCTS configuration.

        Args:
            max_iterations: Maximum simulations per search.
            max_seconds: Maximum wall-clock time in seconds.
            c_puct: PUCT exploration constant.
            rollout_depth: Unused (model rollouts are direct).
        """
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth


class MCTSEngine:
    """Monte Carlo Tree Search engine with UCT selection.

    Performs adversarial minimax lookahead over the draft tree, using
    the MatchNetwork to evaluate leaf nodes and comfort-scaled priors
    for action selection.

    Attributes:
        root: The root MCTSNode.
        config: MCTS configuration.
    """

    def __init__(self, root_state: DraftState, config: MCTSConfig | None = None) -> None:
        """Initialize the MCTS engine.

        Args:
            root_state: The starting DraftState for search.
            config: MCTS configuration (uses defaults if None).
        """
        self.config = config or MCTSConfig()
        self.root = MCTSNode(state=root_state)

    def run(self) -> None:
        """Execute the full MCTS search.

        Runs simulations until max_iterations or max_seconds is reached.
        Each simulation follows the selection-expansion-simulation-
        backpropagation cycle.
        """
        start_time = time.time()

        for _ in range(self.config.max_iterations):
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed >= self.config.max_seconds:
                break

            self._simulate()

    def _simulate(self) -> None:
        """Run a single MCTS simulation cycle.

        Follows the four phases:
        1. Selection: traverse from root using UCT
        2. Expansion: add a child node for an untried action
        3. Simulation: evaluate the leaf state
        4. Backpropagation: update visit counts and rewards
        """
        # Phase 1: Selection - traverse down using UCT
        node = self._select(self.root)

        # Phase 2: Expansion - add a child if not terminal and not fully expanded
        if not node.state.is_terminal() and not node.is_fully_expanded():
            node = self._expand(node)

        # Phase 3: Simulation - evaluate leaf state
        reward = self._simulate_rollout(node)

        # Phase 4: Backpropagation - update statistics up the tree
        self._backpropagate(node, reward)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select the best child using UCT until a leaf is reached.

        Args:
            node: Current node to start selection from.

        Returns:
            The leaf node reached by selection.
        """
        while not node.state.is_terminal() and node.is_fully_expanded():
            best_child = self._best_uct_child(node)
            if best_child is None:
                break
            node = best_child

        return node

    def _best_uct_child(self, node: MCTSNode) -> MCTSNode | None:
        """Find the child with the highest UCT value.

        Args:
            node: Parent node whose children to evaluate.

        Returns:
            The child with highest UCT value, or None if no children.
        """
        if not node.children:
            return None

        best_child = None
        best_uct = float("-inf")

        for child in node.children.values():
            uct = child.uct_value(self.config.c_puct)
            if uct > best_uct:
                best_uct = uct
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand the node by adding one untried action as a child.

        Args:
            node: Node to expand.

        Returns:
            The newly created child node.
        """
        valid_moves = node.state.actions_to_try()
        tried_moves = set(node.children.keys())

        # Find an untried move
        untried = [m for m in valid_moves if m not in tried_moves]
        if not untried:
            return node

        # Select the untried move with highest prior probability
        priors = node.state.get_action_probabilities()
        untried.sort(key=lambda m: priors.get(m.hero_id, 0.0), reverse=True)

        move = untried[0]
        new_state = node.state.next_state(move)
        child = MCTSNode(state=new_state, parent=node, move=move)
        node.children[move] = child

        return child

    def _simulate_rollout(self, node: MCTSNode) -> float:
        """Evaluate the leaf state via model rollout.

        Uses the MatchNetwork to compute the win probability for the
        active team at this state.

        Args:
            node: Leaf node to evaluate.

        Returns:
            Win probability for the active team from this state.
        """
        return node.state.rollout()

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Update visit counts and rewards up the tree.

        Args:
            node: The leaf node where the rollout ended.
            reward: The rollout reward (win probability).
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent

    def get_root(self) -> MCTSNode:
        """Get the root node of the search tree.

        Returns:
            The root MCTSNode after search completes.
        """
        return self.root


def get_recommendations(
    root_node: MCTSNode, top_n: int = 5
) -> list[tuple[DraftMove, int, float]]:
    """Extract top-N recommendations from the search tree.

    Sorts the root node's children by visit count and returns the
    most explored actions with their visit counts and average rewards.

    Args:
        root_node: The root MCTSNode after search.
        top_n: Number of recommendations to return.

    Returns:
        List of (move, visit_count, avg_reward) tuples, sorted by visit count.
    """
    children = list(root_node.children.items())
    # Sort by visit count descending, then by average reward
    children.sort(
        key=lambda item: (item[1].visit_count, item[1].total_reward / max(item[1].visit_count, 1)),
        reverse=True,
    )

    recommendations: list[tuple[DraftMove, int, float]] = []
    for move, child in children[:top_n]:
        avg_reward = child.total_reward / max(child.visit_count, 1)
        recommendations.append((move, child.visit_count, avg_reward))

    return recommendations


def get_principal_variation(root_node: MCTSNode) -> list[DraftMove]:
    """Extract the principal variation (most visited path) from root.

    Walks down the tree always selecting the child with the highest
    visit count until a terminal state is reached.

    Args:
        root_node: The root MCTSNode after search.

    Returns:
        List of DraftMove objects representing the principal variation.
    """
    variation: list[DraftMove] = []
    node: MCTSNode | None = root_node

    while node is not None and not node.state.is_terminal():
        if not node.children:
            break

        # Select child with highest visit count
        best_child = max(node.children.values(), key=lambda c: c.visit_count)
        if best_child.move is not None:
            variation.append(best_child.move)
        node = best_child

    return variation

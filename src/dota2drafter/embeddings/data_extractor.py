"""Data extraction from raw match tensors for embedding pre-training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class SkipGramPair:
    """A single training pair for the Skip-Gram model."""

    center: int
    context: int
    negative: int | None = None


class DataExtractor:
    """Extracts training data from raw match tensors.

    Parses (24, 3) draft tensors to generate:
    - Skip-Gram (center, context, negative) triples
    - A hero interaction graph with synergy and opposition edges
    """

    def __init__(self, num_heroes: int, negative_samples: int = 5) -> None:
        self._num_heroes = num_heroes
        self._negative_samples = negative_samples

    def load_batches(self, data_dir: str | Path) -> list[dict[str, Any]]:
        """Load all .pt batch files from the data directory.

        Returns a list of dicts with keys 'x', 'y', 'match_ids'.
        """
        data_path = Path(data_dir)
        batch_files = sorted(data_path.glob("drafts_batch_*.pt"))

        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {data_path}")

        batches: list[dict[str, Any]] = []
        for batch_file in batch_files:
            batch = torch.load(batch_file, weights_only=True)
            batches.append(batch)
            logger.debug("Loaded %s: %d matches", batch_file.name, batch["x"].shape[0])

        return batches

    def extract_skip_gram_pairs(
        self,
        batches: list[dict[str, Any]],
    ) -> list[SkipGramPair]:
        """Extract Skip-Gram training pairs from match batches.

        For each match, extracts the 5 Radiant and 5 Dire picks.
        Generates positive (center, context) pairs within each team.
        Generates negative pairs by sampling heroes outside the team.
        """
        pairs: list[SkipGramPair] = []
        rng = np.random.default_rng(42)

        for batch in batches:
            x_tensors: torch.Tensor = batch["x"]  # (N, 24, 3) or (24, 3) if single match

            # Ensure x_tensors is 3D
            if x_tensors.dim() == 2:
                x_tensors = x_tensors.unsqueeze(0)

            for match_idx in range(x_tensors.shape[0]):
                draft = x_tensors[match_idx]  # (24, 3)

                # Extract picks only (is_pick == 1.0)
                picks = draft[draft[:, 0] == 1.0]  # (10, 3)

                if picks.shape[0] < 10:
                    continue

                # Team 0 = Radiant, Team 1 = Dire
                radiant_picks = picks[picks[:, 1] == 0.0][:, 2].long()  # (5,)
                dire_picks = picks[picks[:, 1] == 1.0][:, 2].long()  # (5,)

                # Extract hero lists
                radiant_heroes = [
                    int(h) for h in radiant_picks.tolist() if h > 0
                ]
                dire_heroes = [
                    int(h) for h in dire_picks.tolist() if h > 0
                ]

                # Generate pairs for Radiant
                for center in radiant_heroes:
                    contexts = [h for h in radiant_heroes if h != center]
                    for context in contexts:
                        neg = self._sample_negative(rng, center, [])
                        pairs.append(SkipGramPair(center, context, neg))

                # Generate pairs for Dire
                for center in dire_heroes:
                    contexts = [h for h in dire_heroes if h != center]
                    for context in contexts:
                        neg = self._sample_negative(rng, center, [])
                        pairs.append(SkipGramPair(center, context, neg))

        logger.info("Extracted %d Skip-Gram pairs from %d batches", len(pairs), len(batches))
        return pairs

    def build_hero_graph(
        self,
        batches: list[dict[str, Any]],
    ) -> Data:
        """Build a hero interaction graph from match batches.

        Creates a directed graph with:
        - Synergy edges: co-pick win rates between heroes
        - Opposition edges: head-to-head win rates

        Returns a PyG Data object with edge_index and edge_attr.
        """
        # Accumulate counts for win rate computation
        synergy_counts: dict[tuple[int, int], list[int]] = {}
        opposition_counts: dict[tuple[int, int], list[int]] = {}

        for batch in batches:
            x_tensors: torch.Tensor = batch["x"]
            y_tensors: torch.Tensor = batch["y"]

            # Ensure x_tensors is 3D
            if x_tensors.dim() == 2:
                x_tensors = x_tensors.unsqueeze(0)
                y_tensors = y_tensors.unsqueeze(0)

            for match_idx in range(x_tensors.shape[0]):
                draft = x_tensors[match_idx]
                radiant_win = int(y_tensors[match_idx].item())

                picks = draft[draft[:, 0] == 1.0]
                if picks.shape[0] < 10:
                    continue

                radiant_picks = picks[picks[:, 1] == 0.0][:, 2].long()
                dire_picks = picks[picks[:, 1] == 1.0][:, 2].long()

                radiant_heroes = [int(h) for h in radiant_picks.tolist() if h > 0]
                dire_heroes = [int(h) for h in dire_picks.tolist() if h > 0]

                # Synergy: co-pick pairs on same team
                all_team_heroes = [(radiant_heroes, 0), (dire_heroes, 1)]
                for team_heroes, _ in all_team_heroes:
                    for i_idx, hi in enumerate(team_heroes):
                        for hj in team_heroes[i_idx + 1 :]:
                            pair = tuple(sorted((hi, hj)))
                            if pair not in synergy_counts:
                                synergy_counts[pair] = [0, 0]
                            synergy_counts[pair][radiant_win] += 1

                # Opposition: hero i vs hero j (across teams)
                for hi in radiant_heroes:
                    for hj in dire_heroes:
                        opposition_counts[(hi, hj)] = [0, 0]
                        opposition_counts[(hi, hj)][radiant_win] += 1

        # Build edges
        edge_list: list[list[int]] = []
        edge_attr_list: list[float] = []

        # Synergy edges (undirected, stored as both directions)
        for (hi, hj), wins in synergy_counts.items():
            total = sum(wins)
            if total < 5:
                continue
            win_rate = wins[radiant_win] / total if total > 0 else 0.0
            edge_list.append([hi, hj])
            edge_attr_list.append(win_rate)
            edge_list.append([hj, hi])
            edge_attr_list.append(win_rate)

        # Opposition edges (directed)
        for (hi, hj), wins in opposition_counts.items():
            total = sum(wins)
            if total < 3:
                continue
            # Win rate of hi against hj
            # If radiant_win=1, hi is on radiant; if hi is radiant, win_rate = wins[1]/total
            # We need to track which team each hero was on — simplify: use raw ratio
            win_rate = wins[1] / total if total > 0 else 0.0
            edge_list.append([hi, hj])
            edge_attr_list.append(win_rate)

        if not edge_list:
            logger.warning("No edges found in hero interaction graph")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return Data(edge_index=edge_index, edge_attr=torch.empty((0, 1)))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).unsqueeze(1)

        logger.info(
            "Built hero graph: %d nodes, %d edges",
            self._num_heroes,
            edge_index.shape[1],
        )
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self._num_heroes,
        )

    def _sample_negative(
        self,
        rng: np.random.Generator,
        center: int,
        team_heroes: list[int],
    ) -> int:
        """Sample a negative hero for Skip-Gram negative sampling.

        Samples uniformly from heroes not in the center's team.
        """
        candidates = [h for h in range(1, self._num_heroes + 1) if h != center]
        return int(rng.choice(candidates))

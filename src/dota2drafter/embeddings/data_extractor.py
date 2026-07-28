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

# Edge type constants for multi-relational graph
SYNERGY = 0        # r_syn: Co-Picked-Radiant / Co-Picked-Dire
ANTAGONIST = 1     # r_ant: Mechanical counter-picks
BANNED_AGAINST = 2 # r_ban: Banned-Against correlations


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
    - A multi-relational hero interaction graph with synergy,
      antagonist, and banned-against edges
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
        """Build a multi-relational hero interaction graph from match batches.

        Creates a graph with three edge types compatible with PyG's RGCNConv:
        - Edge type 0 (SYNERGY): Co-picked-Radiant/Dire undirected edges
        - Edge type 1 (ANTAGONIST): Directed mechanical counter-pick edges
        - Edge type 2 (BANNED_AGAINST): Directed banned-against edges

        Returns a PyG Data object with edge_index, edge_type, and edge_weight.
        """
        synergy_counts: dict[tuple[int, int], list[int]] = {}
        antagonist_counts: dict[tuple[int, int], list[int]] = {}
        ban_counts: dict[tuple[int, int], int] = {}

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

                # Extract ban steps (is_pick == 0.0)
                ban_steps = draft[draft[:, 0] == 0.0]
                radiant_bans = ban_steps[ban_steps[:, 1] == 0.0][:, 2].long().tolist()
                dire_bans = ban_steps[ban_steps[:, 1] == 1.0][:, 2].long().tolist()

                # --- Synergy edges (type 0, undirected) ---
                # Co-picked pairs on same team
                for i_idx, hi in enumerate(radiant_heroes):
                    for hj in radiant_heroes[i_idx + 1:]:
                        pair = tuple(sorted((hi, hj)))
                        if pair not in synergy_counts:
                            synergy_counts[pair] = [0, 0]
                        synergy_counts[pair][0] += radiant_win
                        synergy_counts[pair][1] += 1

                for i_idx, hi in enumerate(dire_heroes):
                    for hj in dire_heroes[i_idx + 1:]:
                        pair = tuple(sorted((hi, hj)))
                        if pair not in synergy_counts:
                            synergy_counts[pair] = [0, 0]
                        synergy_counts[pair][0] += (1 - radiant_win)
                        synergy_counts[pair][1] += 1

                # --- Antagonist edges (type 1, directed) ---
                # hero i counter-picks hero j: i is on winning team, j on losing team
                for hi in radiant_heroes:
                    for hj in dire_heroes:
                        if (hi, hj) not in antagonist_counts:
                            antagonist_counts[(hi, hj)] = [0, 0]
                        antagonist_counts[(hi, hj)][0] += radiant_win
                        antagonist_counts[(hi, hj)][1] += 1

                        if (hj, hi) not in antagonist_counts:
                            antagonist_counts[(hj, hi)] = [0, 0]
                        antagonist_counts[(hj, hi)][0] += (1 - radiant_win)
                        antagonist_counts[(hj, hi)][1] += 1

                # --- Banned-Against edges (type 2, directed) ---
                # hero i picked by Team A, hero j banned by Team B
                for pick_h in radiant_heroes:
                    for ban_h in dire_bans:
                        ban_h = int(ban_h)
                        if ban_h > 0:
                            ban_counts[(pick_h, ban_h)] = ban_counts.get((pick_h, ban_h), 0) + 1

                for pick_h in dire_heroes:
                    for ban_h in radiant_bans:
                        ban_h = int(ban_h)
                        if ban_h > 0:
                            ban_counts[(pick_h, ban_h)] = ban_counts.get((pick_h, ban_h), 0) + 1

        # Build edges
        edge_list: list[list[int]] = []
        edge_type_list: list[int] = []
        edge_weight_list: list[float] = []

        # Synergy edges (type 0, undirected)
        for (hi, hj), (wins, total) in synergy_counts.items():
            if total < 5:
                continue
            win_rate = wins / total
            edge_list.append([hi, hj])
            edge_type_list.append(SYNERGY)
            edge_weight_list.append(win_rate)
            edge_list.append([hj, hi])
            edge_type_list.append(SYNERGY)
            edge_weight_list.append(win_rate)

        # Antagonist edges (type 1, directed)
        for (u, v), (wins, total) in antagonist_counts.items():
            if total < 3:
                continue
            win_rate = wins / total
            edge_list.append([u, v])
            edge_type_list.append(ANTAGONIST)
            edge_weight_list.append(win_rate)

        # Banned-Against edges (type 2, directed)
        max_count = max(ban_counts.values()) if ban_counts else 1
        for (pick_h, ban_h), count in ban_counts.items():
            normalized = count / max_count
            edge_list.append([pick_h, ban_h])
            edge_type_list.append(BANNED_AGAINST)
            edge_weight_list.append(normalized)

        if not edge_list:
            logger.warning("No edges found in multi-relational hero graph")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
            edge_weight = torch.empty((0, 1), dtype=torch.float32)
            return Data(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_weight=edge_weight,
                num_nodes=self._num_heroes + 1,
            )

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32).unsqueeze(1)

        logger.info(
            "Built multi-relational hero graph: %d nodes, %d edges (syn=%d, ant=%d, ban=%d)",
            self._num_heroes + 1,
            edge_index.shape[1],
            sum(1 for t in edge_type_list if t == SYNERGY),
            sum(1 for t in edge_type_list if t == ANTAGONIST),
            sum(1 for t in edge_type_list if t == BANNED_AGAINST),
        )
        return Data(
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
            num_nodes=self._num_heroes + 1,
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

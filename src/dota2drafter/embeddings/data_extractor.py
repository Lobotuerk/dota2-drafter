"""Data extraction from raw match tensors for embedding pre-training."""

from __future__ import annotations

import logging
import math
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
REQUIRED_BANS = 2  # r_req_ban: Directed required-bans edges (win-rate weighted)


@dataclass
class SkipGramPair:
    """A single training pair for the Skip-Gram model."""

    center: int
    context: int
    negative: int | None = None


class DataExtractor:
    """Extracts training data from raw match tensors.

    Parses (24, 4) draft tensors to generate:
    - Skip-Gram (center, context, negative) triples
    - A multi-relational hero interaction graph with synergy,
      antagonist, and required-bans edges
    """

    def __init__(self, num_heroes: int, negative_samples: int = 5) -> None:
        self._num_heroes = num_heroes
        self._negative_samples = negative_samples

    @staticmethod
    def _wilson_score_eff(p_hat: float, n_eff: float, z: float = 1.96) -> float:
        """Calculate Wilson lower bound score using effective sample size.
        
        Args:
            p_hat: Weighted proportion estimate
            n_eff: Effective sample size (N_w^2 / S_w)
            z: Z-score for confidence level (default 1.96 for 95% confidence)
            
        Returns:
            Wilson lower bound score
        """
        if n_eff <= 0:
            return 0.5
        # Clamp p_hat to ensure numerical safety under sqrt
        p_hat = max(0.0, min(1.0, p_hat))
        denominator = 1 + z**2 / n_eff
        center = p_hat + z**2 / (2 * n_eff)
        spread = z * math.sqrt((p_hat * (1 - p_hat) / n_eff) + z**2 / (4 * n_eff**2))
        return (center - spread) / denominator

    @staticmethod
    def load_batches(data_dir: str | Path) -> list[dict[str, Any]]:
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
            x_tensors: torch.Tensor = batch["x"]  # (N, 24, 4) or (24, 4) if single match

            # Ensure x_tensors is 3D
            if x_tensors.dim() == 2:
                x_tensors = x_tensors.unsqueeze(0)

            for match_idx in range(x_tensors.shape[0]):
                draft = x_tensors[match_idx]  # (24, 4)

                # Extract picks only (is_pick == 1.0)
                picks = draft[draft[:, 0] == 1.0]  # (10, 4)

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

    def build_pruned_hero_graph(
        self,
        batches: list[dict[str, Any]],
        wilson_threshold: float = 0.50,
        gamma: float = 0.80,
        percentile_keep: float | None = None,
    ) -> Data:
        """Build a multi-relational hero graph using Patch-Weighted Wilson Score.
        
        Computes Wilson lower bound scores with patch-distance discounting for edge weighting.
        Prunes edges where Wilson Score <= threshold.
        
        Args:
            batches: List of batch dictionaries containing match data
            wilson_threshold: Minimum Wilson Score to keep an edge (default 0.50)
            gamma: Decay factor per major patch (default 0.80)
            percentile_keep: Deprecated parameter, kept for backward compatibility
            
        Returns:
            PyG Data object with edge_index, edge_type, and edge_weight
        """
        # Harden tensor shape validation
        for batch in batches:
            x_tensors: torch.Tensor = batch["x"]
            if x_tensors.dim() == 2:
                x_tensors = x_tensors.unsqueeze(0)
            if x_tensors.shape[-1] < 4:
                raise ValueError(
                    "REQUIRED_BANS requires step_index as the 4th column in draft tensors."
                )

        # Step 1: Current Patch Discovery
        max_patch_id = 0
        has_patches = False
        for batch in batches:
            p_ids = batch.get("patch_ids")
            if p_ids is not None and p_ids.numel() > 0:
                has_patches = True
                batch_max = int(p_ids.max().item())
                if batch_max > max_patch_id:
                    max_patch_id = batch_max

        P_current = max_patch_id if has_patches else 0

        # Accumulate weighted statistics for each relation type
        # Keys: (hero_pair) -> [N_w, W_w, S_w, total_count]
        synergy_stats: dict[tuple[int, int], list[float]] = {}
        antagonist_stats: dict[tuple[int, int], list[float]] = {}
        req_ban_stats: dict[tuple[int, int], list[float]] = {}

        for batch in batches:
            x_tensors: torch.Tensor = batch["x"]
            y_tensors: torch.Tensor = batch["y"]
            p_ids = batch.get("patch_ids")

            # Ensure x_tensors is 3D
            if x_tensors.dim() == 2:
                x_tensors = x_tensors.unsqueeze(0)
                y_tensors = y_tensors.unsqueeze(0)

            for match_idx in range(x_tensors.shape[0]):
                draft = x_tensors[match_idx]
                radiant_win = int(y_tensors[match_idx].item())

                # Get patch ID for this match
                if has_patches and p_ids is not None and match_idx < len(p_ids):
                    p_id = int(p_ids[match_idx].item())
                else:
                    p_id = P_current

                # Compute patch distance and weight
                delta_p = max(0, P_current - p_id)
                w_m = gamma ** delta_p

                picks = draft[draft[:, 0] == 1.0]
                if picks.shape[0] < 10:
                    continue

                radiant_picks = picks[picks[:, 1] == 0.0][:, 2].long()
                dire_picks = picks[picks[:, 1] == 1.0][:, 2].long()

                radiant_heroes = [int(h) for h in radiant_picks.tolist() if h > 0]
                dire_heroes = [int(h) for h in dire_picks.tolist() if h > 0]

                # --- Synergy edges (type 0, undirected) ---
                for i_idx, hi in enumerate(radiant_heroes):
                    for hj in radiant_heroes[i_idx + 1:]:
                        pair = tuple(sorted((hi, hj)))
                        if pair not in synergy_stats:
                            synergy_stats[pair] = [0.0, 0.0, 0.0, 0]
                        # y_m = radiant_win for Radiant team
                        synergy_stats[pair][0] += w_m  # N_w
                        synergy_stats[pair][1] += radiant_win * w_m  # W_w
                        synergy_stats[pair][2] += w_m ** 2  # S_w
                        synergy_stats[pair][3] += 1  # total_count

                for i_idx, hi in enumerate(dire_heroes):
                    for hj in dire_heroes[i_idx + 1:]:
                        pair = tuple(sorted((hi, hj)))
                        if pair not in synergy_stats:
                            synergy_stats[pair] = [0.0, 0.0, 0.0, 0]
                        # y_m = 1 - radiant_win for Dire team
                        synergy_stats[pair][0] += w_m  # N_w
                        synergy_stats[pair][1] += (1 - radiant_win) * w_m  # W_w
                        synergy_stats[pair][2] += w_m ** 2  # S_w
                        synergy_stats[pair][3] += 1  # total_count

                # --- Antagonist edges (type 1, directed) ---
                for hi in radiant_heroes:
                    for hj in dire_heroes:
                        if (hi, hj) not in antagonist_stats:
                            antagonist_stats[(hi, hj)] = [0.0, 0.0, 0.0, 0]
                        # y_m = radiant_win (hi on Radiant, hj on Dire)
                        antagonist_stats[(hi, hj)][0] += w_m
                        antagonist_stats[(hi, hj)][1] += radiant_win * w_m
                        antagonist_stats[(hi, hj)][2] += w_m ** 2
                        antagonist_stats[(hi, hj)][3] += 1

                        if (hj, hi) not in antagonist_stats:
                            antagonist_stats[(hj, hi)] = [0.0, 0.0, 0.0, 0]
                        # y_m = 1 - radiant_win (hj on Dire, hi on Radiant)
                        antagonist_stats[(hj, hi)][0] += w_m
                        antagonist_stats[(hj, hi)][1] += (1 - radiant_win) * w_m
                        antagonist_stats[(hj, hi)][2] += w_m ** 2
                        antagonist_stats[(hj, hi)][3] += 1

                # --- REQUIRED_BANS edges (type 2, directed) ---
                for pick_row in draft:
                    if pick_row[0] != 1.0:
                        continue
                    p_team = int(pick_row[1])
                    p_hero = int(pick_row[2])
                    p_step = int(pick_row[3])
                    p_win = radiant_win if p_team == 0 else (1 - radiant_win)

                    for ban_row in draft:
                        if ban_row[0] != 0.0:
                            continue
                        b_team = int(ban_row[1])
                        b_hero = int(ban_row[2])
                        b_step = int(ban_row[3])

                        if b_step < p_step or (b_step > p_step and b_team == p_team):
                            if b_hero <= 0:
                                continue
                            if (p_hero, b_hero) not in req_ban_stats:
                                req_ban_stats[(p_hero, b_hero)] = [0.0, 0.0, 0.0, 0]
                            # y_m = p_win
                            req_ban_stats[(p_hero, b_hero)][0] += w_m
                            req_ban_stats[(p_hero, b_hero)][1] += p_win * w_m
                            req_ban_stats[(p_hero, b_hero)][2] += w_m ** 2
                            req_ban_stats[(p_hero, b_hero)][3] += 1

        # Build edges using Wilson Score pruning
        edge_list: list[list[int]] = []
        edge_type_list: list[int] = []
        edge_weight_list: list[float] = []

        # Synergy edges (type 0, undirected)
        for (hi, hj), (n_w, w_w, s_w, total_count) in synergy_stats.items():
            if total_count < 5:  # Hard floor for synergy
                continue
            if n_w <= 0:
                continue
            n_eff = (n_w ** 2) / s_w if s_w > 0 else 0
            p_hat = w_w / n_w
            wilson_score = self._wilson_score_eff(p_hat, n_eff)
            if wilson_score <= wilson_threshold:
                continue
            edge_list.append([hi, hj])
            edge_type_list.append(SYNERGY)
            edge_weight_list.append(wilson_score)
            edge_list.append([hj, hi])
            edge_type_list.append(SYNERGY)
            edge_weight_list.append(wilson_score)

        # Antagonist edges (type 1, directed)
        for (u, v), (n_w, w_w, s_w, total_count) in antagonist_stats.items():
            if total_count < 3:  # Hard floor for antagonist
                continue
            if n_w <= 0:
                continue
            n_eff = (n_w ** 2) / s_w if s_w > 0 else 0
            p_hat = w_w / n_w
            wilson_score = self._wilson_score_eff(p_hat, n_eff)
            if wilson_score <= wilson_threshold:
                continue
            edge_list.append([u, v])
            edge_type_list.append(ANTAGONIST)
            edge_weight_list.append(wilson_score)

        # REQUIRED_BANS edges (type 2, directed)
        for (pick_h, ban_h), (n_w, w_w, s_w, total_count) in req_ban_stats.items():
            if total_count < 3:  # Hard floor for required bans
                continue
            if n_w <= 0:
                continue
            n_eff = (n_w ** 2) / s_w if s_w > 0 else 0
            p_hat = w_w / n_w
            wilson_score = self._wilson_score_eff(p_hat, n_eff)
            if wilson_score <= wilson_threshold:
                continue
            edge_list.append([pick_h, ban_h])
            edge_type_list.append(REQUIRED_BANS)
            edge_weight_list.append(wilson_score)

        if not edge_list:
            logger.warning("No edges found in pruned hero graph")
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
            "Built pruned multi-relational hero graph: %d nodes, %d edges "
            "(syn=%d, ant=%d, req_ban=%d)",
            self._num_heroes + 1,
            edge_index.shape[1],
            sum(1 for t in edge_type_list if t == SYNERGY),
            sum(1 for t in edge_type_list if t == ANTAGONIST),
            sum(1 for t in edge_type_list if t == REQUIRED_BANS),
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

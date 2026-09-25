"""Transformer trainer - training loop, validation, metrics, and checkpointing."""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dota2drafter.models.match_network import MatchNetwork

logger = logging.getLogger(__name__)


def apply_prefix_truncation(
    x_draft: torch.Tensor,
    y_label: torch.Tensor,
    radiant_players: list[int],
    dire_players: list[int],
    player_comfort_map: dict[int, torch.Tensor] | None = None,
    player_input_dim: int = 127,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Apply multi-prefix sequence crop augmentation.

    Extracts partial drafts at multiple truncation points (6, 12, 18, 24) and
    returns augmented samples where steps beyond the truncation point are zeroed out.

    Args:
        x_draft: Draft sequence tensor of shape (24, 4).
        y_label: Label tensor of shape (1,).
        radiant_players: List of Radiant player account IDs.
        dire_players: List of Dire player account IDs.
        player_comfort_map: Optional mapping of account_id -> comfort tensor.
        player_input_dim: Number of features per comfort vector.

    Returns:
        List of (x_draft, player_comfort, y) tuples for each truncation point.
    """
    truncation_points = [7, 9, 12, 18, 22, 24]
    samples = []

    for t in truncation_points:
        x_truncated = x_draft.clone()
        if t < 24:
            x_truncated[t:, :] = 0.0
            x_truncated[t:, 2] = -1.0  # Assign explicit hero padding sentinel

        comfort_rows: list[torch.Tensor] = []
        comfort_map = player_comfort_map or {}
        for account_id in radiant_players:
            if account_id == 0:
                comfort_rows.append(torch.zeros(player_input_dim))
            elif account_id in comfort_map:
                comfort_rows.append(comfort_map[account_id])
            else:
                comfort_rows.append(torch.zeros(player_input_dim))
        for account_id in dire_players:
            if account_id == 0:
                comfort_rows.append(torch.zeros(player_input_dim))
            elif account_id in comfort_map:
                comfort_rows.append(comfort_map[account_id])
            else:
                comfort_rows.append(torch.zeros(player_input_dim))
        player_comfort = torch.stack(comfort_rows)

        samples.append((x_truncated, player_comfort, y_label))

    return samples


def augment_draft_permutations(
    x_draft: torch.Tensor,
    y_label: torch.Tensor,
    radiant_players: list[int],
    dire_players: list[int],
    player_comfort_map: dict[int, torch.Tensor] | None = None,
    player_input_dim: int = 127,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Apply intra-phase draft permutation augmentation.

    Exhaustively generates all possible (64) permutations of same-team, same-phase
    draft actions (which are mathematically order-invariant in Captains Mode).

    Args:
        x_draft: Draft sequence tensor of shape (24, 4).
        y_label: Label tensor of shape (1,).
        radiant_players: List of Radiant player account IDs.
        dire_players: List of Dire player account IDs.
        player_comfort_map: Optional mapping of account_id -> comfort tensor.
        player_input_dim: Number of features per comfort vector.

    Returns:
        List of (x_draft, player_comfort, y) tuples including all permutation combinations.
    """
    samples = []
    comfort_map = player_comfort_map or {}

    # Phase groups: same-team, same-phase draft actions that are order-invariant
    phase_groups = [
        [0, 1],
        [2, 3],
        [5, 6],
        [9, 10],
        [13, 14],
        [15, 16]
    ]

    # Generate all possible permutations for each group
    group_permutations = []
    for group in phase_groups:
        perms = list(itertools.permutations(group))
        group_permutations.append(perms)

    # Compute Cartesian product across all groups to get 2^6 = 64 combinations
    all_perm_combinations = list(itertools.product(*group_permutations))

    # Build player comfort row once (identical for all permutations of this match)
    comfort_rows: list[torch.Tensor] = []
    for account_id in radiant_players:
        if account_id == 0:
            comfort_rows.append(torch.zeros(player_input_dim))
        elif account_id in comfort_map:
            comfort_rows.append(comfort_map[account_id])
        else:
            comfort_rows.append(torch.zeros(player_input_dim))
    for account_id in dire_players:
        if account_id == 0:
            comfort_rows.append(torch.zeros(player_input_dim))
        elif account_id in comfort_map:
            comfort_rows.append(comfort_map[account_id])
        else:
            comfort_rows.append(torch.zeros(player_input_dim))
    player_comfort = torch.stack(comfort_rows)

    for perm_comb in all_perm_combinations:
        x_permuted = x_draft.clone()
        for original_group, perm_group in zip(phase_groups, perm_comb):
            # Safe slice assignment of permuted values
            temp = x_draft[list(perm_group)].clone()
            x_permuted[original_group] = temp
        samples.append((x_permuted, player_comfort, y_label))

    return samples


def is_pub_draft(x_draft: torch.Tensor) -> bool:
    """Check if x_draft is in pub match format (no bans, picks in slots 0..9, padding in 10..23)."""
    has_bans = ((x_draft[:, 0] == 0.0) & (x_draft[:, 2] >= 0.0)).any().item()
    return not has_bans and (x_draft[0, 0] == 1.0).item()


def augment_pub_permutations(
    x_draft: torch.Tensor,
    y_label: torch.Tensor,
    radiant_players: list[int],
    dire_players: list[int],
    player_comfort_map: dict[int, torch.Tensor] | None = None,
    player_input_dim: int = 127,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Apply intra-team pick permutation augmentation for high-MMR pub matches.

    Ranked All Pick matches have 5 Radiant picks in slots 0..4, 5 Dire picks
    in slots 5..9, no bans, and padding in slots 10..23.
    Generates 64 valid, distinct permutations of intra-team pick orders
    while keeping slots 10..23 preserved as padding.

    Args:
        x_draft: Pub draft tensor of shape (24, 4).
        y_label: Label tensor of shape (1,).
        radiant_players: List of Radiant player account IDs.
        dire_players: List of Dire player account IDs.
        player_comfort_map: Optional mapping of account_id -> comfort tensor.
        player_input_dim: Number of features per comfort vector.

    Returns:
        List of (x_draft, player_comfort, y) tuples including 64 permutation combinations.
    """
    comfort_map = player_comfort_map or {}

    comfort_rows: list[torch.Tensor] = []
    for account_id in radiant_players:
        if account_id == 0:
            comfort_rows.append(torch.zeros(player_input_dim))
        elif account_id in comfort_map:
            comfort_rows.append(comfort_map[account_id])
        else:
            comfort_rows.append(torch.zeros(player_input_dim))
    for account_id in dire_players:
        if account_id == 0:
            comfort_rows.append(torch.zeros(player_input_dim))
        elif account_id in comfort_map:
            comfort_rows.append(comfort_map[account_id])
        else:
            comfort_rows.append(torch.zeros(player_input_dim))
    player_comfort = torch.stack(comfort_rows)

    # 8 distinct permutations for Radiant picks (indices 0..4)
    # Covering intra-round swaps: round 1 (0, 1), round 2 (2, 3), and cross-round swaps
    radiant_perms = [
        [0, 1, 2, 3, 4],  # Identity
        [1, 0, 2, 3, 4],  # Swap round 1
        [0, 1, 3, 2, 4],  # Swap round 2
        [1, 0, 3, 2, 4],  # Swap both round 1 and 2
        [4, 1, 2, 3, 0],  # Swap first and last pick
        [0, 4, 2, 3, 1],  # Swap second and last pick
        [2, 3, 0, 1, 4],  # Swap round 1 and round 2 pairs
        [3, 2, 1, 0, 4],  # Reverse first 4 picks
    ]

    # 8 distinct relative permutations for Dire picks (indices 5..9)
    dire_perms = [
        [5, 6, 7, 8, 9],  # Identity
        [6, 5, 7, 8, 9],  # Swap round 1
        [5, 6, 8, 7, 9],  # Swap round 2
        [6, 5, 8, 7, 9],  # Swap both round 1 and 2
        [9, 6, 7, 8, 5],  # Swap first and last pick
        [5, 9, 7, 8, 6],  # Swap second and last pick
        [7, 8, 5, 6, 9],  # Swap round 1 and round 2 pairs
        [8, 7, 6, 5, 9],  # Reverse first 4 picks
    ]

    samples = []
    # Cartesian product: 8 x 8 = 64 distinct permutation combinations
    for r_perm, d_perm in itertools.product(radiant_perms, dire_perms):
        x_permuted = x_draft.clone()
        # Permute Radiant picks
        x_permuted[0:5] = x_draft[r_perm].clone()
        for s in range(5):
            x_permuted[s, 3] = float(s)

        # Permute Dire picks
        x_permuted[5:10] = x_draft[d_perm].clone()
        for s in range(5, 10):
            x_permuted[s, 3] = float(s)

        # Slots 10..23 remain untouched padding

        # Permute player comfort rows correspondingly so players match their heroes
        p_comfort_perm = player_comfort.clone()
        p_comfort_perm[0:5] = player_comfort[r_perm].clone()
        p_comfort_perm[5:10] = player_comfort[d_perm].clone()

        samples.append((x_permuted, p_comfort_perm, y_label))

    return samples


@dataclass
class TrainingConfig:
    """Configuration for the Transformer training loop."""

    learning_rate: float = 1e-4
    lr_backbone: float | None = None
    lr_head: float | None = None
    step_loss_gamma: float = 0.0
    num_epochs: int = 50
    batch_size: int = 64
    val_split: float = 0.2
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    patience: int = 25
    min_delta: float = 1e-4
    label_smoothing_eps: float = 0.15
    augment: int | str | bool = "false"

    # Temperature Annealing parameters for slot routing
    slot_tau_start: float = 0.30
    slot_tau_end: float = 0.05
    slot_tau_decay_epochs: int = 100
    checkpoint_metric: str = "val_auc"

    # AW-MLM Parameters
    aw_tau_start: float = 0.15
    aw_tau_end: float = 0.08
    aw_tau_decay_epochs: int = 50
    aw_clip_min: float = 0.1
    aw_clip_max: float = 10.0

    # Two-Stage Training Parameters
    stage: int | None = None
    draft_sample_weight: float = 5.0
    pub_data_dir: str = "data"
    pub_epochs: int | None = None
    stage1_checkpoint_path: str | Path | None = None


@dataclass
class TrainingMetrics:
    """Tracks training and validation metrics across epochs."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    val_auc_scores: list[float] = field(default_factory=list)
    val_mlm_accuracies: list[float] = field(default_factory=list)
    val_mlm_top5_accuracies: list[float] = field(default_factory=list)
    val_brier_scores: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_mlm_top5_acc: float = float("-inf")
    best_val_auc: float = float("-inf")
    best_val_loss: float = float("inf")
    best_brier_score: float = float("inf")
    best_checkpoint_value: float = float("-inf")
    calibrated_temperature: float = 1.0


class PlayerComfortDataset(Dataset):
    """PyTorch Dataset that yields (x_draft, player_matrices, y) from raw data.

    Dynamically looks up historical player matrices given account IDs.
    Supports multi-prefix sequence crop and intra-phase permutation augmentation.
    """

    def __init__(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]],
        dire_players: list[list[int]],
        player_comfort_map: dict[int, torch.Tensor] | None = None,
        player_input_dim: int = 127,
        augment: int | bool = 0,
        patch_ids: list[torch.Tensor] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            x_drafts: List of draft sequence tensors, each (24, 4).
            y_labels: List of label tensors, each (1,).
            radiant_players: List of Radiant player account ID lists (5 IDs each).
            dire_players: List of Dire player account ID lists (5 IDs each).
            player_comfort_map: Optional mapping of account_id -> comfort tensor (10, C).
            player_input_dim: C, number of features per comfort vector.
            augment: If True, apply all variations (448). If an integer, precomputes all
                     variations but limits the exposed items to `augment` per match.
        """
        self.x_drafts = x_drafts
        self.y_labels = y_labels
        self.radiant_players = radiant_players
        self.dire_players = dire_players
        self.player_comfort_map = player_comfort_map or {}
        self.player_input_dim = player_input_dim
        self.patch_ids = patch_ids

        # Parse augment type and determine limit per original match
        if isinstance(augment, bool):
            if augment:
                self.augment_limit = 448  # Default to exposing all 448 augmentations if True
                self.augment = True
            else:
                self.augment_limit = 0
                self.augment = False
        elif isinstance(augment, int):
            if augment > 0:
                self.augment_limit = augment
                self.augment = True
            else:
                self.augment_limit = 0
                self.augment = False
        else:
            self.augment_limit = 0
            self.augment = False

        # Pre-compute all samples upfront to avoid per-call permutation generation
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.all_augmented_samples: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        self.selected_indices: list[list[int]] = []

        if not self.augment:
            for idx in range(len(x_drafts)):
                self.samples.append(self._build_player_comfort_sample(idx))
        else:
            cm_truncation_points = [6, 8, 11, 17, 21, 23]
            # Balanced crop stages for pub games: (num_radiant_picks, num_dire_picks)
            # Simulating partial draft stages: 1v1, 2v2, 3v3, 4v4, 5v4, 4v5
            pub_crop_stages = [
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 4),
                (4, 5),
            ]

            # Calculate how many permutations to evaluate per match to maintain a sufficient
            # pool for reshuffle_augmentations without allocating tens of millions of tensors.
            # Each permutation yields 7 variants (1 full draft + 6 partial crops).
            needed_variants = self.augment_limit * 2 if self.augment_limit > 0 else 448
            needed_perms = min(64, max(1, math.ceil(needed_variants / 7)))

            for base_idx in range(len(x_drafts)):
                x_draft = x_drafts[base_idx]
                y = y_labels[base_idx]
                radiant = radiant_players[base_idx]
                dire = dire_players[base_idx]
                patch_id = self.patch_ids[base_idx] if self.patch_ids else torch.tensor(0, dtype=torch.long)

                if is_pub_draft(x_draft):
                    perm_samples = augment_pub_permutations(
                        x_draft, y, radiant, dire,
                        self.player_comfort_map, self.player_input_dim,
                    )
                    if needed_perms < len(perm_samples):
                        perm_samples = perm_samples[:needed_perms]

                    match_augmentations = []
                    for perm_idx in range(len(perm_samples)):
                        x_permuted, player_comfort, y_label = perm_samples[perm_idx]
                        match_augmentations.append((x_permuted, player_comfort, y_label, patch_id))

                        for n_r, n_d in pub_crop_stages:
                            x_cropped = x_permuted.clone()
                            # Mask remaining Radiant pick slots (n_r..4)
                            for s in range(n_r, 5):
                                x_cropped[s, 0] = 0.0
                                x_cropped[s, 1] = 0.0
                                x_cropped[s, 2] = -1.0
                            # Mask remaining Dire pick slots (5+n_d..9)
                            for s in range(5 + n_d, 10):
                                x_cropped[s, 0] = 0.0
                                x_cropped[s, 1] = 0.0
                                x_cropped[s, 2] = -1.0
                            match_augmentations.append((x_cropped, player_comfort, y_label, patch_id))

                    self.all_augmented_samples.append(match_augmentations)
                else:
                    perm_samples = augment_draft_permutations(
                        x_draft, y, radiant, dire,
                        self.player_comfort_map, self.player_input_dim,
                    )
                    if needed_perms < len(perm_samples):
                        perm_samples = perm_samples[:needed_perms]

                    match_augmentations = []
                    for perm_idx in range(len(perm_samples)):
                        x_permuted, player_comfort, y_label = perm_samples[perm_idx]
                        match_augmentations.append((x_permuted, player_comfort, y_label, patch_id))

                        for t in cm_truncation_points:
                            x_truncated = x_permuted.clone()
                            x_truncated[t:, :] = 0.0
                            x_truncated[t:, 2] = -1.0  # Assign explicit hero padding sentinel
                            match_augmentations.append((x_truncated, player_comfort, y_label, patch_id))

                    self.all_augmented_samples.append(match_augmentations)

            self.reshuffle_augmentations()

            # Free raw data after pre-computation
            self.x_drafts = []
            self.y_labels = []
            self.radiant_players = []
            self.dire_players = []

    def reshuffle_augmentations(self) -> None:
        """Reselect which augmented samples are visible, ensuring a fresh set of variations."""
        if self.augment_limit > 0:
            self.selected_indices = []
            for base_idx in range(len(self.all_augmented_samples)):
                num_avail = len(self.all_augmented_samples[base_idx])
                if self.augment_limit <= num_avail:
                    indices = random.sample(range(num_avail), self.augment_limit)
                else:
                    indices = random.choices(range(num_avail), k=self.augment_limit)
                self.selected_indices.append(indices)

    def __len__(self) -> int:
        if self.augment_limit > 0:
            return len(self.all_augmented_samples) * self.augment_limit
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (x_draft, player_comfort, y).
            - x_draft: (24, 4)
            - player_comfort: (10, player_input_dim)
            - y: (1,)
        """
        if self.augment_limit > 0:
            base_idx = idx // self.augment_limit
            sub_idx = idx % self.augment_limit
            selected_aug_idx = self.selected_indices[base_idx][sub_idx]
            return self.all_augmented_samples[base_idx][selected_aug_idx]
        return self.samples[idx]

    def _get_basic_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a basic sample without augmentation."""
        x_draft = self.x_drafts[idx]
        y = self.y_labels[idx]
        player_comfort = self._build_player_comfort(idx)
        return x_draft, player_comfort, y

    def _build_player_comfort_sample(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a basic sample tuple (x_draft, player_comfort, y, patch_id) for pre-computation."""
        x_draft = self.x_drafts[idx]
        y = self.y_labels[idx]
        player_comfort = self._build_player_comfort(idx)
        patch_id = self.patch_ids[idx] if self.patch_ids else torch.tensor(0, dtype=torch.long)
        return x_draft, player_comfort, y, patch_id

    def _build_player_comfort(self, idx: int) -> torch.Tensor:
        """Build the (10, C) player comfort tensor for a sample.

        Args:
            idx: Sample index.

        Returns:
            Player comfort tensor of shape (10, player_input_dim).
        """
        comfort_rows: list[torch.Tensor] = []

        for account_id in self.radiant_players[idx]:
            if account_id == 0 or account_id not in self.player_comfort_map:
                anon = torch.zeros(self.player_input_dim, dtype=torch.float32)
                anon[self.player_input_dim // 2 :] = 0.5  # Default Wilson Score to 0.5 (neutral winrate)
                comfort_rows.append(anon)
            else:
                comfort_rows.append(self.player_comfort_map[account_id])

        for account_id in self.dire_players[idx]:
            if account_id == 0 or account_id not in self.player_comfort_map:
                anon = torch.zeros(self.player_input_dim, dtype=torch.float32)
                anon[self.player_input_dim // 2 :] = 0.5  # Default Wilson Score to 0.5 (neutral winrate)
                comfort_rows.append(anon)
            else:
                comfort_rows.append(self.player_comfort_map[account_id])

        player_comfort = torch.stack(comfort_rows)
        return player_comfort


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Compute training/validation metrics.

    Args:
        predictions: Raw logits, shape (B,).
        targets: Ground truth labels, shape (B,) or (B, 1).

    Returns:
        Dictionary with 'accuracy', 'roc_auc', 'bce', 'brier_score' keys.
    """
    bce_loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets).item()

    predicted_labels = (torch.sigmoid(predictions) >= 0.5).float()
    if targets.dim() == 2:
        targets = targets.squeeze(-1)
    accuracy = (predicted_labels == targets).float().mean().item()

    auc = _compute_roc_auc(predictions, targets)

    # Brier score: mean squared error between predicted probabilities and actual binary labels
    probs = torch.sigmoid(predictions)
    brier_score = F.mse_loss(probs, targets).item()

    return {"bce": bce_loss, "accuracy": accuracy, "roc_auc": auc, "brier_score": brier_score}


def _compute_roc_auc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute ROC-AUC score using Mann-Whitney U statistic."""
    if targets.dim() == 2:
        targets = targets.squeeze(-1)

    pos_indices = torch.where(targets == 1)[0]
    neg_indices = torch.where(targets == 0)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return 0.5

    pos_preds = predictions[pos_indices]
    neg_preds = predictions[neg_indices]

    # Mann-Whitney U statistic: add 0.5 for ties
    u = torch.sum(pos_preds.unsqueeze(1) > neg_preds.unsqueeze(0)).float()
    u += 0.5 * torch.sum(pos_preds.unsqueeze(1) == neg_preds.unsqueeze(0)).float()
    auc = u.item() / (len(pos_indices) * len(neg_indices))

    return auc


class TransformerTrainer:
    """Training loop, validation, metrics, and checkpointing for the Match Network.

    Components:
    - DataLoader yielding (x_draft, player_matrices, y).
    - Training loop with label smoothing support.
    - Validation loop evaluating performance on a hold-out set.
    - Metrics tracking: Accuracy, ROC-AUC, BCE.
    - Checkpointing logic to save the best model weights per epoch.
    - MLM pre-training support.
    """

    def __init__(
        self,
        model: MatchNetwork,
        train_config: TrainingConfig | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: MatchNetwork instance to train.
            train_config: Training configuration. Defaults to TrainingConfig().
        """
        self.model = model
        self.config = train_config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.metrics = TrainingMetrics()

        self.model = self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.calibrated_temperature: float = 1.0

        lr_backbone = self.config.lr_backbone
        lr_head = self.config.lr_head

        if lr_backbone is not None or lr_head is not None:
            actual_lr_backbone = (
                lr_backbone if lr_backbone is not None else self.config.learning_rate
            )
            actual_lr_head = (
                lr_head if lr_head is not None else self.config.learning_rate
            )

            logger.info(
                "Using discriminative learning rates: backbone_lr=%.2e, head_lr=%.2e",
                actual_lr_backbone,
                actual_lr_head,
            )

            backbone_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if "set_transformer_head" in name or "mlm_head" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            param_groups = [
                {
                    "params": backbone_params,
                    "lr": actual_lr_backbone,
                    "initial_lr": actual_lr_backbone,
                },
                {
                    "params": head_params,
                    "lr": actual_lr_head,
                    "initial_lr": actual_lr_head,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=0.1,
            )
        else:
            logger.info("Using single learning rate: lr=%.2e", self.config.learning_rate)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-2,
            )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.num_epochs
        )

    @staticmethod
    def _compute_weighted_mlm_loss(
        mlm_logits: torch.Tensor,
        ntp_labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss for masked language modeling."""
        ce_elements = torch.nn.functional.cross_entropy(
            mlm_logits.transpose(1, 2),
            ntp_labels,
            ignore_index=-1,
            reduction="none",
            label_smoothing=0.0,
        )
        weighted_ce = ce_elements * weights
        valid_elements = (ntp_labels != -1).sum().float()
        return weighted_ce.sum() / torch.clamp(valid_elements, min=1.0)

    def _forward_value(
        self,
        x_batch: torch.Tensor,
        patch_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute win probability logits using SetTransformerHead and pure hero embeddings.

        Args:
            x_batch: Draft tensor of shape (batch_size, seq_len, 4).
            patch_batch: Optional patch indices of shape (batch_size,).

        Returns:
            Win probability logits of shape (batch_size,).
        """
        match_net = getattr(self.model, "match_network", self.model)
        hero_indices = x_batch[:, :, 2]
        pure_hero_embeds = match_net.joint_embedding.get_pure_hero_embeddings(
            hero_indices,
            patch_ids=patch_batch,
            for_value=True,
        )
        return match_net.set_transformer_head(pure_hero_embeds, x_batch)

    def _compute_signed_advantages(
        self,
        x_batch: torch.Tensor,
        patch_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute signed marginal advantages delta_t = sigma(t) * (V(s_t) - V(s_{t-1})).

        Generates prefix slices of the draft, computes Radiant win probability for each prefix
        via the SetTransformerHead proxy, and signs advantages based on the acting team:
        +1.0 for Radiant steps, -1.0 for Dire steps.

        Args:
            x_batch: Draft tensor of shape (batch_size, seq_len, 4).
            patch_batch: Optional patch indices of shape (batch_size,).

        Returns:
            Tensor of signed advantages delta_t with shape (batch_size, seq_len).
        """
        match_net = getattr(self.model, "match_network", self.model)
        if not (hasattr(match_net, "joint_embedding") and hasattr(match_net, "set_transformer_head")):
            return torch.zeros(x_batch.size(0), x_batch.size(1), device=x_batch.device)

        batch_size, seq_len, num_features = x_batch.shape

        with torch.no_grad():
            # 1. Expand x_batch to create prefix slices for each step t in [0, seq_len - 1]
            x_prefixes = x_batch.unsqueeze(1).repeat(1, seq_len, 1, 1)

            # 2. For each prefix step t, mask out subsequent steps (> t) by setting hero index to -1.0
            t_idx = torch.arange(seq_len, device=x_batch.device).view(1, seq_len, 1)
            s_idx = torch.arange(seq_len, device=x_batch.device).view(1, 1, seq_len)
            mask_after_t = s_idx > t_idx

            hero_col = x_prefixes[..., 2]
            x_prefixes[..., 2] = torch.where(
                mask_after_t,
                torch.tensor(-1.0, device=x_batch.device),
                hero_col,
            )

            # 3. Reshape to (batch_size * seq_len, seq_len, num_features)
            x_prefixes_flat = x_prefixes.view(batch_size * seq_len, seq_len, num_features)

            # 4. Expand patch_ids if provided
            patch_ids_expanded = (
                patch_batch.repeat_interleave(seq_len)
                if patch_batch is not None
                else None
            )

            # 5. Pass through joint_embedding and frozen set_transformer_head
            was_training = match_net.set_transformer_head.training
            match_net.set_transformer_head.eval()
            try:
                v_logits = self._forward_value(x_prefixes_flat, patch_ids_expanded)
            finally:
                if was_training:
                    match_net.set_transformer_head.train()

            # 6. Apply sigmoid activation to get win probabilities V(s_t) for Radiant team
            calibrated_t = getattr(self, "calibrated_temperature", 1.0)
            v_probs = torch.sigmoid(v_logits / calibrated_t).view(batch_size, seq_len)

            # 7. Compute marginal advantage V(s_t) - V(s_{t-1}) with prior V(s_{-1}) = 0.5
            v_prev = torch.cat(
                [torch.full((batch_size, 1), 0.5, device=x_batch.device), v_probs[:, :-1]],
                dim=1,
            )
            delta_v = v_probs - v_prev

            # 8. Sign advantage based on acting team:
            # If Dire (1.0), multiply by -1 (Dire's advantage is decrease in Radiant's win probability)
            acting_team = x_batch[:, :, 1]
            team_sign = torch.where(acting_team == 1.0, -1.0, 1.0)
            advantages = delta_v * team_sign

        return advantages

    def _compute_advantage_weights(
        self,
        x_batch: torch.Tensor,
        patch_batch: torch.Tensor | None,
        current_aw_tau: float,
    ) -> torch.Tensor:
        """Compute advantage weights for Advantage-Weighted Masked Language Modeling (AW-MLM).

        Generates prefix slices of the draft, computes Radiant win probability for each prefix
        via the frozen SetTransformerHead proxy, determines signed marginal advantages per step,
        and applies temperature scaling and clipping.

        Args:
            x_batch: Draft tensor of shape (batch_size, seq_len, 4).
            patch_batch: Optional patch indices of shape (batch_size,).
            current_aw_tau: Current annealed temperature parameter tau.

        Returns:
            Tensor of advantage weights w_t with shape (batch_size, seq_len).
        """
        advantages = self._compute_signed_advantages(x_batch, patch_batch)
        tau = max(current_aw_tau, 1e-6)
        w_t = torch.exp(advantages / tau)
        w_t = torch.clamp(
            w_t,
            min=self.config.aw_clip_min,
            max=self.config.aw_clip_max,
        )
        return w_t

    def _validate_value(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Validate the Value Head on hold-out dataset tracking BCE, ROC-AUC, and Brier Score."""
        self.model.eval()
        val_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                x_batch = batch_data[0].to(self.device)
                y_batch = batch_data[2].to(self.device).squeeze(-1).float()
                patch_batch = (
                    batch_data[3].to(self.device)
                    if len(batch_data) > 3 and batch_data[3] is not None
                    else None
                )

                logits = self._forward_value(x_batch, patch_batch)
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)

                val_loss += loss.item()
                all_preds.append(logits.cpu())
                all_targets.append(y_batch.cpu())
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        if all_preds:
            all_preds_tensor = torch.cat(all_preds)
            all_targets_tensor = torch.cat(all_targets)
            metrics = compute_metrics(all_preds_tensor, all_targets_tensor)
        else:
            metrics = {"bce": 0.0, "accuracy": 0.0, "roc_auc": 0.5, "brier_score": 0.0}

        return avg_val_loss, metrics

    def calibrate_temperature(self, val_loader: DataLoader) -> float:
        """Calibrate temperature scalar T on validation set using Platt scaling via L-BFGS to minimize Brier score.

        Optimizes T on validation logits to minimize F.mse_loss(torch.sigmoid(logits / T), targets).
        Attaches the fitted scalar temperature to the trainer and model.

        Args:
            val_loader: DataLoader providing validation draft samples and targets.

        Returns:
            Fitted scalar temperature value T.
        """
        self.model.eval()
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_data in val_loader:
                x_batch = batch_data[0].to(self.device)
                y_batch = batch_data[2].to(self.device).squeeze(-1).float()
                patch_batch = (
                    batch_data[3].to(self.device)
                    if len(batch_data) > 3 and batch_data[3] is not None
                    else None
                )

                logits = self._forward_value(x_batch, patch_batch)
                all_logits.append(logits)
                all_targets.append(y_batch)

        if not all_logits:
            logger.warning("Empty validation set provided for temperature calibration; keeping T=1.0")
            return 1.0

        val_logits = torch.cat(all_logits)
        val_targets = torch.cat(all_targets)

        temperature_param = nn.Parameter(torch.tensor([1.0], device=self.device, dtype=torch.float32))
        optimizer = torch.optim.LBFGS([temperature_param], lr=0.05, max_iter=100)

        def closure():
            optimizer.zero_grad()
            t_clamped = torch.clamp(temperature_param, min=1e-3)
            probs = torch.sigmoid(val_logits / t_clamped)
            loss = F.mse_loss(probs, val_targets)
            loss.backward()
            return loss

        optimizer.step(closure)

        calibrated_t = torch.clamp(temperature_param, min=1e-3).detach().item()
        self.calibrated_temperature = calibrated_t
        self.metrics.calibrated_temperature = calibrated_t

        match_net = getattr(self.model, "match_network", self.model)
        match_net.temperature = calibrated_t
        if hasattr(match_net, "set_transformer_head"):
            match_net.set_transformer_head.temperature = calibrated_t

        logger.info("Temperature calibration complete. Fitted T = %.4f", calibrated_t)
        return calibrated_t

    def train_value_head(
        self,
        pubs_loader: DataLoader | None,
        drafts_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingMetrics:
        """Stage 1: Pre-train and calibrate the Value Head (SetTransformerHead).

        Sequential training:
        - Loop 1: Train on pubs_loader (BCE Loss).
        - Loop 2: Fine-tune on drafts_loader (BCE Loss * draft_sample_weight, default 5.0).
        - Checkpoint selection and early stopping monitor patience on minimizing validation Brier Score.
        - Post-hoc temperature calibration via L-BFGS to directly minimize validation Brier score.
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Freeze policy components: only train SetTransformerHead and JointEmbedding value path
        match_net = getattr(self.model, "match_network", self.model)
        for name, param in self.model.named_parameters():
            if (
                "set_transformer_head" in name
                or "project_value" in name
                or "film_" in name
                or "w_patch" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=1e-2,
        )

        # Phase 1: Pre-train on pub games
        pub_epochs = self.config.pub_epochs if self.config.pub_epochs is not None else self.config.num_epochs
        if pubs_loader is not None and len(pubs_loader) > 0:
            logger.info("Stage 1 Phase 1: Pre-training Value Head on pub games (%d epochs)...", pub_epochs)
            patience_counter = 0
            best_pub_brier = float("inf")
            best_pub_state = None

            for epoch in range(1, pub_epochs + 1):
                if hasattr(pubs_loader.dataset, "reshuffle_augmentations"):
                    pubs_loader.dataset.reshuffle_augmentations()
                self.model.train()
                train_loss = 0.0
                train_batches = 0

                for batch_data in tqdm(pubs_loader, desc=f"Stage 1 [Pubs Pre-train] Epoch {epoch}/{pub_epochs}"):
                    x_batch = batch_data[0].to(self.device)
                    y_batch = batch_data[2].to(self.device).squeeze(-1).float()
                    patch_batch = (
                        batch_data[3].to(self.device)
                        if len(batch_data) > 3 and batch_data[3] is not None
                        else None
                    )

                    logits = self._forward_value(x_batch, patch_batch)
                    loss = F.binary_cross_entropy_with_logits(logits, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    train_batches += 1

                avg_train_loss = train_loss / max(train_batches, 1)
                val_loss, val_metrics = self._validate_value(val_loader)
                brier = val_metrics["brier_score"]
                auc = val_metrics["roc_auc"]

                logger.info(
                    "Stage 1 Pubs Epoch %d/%d - Train BCE: %.4f - Val BCE: %.4f - Val Brier: %.4f - Val AUC: %.4f",
                    epoch, pub_epochs, avg_train_loss, val_loss, brier, auc,
                )

                if brier < best_pub_brier - self.config.min_delta:
                    best_pub_brier = brier
                    patience_counter = 0
                    best_pub_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info("Stage 1 Phase 1 early stopping at epoch %d", epoch)
                        break

            if best_pub_state is not None:
                self.model.load_state_dict(best_pub_state)
                logger.info(
                    "Restored best Stage 1 Phase 1 weights (Val Brier: %.4f) before Phase 2 fine-tuning.",
                    best_pub_brier,
                )
        else:
            logger.warning("No pub games DataLoader provided or empty; proceeding directly to draft games fine-tuning.")

        # Phase 2: Fine-tune on draft games with draft_sample_weight
        draft_epochs = self.config.num_epochs
        weight = self.config.draft_sample_weight
        logger.info(
            "Stage 1 Phase 2: Fine-tuning Value Head on draft games (%d epochs, sample_weight=%.1f)...",
            draft_epochs, weight,
        )
        patience_counter = 0
        best_brier = float("inf")
        best_state = None

        for epoch in range(1, draft_epochs + 1):
            if hasattr(drafts_loader.dataset, "reshuffle_augmentations"):
                drafts_loader.dataset.reshuffle_augmentations()
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_data in tqdm(drafts_loader, desc=f"Stage 1 [Draft Fine-tune] Epoch {epoch}/{draft_epochs}"):
                x_batch = batch_data[0].to(self.device)
                y_batch = batch_data[2].to(self.device).squeeze(-1).float()
                patch_batch = (
                    batch_data[3].to(self.device)
                    if len(batch_data) > 3 and batch_data[3] is not None
                    else None
                )

                logits = self._forward_value(x_batch, patch_batch)
                raw_loss = F.binary_cross_entropy_with_logits(logits, y_batch)
                loss = weight * raw_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                train_loss += raw_loss.item()
                train_batches += 1

            avg_train_loss = train_loss / max(train_batches, 1)
            self.metrics.train_losses.append(avg_train_loss)

            val_loss, val_metrics = self._validate_value(val_loader)
            brier = val_metrics["brier_score"]
            auc = val_metrics["roc_auc"]

            self.metrics.val_losses.append(val_loss)
            self.metrics.val_accuracies.append(val_metrics["accuracy"])
            self.metrics.val_auc_scores.append(auc)
            self.metrics.val_brier_scores.append(brier)

            logger.info(
                "Stage 1 Draft Epoch %d/%d - Train BCE: %.4f - Val BCE: %.4f - Val Brier: %.4f - Val AUC: %.4f",
                epoch, draft_epochs, avg_train_loss, val_loss, brier, auc,
            )

            is_better_brier = brier < best_brier - self.config.min_delta
            auc_acceptable = (auc >= self.metrics.best_val_auc - 0.05) if self.metrics.best_val_auc > float("-inf") else True

            if auc > self.metrics.best_val_auc:
                self.metrics.best_val_auc = auc

            if is_better_brier and auc_acceptable:
                best_brier = brier
                self.metrics.best_brier_score = brier
                self.metrics.best_epoch = epoch
                self.metrics.best_checkpoint_value = brier
                patience_counter = 0

                best_state = {
                    "model_state": self.model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_auc": auc,
                    "val_brier_score": brier,
                    "stage": 1,
                }
                torch.save(best_state, os.path.join(self.config.checkpoint_dir, "stage1_best_model.pt"))
                torch.save(best_state, os.path.join(self.config.checkpoint_dir, "best_model.pt"))
                logger.info("  [checkpoint] Saved best Stage 1 Value Head at epoch %d (Brier=%.4f, AUC=%.4f)", epoch, brier, auc)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Stage 1 Phase 2 early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state["model_state"])

        logger.info("Running post-hoc temperature calibration via L-BFGS...")
        calibrated_t = self.calibrate_temperature(val_loader)

        final_checkpoint_path = os.path.join(self.config.checkpoint_dir, "stage1_best_model.pt")
        final_state = {
            "model_state": self.model.state_dict(),
            "epoch": self.metrics.best_epoch,
            "val_auc": self.metrics.best_val_auc,
            "val_brier_score": self.metrics.best_brier_score,
            "temperature": calibrated_t,
            "calibrated_temperature": calibrated_t,
            "stage": 1,
        }
        torch.save(final_state, final_checkpoint_path)
        torch.save(final_state, os.path.join(self.config.checkpoint_dir, "best_model.pt"))
        logger.info("Stage 1 complete. Saved calibrated checkpoint to %s (T=%.4f)", final_checkpoint_path, calibrated_t)

        return self.metrics

    def train_stage_1(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]] | None = None,
        dire_players: list[list[int]] | None = None,
        player_comfort_map: dict[int, torch.Tensor] | None = None,
        patch_ids: list[torch.Tensor] | None = None,
        x_pubs: list[torch.Tensor] | None = None,
        y_pubs: list[torch.Tensor] | None = None,
        patch_ids_pubs: list[torch.Tensor] | None = None,
    ) -> TrainingMetrics:
        """Stage 1 entry point: prepare dataloaders and run train_value_head."""
        n = len(x_drafts)
        radiant_players = radiant_players or [[0] * 5 for _ in range(n)]
        dire_players = dire_players or [[0] * 5 for _ in range(n)]

        latest_patch_id = -1
        if patch_ids is not None and len(patch_ids) > 0:
            latest_patch_id = max([p.item() if hasattr(p, 'item') else p for p in patch_ids])

        latest_patch_indices = []
        older_patch_indices = []

        for i in range(n):
            if patch_ids is not None:
                p_val = patch_ids[i].item() if hasattr(patch_ids[i], 'item') else patch_ids[i]
                if p_val == latest_patch_id:
                    latest_patch_indices.append(i)
                else:
                    older_patch_indices.append(i)
            else:
                older_patch_indices.append(i)

        latest_shuffled = torch.randperm(len(latest_patch_indices)).tolist()
        latest_patch_indices = [latest_patch_indices[i] for i in latest_shuffled]

        older_shuffled = torch.randperm(len(older_patch_indices)).tolist()
        older_patch_indices = [older_patch_indices[i] for i in older_shuffled]

        val_size = int(len(latest_patch_indices) * 0.4)
        val_size = max(1, val_size) if latest_patch_indices else 0

        val_indices = latest_patch_indices[:val_size]
        train_indices = latest_patch_indices[val_size:] + older_patch_indices
        train_shuffled = torch.randperm(len(train_indices)).tolist()
        train_indices = [train_indices[i] for i in train_shuffled]

        player_input_dim = getattr(self.model, "player_input_dim", 127)
        if player_comfort_map:
            first_tensor = next(iter(player_comfort_map.values()))
            player_input_dim = first_tensor.size(0)

        train_dataset = PlayerComfortDataset(
            x_drafts=[x_drafts[i] for i in train_indices],
            y_labels=[y_labels[i] for i in train_indices],
            radiant_players=[radiant_players[i] for i in train_indices],
            dire_players=[dire_players[i] for i in train_indices],
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
            augment=self.config.augment,
            patch_ids=[patch_ids[i] for i in train_indices] if patch_ids else None,
        )

        val_dataset = PlayerComfortDataset(
            x_drafts=[x_drafts[i] for i in val_indices],
            y_labels=[y_labels[i] for i in val_indices],
            radiant_players=[radiant_players[i] for i in val_indices],
            dire_players=[dire_players[i] for i in val_indices],
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
            augment=False,
            patch_ids=[patch_ids[i] for i in val_indices] if patch_ids else None,
        )

        drafts_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        pubs_loader = None
        if x_pubs is not None and len(x_pubs) > 0 and y_pubs is not None and len(y_pubs) > 0:
            num_pubs = len(x_pubs)
            pubs_dataset = PlayerComfortDataset(
                x_drafts=x_pubs,
                y_labels=y_pubs,
                radiant_players=[[0] * 5 for _ in range(num_pubs)],
                dire_players=[[0] * 5 for _ in range(num_pubs)],
                player_comfort_map=None,
                player_input_dim=player_input_dim,
                augment=self.config.augment,
                patch_ids=patch_ids_pubs,
            )
            pubs_loader = DataLoader(pubs_dataset, batch_size=self.config.batch_size, shuffle=True)
        else:
            logger.warning("Pub games dataset not provided or empty; skipping Phase 1.")

        return self.train_value_head(
            pubs_loader=pubs_loader,
            drafts_loader=drafts_loader,
            val_loader=val_loader,
        )

    def train_stage_2(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]],
        dire_players: list[list[int]],
        player_comfort_map: dict[int, torch.Tensor] | None = None,
        patch_ids: list[torch.Tensor] | None = None,
        stage1_checkpoint_path: str | Path | None = None,
    ) -> TrainingMetrics:
        """Stage 2: Freeze Value Head and train Policy Head on draft games trajectories."""
        ckpt_path = (
            stage1_checkpoint_path
            or self.config.stage1_checkpoint_path
            or os.path.join(self.config.checkpoint_dir, "stage1_best_model.pt")
        )
        if not os.path.exists(str(ckpt_path)):
            fallback_ckpt = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            if os.path.exists(fallback_ckpt):
                ckpt_path = fallback_ckpt

        if os.path.exists(str(ckpt_path)):
            logger.info("Loading Stage 1 checkpoint from %s for Stage 2 training...", ckpt_path)
            ckpt = torch.load(ckpt_path, weights_only=True)
            self.model.load_state_dict(ckpt["model_state"], strict=False)
            if "temperature" in ckpt:
                self.calibrated_temperature = float(ckpt["temperature"])
                logger.info("Loaded calibrated temperature T=%.4f from Stage 1 checkpoint", self.calibrated_temperature)
            elif "calibrated_temperature" in ckpt:
                self.calibrated_temperature = float(ckpt["calibrated_temperature"])
        else:
            logger.warning("Stage 1 checkpoint not found at %s. Proceeding with existing weights.", ckpt_path)

        match_net = getattr(self.model, "match_network", self.model)
        for param in match_net.set_transformer_head.parameters():
            param.requires_grad = False
        if hasattr(match_net.joint_embedding, "project_value"):
            for param in match_net.joint_embedding.project_value.parameters():
                param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "set_transformer_head" not in name and "project_value" not in name:
                param.requires_grad = True

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=1e-2,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.num_epochs
        )

        n = len(x_drafts)
        latest_patch_id = -1
        if patch_ids is not None and len(patch_ids) > 0:
            latest_patch_id = max([p.item() if hasattr(p, 'item') else p for p in patch_ids])

        latest_patch_indices = []
        older_patch_indices = []

        for i in range(n):
            if patch_ids is not None:
                p_val = patch_ids[i].item() if hasattr(patch_ids[i], 'item') else patch_ids[i]
                if p_val == latest_patch_id:
                    latest_patch_indices.append(i)
                else:
                    older_patch_indices.append(i)
            else:
                older_patch_indices.append(i)

        latest_shuffled = torch.randperm(len(latest_patch_indices)).tolist()
        latest_patch_indices = [latest_patch_indices[i] for i in latest_shuffled]
        older_shuffled = torch.randperm(len(older_patch_indices)).tolist()
        older_patch_indices = [older_patch_indices[i] for i in older_shuffled]

        val_size = int(len(latest_patch_indices) * 0.4)
        val_size = max(1, val_size) if latest_patch_indices else 0

        val_indices = latest_patch_indices[:val_size]
        train_indices = latest_patch_indices[val_size:] + older_patch_indices
        train_shuffled = torch.randperm(len(train_indices)).tolist()
        train_indices = [train_indices[i] for i in train_shuffled]

        player_input_dim = getattr(self.model, "player_input_dim", 127)
        if player_comfort_map:
            first_tensor = next(iter(player_comfort_map.values()))
            player_input_dim = first_tensor.size(0)

        train_dataset = PlayerComfortDataset(
            x_drafts=[x_drafts[i] for i in train_indices],
            y_labels=[y_labels[i] for i in train_indices],
            radiant_players=[radiant_players[i] for i in train_indices],
            dire_players=[dire_players[i] for i in train_indices],
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
            augment=self.config.augment,
            patch_ids=[patch_ids[i] for i in train_indices] if patch_ids else None,
        )

        val_dataset = PlayerComfortDataset(
            x_drafts=[x_drafts[i] for i in val_indices],
            y_labels=[y_labels[i] for i in val_indices],
            radiant_players=[radiant_players[i] for i in val_indices],
            dire_players=[dire_players[i] for i in val_indices],
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
            augment=False,
            patch_ids=[patch_ids[i] for i in val_indices] if patch_ids else None,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        patience_counter = 0

        if self.config.slot_tau_decay_epochs > 0:
            tau_decay_rate = (self.config.slot_tau_end / self.config.slot_tau_start) ** (
                1.0 / self.config.slot_tau_decay_epochs
            )
        else:
            tau_decay_rate = 1.0

        if self.config.aw_tau_decay_epochs > 0:
            aw_tau_decay_rate = (self.config.aw_tau_end / self.config.aw_tau_start) ** (
                1.0 / self.config.aw_tau_decay_epochs
            )
        else:
            aw_tau_decay_rate = 1.0

        for epoch in range(1, self.config.num_epochs + 1):
            current_tau = self.config.slot_tau_start * (
                tau_decay_rate ** min(epoch - 1, self.config.slot_tau_decay_epochs)
            )
            current_aw_tau = self.config.aw_tau_start * (
                aw_tau_decay_rate ** min(epoch - 1, self.config.aw_tau_decay_epochs)
            )

            if getattr(self.model, "match_network", None) and hasattr(self.model.match_network, "mlm_head"):
                if hasattr(self.model.match_network.mlm_head, "temperature"):
                    self.model.match_network.mlm_head.temperature = current_tau

            if hasattr(train_dataset, "reshuffle_augmentations"):
                train_dataset.reshuffle_augmentations()

            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_data in tqdm(train_loader, desc=f"Stage 2 Epoch {epoch}/{self.config.num_epochs} [Train]"):
                x_batch, player_batch, y_batch = batch_data[:3]
                patch_batch = batch_data[3] if len(batch_data) > 3 else None

                x_batch = x_batch.to(self.device)
                player_batch = player_batch.to(self.device)
                if patch_batch is not None:
                    patch_batch = patch_batch.to(self.device)

                _, mlm_logits = self.model(x_batch, player_batch, patch_ids=patch_batch)

                ntp_labels = x_batch[:, :, 2].long()
                w_t = self._compute_advantage_weights(
                    x_batch=x_batch,
                    patch_batch=patch_batch,
                    current_aw_tau=current_aw_tau,
                )

                mlm_loss = self._compute_weighted_mlm_loss(
                    mlm_logits=mlm_logits,
                    ntp_labels=ntp_labels,
                    weights=w_t,
                )

                entropy_loss = self.model.get_entropy_loss()
                collision_loss = self.model.get_collision_loss()
                composition_loss = self.model.get_composition_loss()

                total_loss = (
                    1.0 * mlm_loss
                    + entropy_loss
                    + 0.05 * collision_loss
                    + 0.10 * composition_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                self.optimizer.step()

                train_loss += total_loss.item()
                train_batches += 1

            avg_train_loss = train_loss / max(train_batches, 1)
            self.metrics.train_losses.append(avg_train_loss)

            val_loss, val_metrics = self._validate(val_loader, condition_on_positive_advantage=True)
            self.metrics.val_losses.append(val_loss)
            self.metrics.val_accuracies.append(val_metrics["accuracy"])
            self.metrics.val_auc_scores.append(val_metrics["roc_auc"])
            self.metrics.val_mlm_accuracies.append(val_metrics["mlm_accuracy"])
            self.metrics.val_mlm_top5_accuracies.append(val_metrics["mlm_top5_accuracy"])
            self.metrics.val_brier_scores.append(val_metrics["brier_score"])

            logger.info(
                "Stage 2 Epoch %d/%d - Policy Loss: %.4f - Pos-Adv Top5: %.4f (P1: %.4f | P2: %.4f | P3: %.4f)",
                epoch,
                self.config.num_epochs,
                avg_train_loss,
                val_metrics["mlm_top5_accuracy"],
                val_metrics["p1_top5_accuracy"],
                val_metrics["p2_top5_accuracy"],
                val_metrics["p3_top5_accuracy"],
            )

            current_score = val_metrics["mlm_top5_accuracy"]
            is_better = current_score > self.metrics.best_mlm_top5_acc + self.config.min_delta

            if current_score > self.metrics.best_mlm_top5_acc:
                self.metrics.best_mlm_top5_acc = current_score

            if is_better:
                self.metrics.best_checkpoint_value = current_score
                self.metrics.best_epoch = epoch
                patience_counter = 0

                best_state = {
                    "model_state": self.model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_auc": val_metrics["roc_auc"],
                    "mlm_top5_accuracy": current_score,
                    "checkpoint_metric": "val_top5_acc",
                    "checkpoint_value": current_score,
                    "temperature": self.calibrated_temperature,
                    "stage": 2,
                }
                torch.save(best_state, os.path.join(self.config.checkpoint_dir, "best_model.pt"))
                torch.save(best_state, os.path.join(self.config.checkpoint_dir, "stage2_best_model.pt"))
                logger.info("  [checkpoint] Saved best Stage 2 Policy Head at epoch %d (Top5=%.4f)", epoch, current_score)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            self.scheduler.step()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return self.metrics

    def train(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]],
        dire_players: list[list[int]],
        player_comfort_map: dict[int, torch.Tensor] | None = None,
        patch_ids: list[torch.Tensor] | None = None,
        x_pubs: list[torch.Tensor] | None = None,
        y_pubs: list[torch.Tensor] | None = None,
        patch_ids_pubs: list[torch.Tensor] | None = None,
        stage1_checkpoint_path: str | Path | None = None,
    ) -> TrainingMetrics:
        """Run training pipeline. Dispatches to train_stage_1 or train_stage_2 if stage is configured."""
        if self.config.stage == 1:
            return self.train_stage_1(
                x_drafts=x_drafts,
                y_labels=y_labels,
                radiant_players=radiant_players,
                dire_players=dire_players,
                player_comfort_map=player_comfort_map,
                patch_ids=patch_ids,
                x_pubs=x_pubs,
                y_pubs=y_pubs,
                patch_ids_pubs=patch_ids_pubs,
            )
        elif self.config.stage == 2:
            return self.train_stage_2(
                x_drafts=x_drafts,
                y_labels=y_labels,
                radiant_players=radiant_players,
                dire_players=dire_players,
                player_comfort_map=player_comfort_map,
                patch_ids=patch_ids,
                stage1_checkpoint_path=stage1_checkpoint_path,
            )
        else:
            return self._train_joint(
                x_drafts=x_drafts,
                y_labels=y_labels,
                radiant_players=radiant_players,
                dire_players=dire_players,
                player_comfort_map=player_comfort_map,
                patch_ids=patch_ids,
            )

    def _train_joint(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]],
        dire_players: list[list[int]],
        player_comfort_map: dict[int, torch.Tensor] | None = None,
        patch_ids: list[torch.Tensor] | None = None,
    ) -> TrainingMetrics:
        """Run the full joint training loop with label smoothing and augmentation.

        Args:
            x_drafts: List of draft sequence tensors, each (24, 4).
            y_labels: List of label tensors, each (1,).
            radiant_players: List of Radiant player account ID lists.
            dire_players: List of Dire player account ID lists.
            player_comfort_map: Optional mapping of account_id -> comfort tensor.

        Returns:
            TrainingMetrics with full training history.
        """
        n = len(x_drafts)
        
        # Filter validation set to ONLY include matches from the latest patch
        latest_patch_id = -1
        if patch_ids is not None and len(patch_ids) > 0:
            latest_patch_id = max([p.item() if hasattr(p, 'item') else p for p in patch_ids])
            
        latest_patch_indices = []
        older_patch_indices = []
        
        for i in range(n):
            if patch_ids is not None:
                p_val = patch_ids[i].item() if hasattr(patch_ids[i], 'item') else patch_ids[i]
                if p_val == latest_patch_id:
                    latest_patch_indices.append(i)
                else:
                    older_patch_indices.append(i)
            else:
                older_patch_indices.append(i)
                
        # Shuffle indices
        # torch.manual_seed(42)
        latest_shuffled = torch.randperm(len(latest_patch_indices)).tolist()
        latest_patch_indices = [latest_patch_indices[i] for i in latest_shuffled]
        
        older_shuffled = torch.randperm(len(older_patch_indices)).tolist()
        older_patch_indices = [older_patch_indices[i] for i in older_shuffled]
        
        val_size = int(len(latest_patch_indices) * 0.4)
        val_size = max(1, val_size) if latest_patch_indices else 0
        
        val_indices = latest_patch_indices[:val_size]
        train_indices = latest_patch_indices[val_size:] + older_patch_indices
            
        # Shuffle training set one more time so old and new patches are mixed
        # torch.manual_seed(42)  # Removed to prevent identical shuffle on every epoch!
        train_shuffled = torch.randperm(len(train_indices)).tolist()
        train_indices = [train_indices[i] for i in train_shuffled]

        player_input_dim = getattr(self.model, "player_input_dim", 127)
        if player_comfort_map:
            first_tensor = next(iter(player_comfort_map.values()))
            player_input_dim = first_tensor.size(0)

        train_dataset = PlayerComfortDataset(
            x_drafts=[x_drafts[i] for i in train_indices],
            y_labels=[y_labels[i] for i in train_indices],
            radiant_players=[radiant_players[i] for i in train_indices],
            dire_players=[dire_players[i] for i in train_indices],
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
            augment=self.config.augment,
            patch_ids=[patch_ids[i] for i in train_indices] if patch_ids else None,
        )

        val_dataset = PlayerComfortDataset(
            x_drafts=[x_drafts[i] for i in val_indices],
            y_labels=[y_labels[i] for i in val_indices],
            radiant_players=[radiant_players[i] for i in val_indices],
            dire_players=[dire_players[i] for i in val_indices],
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
            augment=False,
            patch_ids=[patch_ids[i] for i in val_indices] if patch_ids else None,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        patience_counter = 0

        # Calculate decay rate for slot attention temperature
        if self.config.slot_tau_decay_epochs > 0:
            tau_decay_rate = (self.config.slot_tau_end / self.config.slot_tau_start) ** (
                1.0 / self.config.slot_tau_decay_epochs
            )
        else:
            tau_decay_rate = 1.0

        # Calculate decay rate for AW-MLM temperature
        if self.config.aw_tau_decay_epochs > 0:
            aw_tau_decay_rate = (self.config.aw_tau_end / self.config.aw_tau_start) ** (
                1.0 / self.config.aw_tau_decay_epochs
            )
        else:
            aw_tau_decay_rate = 1.0

        for epoch in range(1, self.config.num_epochs + 1):
            # Anneal the temperature
            current_tau = self.config.slot_tau_start * (
                tau_decay_rate ** min(epoch - 1, self.config.slot_tau_decay_epochs)
            )
            current_aw_tau = self.config.aw_tau_start * (
                aw_tau_decay_rate ** min(epoch - 1, self.config.aw_tau_decay_epochs)
            )
            # Inject dynamic temperature into the MLM Head if present
            if getattr(self.model, "match_network", None) and hasattr(self.model.match_network, "mlm_head"):
                if hasattr(self.model.match_network.mlm_head, "temperature"):
                    self.model.match_network.mlm_head.temperature = current_tau

            if hasattr(train_dataset, "reshuffle_augmentations"):
                train_dataset.reshuffle_augmentations()

            self.model.train()
            train_loss = 0.0
            train_batches = 0

            eps = self.config.label_smoothing_eps

            for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]"):
                x_batch, player_batch, y_batch = batch_data[:3]
                patch_batch = batch_data[3] if len(batch_data) > 3 else None

                x_batch = x_batch.to(self.device)
                player_batch = player_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)
                if patch_batch is not None:
                    patch_batch = patch_batch.to(self.device)

                # Pass clean x_batch directly; HierarchicalTransformer right-shifts internally
                logits, mlm_logits = self.model(x_batch, player_batch, patch_ids=patch_batch)

                # Label smoothing for win probability loss
                y_smoothed = y_batch * (1.0 - eps) + (eps / 2.0)

                if self.config.step_loss_gamma > 0.0:
                    loss_elements = F.binary_cross_entropy_with_logits(
                        logits, y_smoothed, reduction="none"
                    )
                    # t is the active draft length for each sample in the batch
                    t = torch.sum(torch.sum(torch.abs(x_batch), dim=-1) > 0, dim=-1).float()
                    weights = (t / 24.0) ** self.config.step_loss_gamma
                    loss = torch.mean(weights * loss_elements)
                else:
                    loss = self.criterion(logits, y_smoothed)
                
                # NTP loss: targets are the true heroes at all sequence positions
                ntp_labels = x_batch[:, :, 2].long()
                
                # AW-MLM: Dynamically compute advantage weights
                w_t = self._compute_advantage_weights(
                    x_batch=x_batch,
                    patch_batch=patch_batch,
                    current_aw_tau=current_aw_tau,
                )
                
                mlm_loss = self._compute_weighted_mlm_loss(
                    mlm_logits=mlm_logits,
                    ntp_labels=ntp_labels,
                    weights=w_t,
                )
                
                # Slot attention entropy regularization
                entropy_loss = self.model.get_entropy_loss()
                collision_loss = self.model.get_collision_loss()
                composition_loss = self.model.get_composition_loss()

                total_loss = (
                    1.0 * loss               # Win BCE loss
                    + 1.0 * mlm_loss          # Draft Policy NTP loss
                    + entropy_loss            # Role Head Sharpness
                    + 0.05 * collision_loss   # Subtractive Repulsion
                    + 0.10 * composition_loss # Role Composition Constraint
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            avg_train_loss = train_loss / max(train_batches, 1)
            self.metrics.train_losses.append(avg_train_loss)

            val_loss, val_metrics = self._validate(val_loader)
            self.metrics.val_losses.append(val_loss)
            self.metrics.val_accuracies.append(val_metrics["accuracy"])
            self.metrics.val_auc_scores.append(val_metrics["roc_auc"])
            self.metrics.val_mlm_accuracies.append(val_metrics["mlm_accuracy"])
            self.metrics.val_mlm_top5_accuracies.append(val_metrics["mlm_top5_accuracy"])
            self.metrics.val_brier_scores.append(val_metrics["brier_score"])

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f - Val AUC: %.4f - Top5: %.4f (P1: %.4f | P2: %.4f | P3: %.4f)",
                epoch,
                self.config.num_epochs,
                avg_train_loss,
                val_loss,
                val_metrics["roc_auc"],
                val_metrics["mlm_top5_accuracy"],
                val_metrics["p1_top5_accuracy"],
                val_metrics["p2_top5_accuracy"],
                val_metrics["p3_top5_accuracy"],
            )

            if wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "val_top5_acc": val_metrics["mlm_top5_accuracy"],
                        "val_p1_top5_acc": val_metrics["p1_top5_accuracy"],
                        "val_p2_top5_acc": val_metrics["p2_top5_accuracy"],
                        "val_p3_top5_acc": val_metrics["p3_top5_accuracy"],
                        "val_auc": val_metrics["roc_auc"],
                    }
                )

            metric_choice = self.config.checkpoint_metric.lower().strip()
            if metric_choice in ("val_auc", "auc", "roc_auc"):
                current_score = val_metrics["roc_auc"]
                is_better = current_score > self.metrics.best_val_auc + self.config.min_delta
                score_str = f"Val AUC={current_score:.4f}"
            elif metric_choice in (
                "val_top5_acc", "top5", "val_top5_accuracy", "mlm_top5_accuracy"
            ):
                current_score = val_metrics["mlm_top5_accuracy"]
                is_better = current_score > self.metrics.best_mlm_top5_acc + self.config.min_delta
                score_str = f"Val Top5={current_score:.4f}"
            elif metric_choice in ("val_loss", "loss"):
                current_score = val_loss
                is_better = current_score < self.metrics.best_val_loss - self.config.min_delta
                score_str = f"Val Loss={current_score:.4f}"
            else:
                raise ValueError(
                    f"Unsupported checkpoint_metric '{self.config.checkpoint_metric}'. "
                    "Expected one of: 'val_auc', 'val_top5_acc', 'val_loss'"
                )

            # Update tracked best values
            if val_metrics["roc_auc"] > self.metrics.best_val_auc:
                self.metrics.best_val_auc = val_metrics["roc_auc"]
            if val_metrics["mlm_top5_accuracy"] > self.metrics.best_mlm_top5_acc:
                self.metrics.best_mlm_top5_acc = val_metrics["mlm_top5_accuracy"]
            if val_loss < self.metrics.best_val_loss:
                self.metrics.best_val_loss = val_loss

            if is_better:
                self.metrics.best_checkpoint_value = current_score
                self.metrics.best_epoch = epoch
                patience_counter = 0

                best_state = {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_auc": val_metrics["roc_auc"],
                    "mlm_top5_accuracy": val_metrics["mlm_top5_accuracy"],
                    "checkpoint_metric": metric_choice,
                    "checkpoint_value": current_score,
                }

                checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                torch.save(best_state, checkpoint_path)
                logger.info("  [checkpoint] Saved best model at epoch %d (%s)", epoch, score_str)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            self.scheduler.step()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return self.metrics

    def _validate(
        self,
        val_loader: DataLoader,
        condition_on_positive_advantage: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """Run validation loop with original (unsmoothed) targets for metrics."""
        self.model.eval()
        val_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        val_batches = 0
        
        mlm_correct = 0
        mlm_top5_correct = 0
        mlm_total = 0
        
        # Track Top-5 hits and totals by draft phase
        p1_hits = 0
        p2_hits = 0
        p3_hits = 0
        p1_total = 0
        p2_total = 0
        p3_total = 0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                x_batch, player_batch, y_batch = batch_data[:3]
                patch_batch = batch_data[3] if len(batch_data) > 3 else None

                x_batch = x_batch.to(self.device)
                player_batch = player_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)
                if patch_batch is not None:
                    patch_batch = patch_batch.to(self.device)

                # Pass clean x_batch directly; HierarchicalTransformer right-shifts internally
                logits, mlm_logits = self.model(x_batch, player_batch, patch_ids=patch_batch)

                if self.config.step_loss_gamma > 0.0:
                    loss_elements = F.binary_cross_entropy_with_logits(
                        logits, y_batch, reduction="none"
                    )
                    # t is the active draft length for each sample in the batch
                    t = torch.sum(torch.sum(torch.abs(x_batch), dim=-1) > 0, dim=-1).float()
                    weights = (t / 24.0) ** self.config.step_loss_gamma
                    loss = torch.mean(weights * loss_elements)
                else:
                    loss = self.criterion(logits, y_batch)

                # Slot attention entropy regularization
                entropy_loss = self.model.get_entropy_loss()
                collision_loss = self.model.get_collision_loss()
                composition_loss = self.model.get_composition_loss()

                val_loss += (
                    loss
                    + entropy_loss
                    + 0.20 * collision_loss
                    + 1.0 * composition_loss
                ).item()

                all_preds.append(logits.cpu())
                all_targets.append(y_batch.cpu())
                val_batches += 1
                
                # NTP Policy accuracy evaluation
                # NTP target labels: all 24 heroes (h_0 ... h_23)
                ntp_labels = x_batch[:, :, 2].long()
                valid_mask = ntp_labels != -1  # evaluates real picks/bans

                if condition_on_positive_advantage or self.config.stage == 2:
                    advantages = self._compute_signed_advantages(x_batch, patch_batch)
                    eval_mask = valid_mask & (advantages > 0.0)
                else:
                    eval_mask = valid_mask
                
                if eval_mask.any():
                    # Compute Top-1 accuracy over valid steps
                    preds = mlm_logits.argmax(dim=-1)
                    mlm_correct += (preds[eval_mask] == ntp_labels[eval_mask]).sum().item()
                    
                    # Compute Top-5 accuracy over valid steps
                    top5_preds = mlm_logits.topk(k=5, dim=-1).indices
                    expanded_labels = ntp_labels.unsqueeze(-1)
                    
                    # Shape: (B, 24)
                    hits_mask = (top5_preds == expanded_labels).any(dim=-1) & eval_mask
                    mlm_top5_correct += hits_mask.sum().item()
                    mlm_total += eval_mask.sum().item()
                    
                    # Separate hit counts and totals by draft phase
                    # Phase 1: Steps 0-7 (Flex/Meta)
                    p1_hits += hits_mask[:, :8].sum().item()
                    p1_total += eval_mask[:, :8].sum().item()
                    
                    # Phase 2: Steps 8-15 (Core Structure)
                    p2_hits += hits_mask[:, 8:16].sum().item()
                    p2_total += eval_mask[:, 8:16].sum().item()
                    
                    # Phase 3: Steps 16-23 (Counter/Last Picks)
                    p3_hits += hits_mask[:, 16:24].sum().item()
                    p3_total += eval_mask[:, 16:24].sum().item()

        avg_val_loss = val_loss / max(val_batches, 1)
        mlm_accuracy = (mlm_correct / mlm_total) if mlm_total > 0 else 0.0
        mlm_top5_accuracy = (mlm_top5_correct / mlm_total) if mlm_total > 0 else 0.0
        
        p1_accuracy = (p1_hits / p1_total) if p1_total > 0 else 0.0
        p2_accuracy = (p2_hits / p2_total) if p2_total > 0 else 0.0
        p3_accuracy = (p3_hits / p3_total) if p3_total > 0 else 0.0

        if all_preds:
            all_preds_tensor = torch.cat(all_preds)
            all_targets_tensor = torch.cat(all_targets)
            metrics = compute_metrics(all_preds_tensor, all_targets_tensor)
        else:
            metrics = {"bce": 0.0, "accuracy": 0.0, "roc_auc": 0.5, "brier_score": 0.0}
            
        metrics["mlm_accuracy"] = mlm_accuracy
        metrics["mlm_top5_accuracy"] = mlm_top5_accuracy
        metrics["p1_top5_accuracy"] = p1_accuracy
        metrics["p2_top5_accuracy"] = p2_accuracy
        metrics["p3_top5_accuracy"] = p3_accuracy

        return avg_val_loss, metrics

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the current model state.

        Args:
            path: Path to save the checkpoint.
        """
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "temperature": self.calibrated_temperature,
            "metrics": {
                "train_losses": self.metrics.train_losses,
                "val_losses": self.metrics.val_losses,
                "val_accuracies": self.metrics.val_accuracies,
                "val_auc_scores": self.metrics.val_auc_scores,
                "val_mlm_accuracies": self.metrics.val_mlm_accuracies,
                "val_mlm_top5_accuracies": self.metrics.val_mlm_top5_accuracies,
                "best_epoch": self.metrics.best_epoch,
                "best_mlm_top5_acc": self.metrics.best_mlm_top5_acc,
            },
        }
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load a model checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Loaded checkpoint dictionary.
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "temperature" in checkpoint:
            self.calibrated_temperature = float(checkpoint["temperature"])
            self.metrics.calibrated_temperature = self.calibrated_temperature
        elif "calibrated_temperature" in checkpoint:
            self.calibrated_temperature = float(checkpoint["calibrated_temperature"])
            self.metrics.calibrated_temperature = self.calibrated_temperature

        if "metrics" in checkpoint:
            m = checkpoint["metrics"]
            self.metrics.train_losses = m.get("train_losses", [])
            self.metrics.val_losses = m.get("val_losses", [])
            self.metrics.val_accuracies = m.get("val_accuracies", [])
            self.metrics.val_auc_scores = m.get("val_auc_scores", [])
            self.metrics.val_mlm_accuracies = m.get("val_mlm_accuracies", [])
            self.metrics.val_mlm_top5_accuracies = m.get("val_mlm_top5_accuracies", [])
            self.metrics.best_epoch = m.get("best_epoch", 0)
            self.metrics.best_mlm_top5_acc = m.get("best_mlm_top5_acc", float("-inf"))

        logger.info("Loaded checkpoint from %s", path)
        return checkpoint
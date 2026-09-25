"""Transformer trainer - training loop, validation, metrics, and checkpointing."""

from __future__ import annotations

import logging
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
    best_checkpoint_value: float = float("-inf")


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
            truncation_points = [6, 8, 11, 17, 21, 23]
            for base_idx in range(len(x_drafts)):
                x_draft = x_drafts[base_idx]
                y = y_labels[base_idx]
                radiant = radiant_players[base_idx]
                dire = dire_players[base_idx]
                patch_id = self.patch_ids[base_idx] if self.patch_ids else torch.tensor(0, dtype=torch.long)

                perm_samples = augment_draft_permutations(
                    x_draft, y, radiant, dire,
                    self.player_comfort_map, self.player_input_dim,
                )

                match_augmentations = []
                for perm_idx in range(len(perm_samples)):
                    x_permuted, player_comfort, y_label = perm_samples[perm_idx]

                    match_augmentations.append((x_permuted, player_comfort, y_label, patch_id))

                    for t in truncation_points:
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

    def train(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]],
        dire_players: list[list[int]],
        player_comfort_map: dict[int, torch.Tensor] | None = None,
        patch_ids: list[torch.Tensor] | None = None,
    ) -> TrainingMetrics:
        """Run the full training loop with label smoothing and augmentation.

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

        for epoch in range(1, self.config.num_epochs + 1):
            # Anneal the temperature
            current_tau = self.config.slot_tau_start * (
                tau_decay_rate ** min(epoch - 1, self.config.slot_tau_decay_epochs)
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
                
                # Step-Weighted Cross Entropy:
                # w_t = 0.5 + 1.0 * (t / 23) => 0.5 at Step 0, 1.5 at Last Pick
                seq_len = mlm_logits.size(1)
                t_idx = torch.arange(seq_len, device=self.device).float()
                step_weights = 0.5 + 1.0 * (t_idx / max(1.0, float(seq_len - 1)))
                
                ce_elements = torch.nn.functional.cross_entropy(
                    mlm_logits.transpose(1, 2), 
                    ntp_labels, 
                    ignore_index=-1,
                    reduction="none",
                    label_smoothing=0.0
                )
                
                weighted_ce = ce_elements * step_weights.unsqueeze(0)
                valid_elements = (ntp_labels != -1).sum().float()
                
                mlm_loss = weighted_ce.sum() / torch.clamp(valid_elements, min=1.0)
                
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

    def _validate(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
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
                
                if valid_mask.any():
                    # Compute Top-1 accuracy over valid steps
                    preds = mlm_logits.argmax(dim=-1)
                    mlm_correct += (preds[valid_mask] == ntp_labels[valid_mask]).sum().item()
                    
                    # Compute Top-5 accuracy over valid steps
                    top5_preds = mlm_logits.topk(k=5, dim=-1).indices
                    expanded_labels = ntp_labels.unsqueeze(-1)
                    
                    # Shape: (B, 24)
                    hits_mask = (top5_preds == expanded_labels).any(dim=-1) & valid_mask
                    mlm_top5_correct += hits_mask.sum().item()
                    mlm_total += valid_mask.sum().item()
                    
                    # Separate hit counts and totals by draft phase
                    # Phase 1: Steps 0-7 (Flex/Meta)
                    p1_hits += hits_mask[:, :8].sum().item()
                    p1_total += valid_mask[:, :8].sum().item()
                    
                    # Phase 2: Steps 8-15 (Core Structure)
                    p2_hits += hits_mask[:, 8:16].sum().item()
                    p2_total += valid_mask[:, 8:16].sum().item()
                    
                    # Phase 3: Steps 16-23 (Counter/Last Picks)
                    p3_hits += hits_mask[:, 16:24].sum().item()
                    p3_total += valid_mask[:, 16:24].sum().item()

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
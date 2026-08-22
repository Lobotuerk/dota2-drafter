"""Transformer trainer - training loop, validation, metrics, and checkpointing."""

from __future__ import annotations

import logging
import os
import random
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    patience: int = 10
    min_delta: float = 1e-4
    label_smoothing_eps: float = 0.15
    augment: int | bool = True


@dataclass
class TrainingMetrics:
    """Tracks training and validation metrics across epochs."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    val_auc_scores: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_roc_auc: float = float("-inf")


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

                perm_samples = augment_draft_permutations(
                    x_draft, y, radiant, dire,
                    self.player_comfort_map, self.player_input_dim,
                )

                match_augmentations = []
                for perm_idx in range(len(perm_samples)):
                    x_permuted, player_comfort, y_label = perm_samples[perm_idx]

                    match_augmentations.append((x_permuted, player_comfort, y_label))

                    for t in truncation_points:
                        x_truncated = x_permuted.clone()
                        x_truncated[t:, :] = 0.0
                        match_augmentations.append((x_truncated, player_comfort, y_label))
                
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
            if account_id == 0:
                comfort_rows.append(torch.zeros(self.player_input_dim))
            elif account_id in self.player_comfort_map:
                comfort_rows.append(self.player_comfort_map[account_id])
            else:
                comfort_rows.append(torch.zeros(self.player_input_dim))

        for account_id in self.dire_players[idx]:
            if account_id == 0:
                comfort_rows.append(torch.zeros(self.player_input_dim))
            elif account_id in self.player_comfort_map:
                comfort_rows.append(self.player_comfort_map[account_id])
            else:
                comfort_rows.append(torch.zeros(self.player_input_dim))

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
        Dictionary with 'accuracy', 'roc_auc', 'bce' keys.
    """
    bce_loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets).item()

    predicted_labels = (torch.sigmoid(predictions) >= 0.5).float()
    if targets.dim() == 2:
        targets = targets.squeeze(-1)
    accuracy = (predicted_labels == targets).float().mean().item()

    auc = _compute_roc_auc(predictions, targets)

    return {"bce": bce_loss, "accuracy": accuracy, "roc_auc": auc}


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
                if "output_head" in name or "mlm_head" in name:
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
        if patch_ids is not None:
            latest_patch_id = max([p.item() for p in patch_ids])
            
        latest_patch_indices = []
        older_patch_indices = []
        
        for i in range(n):
            if patch_ids is not None and patch_ids[i].item() == latest_patch_id:
                latest_patch_indices.append(i)
            else:
                older_patch_indices.append(i)
                
        # Shuffle indices
        # torch.manual_seed(42)
        latest_shuffled = torch.randperm(len(latest_patch_indices)).tolist()
        latest_patch_indices = [latest_patch_indices[i] for i in latest_shuffled]
        
        older_shuffled = torch.randperm(len(older_patch_indices)).tolist()
        older_patch_indices = [older_patch_indices[i] for i in older_shuffled]
        
        # Calculate exactly how many matches we need for the validation set
        val_size = int(n * self.config.val_split)
        
        # Pull entirely from the latest patch for validation
        if len(latest_patch_indices) >= val_size:
            val_indices = latest_patch_indices[:val_size]
            # Put the remaining latest patch matches into the train set
            train_indices = older_patch_indices + latest_patch_indices[val_size:]
        else:
            # If we don't have enough latest patch matches, use all of them and pad with older ones
            # (Though in a real scenario, you almost always have enough recent matches)
            val_indices = latest_patch_indices + older_patch_indices[:(val_size - len(latest_patch_indices))]
            train_indices = older_patch_indices[(val_size - len(latest_patch_indices)):]
            
        # Shuffle training set one more time so old and new patches are mixed
        # torch.manual_seed(42)  # Removed to prevent identical shuffle on every epoch!
        train_shuffled = torch.randperm(len(train_indices)).tolist()
        train_indices = [train_indices[i] for i in train_shuffled]

        # Determine player_input_dim: first try model, then fallback to comfort map or default
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
        best_state = None

        for epoch in range(1, self.config.num_epochs + 1):
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

                # 1. Main win-prediction forward pass
                logits = self.model(x_batch, player_batch, mlm_mode=False, patch_ids=patch_batch)

                # Label smoothing
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

                # 2. Parallel MLM Policy task (to train the priors simultaneously)
                # Apply 15% random masking to the current batch
                mlm_x_batch = x_batch.clone()
                mlm_labels = mlm_x_batch[:, :, 2].clone().long()
                
                # Create mask (only mask valid heroes, not -1 padding)
                rand_mask = torch.rand(mlm_x_batch.shape[:2], device=self.device) < 0.15
                valid_mask = mlm_x_batch[:, :, 2] != -1.0
                mask = rand_mask & valid_mask
                
                mlm_x_batch[mask, 2] = -1.0
                mlm_labels[~mask] = -1
                
                # Forward pass for MLM
                mlm_logits = self.model(mlm_x_batch, player_batch, mlm_mode=True, patch_ids=patch_batch)
                
                mlm_loss = torch.nn.functional.cross_entropy(
                    mlm_logits.view(-1, mlm_logits.size(-1)), 
                    mlm_labels.view(-1), 
                    ignore_index=-1
                )
                
                # Combine losses (AlphaZero-style dual objective)
                total_loss = loss + (0.5 * mlm_loss)

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

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f - Val Acc: %.4f - Val AUC: %.4f",
                epoch,
                self.config.num_epochs,
                avg_train_loss,
                val_loss,
                val_metrics["accuracy"],
                val_metrics["roc_auc"],
            )

            if val_metrics["roc_auc"] > self.metrics.best_roc_auc - self.config.min_delta:
                self.metrics.best_roc_auc = val_metrics["roc_auc"]
                self.metrics.best_epoch = epoch
                patience_counter = 0

                best_state = {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "roc_auc": val_metrics["roc_auc"],
                }

                checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                torch.save(best_state, checkpoint_path)
                logger.info("  [checkpoint] Saved best model at epoch %d (Val AUC=%.4f)", epoch, val_metrics["roc_auc"])
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            self.scheduler.step()

        return self.metrics

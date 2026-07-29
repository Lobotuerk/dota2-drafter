"""Transformer trainer - training loop, validation, metrics, and checkpointing."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
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
    player_input_dim: int = 10,
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
    truncation_points = [6, 12, 18, 24]
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
    player_input_dim: int = 10,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Apply intra-phase draft permutation augmentation.

    Generates 1-2 random valid permutations by swapping same-team, same-phase
    draft actions (order-invariant in Captains Mode).

    Args:
        x_draft: Draft sequence tensor of shape (24, 4).
        y_label: Label tensor of shape (1,).
        radiant_players: List of Radiant player account IDs.
        dire_players: List of Dire player account IDs.
        player_comfort_map: Optional mapping of account_id -> comfort tensor.
        player_input_dim: Number of features per comfort vector.

    Returns:
        List of (x_draft, player_comfort, y) tuples including original + permutations.
    """
    samples = []
    comfort_map = player_comfort_map or {}

    # Phase groups: same-team, same-phase draft actions that are order-invariant
    phase_groups = [
        [0, 2, 4],
        [1, 3, 5],
        [6, 8, 10, 12, 14, 16],
        [7, 9, 11, 13, 15, 17],
        [18, 20],
        [19, 21, 22, 23],
    ]

    num_permutations = random.randint(1, 2)
    for _ in range(num_permutations):
        x_permuted = x_draft.clone()
        for group in phase_groups:
            if len(group) >= 2 and random.random() < 0.5:
                i, j = group[0], group[1]
                x_permuted[i], x_permuted[j] = x_permuted[j].clone(), x_permuted[i].clone()

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

        samples.append((x_permuted, player_comfort, y_label))

    return samples


@dataclass
class TrainingConfig:
    """Configuration for the Transformer training loop."""

    learning_rate: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 64
    val_split: float = 0.2
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    patience: int = 10
    min_delta: float = 1e-4
    label_smoothing_eps: float = 0.15


@dataclass
class TrainingMetrics:
    """Tracks training and validation metrics across epochs."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    val_auc_scores: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


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
        player_input_dim: int = 10,
        augment: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            x_drafts: List of draft sequence tensors, each (24, 4).
            y_labels: List of label tensors, each (1,).
            radiant_players: List of Radiant player account ID lists (5 IDs each).
            dire_players: List of Dire player account ID lists (5 IDs each).
            player_comfort_map: Optional mapping of account_id -> comfort tensor (10, C).
            player_input_dim: C, number of features per comfort vector.
            augment: If True, apply prefix truncation and permutation augmentation.
        """
        self.x_drafts = x_drafts
        self.y_labels = y_labels
        self.radiant_players = radiant_players
        self.dire_players = dire_players
        self.player_comfort_map = player_comfort_map or {}
        self.player_input_dim = player_input_dim
        self.augment = augment

    def __len__(self) -> int:
        if self.augment:
            return len(self.x_drafts) * 10
        return len(self.x_drafts)

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
        if self.augment:
            return self._get_augmented_sample(idx)
        return self._get_basic_sample(idx)

    def _get_basic_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a basic sample without augmentation."""
        x_draft = self.x_drafts[idx]
        y = self.y_labels[idx]
        player_comfort = self._build_player_comfort(idx)
        return x_draft, player_comfort, y

    def _get_augmented_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an augmented sample using prefix truncation and permutations."""
        base_idx = idx // 10
        sub_idx = idx % 10

        x_draft = self.x_drafts[base_idx]
        y = self.y_labels[base_idx]
        radiant_players = self.radiant_players[base_idx]
        dire_players = self.dire_players[base_idx]

        # sub_idx 0-3: prefix truncation at 6, 12, 18, 24
        if sub_idx < 4:
            truncation_points = [6, 12, 18, 24]
            t = truncation_points[sub_idx]
            x_truncated = x_draft.clone()
            if t < 24:
                x_truncated[t:, :] = 0.0
            player_comfort = self._build_player_comfort(base_idx)
            return x_truncated, player_comfort, y

        # sub_idx 4-9: permutation augmentations (2 permutations)
        perm_samples = augment_draft_permutations(
            x_draft, y, radiant_players, dire_players,
            self.player_comfort_map, self.player_input_dim,
        )
        perm_idx = sub_idx - 4
        if perm_idx < len(perm_samples):
            return perm_samples[perm_idx]

        # Fallback to basic sample
        return self._get_basic_sample(base_idx)

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

    u = torch.sum(pos_preds.unsqueeze(1) > neg_preds.unsqueeze(0)).float()
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]],
        dire_players: list[list[int]],
        player_comfort_map: dict[int, torch.Tensor] | None = None,
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
        val_size = int(n * self.config.val_split)
        indices = list(range(n))
        torch.manual_seed(42)
        indices = torch.randperm(n).tolist()

        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        player_input_dim = 10
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
            augment=True,
        )

        val_dataset = PlayerComfortDataset(
            x_drafts=[x_drafts[i] for i in val_indices],
            y_labels=[y_labels[i] for i in val_indices],
            radiant_players=[radiant_players[i] for i in val_indices],
            dire_players=[dire_players[i] for i in val_indices],
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
            augment=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        patience_counter = 0
        best_state = None

        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            eps = self.config.label_smoothing_eps

            for x_batch, player_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]"):
                x_batch = x_batch.to(self.device)
                player_batch = player_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)

                logits = self.model(x_batch, player_batch)

                # Label smoothing
                y_smoothed = y_batch * (1.0 - eps) + (eps / 2.0)
                loss = self.criterion(logits, y_smoothed)

                self.optimizer.zero_grad()
                loss.backward()
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

            if val_loss < self.metrics.best_val_loss - self.config.min_delta:
                self.metrics.best_val_loss = val_loss
                self.metrics.best_epoch = epoch
                patience_counter = 0

                best_state = {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }

                checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                torch.save(best_state, checkpoint_path)
                logger.info("  [checkpoint] Saved best model at epoch %d (val_loss=%.4f)", epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        return self.metrics

    def mlm_train(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]],
        dire_players: list[list[int]],
        player_comfort_map: dict[int, torch.Tensor] | None = None,
        num_epochs: int = 20,
        mlm_probability: float = 0.15,
    ) -> TrainingMetrics:
        """Pre-train the Transformer using Masked Language Modeling.

        Stage 1 of two-stage training: freeze the Transformer body and train
        only the MLM head to predict masked hero identities.

        Args:
            x_drafts: List of draft sequence tensors, each (24, 4).
            y_labels: List of label tensors, each (1,).
            radiant_players: List of Radiant player account ID lists.
            dire_players: List of Dire player account ID lists.
            player_comfort_map: Optional mapping of account_id -> comfort tensor.
            num_epochs: Number of MLM pre-training epochs.
            mlm_probability: Probability of masking each hero (default 0.15).

        Returns:
            TrainingMetrics with MLM training history.
        """
        n = len(x_drafts)
        indices = list(range(n))
        torch.manual_seed(42)
        indices = torch.randperm(n).tolist()

        player_input_dim = 10
        if player_comfort_map:
            first_tensor = next(iter(player_comfort_map.values()))
            player_input_dim = first_tensor.size(0)

        mlm_dataset = _MLMDataset(
            x_drafts=x_drafts,
            y_labels=y_labels,
            radiant_players=radiant_players,
            dire_players=dire_players,
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
            mlm_probability=mlm_probability,
        )

        mlm_loader = DataLoader(mlm_dataset, batch_size=self.config.batch_size, shuffle=True)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Freeze the Transformer body, train only MLM head
        for name, param in self.model.named_parameters():
            if "mlm_head" not in name:
                param.requires_grad = False

        mlm_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )
        mlm_criterion = nn.CrossEntropyLoss()

        mlm_metrics = TrainingMetrics()
        patience_counter = 0
        best_mlm_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            batches = 0

            for x_batch, player_batch, mlm_mask, mlm_targets in tqdm(
                mlm_loader, desc=f"MLM Epoch {epoch}/{num_epochs}"
            ):
                x_batch = x_batch.to(self.device)
                player_batch = player_batch.to(self.device)
                mlm_mask = mlm_mask.to(self.device)
                mlm_targets = mlm_targets.to(self.device)

                mlm_logits = self.model(x_batch, player_batch, mlm_mode=True)

                # Reshape for cross-entropy: (B*24, num_heroes+1)
                B, S, C = mlm_logits.shape
                mlm_logits_flat = mlm_logits.view(B * S, C)
                mlm_targets_flat = mlm_targets.view(B * S)

                loss = mlm_criterion(mlm_logits_flat, mlm_targets_flat)

                mlm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=1.0,
                )
                mlm_optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / max(batches, 1)
            mlm_metrics.train_losses.append(avg_loss)

            logger.info(
                "MLM Epoch %d/%d - Avg Loss: %.4f",
                epoch, num_epochs, avg_loss,
            )

            if avg_loss < best_mlm_loss - self.config.min_delta:
                best_mlm_loss = avg_loss
                patience_counter = 0
                best_state = {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": mlm_optimizer.state_dict(),
                }
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir, "mlm_pretrained.pt"
                )
                torch.save(best_state, checkpoint_path)
                logger.info("  [checkpoint] Saved MLM pretrained model at epoch %d", epoch)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("MLM early stopping at epoch %d", epoch)
                    break

        return mlm_metrics

    def _validate(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Run validation loop with original (unsmoothed) targets for metrics."""
        self.model.eval()
        val_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        val_batches = 0

        with torch.no_grad():
            for x_batch, player_batch, y_batch in tqdm(val_loader, desc="Validation"):
                x_batch = x_batch.to(self.device)
                player_batch = player_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)

                logits = self.model(x_batch, player_batch)
                loss = self.criterion(logits, y_batch)

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
            metrics = {"bce": 0.0, "accuracy": 0.0, "roc_auc": 0.5}

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
                "best_epoch": self.metrics.best_epoch,
                "best_val_loss": self.metrics.best_val_loss,
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
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "metrics" in checkpoint:
            m = checkpoint["metrics"]
            self.metrics.train_losses = m.get("train_losses", [])
            self.metrics.val_losses = m.get("val_losses", [])
            self.metrics.val_accuracies = m.get("val_accuracies", [])
            self.metrics.val_auc_scores = m.get("val_auc_scores", [])
            self.metrics.best_epoch = m.get("best_epoch", 0)
            self.metrics.best_val_loss = m.get("best_val_loss", float("inf"))

        logger.info("Loaded checkpoint from %s", path)
        return checkpoint


class _MLMDataset(Dataset):
    """Dataset for Masked Language Modeling pre-training."""

    def __init__(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]],
        dire_players: list[list[int]],
        player_comfort_map: dict[int, torch.Tensor] | None = None,
        player_input_dim: int = 10,
        mlm_probability: float = 0.15,
    ) -> None:
        """Initialize MLM dataset.

        Args:
            x_drafts: List of draft sequence tensors, each (24, 4).
            y_labels: List of label tensors, each (1,).
            radiant_players: List of Radiant player account ID lists.
            dire_players: List of Dire player account ID lists.
            player_comfort_map: Optional mapping of account_id -> comfort tensor.
            player_input_dim: Number of features per comfort vector.
            mlm_probability: Probability of masking each hero (default 0.15).
        """
        self.x_drafts = x_drafts
        self.y_labels = y_labels
        self.radiant_players = radiant_players
        self.dire_players = dire_players
        self.player_comfort_map = player_comfort_map or {}
        self.player_input_dim = player_input_dim
        self.mlm_probability = mlm_probability

    def __len__(self) -> int:
        return len(self.x_drafts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single MLM sample with masked hero indices.

        Returns:
            Tuple of (x_draft, player_comfort, mlm_mask, mlm_targets).
            - x_draft: (24, 4) with masked heroes set to hero index 0
            - player_comfort: (10, player_input_dim)
            - mlm_mask: (24,) boolean mask where True = was masked
            - mlm_targets: (24,) original hero indices at masked positions
        """
        x_draft = self.x_drafts[idx].clone()
        player_comfort = self._build_player_comfort(idx)

        # Extract hero indices (column 2)
        hero_indices = x_draft[:, 2].long()  # (24,)

        # Create MLM mask: 15% probability per position
        mask = torch.rand(24) < self.mlm_probability  # (24,)

        # Save original hero indices for targets
        targets = hero_indices.clone()

        # Apply masking: replace masked positions with 0 (MASK token)
        x_draft[mask, 2] = 0

        return x_draft, player_comfort, mask.float(), targets.long()

    def _build_player_comfort(self, idx: int) -> torch.Tensor:
        """Build the (10, C) player comfort tensor for a sample."""
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

        return torch.stack(comfort_rows)

"""Transformer trainer - training loop, validation, metrics, and checkpointing."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dota2drafter.models.match_network import MatchNetwork

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the Transformer training loop."""

    learning_rate: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 64
    val_split: float = 0.2
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    patience: int = 10  # Early stopping patience
    min_delta: float = 1e-4  # Minimum change to qualify as an improvement


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
    """

    def __init__(
        self,
        x_drafts: list[torch.Tensor],
        y_labels: list[torch.Tensor],
        radiant_players: list[list[int]],
        dire_players: list[list[int]],
        player_comfort_map: dict[int, torch.Tensor] | None = None,
        player_input_dim: int = 10,
    ) -> None:
        """Initialize the dataset.

        Args:
            x_drafts: List of draft sequence tensors, each (24, 4).
            y_labels: List of label tensors, each (1,).
            radiant_players: List of Radiant player account ID lists (5 IDs each).
            dire_players: List of Dire player account ID lists (5 IDs each).
            player_comfort_map: Optional mapping of account_id -> comfort tensor (10, C).
            player_input_dim: C, number of features per player comfort vector.
        """
        self.x_drafts = x_drafts
        self.y_labels = y_labels
        self.radiant_players = radiant_players
        self.dire_players = dire_players
        self.player_comfort_map = player_comfort_map or {}
        self.player_input_dim = player_input_dim

    def __len__(self) -> int:
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
        x_draft = self.x_drafts[idx]
        y = self.y_labels[idx]

        # Build player comfort tensor: 5 Radiant + 5 Dire = 10 players
        player_comfort = self._build_player_comfort(idx)

        return x_draft, player_comfort, y

    def _build_player_comfort(self, idx: int) -> torch.Tensor:
        """Build the (10, C) player comfort tensor for a sample.

        Args:
            idx: Sample index.

        Returns:
            Player comfort tensor of shape (10, player_input_dim).
        """
        comfort_rows: list[torch.Tensor] = []

        # Radiant players (first 5)
        for account_id in self.radiant_players[idx]:
            if account_id == 0:
                # Anonymous account: secure zero vector
                comfort_rows.append(torch.zeros(self.player_input_dim))
            elif account_id in self.player_comfort_map:
                comfort_rows.append(self.player_comfort_map[account_id])
            else:
                # Default: zero vector if no comfort data available
                comfort_rows.append(torch.zeros(self.player_input_dim))

        # Dire players (next 5)
        for account_id in self.dire_players[idx]:
            if account_id == 0:
                # Anonymous account: secure zero vector
                comfort_rows.append(torch.zeros(self.player_input_dim))
            elif account_id in self.player_comfort_map:
                comfort_rows.append(self.player_comfort_map[account_id])
            else:
                comfort_rows.append(torch.zeros(self.player_input_dim))

        player_comfort = torch.stack(comfort_rows)  # (10, C)
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
    # BCE loss
    bce_loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets).item()

    # Accuracy
    predicted_labels = (torch.sigmoid(predictions) >= 0.5).float()
    if targets.dim() == 2:
        targets = targets.squeeze(-1)
    accuracy = (predicted_labels == targets).float().mean().item()

    # ROC-AUC (simple implementation without sklearn dependency)
    auc = _compute_roc_auc(predictions, targets)

    return {"bce": bce_loss, "accuracy": accuracy, "roc_auc": auc}


def _compute_roc_auc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute ROC-AUC score.

    Uses a simple rank-based approach compatible without sklearn.

    Args:
        predictions: Raw logits, shape (B,).
        targets: Ground truth labels, shape (B,).

    Returns:
        ROC-AUC score between 0 and 1.
    """
    if targets.dim() == 2:
        targets = targets.squeeze(-1)

    # Sort by predictions
    sorted_indices = torch.argsort(predictions, descending=True)
    sorted_targets = targets[sorted_indices]

    # Count positives and negatives
    num_positives = targets.sum().item()
    num_negatives = len(targets) - num_positives

    if num_positives == 0 or num_negatives == 0:
        return 0.5  # Undefined, return chance

    # Compute AUC using trapezoidal rule on ROC curve
    tp = 0
    fp = 0
    prev_tp = 0
    prev_fp = 0
    auc = 0.0
    prev_score = float("inf")

    for target in sorted_targets:
        score = predictions[sorted_indices[sorted_targets.tolist().index(target.item())]].item() if target.item() in sorted_targets.tolist() else predictions[sorted_indices[0]].item()
        # Use index-based approach instead
        pass

    # Simpler: Mann-Whitney U statistic
    pos_indices = torch.where(targets == 1)[0]
    neg_indices = torch.where(targets == 0)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return 0.5

    pos_preds = predictions[pos_indices]
    neg_preds = predictions[neg_indices]

    # AUC = (sum of ranks of positives - N_pos*(N_pos+1)/2) / (N_pos * N_neg)
    # Using Mann-Whitney U
    u = torch.sum(pos_preds.unsqueeze(1) > neg_preds.unsqueeze(0)).float()
    auc = u.item() / (len(pos_indices) * len(neg_indices))

    return auc


class TransformerTrainer:
    """Training loop, validation, metrics, and checkpointing for the Match Network.

    Components:
    - DataLoader yielding (x_draft, player_matrices, y).
    - Training loop utilizing BCEWithLogitsLoss.
    - Validation loop evaluating performance on a hold-out set.
    - Metrics tracking: Accuracy, ROC-AUC, BCE.
    - Checkpointing logic to save the best model weights per epoch.
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

        # Move model to device
        self.model = self.model.to(self.device)

        # Loss function and optimizer
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
        """Run the full training loop.

        Args:
            x_drafts: List of draft sequence tensors, each (24, 4).
            y_labels: List of label tensors, each (1,).
            radiant_players: List of Radiant player account ID lists.
            dire_players: List of Dire player account ID lists.
            player_comfort_map: Optional mapping of account_id -> comfort tensor.

        Returns:
            TrainingMetrics with full training history.
        """
        # Split into train/val sets
        n = len(x_drafts)
        val_size = int(n * self.config.val_split)
        indices = list(range(n))
        torch.manual_seed(42)
        torch.manual_seed(42)
        indices = torch.randperm(n).tolist()

        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        # Determine player_input_dim from the first comfort tensor in the map
        player_input_dim = 10  # default fallback
        if player_comfort_map:
            first_tensor = next(iter(player_comfort_map.values()))
            player_input_dim = first_tensor.size(0)

        # Create datasets
        train_dataset = PlayerComfortDataset(
            x_drafts=[x_drafts[i] for i in train_indices],
            y_labels=[y_labels[i] for i in train_indices],
            radiant_players=[radiant_players[i] for i in train_indices],
            dire_players=[dire_players[i] for i in train_indices],
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
        )

        val_dataset = PlayerComfortDataset(
            x_drafts=[x_drafts[i] for i in val_indices],
            y_labels=[y_labels[i] for i in val_indices],
            radiant_players=[radiant_players[i] for i in val_indices],
            dire_players=[dire_players[i] for i in val_indices],
            player_comfort_map=player_comfort_map,
            player_input_dim=player_input_dim,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Training loop
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.config.num_epochs + 1):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for x_batch, player_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]"):
                x_batch = x_batch.to(self.device)
                player_batch = player_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)

                # Forward pass
                logits = self.model(x_batch, player_batch)

                # Compute loss
                loss = self.criterion(logits, y_batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            avg_train_loss = train_loss / max(train_batches, 1)
            self.metrics.train_losses.append(avg_train_loss)

            # Validation phase
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

            # Checkpointing: save best model
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

    def _validate(self, val_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Run validation loop.

        Args:
            val_loader: DataLoader yielding (x_draft, player_comfort, y) batches.

        Returns:
            Tuple of (average_val_loss, aggregate_metrics).
        """
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

        # Compute aggregate metrics
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

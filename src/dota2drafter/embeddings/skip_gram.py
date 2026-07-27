"""Skip-Gram model for learning semantic hero embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dota2drafter.embeddings.data_extractor import SkipGramPair

logger = logging.getLogger(__name__)


class SkipGramDataset(Dataset):
    """PyTorch Dataset for Skip-Gram training pairs."""

    def __init__(self, pairs: list[SkipGramPair]) -> None:
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[int, int, int | None]:
        pair = self._pairs[idx]
        return pair.center, pair.context, pair.negative


class SkipGramModel(nn.Module):
    """Skip-Gram model with negative sampling.

    Uses two embedding matrices (target and context) optimized via
    Binary Cross Entropy loss with negative sampling.
    """

    def __init__(self, num_heroes: int, embed_dim: int) -> None:
        super().__init__()
        self.num_heroes = num_heroes
        self.embed_dim = embed_dim

        self.target_embedding = nn.Embedding(num_heroes + 1, embed_dim, padding_idx=0)
        self.context_embedding = nn.Embedding(num_heroes + 1, embed_dim, padding_idx=0)

        self._init_embeddings()

    def _init_embeddings(self) -> None:
        """Initialize embeddings with uniform distribution."""
        init_range = 1.0 / self.embed_dim
        self.target_embedding.weight.data.uniform_(-init_range, init_range)
        self.context_embedding.weight.data.uniform_(-init_range, init_range)

    def forward(
        self,
        centers: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute predictions for positive and negative pairs.

        Args:
            centers: Target hero indices, shape (batch_size,)
            contexts: Context hero indices, shape (batch_size,)
            negatives: Negative hero indices, shape (batch_size, num_negatives)

        Returns:
            Positive scores if negatives is None, else (pos_scores, neg_scores)
        """
        center_vec = self.target_embedding(centers)  # (B, D)
        context_vec = self.context_embedding(contexts)  # (B, D)

        # Positive scores: dot product of center and context
        pos_scores = torch.sum(center_vec * context_vec, dim=1)  # (B,)

        if negatives is None:
            return pos_scores

        neg_vec = self.context_embedding(negatives)  # (B, N, D)
        neg_scores = torch.sum(center_vec.unsqueeze(1) * neg_vec, dim=2)  # (B, N)

        return pos_scores, neg_scores

    def compute_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Binary Cross Entropy loss with negative sampling.

        Args:
            pos_scores: Positive pair scores, shape (batch_size,)
            neg_scores: Negative pair scores, shape (batch_size, num_negatives)

        Returns:
            Scalar loss
        """
        # Positive samples: label = 1
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )

        # Negative samples: label = 0, average over negatives then batch
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        ).mean()

        return pos_loss + neg_loss

    def get_embeddings(self) -> torch.Tensor:
        """Return the trained target embeddings (num_heroes+1, embed_dim)."""
        return self.target_embedding.weight.data

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
    ) -> float:
        """Train for one epoch.

        Args:
            dataloader: DataLoader yielding (center, context, negative) tuples
            optimizer: Optimizer
            device: Device to train on

        Returns:
            Average loss for the epoch
        """
        self.train()
        self.to(device)
        total_loss = 0.0
        num_batches = 0

        for centers, contexts, negatives in dataloader:
            centers = centers.to(device)
            contexts = contexts.to(device)
            negatives = negatives.to(device) if negatives[0] is not None else None

            optimizer.zero_grad()

            pos_scores, neg_scores = self(centers, contexts, negatives)
            loss = self.compute_loss(pos_scores, neg_scores)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def save(self, path: str | Path) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)
        logger.info("Saved Skip-Gram model to %s", path)

    @classmethod
    def load(cls, path: str | Path, num_heroes: int, embed_dim: int) -> SkipGramModel:
        """Load a trained model from disk."""
        model = cls(num_heroes, embed_dim)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        logger.info("Loaded Skip-Gram model from %s", path)
        return model

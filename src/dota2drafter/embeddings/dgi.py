"""Deep Graph Infomax (DGI) model for structural hero embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class DGIEncoder(nn.Module):
    """GCN encoder for DGI.

    A two-layer Graph Convolutional Network that maps node features
    to structural embeddings.
    """

    def __init__(self, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim or embed_dim
        self._input_dim = embed_dim

        self.conv1 = GCNConv(embed_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GCN encoder.

        Args:
            x: Node features, shape (num_nodes, embed_dim)
            edge_index: Graph connectivity, shape (2, num_edges)

        Returns:
            Encoded node representations, shape (num_nodes, embed_dim)
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class DGIModel(nn.Module):
    """Deep Graph Infomax model.

    Learns structural embeddings by maximizing mutual information
    between local node representations (after GCN encoding) and
    a global graph summary. Uses row-wise permutation for corruption.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = DGIEncoder(embed_dim, hidden_dim)
        self.sigm = nn.Sigmoid()

        # Sigmoid readout function for global graph summary
        self.readout = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )

        # Feature projection: maps arbitrary input dim to embed_dim
        self.feature_proj = nn.LazyLinear(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Node features, shape (num_nodes, input_dim)
            edge_index: Graph connectivity, shape (2, num_edges)
            s: Global summary vector (optional, computed from x if not provided)

        Returns:
            Local representations and global summary
        """
        # Project input features to embed_dim if needed
        if x.shape[1] != self.embed_dim:
            x = self.feature_proj(x)

        h = self.encoder(x, edge_index)

        if s is None:
            s = self._readout(h)

        return h, s

    def _readout(self, h: torch.Tensor) -> torch.Tensor:
        """Compute global graph summary via mean pooling + transform."""
        return self.readout(h.mean(dim=0))

    def compute_loss(
        self,
        pos_local: torch.Tensor,
        pos_global: torch.Tensor,
        neg_local: torch.Tensor,
        neg_global: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DGI binary cross-entropy loss.

        Maximizes mutual information between local and global
        representations for both positive and negative samples.

        Args:
            pos_local: Local representations for positive samples
            pos_global: Global summary for positive graphs
            neg_local: Local representations for negative samples
            neg_global: Global summary for negative (corrupted) graphs

        Returns:
            Scalar loss
        """
        pos_scores = torch.sum(pos_local * pos_global, dim=1)
        neg_scores = torch.sum(neg_local * neg_global, dim=1)

        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

        return pos_loss.mean() + neg_loss.mean()

    def train_epoch(
        self,
        graph: Data,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        num_negatives: int = 1,
    ) -> float:
        """Train for one epoch.

        Args:
            graph: PyG Data object with edge_index, edge_attr, num_nodes
            optimizer: Optimizer
            device: Device to train on
            num_negatives: Number of negative samples per forward pass

        Returns:
            Average loss for the epoch
        """
        self.train()
        self.to(device)

        edge_index = graph.edge_index.to(device)
        num_nodes = graph.num_nodes

        # Use edge_attr as initial node features if available, else identity
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            # edge_attr is (num_edges, 1); derive node features from edge statistics
            x = self._derive_node_features(graph, device)
        else:
            x = torch.eye(num_nodes, device=device)

        # Compute global summary
        h, s = self(x, edge_index)
        s_global = self._readout(h)

        total_loss = 0.0
        num_batches = 0

        # Generate negative samples via row-wise corruption
        for _ in range(max(1, num_negatives)):
            corrupted_x = self._corrupt(x)
            h_neg, s_neg = self(corrupted_x, edge_index, s=s_global)

            optimizer.zero_grad()
            loss = self.compute_loss(h, s_global, h_neg, s_neg)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def get_embeddings(self, graph: Data, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Extract final node embeddings from the trained encoder.

        Args:
            graph: PyG Data object
            device: Device to compute on

        Returns:
            Node embeddings, shape (num_nodes, embed_dim)
        """
        self.eval()
        edge_index = graph.edge_index.to(device)

        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            x = self._derive_node_features(graph, device)
        else:
            x = torch.eye(graph.num_nodes, device=device)

        # Project to embed_dim if needed
        if x.shape[1] != self.embed_dim:
            x = self.feature_proj(x.to(device))

        with torch.no_grad():
            embeddings = self.encoder(x, edge_index)

        return embeddings.cpu()

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """Row-wise permutation of node features for negative sampling."""
        perm = torch.randperm(x.shape[0], device=x.device)
        return x[perm]

    def _derive_node_features(self, graph: Data, device: torch.device) -> torch.Tensor:
        """Derive node features from edge attributes.

        For each node, aggregates edge attributes of incident edges
        via mean pooling.
        """
        num_nodes = graph.num_nodes
        edge_attr = graph.edge_attr.to(device).float()  # (num_edges, 1)
        edge_index = graph.edge_index.to(device)  # (2, num_edges)

        # Initialize node features
        node_features = torch.zeros(num_nodes, edge_attr.shape[1], device=device)
        edge_counts = torch.zeros(num_nodes, device=device)

        # Aggregate edge attributes per node
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        node_features.scatter_add_(0, src_nodes.unsqueeze(1).expand(-1, edge_attr.shape[1]), edge_attr)
        node_features.scatter_add_(0, dst_nodes.unsqueeze(1).expand(-1, edge_attr.shape[1]), edge_attr)

        edge_counts.scatter_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=edge_counts.dtype))
        edge_counts.scatter_add_(0, dst_nodes, torch.ones_like(dst_nodes, dtype=edge_counts.dtype))

        # Mean pool
        counts = edge_counts.clamp(min=1).unsqueeze(1)
        node_features = node_features / counts

        return node_features

    def save(self, path: str | Path) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)
        logger.info("Saved DGI model to %s", path)

    @classmethod
    def load(cls, path: str | Path, embed_dim: int) -> DGIModel:
        """Load a trained model from disk."""
        try:
            torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])
        except AttributeError:
            pass
        model = cls(embed_dim)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        logger.info("Loaded DGI model from %s", path)
        return model

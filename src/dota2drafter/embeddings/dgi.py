"""Deep Graph Infomax (DGI) model for structural hero embeddings."""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

class DGIEncoder(nn.Module):
    """GCN encoder for DGI."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim or embed_dim
        self.embed_dim = embed_dim

        self.conv1 = GCNConv(embed_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, embed_dim)
        self.prelu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def corruption(x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise permutation of node features for negative sampling."""
    return x[torch.randperm(x.size(0))], edge_index

class DGIModel(nn.Module):
    """Deep Graph Infomax wrapper compatible with pretrainer.py"""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = DGIEncoder(embed_dim, hidden_dim)
        
        def summary(z, *args, **kwargs):
            return torch.sigmoid(z.mean(dim=0))

        self.model = DeepGraphInfomax(
            hidden_channels=embed_dim,
            encoder=self.encoder,
            summary=summary,
            corruption=corruption
        )

        self.feature_proj = nn.LazyLinear(embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[1] != self.embed_dim:
            x = self.feature_proj(x)
        pos_z, neg_z, summary_vec = self.model(x, edge_index)
        return pos_z, neg_z, summary_vec

    def train_epoch(
        self,
        graph: Data,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        num_negatives: int = 1, # Kept for API compatibility, not explicitly needed with PyG
    ) -> float:
        self.train()
        self.to(device)

        edge_index = graph.edge_index.to(device)
        num_nodes = graph.num_nodes

        if hasattr(graph, "x") and graph.x is not None:
            x = graph.x.to(device)
        else:
            x = torch.eye(num_nodes, device=device)

        if x.shape[1] != self.embed_dim:
            x = self.feature_proj(x)

        total_loss = 0.0
        
        for _ in range(max(1, num_negatives)):
            optimizer.zero_grad()
            pos_z, neg_z, summary_vec = self.model(x, edge_index)
            loss = self.model.loss(pos_z, neg_z, summary_vec)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, num_negatives)

    def get_embeddings(self, graph: Data, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        self.eval()
        self.to(device)
        
        edge_index = graph.edge_index.to(device)
        
        if hasattr(graph, "x") and graph.x is not None:
            x = graph.x.to(device)
        else:
            x = torch.eye(graph.num_nodes, device=device)

        if x.shape[1] != self.embed_dim:
            x = self.feature_proj(x)

        with torch.no_grad():
            pos_z, _, _ = self.model(x, edge_index)
            
        return pos_z

"""Relational Graph Convolutional Network for hero embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import index_sort

logger = logging.getLogger(__name__)


class WeightedRGCNConv(RGCNConv):
    """Subclass of RGCNConv that overrides forward, propagate, and message to apply edge weights."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._temp_edge_weight = None
        self._temp_edge_type = None
        self._loop_idx = 0

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> torch.Tensor:
        if edge_weight is None:
            return super().forward(x, edge_index, edge_type)

        # Sort edges and weights by edge_type if not already sorted
        if not self.is_sorted and edge_type is not None:
            if (edge_type[1:] < edge_type[:-1]).any():
                edge_type, perm = index_sort(edge_type, max_value=self.num_relations)
                edge_index = edge_index[:, perm]
                edge_weight = edge_weight[perm]

        self._temp_edge_weight = edge_weight
        self._temp_edge_type = edge_type
        self._loop_idx = 0

        orig_is_sorted = self.is_sorted
        self.is_sorted = True

        try:
            out = super().forward(x, edge_index, edge_type)
        finally:
            self.is_sorted = orig_is_sorted
            self._temp_edge_weight = None
            self._temp_edge_type = None

        return out

    def propagate(self, edge_index: Adj, size=None, **kwargs):
        temp_edge_weight = getattr(self, "_temp_edge_weight", None)
        if temp_edge_weight is not None:
            if kwargs.get("edge_type_ptr") is not None:
                self._current_edge_weight = temp_edge_weight
            else:
                mask = self._temp_edge_type == self._loop_idx
                self._current_edge_weight = temp_edge_weight[mask]
                self._loop_idx += 1
        else:
            self._current_edge_weight = None
        return super().propagate(edge_index, size, **kwargs)

    def message(
        self,
        x_j: torch.Tensor,
        edge_type_ptr: OptTensor = None,
    ) -> torch.Tensor:
        out = super().message(x_j, edge_type_ptr)
        current_edge_weight = getattr(self, "_current_edge_weight", None)
        if current_edge_weight is not None:
            return current_edge_weight.view(-1, 1) * out
        return out


class HeroRGCN(nn.Module):
    """Relational Graph Convolutional Network for hero embeddings.

    Consumes frozen hero embeddings (from Skip-Gram + DGI) as initial
    node features and processes them over a multi-relational hero graph
    to produce relation-aware structural embeddings H_GNN.

    Uses PyG's standard RGCNConv with relation-specific projection matrices.
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        num_relations: int,
        frozen_embeddings: torch.Tensor,
        hidden_dim: int | None = None,
        num_layers: int = 2,
    ) -> None:
        """Initialize the HeroRGCN model.

        Args:
            num_nodes: Number of hero nodes (K+1, including padding at index 0).
            d_model: Embedding dimension (e.g., 64).
            num_relations: Number of edge types (3: synergy, antagonist, banned-against).
            frozen_embeddings: Pre-trained tensor from DGI stage, shape (K+1, d_model).
            hidden_dim: Hidden dimension for RGCN layers. Defaults to d_model.
            num_layers: Number of RGCN layers (1-2 recommended).
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim or d_model

        # Frozen embedding layer initialized with DGI pre-trained weights
        self.embedding = nn.Embedding.from_pretrained(
            frozen_embeddings,
            freeze=True,
            padding_idx=0,
        )

        # RGCN layers
        layers: list[nn.Module] = []
        input_dim = d_model

        for layer_idx in range(num_layers):
            output_dim = self.hidden_dim if layer_idx < num_layers - 1 else d_model
            layers.append(
                WeightedRGCNConv(
                    input_dim,
                    output_dim,
                    num_relations=num_relations,
                    num_bases=None,
                )
            )
            input_dim = output_dim

        self.rgcn_layers = nn.ModuleList(layers)
        self.activation = nn.LeakyReLU()

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor | None = None,  # Pass edge_weight
    ) -> torch.Tensor:
        """Forward pass through the HeroRGCN.

        Args:
            edge_index: Graph connectivity, shape (2, num_edges).
            edge_type: Edge type tensor, shape (num_edges,), values in {0, 1, 2}.

        Returns:
            Updated embedding matrix H_GNN, shape (num_nodes, d_model).
        """
        # Get frozen node features
        h = self.embedding.weight  # (num_nodes, d_model)

        for i, rgcn_layer in enumerate(self.rgcn_layers):
            h_in = h
            h_out = rgcn_layer(h, edge_index, edge_type, edge_weight=edge_weight)

            # If input and output dimensions match, we can do a residual skip connection
            if h_in.shape == h_out.shape:
                h = h_out + h_in
            else:
                h = h_out

            # Apply activation
            if i < len(self.rgcn_layers) - 1:
                h = self.activation(h)


        return h

    def get_embeddings(
        self,
        graph: Data,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Extract H_GNN embeddings from the trained model.

        Args:
            graph: PyG Data object with edge_index, edge_type, num_nodes.
            device: Device to compute on.

        Returns:
            H_GNN embeddings, shape (num_nodes, d_model).
        """
        self.eval()
        self.to(device)

        edge_index = graph.edge_index.to(device)
        edge_type = graph.edge_type.to(device)
        edge_weight = (
            graph.edge_weight.to(device)
            if hasattr(graph, "edge_weight") and graph.edge_weight is not None
            else None
        )

        with torch.no_grad():
            h_gnn = self(edge_index, edge_type, edge_weight=edge_weight)

        return h_gnn.cpu()

    def save(self, path: str | Path) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)
        logger.info("Saved HeroRGCN model to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        frozen_embeddings: torch.Tensor,
        d_model: int,
        num_relations: int = 3,
    ) -> HeroRGCN:
        """Load a trained model from disk.

        Args:
            path: Path to saved model weights.
            frozen_embeddings: Pre-trained tensor, shape (K+1, d_model).
            d_model: Embedding dimension.
            num_relations: Number of edge types.

        Returns:
            Loaded HeroRGCN model in eval mode.
        """
        try:
            torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])
        except AttributeError:
            pass

        state = torch.load(path, weights_only=True)
        num_layers = len([k for k in state if k.startswith("rgcn_layers.")])
        num_nodes = frozen_embeddings.shape[0]

        model = cls(
            num_nodes=num_nodes,
            d_model=d_model,
            num_relations=num_relations,
            frozen_embeddings=frozen_embeddings,
        )
        model.load_state_dict(state)
        model.eval()
        logger.info("Loaded HeroRGCN model from %s", path)
        return model

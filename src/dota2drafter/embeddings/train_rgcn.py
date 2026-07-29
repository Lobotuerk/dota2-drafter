"""Standalone RGCN training pipeline for hero embeddings.

This module provides a separate stage from pretrainer.py that:
1. Loads match data batches
2. Builds the multi-relational hero graph
3. Loads frozen DGI embeddings as initial node features
4. Trains the HeroRGCN model
5. Yields H_GNN embeddings for downstream consumption
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from dota2drafter.embeddings.data_extractor import DataExtractor
from dota2drafter.embeddings.rgcn import HeroRGCN

logger = logging.getLogger(__name__)


class LinkPredictionDecoder(nn.Module):
    """Bilinear / DistMult link prediction decoder for multi-relational RGCN graph."""

    def __init__(self, d_model: int, num_relations: int):
        super().__init__()
        # DistMult relation diagonal matrices parameter
        self.rel_emb = nn.Parameter(torch.Tensor(num_relations, d_model))
        nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        # Retrieve relation diagonal matrices
        r = self.rel_emb[edge_type]  # Shape: [E, d_model]
        
        # Bilinear dot product: (h_src * r) . h_dst
        scores = torch.sum(h_src * r * h_dst, dim=-1)
        return scores


def train_rgcn(
    data_dir: str | Path,
    frozen_embeddings_path: str | Path,
    output_file: str | Path,
    d_model: int = 64,
    num_relations: int = 3,
    rgcn_epochs: int = 20,
    learning_rate: float = 1.5e-3,
    device: str | None = None,
    hidden_dim: int | None = None,
    num_layers: int = 2,
    percentile_keep: float = 0.80,
) -> Path:
    """Train the HeroRGCN model on the multi-relational hero graph.

    This is a separate stage from the pretrainer.py pipeline. It loads
    frozen embeddings from the DGI/Skip-Gram stage and trains the RGCN
    to produce relation-aware structural embeddings.

    Args:
        data_dir: Directory containing draft batch .pt files.
        frozen_embeddings_path: Path to saved DGI embeddings (.pt tensor).
        output_file: Path to save the trained RGCN model weights.
        d_model: Embedding dimension.
        num_relations: Number of edge types (default 3).
        rgcn_epochs: Number of RGCN training epochs.
        learning_rate: Learning rate for the optimizer.
        device: Device to train on (auto-detected if None).
        hidden_dim: Hidden dimension for RGCN layers. Defaults to d_model.
        num_layers: Number of RGCN layers (1-2 recommended).
        percentile_keep: Percentile threshold to keep only top-N strongest edges (default 0.80).

    Returns:
        Path to the saved RGCN model weights.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_file)

    # Step 1: Load data
    logger.info("Step 1: Loading data from %s", data_dir)
    extractor = DataExtractor(num_heroes=127)
    batches = extractor.load_batches(data_dir)

    num_heroes = extractor._num_heroes

    # Step 2: Build and prune multi-relational graph
    logger.info("Step 2: Building and pruning multi-relational hero graph (percentile_keep=%.2f)...", percentile_keep)
    hero_graph = extractor.build_pruned_hero_graph(batches, percentile_keep=percentile_keep)
    hero_graph = hero_graph.to(device)

    # Step 3: Load frozen DGI embeddings
    logger.info("Step 3: Loading frozen embeddings from %s", frozen_embeddings_path)
    frozen_weights = torch.load(frozen_embeddings_path, weights_only=True)

    if not isinstance(frozen_weights, torch.Tensor):
        raise ValueError(f"Expected a tensor of embeddings, got {type(frozen_weights)}")

    logger.info("Frozen embeddings shape: %s", tuple(frozen_weights.shape))

    # Step 4: Initialize HeroRGCN model
    logger.info(
        "Step 4: Initializing HeroRGCN (d_model=%d, num_relations=%d, num_layers=%d)",
        d_model,
        num_relations,
        num_layers,
    )
    rgcn = HeroRGCN(
        num_nodes=num_heroes,
        d_model=d_model,
        num_relations=num_relations,
        frozen_embeddings=frozen_weights,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    rgcn = rgcn.to(device)

    # Step 5: Train RGCN
    # Use standard DistMult / Bilinear scoring link prediction decoder
    logger.info("Step 5: Training RGCN with DistMult Link Prediction Decoder (epochs=%d, lr=%.2e)", rgcn_epochs, learning_rate)
    decoder = LinkPredictionDecoder(d_model, num_relations).to(device)
    optimizer = torch.optim.AdamW(
        list(rgcn.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rgcn_epochs)

    for epoch in range(1, rgcn_epochs + 1):
        rgcn.train()
        decoder.train()
        edge_index = hero_graph.edge_index
        edge_type = hero_graph.edge_type

        # Forward pass
        h_gnn = rgcn(edge_index, edge_type)

        # 1. Positive samples: true edges in graph
        h_src = h_gnn[edge_index[0]]
        h_dst = h_gnn[edge_index[1]]
        pos_scores = decoder(h_src, h_dst, edge_type)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )

        # 2. Negative samples: row-wise permutation of destination nodes
        perm = torch.randperm(h_gnn.shape[0], device=device)
        neg_edge_dst = perm[edge_index[1] % h_gnn.shape[0]]
        h_neg_dst = h_gnn[neg_edge_dst]
        neg_scores = decoder(h_src, h_neg_dst, edge_type)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        loss = pos_loss.mean() + neg_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "RGCN epoch %d/%d, loss: %.4f (pos: %.4f, neg: %.4f)",
                epoch,
                rgcn_epochs,
                loss.item(),
                pos_loss.mean().item(),
                neg_loss.mean().item(),
            )

    # Save model
    rgcn.save(output_path)

    # Extract final H_GNN embeddings
    h_gnn_final = rgcn.get_embeddings(hero_graph, device)
    logger.info("Extracted H_GNN embeddings: shape %s", tuple(h_gnn_final.shape))

    return output_path


def load_rgcn_embeddings(
    model_path: str | Path,
    frozen_embeddings_path: str | Path,
    d_model: int,
    num_heroes: int,
    num_relations: int = 3,
) -> HeroRGCN:
    """Load a trained HeroRGCN model from disk.

    Args:
        model_path: Path to saved RGCN model weights.
        frozen_embeddings_path: Path to the frozen DGI embeddings.
        d_model: Embedding dimension.
        num_heroes: Number of heroes (K).
        num_relations: Number of edge types.

    Returns:
        Loaded HeroRGCN model in eval mode.
    """
    frozen_weights = torch.load(frozen_embeddings_path, weights_only=True)
    if not isinstance(frozen_weights, torch.Tensor):
        raise ValueError(f"Expected a tensor of embeddings, got {type(frozen_weights)}")

    num_nodes = frozen_weights.shape[0]
    model = HeroRGCN.load(
        path=model_path,
        frozen_embeddings=frozen_weights,
        d_model=d_model,
        num_relations=num_relations,
    )
    logger.info(
        "Loaded HeroRGCN: %d heroes, %d dim, %d relations",
        num_heroes,
        d_model,
        num_relations,
    )
    return model

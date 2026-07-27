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


def train_rgcn(
    data_dir: str | Path,
    frozen_embeddings_path: str | Path,
    output_file: str | Path,
    d_model: int = 64,
    num_relations: int = 3,
    rgcn_epochs: int = 20,
    learning_rate: float = 1e-2,
    device: str | None = None,
    hidden_dim: int | None = None,
    num_layers: int = 2,
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

    Returns:
        Path to the saved RGCN model weights.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_file)

    # Step 1: Load data
    logger.info("Step 1: Loading data from %s", data_dir)
    extractor = DataExtractor(num_heroes=124)
    batches = extractor.load_batches(data_dir)

    num_heroes = extractor._num_heroes

    # Step 2: Build multi-relational graph
    logger.info("Step 2: Building multi-relational hero graph...")
    hero_graph = extractor.build_hero_graph(batches)
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
    # Use a simplified training objective: maximize mutual information
    # between node embeddings and graph summary (inspired by DGI)
    logger.info("Step 5: Training RGCN (epochs=%d, lr=%.2e)", rgcn_epochs, learning_rate)
    optimizer = torch.optim.Adam(rgcn.parameters(), lr=learning_rate)

    # Readout function for global graph summary
    readout = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.Tanh(),
    ).to(device)

    for epoch in range(1, rgcn_epochs + 1):
        rgcn.train()
        edge_index = hero_graph.edge_index
        edge_type = hero_graph.edge_type

        # Forward pass
        h_gnn = rgcn(edge_index, edge_type)

        # Compute global summary
        s_global = readout(h_gnn.mean(dim=0))

        # Positive samples: original embeddings
        pos_scores = torch.sum(h_gnn * s_global, dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )

        # Negative samples: row-wise permutation
        perm = torch.randperm(h_gnn.shape[0], device=device)
        h_neg = h_gnn[perm]
        neg_scores = torch.sum(h_neg * s_global, dim=1)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        loss = pos_loss.mean() + neg_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

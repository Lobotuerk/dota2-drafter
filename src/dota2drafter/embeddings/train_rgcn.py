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
    """Directed link prediction decoder for multi-relational RGCN graph.
    
    Replaces symmetric DistMult with a directed MLP to properly model
    directed antagonist and required-bans edges.
    """

    def __init__(self, d_model: int, num_relations: int):
        super().__init__()
        self.r_embed = nn.Embedding(num_relations, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(2 * d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, 1)
        )

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        r = self.r_embed(edge_type)
        # Concatenate src, relation, and dst to preserve directionality
        cat = torch.cat([h_src, r, h_dst], dim=-1)
        scores = self.decoder(cat).squeeze(-1)
        return scores


    def get_embeddings(self, hero_graph, device):
        # Existing method to retrieve embeddings for evaluation
        edge_index = hero_graph.edge_index
        edge_type = hero_graph.edge_type
        return self(edge_index, edge_type)

def evaluate_graph_quality(rgcn, decoder, hero_graph, val_edges, val_edge_types, device):
    """Evaluate link prediction on a validation edge split using Hits@10 and MRR."""
    rgcn.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Get full node embeddings from the entire graph structure
        # Ensure the graph passed here is properly moved to the device first
        hero_graph = hero_graph.to(device)
        h_gnn = rgcn.get_embeddings(hero_graph, device).to(device)
        
        num_nodes = h_gnn.shape[0]
        num_val_edges = val_edges.shape[1]
        
        # We will rank the true destination node against ALL possible destination nodes
        mrr_sum = 0.0
        hits_at_10 = 0
        
        for i in range(num_val_edges):
            src = val_edges[0, i]
            true_dst = val_edges[1, i]
            edge_type = val_edge_types[i].to(device)
            
            # (1, d_model)
            h_src = h_gnn[src].unsqueeze(0).to(device)
            
            # Replicate src for all possible destinations (num_nodes, d_model)
            h_src_expand = h_src.expand(num_nodes, -1)
            h_dst_all = h_gnn  # (num_nodes, d_model) already on device
            
            # Predict scores for src -> all possible nodes
            # Note: edge_type must be broadcast correctly. 
            # It expects (B,) where B is num_nodes
            edge_type_expand = edge_type.unsqueeze(0).expand(num_nodes)
            
            scores = decoder(h_src_expand, h_dst_all, edge_type_expand)
            
            # Rank scores descending
            # We want to find the rank of 'true_dst'
            # To get rank, we can count how many nodes scored higher than the true node
            true_score = scores[true_dst].item()
            
            # Rank = 1 + number of scores strictly greater than the true score
            # (In practice, ties should be broken randomly or averaged, but strict > is common for simplicity)
            rank = 1 + (scores > true_score).sum().item()
            
            mrr_sum += 1.0 / rank
            if rank <= 10:
                hits_at_10 += 1
                
        mrr = mrr_sum / num_val_edges
        hits_10_ratio = hits_at_10 / num_val_edges
        
    return mrr, hits_10_ratio


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
    wilson_threshold: float = 0.50,
    gamma: float = 0.80,
    percentile_keep: float | None = None,
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
    logger.info("Step 2: Building and pruning multi-relational hero graph (wilson_threshold=%.2f, gamma=%.2f)...", wilson_threshold, gamma)
    full_hero_graph = extractor.build_pruned_hero_graph(
        batches,
        wilson_threshold=wilson_threshold,
        gamma=gamma,
        percentile_keep=percentile_keep,
    )
    
    # --- Edge Splitting for Validation ---
    num_edges = full_hero_graph.edge_index.shape[1]
    perm = torch.randperm(num_edges)
    val_size = int(num_edges * 0.1) # 10% for validation
    
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    
    # Train graph (used for structural message passing AND positive training samples)
    train_edge_index = full_hero_graph.edge_index[:, train_idx].to(device)
    train_edge_type = full_hero_graph.edge_type[train_idx].to(device)
    
    # Note: We must construct a new Data object (or similar) to pass to the RGCN if it expects `hero_graph.edge_index`
    # We'll just patch the properties on a copy or pass them directly.
    import copy
    train_hero_graph = copy.copy(full_hero_graph)
    train_hero_graph.edge_index = train_edge_index
    train_hero_graph.edge_type = train_edge_type
    
    # Validation edges (to rank)
    val_edge_index = full_hero_graph.edge_index[:, val_idx].to(device)
    val_edge_type = full_hero_graph.edge_type[val_idx].to(device)
    # ------------------------------------

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
    logger.info("Step 5: Training RGCN with Asymmetric MLP Link Prediction Decoder (epochs=%d, lr=%.2e)", rgcn_epochs, learning_rate)
    decoder = LinkPredictionDecoder(d_model, num_relations).to(device)
    optimizer = torch.optim.AdamW(
        list(rgcn.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    # Using ReduceLROnPlateau monitoring validation MRR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    best_mrr = -1.0
    best_model_state = None
    early_stop_patience = 60
    early_stop_counter = 0

    for epoch in range(1, rgcn_epochs + 1):
        rgcn.train()
        decoder.train()
        
        edge_index = train_hero_graph.edge_index
        edge_type = train_hero_graph.edge_type

        # Forward pass
        h_gnn = rgcn(edge_index, edge_type)

        # 1. Positive samples: true edges in train graph
        h_src = h_gnn[edge_index[0]]
        h_dst = h_gnn[edge_index[1]]
        pos_scores = decoder(h_src, h_dst, edge_type)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )

        # 2. Negative samples: 5:1 ratio with Hard Negative Filtering
        neg_ratio = 5
        h_src_neg = h_src.repeat(neg_ratio, 1)
        edge_type_neg = edge_type.repeat(neg_ratio)
        src_rep = edge_index[0].repeat(neg_ratio)
        
        num_nodes_curr = h_gnn.shape[0]
        num_neg_samples = edge_index.shape[1] * neg_ratio
        
        # Vectorized check to mask out negative samples that accidentally land on true positive training edges
        # (Equivalent to checking against a set of tuples but runs natively on GPU)
        is_pos_edge = torch.zeros((num_nodes_curr, num_relations, num_nodes_curr), dtype=torch.bool, device=device)
        is_pos_edge[edge_index[0], edge_type, edge_index[1]] = True
        
        neg_dst = torch.randint(0, num_nodes_curr, (num_neg_samples,), device=device)
        invalid = is_pos_edge[src_rep, edge_type_neg, neg_dst]
        
        # Resample any destination node that forms an existing positive edge
        while invalid.any():
            new_dst = torch.randint(0, num_nodes_curr, (invalid.sum(),), device=device)
            neg_dst[invalid] = new_dst
            invalid = is_pos_edge[src_rep, edge_type_neg, neg_dst]
            
        h_neg_dst = h_gnn[neg_dst]
        
        neg_scores = decoder(h_src_neg, h_neg_dst, edge_type_neg)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        loss = pos_loss.mean() + neg_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation every epoch
        val_mrr, val_hits_10 = evaluate_graph_quality(
            rgcn, decoder, train_hero_graph, val_edge_index, val_edge_type, device
        )
        
        scheduler.step(val_mrr)

        if val_mrr > best_mrr:
            best_mrr = val_mrr
            early_stop_counter = 0
            # Cache best model weights
            best_model_state = copy.deepcopy(rgcn.state_dict())
        else:
            early_stop_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "RGCN epoch %d/%d, loss: %.4f (pos: %.4f, neg: %.4f) | Val MRR: %.4f, Hits@10: %.4f",
                epoch,
                rgcn_epochs,
                loss.item(),
                pos_loss.mean().item(),
                neg_loss.mean().item(),
                val_mrr,
                val_hits_10
            )

        if early_stop_counter >= early_stop_patience:
            logger.info("Early stopping RGCN after %d epochs due to Val MRR plateau.", epoch)
            break

    # Save best model
    if best_model_state is not None:
        rgcn.load_state_dict(best_model_state)
    rgcn.save(output_path)

    # Extract final H_GNN embeddings using the FULL graph for the final exported state
    full_hero_graph = full_hero_graph.to(device)
    h_gnn_final = rgcn.get_embeddings(full_hero_graph, device)
    logger.info("Extracted H_GNN embeddings: shape %s. Best Val MRR: %.4f", tuple(h_gnn_final.shape), best_mrr)

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

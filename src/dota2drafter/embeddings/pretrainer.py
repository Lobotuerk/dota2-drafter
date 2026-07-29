"""Orchestration module for embedding pre-training pipeline.

Provides a clean, importable API to train and consume embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dota2drafter.embeddings.data_extractor import DataExtractor
from dota2drafter.embeddings.skip_gram import SkipGramDataset, SkipGramModel
from dota2drafter.embeddings.dgi import DGIModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def check_embedding_health(model: DGIModel, x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Print the similarity distribution metrics to monitor embedding health during training."""
    model.eval()
    if x.shape[1] != model.embed_dim:
        x = model.feature_proj(x.to(edge_index.device))
        
    emb = model.encoder(x, edge_index)  # Shape: [num_nodes, embed_dim]
    emb = F.normalize(emb, p=2, dim=1)  # Ensure L2 normalized
    
    # Compute full pairwise cosine similarity
    sim_matrix = torch.mm(emb, emb.T)
    
    # Mask out self-similarity (diagonal)
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=emb.device)
    off_diag_sim = sim_matrix[mask]
    
    mean_sim = off_diag_sim.mean().item()
    min_sim  = off_diag_sim.min().item()
    max_sim  = off_diag_sim.max().item()
    
    print(f"Stats -> Mean Sim: {mean_sim:.3f} | Min: {min_sim:.3f} | Max: {max_sim:.3f}")
    return mean_sim


def train_embeddings(
    data_dir: str | Path,
    output_file: str | Path,
    embed_dim: int = 64,
    skip_gram_epochs: int = 10,
    dgi_epochs: int = 20,
    skip_gram_lr: float = 1e-2,
    dgi_lr: float = 1e-2,
    batch_size: int = 256,
    device: str | None = None,
) -> Path:
    """End-to-end embedding pre-training pipeline.

    Trains Skip-Gram embeddings first, then uses them as initial
    node features for DGI structural embeddings.

    Args:
        data_dir: Directory containing draft batch .pt files
        output_file: Path to save the final embedding weights
        embed_dim: Dimension of the embedding space
        skip_gram_epochs: Number of Skip-Gram training epochs
        dgi_epochs: Number of DGI training epochs
        skip_gram_lr: Learning rate for Skip-Gram optimizer
        dgi_lr: Learning rate for DGI optimizer
        batch_size: Batch size for Skip-Gram DataLoader
        device: Device to train on (auto-detected if None)

    Returns:
        Path to the saved embedding weights
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_file)

    # Step 1: Extract data
    logger.info("Step 1: Extracting data from %s", data_dir)
    batches = DataExtractor.load_batches(data_dir)

    max_hero_idx = 0
    for batch in batches:
        x_tensors = batch["x"]
        if x_tensors.dim() == 3:
            max_hero_idx = max(max_hero_idx, int(x_tensors[:, :, 2].max().item()))
        else:
            max_hero_idx = max(max_hero_idx, int(x_tensors[:, 2].max().item()))

    max_hero_idx = max(max_hero_idx, 124)
    logger.info("Detected actual maximum hero index in dataset: %d", max_hero_idx)
    extractor = DataExtractor(num_heroes=max_hero_idx)

    num_heroes = extractor._num_heroes
    logger.info("Extracting Skip-Gram pairs and building hero graph...")
    pairs = extractor.extract_skip_gram_pairs(batches)
    hero_graph = extractor.build_hero_graph(batches)

    if len(pairs) == 0:
        raise ValueError("No Skip-Gram pairs extracted. Check input data.")

    # Step 2: Train Skip-Gram
    logger.info("Step 2: Training Skip-Gram model (embed_dim=%d, epochs=%d)", embed_dim, skip_gram_epochs)
    skip_gram = SkipGramModel(num_heroes, embed_dim)

    dataset = SkipGramDataset(pairs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    skip_optimizer = torch.optim.Adam(skip_gram.parameters(), lr=skip_gram_lr)

    for epoch in range(1, skip_gram_epochs + 1):
        avg_loss = skip_gram.train_epoch(dataloader, skip_optimizer, device)
        logger.info("Skip-Gram epoch %d/%d, loss: %.4f", epoch, skip_gram_epochs, avg_loss)

    # Save Skip-Gram embeddings
    skip_gram_path = output_path.with_suffix(".skipgram.pt")
    skip_gram.save(skip_gram_path)
    sg_embeddings = skip_gram.get_embeddings().to(device)
    logger.info("Saved Skip-Gram embeddings to %s", skip_gram_path)

    # Step 3: Train DGI
    logger.info("Step 3: Training DGI model (epochs=%d)", dgi_epochs)
    dgi = DGIModel(embed_dim)
    dgi_optimizer = torch.optim.Adam(dgi.parameters(), lr=dgi_lr, weight_decay=1e-4)
    dgi_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dgi_optimizer, T_max=dgi_epochs)

    # Move graph to device
    hero_graph = hero_graph.to(device)

    # Use Skip-Gram embeddings as node features for DGI
    initial_x = sg_embeddings[:hero_graph.num_nodes]

    # Temporarily set node features on graph for DGI
    hero_graph.x = initial_x

    for epoch in range(1, dgi_epochs + 1):
        avg_loss = dgi.train_epoch(hero_graph, dgi_optimizer, device)
        logger.info("DGI epoch %d/%d, loss: %.4f", epoch, dgi_epochs, avg_loss)
        if epoch % 10 == 0:
            check_embedding_health(dgi, hero_graph.x, hero_graph.edge_index)
        dgi_scheduler.step()

    # Extract final DGI embeddings
    final_embeddings = dgi.get_embeddings(hero_graph, device)
    logger.info("Extracted DGI embeddings: shape %s", tuple(final_embeddings.shape))

    # Zero out row 0 to ensure index 0 is a clean padding vector (heroes are 1-indexed)
    final_embeddings[0] = 0.0

    # Save final embeddings
    torch.save(final_embeddings.cpu(), output_path)
    logger.info("Saved final embeddings to %s", output_path)

    return output_path


def load_frozen_embeddings(
    weights_path: str | Path,
    embed_dim: int,
    num_heroes: int,
) -> nn.Embedding:
    """Load a frozen nn.Embedding module from saved weights.

    Validates the loaded state dict shape against expected parameters
    before initializing the frozen module.

    Args:
        weights_path: Path to saved embedding weights (.pt file)
        embed_dim: Expected embedding dimension
        num_heroes: Expected number of heroes (K)

    Returns:
        A frozen nn.Embedding module ready for injection into
        the downstream Graph-Augmented Transformer
    """
    weights = torch.load(weights_path, weights_only=True)

    if not isinstance(weights, torch.Tensor):
        raise ValueError(f"Expected a tensor of embeddings, got {type(weights)}")

    actual_num_heroes = weights.shape[0] - 1
    if weights.shape[1] != embed_dim:
        raise ValueError(
            f"Embedding shape mismatch: expected second dimension to be {embed_dim}, "
            f"got {weights.shape[1]}. "
            f"Check embed_dim={embed_dim}."
        )

    if actual_num_heroes != num_heroes:
        logger.warning(
            "Embedding shape mismatch: expected %d heroes, but the saved weights "
            "contain embeddings for %d heroes. Adapting dynamically to %d heroes.",
            num_heroes,
            actual_num_heroes,
            actual_num_heroes,
        )

    embedding = nn.Embedding.from_pretrained(weights, freeze=True)
    logger.info(
        "Loaded frozen embedding: %d heroes, %d dim",
        actual_num_heroes,
        embed_dim,
    )
    return embedding

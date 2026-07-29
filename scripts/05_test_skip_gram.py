#!/usr/bin/env python3
"""Test Skip-Gram + DGI embeddings by finding closest heroes.

Accepts a hero name and prints the 5 closest heroes by cosine similarity
in the embedding space.

Usage::

    python scripts/05_test_skip_gram.py --hero_name "Anti-Mage"

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from torch.nn.functional import cosine_similarity

from dota2drafter.embeddings.pretrainer import load_frozen_embeddings

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the 5 closest heroes to a given hero in embedding space.",
    )
    parser.add_argument(
        "--hero_name",
        type=str,
        required=True,
        help='Hero name (e.g., "Anti-Mage")',
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/skip_gram_dgi.pt",
        help="Path to pre-trained embeddings (default: models/skip_gram_dgi.pt)",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="hero_mapping.json",
        help="Path to hero mapping file (default: hero_mapping.json)",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=32, help="Embedding dimension (default: 32)"
    )
    parser.add_argument(
        "--num_heroes", type=int, default=128, help="Number of heroes (default: 128)"
    )
    return parser.parse_args()


def load_hero_mapping(mapping_path: str) -> dict[str, str]:
    """Load hero mapping from JSON file."""
    path = Path(mapping_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Hero mapping file not found: {path}")
        sys.exit(1)
    import json

    with open(path) as f:
        return json.load(f)


def resolve_hero_id(
    hero_name: str, mapping: dict[str, str]
) -> tuple[str, int]:
    """Resolve hero name to ID via case-insensitive matching.

    Returns:
        Tuple of (canonical_name, hero_id)

    Raises:
        SystemExit if hero not found.
    """
    name_lower = hero_name.lower()
    for hero_id, canonical_name in mapping.items():
        if canonical_name.lower() == name_lower:
            return canonical_name, int(hero_id)

    console.print(f"[bold red]Error:[/bold red] Hero '{hero_name}' not found.")
    console.print("Available heroes:")
    for hero_id, canonical_name in sorted(
        mapping.items(), key=lambda x: int(x[0])
    ):
        console.print(f"  {hero_id}: {canonical_name}")
    sys.exit(1)


def find_closest_heroes(
    embeddings: torch.nn.Embedding,
    contiguous_idx: int,
    sorted_keys: list[str],
    mapping: dict[str, str],
    num_results: int = 5,
) -> list[tuple[str, float]]:
    """Compute cosine similarity and return top-N closest heroes.

    Args:
        embeddings: Frozen embedding layer.
        contiguous_idx: Query contiguous hero index (1-based).
        sorted_keys: Linear ordered list of hero API ID keys from mapping file.
        mapping: Hero ID -> name mapping.
        num_results: Number of closest heroes to return.

    Returns:
        List of (hero_name, similarity_score) tuples sorted descending.
    """
    target_emb = embeddings.weight[contiguous_idx].unsqueeze(0)
    all_embs = embeddings.weight

    similarities = cosine_similarity(target_emb, all_embs, dim=1)

    # Sort descending
    sorted_indices = similarities.argsort(descending=True)

    results: list[tuple[str, float]] = []
    for idx in sorted_indices.tolist():
        if idx == contiguous_idx or idx == 0:  # Skip self and index 0 (unmapped/padding)
            continue

        # Convert contiguous 1-based index back to API ID key by linear keys position
        if idx - 1 < len(sorted_keys):
            api_id = sorted_keys[idx - 1]
            name = mapping.get(api_id, f"Hero-{api_id}")
            score = similarities[idx].item()
            results.append((name, score))
            
        if len(results) >= num_results:
            break

    return results


def print_results(
    query_name: str, results: list[tuple[str, float]]
) -> None:
    """Print formatted results table."""
    table = Table(
        title=f"Top 5 Closest Heroes to [bold]{query_name}[/bold]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Hero", style="bold white")
    table.add_column("Similarity", justify="right", style="green")

    for i, (name, score) in enumerate(results, 1):
        table.add_row(str(i), name, f"{score:.4f}")

    console.print()
    console.print(table)
    console.print()


def main() -> None:
    args = parse_args()

    # Load hero mapping (API ID -> Name)
    mapping = load_hero_mapping(args.mapping_file)

    # Resolve hero name to API ID
    canonical_name, api_hero_id = resolve_hero_id(args.hero_name, mapping)

    # Reconstruct linear sorted keys list for contiguous 1-based mapping
    sorted_keys = sorted(mapping.keys(), key=lambda x: int(x))

    # Get contiguous index for our query hero (1-based index)
    try:
        contiguous_idx = sorted_keys.index(str(api_hero_id)) + 1
    except ValueError:
        console.print(f"[bold red]Error:[/bold red] Hero '{canonical_name}' (API ID {api_hero_id}) is not present in the mapping keys.")
        sys.exit(1)

    # Load embeddings
    model_path = Path(args.model_path)
    if not model_path.exists():
        console.print(f"[bold red]Error:[/bold red] Embedding file not found: {model_path}")
        console.print("Train embeddings first: python scripts/02_train_embeddings.py --mode train")
        sys.exit(1)

    console.print(f"[bold blue]Loading embeddings from: {model_path}[/bold blue]")
    embeddings = load_frozen_embeddings(
        weights_path=args.model_path,
        embed_dim=args.embed_dim,
        num_heroes=args.num_heroes,
    )

    # Mean-center the raw embeddings
    mu = embeddings.weight.mean(dim=0, keepdim=True)
    centered_emb_weight = embeddings.weight - mu
    centered_emb_weight[0] = 0.0  # Keep padding at 0
    
    # Create a new embedding layer with centered weights
    centered_embeddings = torch.nn.Embedding.from_pretrained(centered_emb_weight)

    # Find closest heroes using centered embeddings
    results = find_closest_heroes(centered_embeddings, contiguous_idx, sorted_keys, mapping)

    # Print results
    print_results(canonical_name, results)


if __name__ == "__main__":
    main()

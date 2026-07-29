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
        help="Path to DGI embeddings (default: models/skip_gram_dgi.pt)",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="hero_mapping.json",
        help="Path to hero mapping file (default: hero_mapping.json)",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=64, help="Embedding dimension (default: 64)"
    )
    parser.add_argument(
        "--num_heroes", type=int, default=124, help="Number of heroes (default: 124)"
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
    hero_id: int,
    mapping: dict[str, str],
    num_results: int = 5,
) -> list[tuple[str, float]]:
    """Compute cosine similarity and return top-N closest heroes.

    Args:
        embeddings: Frozen embedding layer.
        hero_id: Query hero ID.
        mapping: Hero ID -> name mapping.
        num_results: Number of closest heroes to return.

    Returns:
        List of (hero_name, similarity_score) tuples sorted descending.
    """
    target_emb = embeddings.weight[hero_id].unsqueeze(0)
    all_embs = embeddings.weight

    similarities = cosine_similarity(target_emb, all_embs, dim=1)

    # Sort descending
    sorted_indices = similarities.argsort(descending=True)

    results: list[tuple[str, float]] = []
    for idx in sorted_indices.tolist():
        if idx == hero_id:
            continue
        name = mapping.get(str(idx), f"Hero-{idx}")
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

    # Load hero mapping
    mapping = load_hero_mapping(args.mapping_file)

    # Resolve hero name to ID
    canonical_name, hero_id = resolve_hero_id(args.hero_name, mapping)

    # Load embeddings
    model_path = Path(args.model_path)
    if not model_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] Embedding file not found: {model_path}"
        )
        console.print(
            "Train embeddings first: python scripts/02_train_embeddings.py --mode train"
        )
        sys.exit(1)

    console.print(
        f"[bold blue]Loading embeddings from: {model_path}[/bold blue]"
    )
    embeddings = load_frozen_embeddings(
        weights_path=args.model_path,
        embed_dim=args.embed_dim,
        num_heroes=args.num_heroes,
    )

    # Find closest heroes
    results = find_closest_heroes(embeddings, hero_id, mapping)

    # Print results
    print_results(canonical_name, results)


if __name__ == "__main__":
    main()

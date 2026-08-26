#!/usr/bin/env python3
"""Test RGCN graph by querying hero relationships.

Accepts a hero name and prints:
  - 5 best synergy partners (highest co-pick win rate)
  - 5 heroes best against (highest counter-pick win rate)
  - 5 heroes required to ban (highest win rate with heroes banned)

Usage::

    python scripts/06_test_rgcn.py --hero_name "Anti-Mage"

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch_geometric.data
from rich.console import Console
from rich.table import Table

from dota2drafter.embeddings.data_extractor import (
    ANTAGONIST,
REQUIRED_BANS,
    SYNERGY,
    DataExtractor,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query RGCN graph for hero relationships (synergy, counters, bans).",
    )
    parser.add_argument(
        "--hero_name",
        type=str,
        required=True,
        help='Hero name (e.g., "Anti-Mage")',
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory with .pt match batches (default: data)",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="hero_mapping.json",
        help="Path to hero mapping file (default: hero_mapping.json)",
    )
    parser.add_argument(
        "--num_heroes", type=int, default=128, help="Number of heroes (default: 128)"
    )
    parser.add_argument(
        "--wilson_threshold",
        type=float,
        default=0.50,
        help="Wilson Score threshold for pruning edges (default: 0.50)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.80,
        help="Decay factor per major patch (default: 0.80)",
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


def query_edges(
    hero_graph: torch_geometric.data.Data,
    contiguous_idx: int,
    edge_type: int,
    sorted_keys: list[str],
    mapping: dict[str, str],
    num_results: int = 5,
) -> list[tuple[str, float]]:
    """Extract and rank edges of a given type from a hero.

    Args:
        hero_graph: PyG Data object with edge_index, edge_type, edge_weight.
        contiguous_idx: Source contiguous hero index (1-based).
        edge_type: Edge type to query (SYNERGY, ANTAGONIST, REQUIRED_BANS).
        sorted_keys: Linear ordered list of hero API ID keys from mapping file.
        mapping: Hero ID -> name mapping.
        num_results: Number of results to return.

    Returns:
        List of (hero_name, weight) tuples sorted descending by weight.
    """
    edge_index = hero_graph.edge_index  # (2, num_edges)
    edge_types = hero_graph.edge_type   # (num_edges,)
    edge_weights = hero_graph.edge_weight.squeeze(-1)  # (num_edges,)

    # Filter edges where source == contiguous_idx and type matches
    mask = (
        (edge_index[0] == contiguous_idx)
        & (edge_types == edge_type)
    )
    edge_ids = torch.where(mask)[0]

    if edge_ids.numel() == 0:
        return []

    # Sort by weight descending
    weights = edge_weights[edge_ids]
    sorted_local = weights.argsort(descending=True)
    top_ids = edge_ids[sorted_local[:num_results]]

    results: list[tuple[str, float]] = []
    for eid in top_ids.tolist():
        target_idx = int(edge_index[1, eid])

        # Convert contiguous 1-based index back to API ID key by linear keys position
        if target_idx - 1 < len(sorted_keys):
            api_id = sorted_keys[target_idx - 1]
            name = mapping.get(api_id, f"Hero-{api_id}")
            w = edge_weights[eid].item()
            results.append((name, w))

    return results


def print_category(
    title: str,
    results: list[tuple[str, float]],
    unit: str = "",
) -> None:
    """Print a formatted results table for one category."""
    if not results:
        console.print(f"\n[bold yellow]{title}[/bold yellow]")
        console.print("  No data available.")
        return

    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Hero", style="bold white")
    table.add_column(unit, justify="right", style="green")

    for i, (name, score) in enumerate(results, 1):
        table.add_row(str(i), name, f"{score:.4f}")

    console.print()
    console.print(table)


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
        console.print(
            f"[bold red]Error:[/bold red] Hero '{canonical_name}' (API ID {api_hero_id}) "
            "is not present in the mapping keys."
        )
        sys.exit(1)

    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Data directory not found: {data_dir}")
        console.print("Gather data first: python scripts/01b_gather_matches.py")
        sys.exit(1)

    console.print(f"[bold blue]Building hero graph from: {data_dir}[/bold blue]")

    # Build and prune graph
    extractor = DataExtractor(num_heroes=args.num_heroes)
    batches = extractor.load_batches(data_dir)
    hero_graph = extractor.build_pruned_hero_graph(
        batches,
        wilson_threshold=args.wilson_threshold,
        gamma=args.gamma,
    )

    console.print(
        f"[bold blue]Graph built: {hero_graph.num_nodes} nodes, "
        f"{hero_graph.edge_index.shape[1]} edges[/bold blue]"
    )

    # Query each relationship type using the linear contiguous indices
    synergy_results = query_edges(hero_graph, contiguous_idx, SYNERGY, sorted_keys, mapping)
    antagonist_results = query_edges(hero_graph, contiguous_idx, ANTAGONIST, sorted_keys, mapping)
    banned_results = query_edges(hero_graph, contiguous_idx, REQUIRED_BANS, sorted_keys, mapping)

    # Print results
    print_category(
        f"[bold]Top 5 Synergy Partners for [green]{canonical_name}[/green][/bold]",
        synergy_results,
        unit="Win Rate",
    )
    print_category(
        f"[bold]Top 5 Heroes [green]{canonical_name}[/green] is Best Against[/bold]",
        antagonist_results,
        unit="Win Rate",
    )
    print_category(
        f"[bold]Top 5 Heroes that require [red]Ban[/red] with [green]{canonical_name}[/green][/bold]",
        banned_results,
        unit="Ban Score",
    )


if __name__ == "__main__":
    main()

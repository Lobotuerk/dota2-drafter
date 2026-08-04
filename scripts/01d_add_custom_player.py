#!/usr/bin/env python3
"""Add or update a custom player's comfort vector in the player comfort map.

Resolves hero names to vector slots via ``hero_mapping.json`` and the hero
indexer, builds a mask-style comfort vector (1.0 at comfort hero slots,
0.0 elsewhere), L2-normalizes it, and merges it into the existing
``player_comfort.pt`` file.

Usage::

    python scripts/01d_add_custom_player.py --id 12345 --heroes Pudge,Anti-Mage
    python scripts/01d_add_custom_player.py --id 12345 --heroes Pudge --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add or update a custom player's comfort vector.",
    )
    parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="Numeric Steam account ID of the custom player.",
    )
    parser.add_argument(
        "--heroes",
        type=str,
        required=True,
        help="Comma-separated list of hero names (e.g. Pudge,Anti-Mage).",
    )
    parser.add_argument(
        "--comfort",
        type=str,
        default="data/player_comfort.pt",
        help="Path to the existing player comfort PyTorch file (default: data/player_comfort.pt)",
    )
    parser.add_argument(
        "--hero_mapping",
        type=str,
        default="hero_mapping.json",
        help="Path to the hero name mapping JSON (default: hero_mapping.json)",
    )
    parser.add_argument(
        "--hero_indexer",
        type=str,
        default="data/hero_indexer.json",
        help="Path to the hero indexer JSON (default: data/hero_indexer.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing player ID in the comfort map.",
    )
    return parser.parse_args()


def load_hero_name_to_api_id(path: str) -> dict[str, str]:
    """Load hero_mapping.json and invert it: hero_name -> api_id string."""
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Hero mapping not found: {mapping_path}")

    with open(mapping_path) as f:
        raw = json.load(f)

    name_to_api_id: dict[str, str] = {}
    for api_id, name in raw.items():
        name_to_api_id[name.strip()] = api_id

    return name_to_api_id


def build_hero_indexer(path: str):
    """Build a HeroIndexer from hero_indexer.json.

    The hero_indexer.json format is a dict mapping API id strings to hero
    objects (or simply a flat dict of API ids).  We construct the list of
    hero dicts expected by ``HeroIndexer.build_mapping``.
    """
    from dota2drafter.processor.hero_indexer import HeroIndexer

    indexer_path = Path(path)
    if not indexer_path.exists():
        raise FileNotFoundError(f"Hero indexer not found: {indexer_path}")

    with open(indexer_path) as f:
        hero_data = json.load(f)

    # hero_indexer.json is typically a flat dict: {api_id_str: {...}}
    heroes = [{"id": int(api_id), "playable": True} for api_id in hero_data.keys()]

    indexer = HeroIndexer()
    indexer.build_mapping(heroes)
    return indexer


def resolve_hero_indices(
    hero_names_str: str,
    name_to_api_id: dict[str, str],
    indexer,
) -> list[int]:
    """Resolve a comma-separated hero name string to contiguous indices.

    Validates every name eagerly and raises on any failure so that no
    intermediate state is left.
    """
    raw_names = [n.strip() for n in hero_names_str.split(",") if n.strip()]
    if not raw_names:
        raise ValueError("No hero names provided.")

    indices: list[int] = []
    for name in raw_names:
        api_id_str = name_to_api_id.get(name)
        if api_id_str is None:
            raise ValueError(f"Hero not found in mapping: '{name}'")

        api_id = int(api_id_str)
        idx = indexer.map_hero_id(api_id)
        if idx is None:
            raise ValueError(f"Hero '{name}' (API id {api_id}) not in indexer.")

        indices.append(idx)

    return indices


def main() -> None:
    args = parse_args()

    # 1. Load hero name -> API id mapping
    console.print("[bold blue]Loading hero mapping...[/bold blue]")
    name_to_api_id = load_hero_name_to_api_id(args.hero_mapping)

    # 2. Load / build hero indexer
    console.print("[bold blue]Loading hero indexer...[/bold blue]")
    indexer = build_hero_indexer(args.hero_indexer)

    # 3. Resolve hero names to contiguous indices (eager validation)
    console.print(f"[bold blue]Resolving heroes: {args.heroes}...[/bold blue]")
    indices = resolve_hero_indices(args.heroes, name_to_api_id, indexer)

    # 4. Load comfort map
    comfort_path = Path(args.comfort)
    if not comfort_path.exists():
        raise FileNotFoundError(f"Comfort file not found: {comfort_path}")

    console.print("[bold blue]Loading comfort map...[/bold blue]")
    comfort_map: dict[int, torch.Tensor] = torch.load(comfort_path, weights_only=True)

    # Check for existing ID
    if args.id in comfort_map and not args.force:
        raise ValueError(
            f"Player ID {args.id} already exists in comfort map. Use --force to overwrite."
        )

    # 5. Determine tensor dimension (vocab_size)
    if comfort_map:
        vocab_size = next(iter(comfort_map.values())).size(0)
    else:
        vocab_size = indexer.get_contiguous_count()

    # Validate that all indices fit within vocab_size
    max_idx = max(indices)
    if max_idx >= vocab_size:
        raise ValueError(
            f"Hero index {max_idx} exceeds vocab_size {vocab_size}. "
            f"The comfort file may be outdated or the indexer mismatched."
        )

    # 6. Construct the vector
    raw_vector = torch.zeros(vocab_size, dtype=torch.float32)
    for idx in indices:
        raw_vector[idx] = 1.0

    # L2 normalization: divide by max(1.0, L2_norm) -- matches 01c_build_comfort.py
    l2_norm = raw_vector.norm().item()
    normalized_vector = raw_vector / max(1.0, l2_norm)

    # 7. Save
    comfort_map[args.id] = normalized_vector
    torch.save(comfort_map, comfort_path)

    hero_names = [name.strip() for name in args.heroes.split(",")]
    console.print(
        f"[bold green]Added player {args.id} with comfort heroes: {', '.join(hero_names)}"
        f" (vector dim={vocab_size}, L2 norm={l2_norm:.4f}) -> {comfort_path}[/bold green]"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        console.print_exception(show_locals=True)
        logger.exception("Add custom player script failed with an error:")
        sys.exit(1)

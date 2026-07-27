#!/usr/bin/env python3
"""Build historical player comfort data for transformer training.

Iterates through match batches, extracts unique player account IDs,
and assigns a zero-initialized (or randomly initialized) tensor to each.
The resulting dictionary is saved as a single ``.pt`` file.

Usage::

    python scripts/01b_build_comfort.py --data_dir data --output data/player_comfort.pt
    python scripts/01b_build_comfort.py --data_dir data --output data/player_comfort.pt --dim 10 --random
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build player comfort map from match batches.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory with .pt match batches (default: data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/player_comfort.pt",
        help="Path to save the comfort map (default: data/player_comfort.pt)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=10,
        help="Player input dimension (default: 10)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random initialization instead of zeros",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Data directory not found: {data_dir}")
        sys.exit(1)

    batch_files = sorted(data_dir.glob("drafts_batch_*.pt"))
    if not batch_files:
        console.print(f"[bold red]Error:[/bold red] No batch files found in {data_dir}")
        sys.exit(1)

    console.print(f"[bold blue]Building player comfort map from {len(batch_files)} batches...[/bold blue]")

    comfort_map: dict[int, torch.Tensor] = {}
    rng = torch.Generator()
    rng.manual_seed(42)

    for batch_file in batch_files:
        try:
            batch = torch.load(batch_file, weights_only=True)
        except FileNotFoundError:
            console.print(f"[bold yellow]Warning:[/bold yellow] Could not load {batch_file}, skipping.")
            continue

        radiant_players = batch.get("radiant_players", [])
        dire_players = batch.get("dire_players", [])

        for player_ids in radiant_players + dire_players:
            for account_id in player_ids:
                if account_id not in comfort_map:
                    if args.random:
                        comfort_map[account_id] = torch.randn(args.dim, generator=rng)
                    else:
                        comfort_map[account_id] = torch.zeros(args.dim)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(comfort_map, output_path)

    console.print(f"[bold green]Saved player comfort map: {len(comfort_map)} unique players -> {output_path}[/bold green]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build historical player comfort data for transformer training.

Iterates through match batches, computes a per-player comfort vector
W_comfort(p, h) = W_p(h) - L_p(h) (wins minus losses per hero),
then applies L2 normalization. The resulting dictionary is saved as a
single ``.pt`` file.

Usage::

    python scripts/01b_build_comfort.py --data_dir data --output data/player_comfort.pt
    python scripts/01b_build_comfort.py --data_dir data --output data/player_comfort.pt --vocab_size 124
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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
        "--vocab_size",
        type=int,
        default=124,
        help="Number of heroes (vocab size, default: 124)",
    )
    parser.add_argument(
        "--hero_indexer",
        type=str,
        default=None,
        help="Path to hero_indexer.json to derive vocab_size dynamically",
    )
    return parser.parse_args()


def load_vocab_size(args: argparse.Namespace) -> int:
    """Determine the vocabulary size (number of heroes).

    Priority:
    1. --vocab_size CLI argument
    2. data/hero_indexer.json (if --hero_indexer is set)
    """
    if args.hero_indexer:
        indexer_path = Path(args.hero_indexer)
        if indexer_path.exists():
            with open(indexer_path, "r") as f:
                indexer_data = json.load(f)
            # hero_indexer.json stores the mapping; count keys for vocab size
            return len(indexer_data)
    return args.vocab_size


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

    vocab_size = load_vocab_size(args)
    console.print(f"[bold blue]Building player comfort map from {len(batch_files)} batches...[/bold blue]")
    console.print(f"[bold blue]Vocab size (heroes): {vocab_size}[/bold blue]")

    # Track wins and losses per (account_id, hero_id) pair
    # win_count[(account_id, hero_id)] = number of wins
    # loss_count[(account_id, hero_id)] = number of losses
    win_count: dict[tuple[int, int], int] = defaultdict(int)
    loss_count: dict[tuple[int, int], int] = defaultdict(int)

    for batch_file in batch_files:
        try:
            batch = torch.load(batch_file, weights_only=True)
        except FileNotFoundError:
            console.print(f"[bold yellow]Warning:[/bold yellow] Could not load {batch_file}, skipping.")
            continue

        radiant_players = batch.get("radiant_players", [])
        dire_players = batch.get("dire_players", [])
        radiant_heroes = batch.get("radiant_heroes", [])
        dire_heroes = batch.get("dire_heroes", [])
        y_labels = batch.get("y", [])

        for i in range(len(radiant_players)):
            radiant_win = y_labels[i].item() == 1.0 if hasattr(y_labels[i], "item") else y_labels[i] == 1

            # Process radiant players
            for j, (account_id, hero_id) in enumerate(zip(radiant_players[i], radiant_heroes[i])):
                # Skip anonymous players and unmapped heroes
                if account_id == 0 or hero_id == -1:
                    continue
                pair = (account_id, hero_id)
                if radiant_win:
                    win_count[pair] += 1
                else:
                    loss_count[pair] += 1

            # Process dire players
            for j, (account_id, hero_id) in enumerate(zip(dire_players[i], dire_heroes[i])):
                if account_id == 0 or hero_id == -1:
                    continue
                pair = (account_id, hero_id)
                if not radiant_win:  # Dire wins when radiant loses
                    win_count[pair] += 1
                else:
                    loss_count[pair] += 1

    # Build comfort map
    comfort_map: dict[int, torch.Tensor] = {}
    all_player_ids = set()
    for (account_id, _hero_id) in win_count.keys():
        all_player_ids.add(account_id)
    for (account_id, _hero_id) in loss_count.keys():
        all_player_ids.add(account_id)

    for account_id in all_player_ids:
        # Build raw comfort vector: W_p(h) - L_p(h) for each hero
        raw_vector = torch.zeros(vocab_size, dtype=torch.float32)
        for hero_idx in range(vocab_size):
            pair = (account_id, hero_idx)
            w = win_count.get(pair, 0)
            l = loss_count.get(pair, 0)
            raw_vector[hero_idx] = float(w - l)

        # L2 normalization: divide by max(1.0, L2_norm)
        l2_norm = raw_vector.norm().item()
        normalized_vector = raw_vector / max(1.0, l2_norm)

        comfort_map[account_id] = normalized_vector

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(comfort_map, output_path)

    console.print(f"[bold green]Saved player comfort map: {len(comfort_map)} unique players -> {output_path}[/bold green]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print_exception(show_locals=True)
        logger.exception("Build comfort script failed with an error:")
        sys.exit(1)

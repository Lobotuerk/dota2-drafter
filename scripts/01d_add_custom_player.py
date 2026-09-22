#!/usr/bin/env python3
"""Add a custom player and their manually specified hero stats to the comfort map.

Usage:
    python scripts/01d_add_custom_player.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

from dota2drafter.processor.hero_indexer import HeroIndexer

console = Console()


def wilson_score(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.5
    p = wins / n
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2))
    return (center - spread) / denominator


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a custom player and hero stats to the comfort map.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory with match batches and hero_indexer.json (default: data)",
    )
    parser.add_argument(
        "--comfort_path",
        type=str,
        default="data/player_comfort.pt",
        help="Path to player comfort .pt file (default: data/player_comfort.pt)",
    )
    parser.add_argument(
        "--hero_indexer",
        type=str,
        default="data/hero_indexer.json",
        help="Path to hero indexer JSON (default: data/hero_indexer.json)",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    # If invoked directly in test environments where sys.argv has pytest flags, parse empty list
    if args is None:
        if len(sys.argv) > 0 and ("pytest" in sys.argv[0] or "py.test" in sys.argv[0]):
            parsed_args = parse_args([])
        else:
            parsed_args = parse_args()
    else:
        parsed_args = parse_args(args)

    comfort_path = Path(parsed_args.comfort_path)
    if not comfort_path.exists():
        raise FileNotFoundError(f"Player comfort file not found: {comfort_path}")

    comfort_map = torch.load(comfort_path, weights_only=True)

    indexer_path = Path(parsed_args.hero_indexer)
    if not indexer_path.exists():
        raise FileNotFoundError(f"Hero indexer not found: {indexer_path}")

    # Load hero mapping & build HeroIndexer for contiguous indexing
    with open(indexer_path) as f:
        hero_data = json.load(f)

    hero_indexer = HeroIndexer()
    heroes = [{"id": int(api_id), "playable": True} for api_id in hero_data.keys()]
    hero_indexer.build_mapping(heroes)

    # Determine vocab_size (ensuring compatibility with existing comfort tensors)
    if comfort_map:
        vocab_size = next(iter(comfort_map.values())).size(0) // 2
    else:
        vocab_size = max([int(v) for v in hero_data.keys()])

    # Map hero names to 0-based contiguous slot
    name_to_idx = {
        name.lower(): hero_indexer.map_hero_id(int(api_id)) - 1
        for api_id, name in hero_data.items()
        if hero_indexer.map_hero_id(int(api_id)) is not None
    }

    account_id = IntPrompt.ask("Enter a custom Account ID (e.g. 999999999)")

    # Start fresh or edit existing
    if account_id in comfort_map:
        console.print(f"[yellow]Player {account_id} already exists! Overwriting...[/yellow]")

    vector = torch.zeros(vocab_size * 2, dtype=torch.float32)
    vector[vocab_size:] = 0.5  # Default wilson

    total_games = IntPrompt.ask(
        "Enter the total estimated matches played by this player (for affinity scaling)"
    )

    console.print("\n[bold]Enter hero stats (type 'done' as Hero Name to finish)[/bold]")
    while True:
        hero = Prompt.ask("Hero Name").lower()
        if hero == "done":
            break

        if hero not in name_to_idx:
            # Try partial match
            matches = [name for name in name_to_idx.keys() if hero in name]
            if not matches:
                console.print(f"[red]Could not find hero: {hero}[/red]")
                continue
            elif len(matches) > 1:
                console.print(f"[yellow]Multiple matches found: {matches}[/yellow]")
                continue
            hero = matches[0]

        hero_idx = name_to_idx[hero]
        games = IntPrompt.ask(f"Games played on {hero.title()}")
        wins = IntPrompt.ask(f"Wins on {hero.title()}")

        affinity = games / max(total_games, 1)
        w_score = wilson_score(wins, games)

        vector[hero_idx] = affinity
        vector[vocab_size + hero_idx] = w_score

        msg = (
            f"[green]Added {hero.title()} (Affinity: {affinity:.2f}, "
            f"Wilson: {w_score:.2f})[/green]"
        )
        console.print(f"{msg}\n")

    comfort_map[account_id] = vector
    torch.save(comfort_map, comfort_path)
    console.print(f"\n[bold green]Saved player {account_id} to {comfort_path}[/bold green]")


if __name__ == "__main__":
    main()

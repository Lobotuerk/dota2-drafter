#!/usr/bin/env python3
import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player comfort mapping.")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory with match batches"
    )
    parser.add_argument(
        "--output", type=str, default="data/player_comfort.pt", help="Output pt file"
    )
    return parser.parse_args()

def load_vocab_size(args: argparse.Namespace) -> int:
    import json
    indexer_path = Path(args.data_dir) / "hero_indexer.json"
    if not indexer_path.exists():
        console.print("[bold yellow]Warning:[/bold yellow] hero_indexer.json not found, defaulting to K=127")
        return 127
    with open(indexer_path, "r") as f:
        data = json.load(f)
    return max([int(v) for v in data.keys()])

def wilson_score(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.5
    p = wins / n
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2))
    return (center - spread) / denominator

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    batch_files = sorted(data_dir.glob("drafts_batch_*.pt"))
    if not batch_files:
        console.print(f"[bold red]Error:[/bold red] No drafts_batch_*.pt found in {data_dir}")
        sys.exit(1)

    vocab_size = load_vocab_size(args)
    console.print(f"[bold blue]Building Hybrid Player Comfort map from {len(batch_files)} batches...[/bold blue]")
    console.print(f"[bold blue]Vocab size (heroes): {vocab_size}[/bold blue]")
    console.print(f"[bold blue]Player vector size: {vocab_size * 2} (Affinity + Wilson Score)[/bold blue]")

    # Tracking dictionaries
    win_count: dict[tuple[int, int], int] = defaultdict(int)
    match_count: dict[tuple[int, int], int] = defaultdict(int)
    player_totals: dict[int, int] = defaultdict(int)

    for batch_path in batch_files:
        batch = torch.load(batch_path, weights_only=True)
        y = batch["y"]  # (N, 1) or (N,)
        
        if y.dim() == 2:
            y = y.squeeze(-1)
            
        r_players = batch["radiant_players"]  # list of lists or (N, 5)
        d_players = batch["dire_players"]
        r_heroes = batch["radiant_heroes"]
        d_heroes = batch["dire_heroes"]

        n_matches = len(y)
        for i in range(n_matches):
            radiant_win = y[i].item() == 1.0

            for p_acc, h_idx in zip(r_players[i], r_heroes[i]):
                if h_idx == -1 or p_acc == 0:
                    continue
                pair = (int(p_acc), int(h_idx) - 1)  # 0-indexed for tensors
                match_count[pair] += 1
                player_totals[int(p_acc)] += 1
                if radiant_win:
                    win_count[pair] += 1

            for p_acc, h_idx in zip(d_players[i], d_heroes[i]):
                if h_idx == -1 or p_acc == 0:
                    continue
                pair = (int(p_acc), int(h_idx) - 1)
                match_count[pair] += 1
                player_totals[int(p_acc)] += 1
                if not radiant_win:
                    win_count[pair] += 1

    # Compile the final mapping
    comfort_map: dict[int, torch.Tensor] = {}
    unique_players = set([p for (p, h) in match_count.keys()])

    for account_id in unique_players:
        total_games = player_totals[account_id]
        if total_games == 0:
            continue

        vector = torch.zeros(vocab_size * 2, dtype=torch.float32)
        
        # Default all Wilson scores to 0.5 initially
        vector[vocab_size:] = 0.5 

        for hero_idx in range(vocab_size):
            pair = (account_id, hero_idx)
            games_on_hero = match_count.get(pair, 0)
            
            if games_on_hero > 0:
                wins_on_hero = win_count.get(pair, 0)
                
                # 1. Affinity (Play rate)
                affinity = games_on_hero / total_games
                
                # 2. Wilson Score
                w_score = wilson_score(wins_on_hero, games_on_hero)
                
                vector[hero_idx] = affinity
                vector[vocab_size + hero_idx] = w_score

        comfort_map[account_id] = vector

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(comfort_map, output_path)

    console.print(f"[bold green]Saved Hybrid player comfort map: {len(comfort_map)} unique players -> {output_path}[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print_exception(show_locals=True)
        sys.exit(1)

#!/usr/bin/env python3
"""Build point-in-time and career player comfort mapping.

Processes matches chronologically so that each match receives a point-in-time
comfort tensor (10, C) built strictly from matches played before it, eliminating
temporal and validation data leakage. Also compiles the final career snapshot
for live inference and scouting.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time player comfort mapping.")
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


def get_match_sort_key(m: dict[str, Any]) -> tuple[int, int | str]:
    """Sort key for chronological ordering of matches."""
    patch_id = m.get("patch_id", 0)
    if hasattr(patch_id, "item"):
        patch_id = patch_id.item()
    elif isinstance(patch_id, torch.Tensor):
        patch_id = int(patch_id.item())
    patch_val = int(patch_id) if patch_id is not None else 0

    mid = m.get("match_id", "")
    try:
        return (patch_val, int(mid))
    except (ValueError, TypeError):
        return (patch_val, str(mid))


def compute_player_vector(
    account_id: int,
    vocab_size: int,
    hero_match_count: dict[tuple[int, int], int],
    hero_win_count: dict[tuple[int, int], int],
    player_totals: dict[int, int],
) -> torch.Tensor:
    """Compute (2 * vocab_size,) comfort vector for an account given current statistics."""
    vector = torch.zeros(vocab_size * 2, dtype=torch.float32)
    vector[vocab_size:] = 0.5  # default Wilson score

    total_games = player_totals.get(account_id, 0)
    if total_games > 0 and account_id != 0:
        for hero_idx in range(vocab_size):
            pair = (account_id, hero_idx)
            games = hero_match_count.get(pair, 0)
            if games > 0:
                wins = hero_win_count.get(pair, 0)
                vector[hero_idx] = games / total_games
                vector[vocab_size + hero_idx] = wilson_score(wins, games)

    return vector


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    batch_files = sorted(data_dir.glob("drafts_batch_*.pt"))
    if not batch_files:
        console.print(f"[bold red]Error:[/bold red] No drafts_batch_*.pt found in {data_dir}")
        sys.exit(1)

    vocab_size = load_vocab_size(args)
    console.print(f"[bold blue]Building Point-in-Time Player Comfort map from {len(batch_files)} batches...[/bold blue]")
    console.print(f"[bold blue]Vocab size (heroes): {vocab_size}[/bold blue]")
    console.print(f"[bold blue]Player vector size: {vocab_size * 2} (Affinity + Wilson Score)[/bold blue]")

    # 1. Load all match records
    all_matches: list[dict[str, Any]] = []
    for batch_idx, batch_path in enumerate(batch_files):
        batch = torch.load(batch_path, weights_only=True)
        y = batch["y"]
        if y.dim() == 2:
            y = y.squeeze(-1)

        r_players = batch["radiant_players"]
        d_players = batch["dire_players"]
        r_heroes = batch["radiant_heroes"]
        d_heroes = batch["dire_heroes"]
        patch_ids = batch.get("patch_ids")
        match_ids = batch.get("match_ids")

        n_matches = len(y)
        for i in range(n_matches):
            if match_ids and i < len(match_ids):
                mid = str(match_ids[i])
            else:
                mid = f"batch_{batch_idx}_match_{i}"

            pid = patch_ids[i] if patch_ids is not None and i < len(patch_ids) else 0

            all_matches.append(
                {
                    "match_id": mid,
                    "patch_id": pid,
                    "y": y[i].item() if hasattr(y[i], "item") else float(y[i]),
                    "radiant_players": [int(p) for p in r_players[i]],
                    "dire_players": [int(p) for p in d_players[i]],
                    "radiant_heroes": [int(h) for h in r_heroes[i]],
                    "dire_heroes": [int(h) for h in d_heroes[i]],
                }
            )

    # 2. Sort matches chronologically: (patch_id, numeric_match_id)
    all_matches.sort(key=get_match_sort_key)
    console.print(f"  Sorted {len(all_matches)} matches chronologically for leakage-free point-in-time calculation.")

    # 3. Chronological simulation for point-in-time per-match comfort
    hero_match_count: dict[tuple[int, int], int] = defaultdict(int)
    hero_win_count: dict[tuple[int, int], int] = defaultdict(int)
    player_totals: dict[int, int] = defaultdict(int)

    match_comfort_map: dict[str, torch.Tensor] = {}

    for m in all_matches:
        mid = str(m["match_id"])
        all_players = m["radiant_players"] + m["dire_players"]

        # Step A: Build comfort tensor for match m using ONLY prior matches
        comfort_rows: list[torch.Tensor] = []
        for account_id in all_players:
            vec = compute_player_vector(
                account_id=account_id,
                vocab_size=vocab_size,
                hero_match_count=hero_match_count,
                hero_win_count=hero_win_count,
                player_totals=player_totals,
            )
            comfort_rows.append(vec)

        match_comfort_map[mid] = torch.stack(comfort_rows)  # shape (10, vocab_size * 2)

        # Step B: Update running stats with match m outcome AFTER building the vector
        radiant_win = (m["y"] == 1.0)
        for p_acc, h_idx in zip(m["radiant_players"], m["radiant_heroes"]):
            if h_idx != -1 and p_acc != 0:
                pair = (p_acc, h_idx - 1)
                hero_match_count[pair] += 1
                player_totals[p_acc] += 1
                if radiant_win:
                    hero_win_count[pair] += 1

        for p_acc, h_idx in zip(m["dire_players"], m["dire_heroes"]):
            if h_idx != -1 and p_acc != 0:
                pair = (p_acc, h_idx - 1)
                hero_match_count[pair] += 1
                player_totals[p_acc] += 1
                if not radiant_win:
                    hero_win_count[pair] += 1

    # 4. Compile final career snapshot across all matches (for live inference)
    player_comfort_map: dict[int, torch.Tensor] = {}
    unique_players = set([p for (p, h) in hero_match_count.keys()])

    for account_id in unique_players:
        if account_id == 0 or player_totals[account_id] == 0:
            continue
        player_comfort_map[account_id] = compute_player_vector(
            account_id=account_id,
            vocab_size=vocab_size,
            hero_match_count=hero_match_count,
            hero_win_count=hero_win_count,
            player_totals=player_totals,
        )

    # 5. Save unified comfort structure
    final_comfort_map: dict[int | str, Any] = {}
    final_comfort_map.update(player_comfort_map)
    final_comfort_map.update(match_comfort_map)
    final_comfort_map["player_comfort"] = player_comfort_map
    final_comfort_map["match_comfort"] = match_comfort_map

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_comfort_map, output_path)

    console.print(
        f"[bold green]Saved Hybrid player comfort map:[/bold green] "
        f"{len(player_comfort_map)} career players, {len(match_comfort_map)} point-in-time matches -> {output_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print_exception(show_locals=True)
        sys.exit(1)

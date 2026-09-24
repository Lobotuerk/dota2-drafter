#!/usr/bin/env python3
"""Script to gather recent high-rank Immortal pub matches and build dataset batches.

Crawls high-rank Ranked matches from Immortal leaderboard players, maps hero picks,
and saves the dataset as PyTorch .pt batches with the format games_batch_*.pt under data/.

Usage::

    python scripts/01f_gather_high_pubs.py
    python scripts/01f_gather_high_pubs.py --limit 2000 --min_rank 80
    python scripts/01f_gather_high_pubs.py config.yaml --limit 5000
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
import torch

from dota2drafter.api.stratz_client import StratzClient
from dota2drafter.config import OutputConfig, StratzConfig, load_config
from dota2drafter.dataset.builder import DatasetBuilder
from dota2drafter.processor.hero_indexer import HeroIndexer
from dota2drafter.processor.tensor_transformer import ProcessedMatch, get_patch_id

logger = logging.getLogger(__name__)
console = Console()

REGION_SERVER_IDS: dict[str, set[int]] = {
    "europe": {3, 8, 9, 28},  # Europe West, Stockholm, Austria, Poland
    "americas": {1, 2, 10, 14, 15, 27, 31, 38},  # US West/East, Brazil, Chile, Peru, Argentina
    "sea": {5, 6, 7, 16, 19, 24, 37},  # Singapore, Dubai, Australia, Japan, HK, Taiwan
    "china": {12, 13, 17, 18, 20, 25, 42, 46, 49, 51},  # Perfect World servers
}

REGION_DIVISIONS: dict[str, str] = {
    "europe": "EUROPE",
    "americas": "AMERICAS",
    "sea": "SE_ASIA",
    "china": "CHINA",
}


def load_hero_indexer(data_dir: str | Path) -> HeroIndexer:
    """Load HeroIndexer from data/hero_indexer.json."""
    indexer_path = Path(data_dir) / "hero_indexer.json"
    if not indexer_path.exists():
        indexer_path = Path("data") / "hero_indexer.json"
    if not indexer_path.exists():
        raise FileNotFoundError(f"Hero indexer not found at: {indexer_path}")
    import json
    with open(indexer_path, "r", encoding="utf-8") as f:
        hero_data = json.load(f)
    heroes = [{"id": int(api_id), "playable": True} for api_id in hero_data.keys()]
    indexer = HeroIndexer()
    indexer.build_mapping(heroes)
    return indexer


def parse_seed_players(players_file: str | Path) -> list[int]:
    """Parse player account IDs from players.txt or dotabuff links."""
    path = Path(players_file)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Match dotabuff URLs or bare digit strings
    url_ids = [int(x) for x in re.findall(r"/players/(\d+)", text)]
    bare_ids = [int(x) for x in re.findall(r"^\s*(\d{6,10})\s*$", text, re.MULTILINE)]
    return list(dict.fromkeys(url_ids + bare_ids))


def load_existing_match_ids(output_dir: Path) -> set[str]:
    """Scan existing drafts_batch_*.pt and games_batch_*.pt to avoid duplicates."""
    seen_ids: set[str] = set()
    patterns = ["drafts_batch_*.pt", "games_batch_*.pt"]
    for pattern in patterns:
        for batch_file in output_dir.glob(pattern):
            try:
                data = torch.load(batch_file, weights_only=False)
                for mid in data.get("match_ids", []):
                    seen_ids.add(str(mid))
            except Exception as e:
                logger.warning("Could not read match IDs from %s: %s", batch_file.name, e)
    return seen_ids


def process_pub_match(
    match_data: dict[str, Any],
    hero_indexer: HeroIndexer,
) -> ProcessedMatch | None:
    """Transform a pub match payload into a ProcessedMatch."""
    match_id = str(match_data.get("id", ""))
    if not match_id:
        return None

    radiant_win = match_data.get("didRadiantWin")
    if radiant_win is None:
        radiant_win = match_data.get("radiantWin")
    if radiant_win is None:
        return None

    players = match_data.get("players", [])
    if len(players) != 10:
        return None

    radiant_players: list[int] = []
    radiant_heroes: list[int] = []
    dire_players: list[int] = []
    dire_heroes: list[int] = []

    for p in players:
        is_radiant = p.get("isRadiant")
        if is_radiant is None and "team" in p:
            is_radiant = (p["team"] in (0, 1))

        acc_id = p.get("steamAccountId") or p.get("accountid") or 0
        h_id = p.get("heroId")
        if h_id is None or h_id <= 0:
            return None

        if is_radiant:
            radiant_players.append(int(acc_id))
            radiant_heroes.append(int(h_id))
        else:
            dire_players.append(int(acc_id))
            dire_heroes.append(int(h_id))

    if len(radiant_heroes) != 5 or len(dire_heroes) != 5:
        return None

    # Check for 24-step picksBans if available
    draft = match_data.get("draft") or {}
    picks_bans = draft.get("picksBans") or []

    steps: list[list[float]] = []
    if len(picks_bans) == 24:
        for step_idx, pb in enumerate(picks_bans):
            is_pick = 1.0 if pb.get("type") == "pick" or pb.get("isPick") else 0.0
            team = float(pb.get("team", 0)) if "team" in pb else (0.0 if pb.get("isRadiant") else 1.0)
            h = pb.get("hero", {}).get("id") if isinstance(pb.get("hero"), dict) else (pb.get("hero") or pb.get("heroId"))
            if h is not None:
                h_idx = hero_indexer.map_hero_id(int(h))
                h_val = float(h_idx) if h_idx is not None else -1.0
            else:
                h_val = -1.0
            steps.append([is_pick, team, h_val, float(step_idx)])
    else:
        # Standard Ranked All Pick: 5 Radiant picks, 5 Dire picks, 14 padding slots
        for step_idx, h_id in enumerate(radiant_heroes):
            h_idx = hero_indexer.map_hero_id(h_id)
            if h_idx is None:
                return None
            steps.append([1.0, 0.0, float(h_idx), float(step_idx)])

        for i, h_id in enumerate(dire_heroes):
            step_idx = i + 5
            h_idx = hero_indexer.map_hero_id(h_id)
            if h_idx is None:
                return None
            steps.append([1.0, 1.0, float(h_idx), float(step_idx)])

        for step_idx in range(10, 24):
            steps.append([0.0, 0.0, -1.0, float(step_idx)])

    x_tensor = torch.tensor(steps, dtype=torch.float32)  # (24, 4)
    y_tensor = torch.tensor([1.0 if radiant_win else 0.0], dtype=torch.float32)  # (1,)
    timestamp = match_data.get("startDateTime") or match_data.get("start_time")
    patch_id = get_patch_id(timestamp)
    match_data["patch_id"] = patch_id

    return ProcessedMatch(
        x_tensor=x_tensor,
        y_tensor=y_tensor,
        match_id=match_id,
        patch_id=patch_id,
        radiant_players=radiant_players,
        dire_players=dire_players,
        radiant_heroes=radiant_heroes,
        dire_heroes=dire_heroes,
    )


async def gather_high_pubs(
    config_path: str,
    limit: int = 5000,
    chunk_size: int | None = None,
    min_rank: int = 80,
    min_patch: int | None = None,
    target_patch: int | None = None,
    region: str = "europe",
    top_leaderboard: int = 3000,
    players_file: str = "players.txt",
    output_dir: str | None = None,
) -> None:
    """Orchestrates gathering and batching high-rank public matches."""
    console.rule("[bold blue]Dota 2 High MMR Pubs Match Gather Pipeline[/bold blue]")

    config = load_config(config_path)

    out_dir = Path(output_dir or config.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_chunk_size = chunk_size or config.output.chunk_size

    output_config = OutputConfig(directory=str(out_dir), chunk_size=out_chunk_size)
    dataset_builder = DatasetBuilder(output_config, prefix="games_batch_")

    region_lower = region.lower()
    allowed_server_ids = REGION_SERVER_IDS.get(region_lower)

    console.print(f"Output Directory: [cyan]{out_dir}[/cyan]")
    console.print(f"Batch Size (Chunk Size): [cyan]{out_chunk_size}[/cyan]")
    console.print(f"Target New Matches: [cyan]{limit}[/cyan]")
    console.print(f"Minimum Rank: [cyan]{min_rank}[/cyan] (80 = Immortal)")
    console.print(f"Target Region: [cyan]{region.upper()}[/cyan]")
    if allowed_server_ids:
        console.print(f"Allowed Server Region IDs: [cyan]{sorted(allowed_server_ids)}[/cyan]")
    if region_lower in REGION_DIVISIONS:
        console.print(f"Top Leaderboard Depth: [cyan]{top_leaderboard}[/cyan] ({REGION_DIVISIONS[region_lower]})")
    if target_patch is not None:
        console.print(f"Target Patch ID: [cyan]{target_patch}[/cyan]")
    elif min_patch is not None:
        console.print(f"Minimum Patch ID: [cyan]{min_patch}[/cyan]")

    hero_indexer = load_hero_indexer(out_dir)
    console.print(f"Loaded Hero Indexer with [green]{hero_indexer.get_contiguous_count()}[/green] heroes.")

    # Load existing match IDs
    existing_ids = load_existing_match_ids(out_dir)
    console.print(f"Found [yellow]{len(existing_ids):,}[/yellow] existing match IDs across batch files.")

    stratz_config = StratzConfig(
        api_key=config.stratz.api_key,
        base_url=config.stratz.base_url,
    )

    player_queue: deque[int] = deque()
    visited_players: set[int] = set()
    patch_counts: dict[int, int] = {}

    # 1. Seed from players.txt
    file_players = parse_seed_players(players_file)
    for pid in file_players:
        if pid not in visited_players:
            player_queue.append(pid)
    console.print(f"Seeded [cyan]{len(file_players)}[/cyan] players from {players_file}.")

    # 2. Seed from Stratz leaderboards
    async with StratzClient(stratz_config) as client:
        if region_lower in REGION_DIVISIONS:
            div = REGION_DIVISIONS[region_lower]
            console.print(f"Fetching top [cyan]{top_leaderboard}[/cyan] {div} leaderboard seeds...")
            chunk_take = 500
            for skip in range(0, top_leaderboard, chunk_take):
                take = min(chunk_take, top_leaderboard - skip)
                try:
                    lead_players = await client.fetch_leaderboard_players(division=div, take=take, skip=skip)
                    for p in lead_players:
                        acc_id = p.get("steamAccountId")
                        if acc_id and acc_id not in visited_players:
                            player_queue.append(acc_id)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not fetch {div} leaderboard at skip {skip}: {e}[/yellow]")
                    break
        else:
            divisions = ["AMERICAS", "EUROPE", "SE_ASIA", "CHINA"]
            for div in divisions:
                try:
                    lead_players = await client.fetch_leaderboard_players(division=div, take=100, skip=0)
                    for p in lead_players:
                        acc_id = p.get("steamAccountId")
                        if acc_id and acc_id not in visited_players:
                            player_queue.append(acc_id)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not fetch leaderboard for {div}: {e}[/yellow]")

        console.print(f"Initial high-rank player queue: [bold green]{len(player_queue)}[/bold green] players.")

        total_collected = 0
        rate_limit_delay = 1.0 / max(1, config.concurrency.rate_limit_per_second)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Gathering matches...", total=limit)

            try:
                while player_queue and total_collected < limit:
                    current_player = player_queue.popleft()
                    if current_player in visited_players:
                        continue
                    visited_players.add(current_player)

                    try:
                        matches = await client.fetch_player_matches(current_player, take=50)
                    except Exception as e:
                        logger.warning("Error fetching matches for player %s: %s", current_player, e)
                        await asyncio.sleep(2.0)
                        continue

                    for m in matches:
                        mid = str(m.get("id", ""))
                        if not mid or mid in existing_ids:
                            continue

                        # Filter: must be ranked
                        lobby = m.get("lobbyType")
                        if lobby not in ("RANKED", "7", 7):
                            continue

                        # Filter: must match server region if specified
                        if allowed_server_ids is not None:
                            match_region = m.get("regionId")
                            if match_region is not None and match_region not in allowed_server_ids:
                                continue

                        # Filter: must meet rank threshold
                        rank = m.get("rank") or m.get("actualRank") or 0
                        bracket = m.get("bracket") or 0
                        if rank < min_rank and bracket < 8:
                            continue

                        # Process match into tensor
                        pm = process_pub_match(m, hero_indexer)
                        if pm is None:
                            continue

                        # Filter: patch threshold if requested
                        if target_patch is not None and pm.patch_id != target_patch:
                            continue
                        if min_patch is not None and pm.patch_id < min_patch:
                            continue

                        # Add to builder
                        dataset_builder.add(pm)
                        existing_ids.add(mid)
                        patch_counts[pm.patch_id] = patch_counts.get(pm.patch_id, 0) + 1
                        total_collected += 1
                        progress.update(task, advance=1)

                        # Enqueue co-players for graph expansion
                        for p in m.get("players", []):
                            co_id = p.get("steamAccountId")
                            if co_id and co_id not in visited_players:
                                player_queue.append(co_id)

                        if total_collected >= limit:
                            break

                    progress.update(
                        task,
                        description=f"[green]Matches: {total_collected}/{limit} | Queue: {len(player_queue)} | Batches: {dataset_builder.get_stats()['batches_saved']}",
                    )
                    await asyncio.sleep(rate_limit_delay)

            finally:
                dataset_builder.flush()

        stats = dataset_builder.get_stats()
        console.print("\n[bold green]Gather Pipeline Complete![/bold green]")
        console.print(f"  New matches collected: [bold cyan]{total_collected}[/bold cyan]")
        console.print(f"  Total batches saved:   [bold cyan]{stats['batches_saved']}[/bold cyan]")
        console.print(f"  Output directory:      [bold cyan]{stats['output_dir']}[/bold cyan]")
        if patch_counts:
            console.print("  [bold blue]Patch Distribution:[/bold blue]")
            for pid in sorted(patch_counts.keys()):
                console.print(f"    Patch {pid:2d}: {patch_counts[pid]:5d} matches")


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    parser = argparse.ArgumentParser(
        description="Gather high-rank Immortal pub matches and build games_batch_*.pt batches."
    )
    parser.add_argument(
        "config_pos",
        nargs="?",
        default=None,
        help="Path to config.yaml (positional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of new matches to collect (default: 5000)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Batch chunk size (default: config output.chunk_size, typically 1000)",
    )
    parser.add_argument(
        "--min_rank",
        type=int,
        default=80,
        help="Minimum rank tier to keep (default: 80 = Immortal)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="europe",
        choices=["europe", "americas", "sea", "china", "all"],
        help="Server region to restrict games to (default: europe)",
    )
    parser.add_argument(
        "--top_leaderboard",
        type=int,
        default=3000,
        help="Top N leaderboard players to seed and crawl from (default: 3000)",
    )
    parser.add_argument(
        "--min_patch",
        type=int,
        default=None,
        help="Minimum patch ID to keep based on timestamp (optional)",
    )
    parser.add_argument(
        "--patch_id",
        type=int,
        default=None,
        help="Specific target patch ID to keep (optional)",
    )
    parser.add_argument(
        "--players_file",
        type=str,
        default="players.txt",
        help="File to seed player account IDs from (default: players.txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save games_batch_*.pt files (default: config output.directory)",
    )

    args = parser.parse_args()
    config_file = args.config_pos or args.config

    try:
        asyncio.run(
            gather_high_pubs(
                config_path=config_file,
                limit=args.limit,
                chunk_size=args.chunk_size,
                min_rank=args.min_rank,
                min_patch=args.min_patch,
                target_patch=args.patch_id,
                region=args.region,
                top_leaderboard=args.top_leaderboard,
                players_file=args.players_file,
                output_dir=args.output_dir,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Process interrupted by user. Saved all completed batches.[/bold yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()

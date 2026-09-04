#!/usr/bin/env python3
"""Remove matches from unapproved leagues and re-chunk the dataset uniformly.

Loads the approved league ids from ``data/leagues.json``, deletes every
completed match in ``state.db`` that belongs to an unapproved league, removes
those matches from the saved ``.pt`` batches, and re-chunks the surviving data
into evenly sized batches (default ``chunk_size`` from ``config.yaml``).

Usage::

    python scripts/01e_cleanup_unapproved.py
    python scripts/01e_cleanup_unapproved.py custom_config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rich.console import Console

from dota2drafter.config import load_config
from dota2drafter.dataset.builder import DatasetBuilder
from dota2drafter.discovery.league_manifest import approved_entries, load_manifest, manifest_path
from dota2drafter.processor.tensor_transformer import ProcessedMatch
from dota2drafter.state import StateDatabase

console = Console()


def _load_samples(batch_path: Path) -> list[ProcessedMatch]:
    """Rebuild ProcessedMatch objects from a saved batch file."""
    batch = torch.load(batch_path, weights_only=True)
    samples = []
    
    # Safely handle patch_ids for backward compatibility
    patch_ids = batch.get("patch_ids")
    
    for i, match_id in enumerate(batch["match_ids"]):
        p_id = int(patch_ids[i].item()) if patch_ids is not None else 21  # Default to latest 7.41e
        samples.append(
            ProcessedMatch(
                x_tensor=batch["x"][i],
                y_tensor=batch["y"][i].reshape(1),
                match_id=match_id,
                patch_id=p_id,
                radiant_players=batch["radiant_players"][i],
                dire_players=batch["dire_players"][i],
                radiant_heroes=batch["radiant_heroes"][i],
                dire_heroes=batch["dire_heroes"][i],
            )
        )
    return samples


def main() -> None:
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    output_dir = Path(config.output.directory)
    manifest_file = manifest_path(output_dir)

    approved = approved_entries(load_manifest(manifest_file))
    approved_league_ids = {str(entry["id"]) for entry in approved}
    console.print(
        f"[bold blue]Using {len(approved_league_ids)} approved leagues "
        f"from {manifest_file}[/bold blue]"
    )

    # Identify completed matches owned by unapproved leagues.
    state_db = StateDatabase(config.state.database_path)
    completed = state_db.get_completed_matches_by_league()
    doomed_match_ids = {
        match_id
        for match_id, league_id in completed
        if league_id is None or league_id not in approved_league_ids
    }
    console.print(
        f"Flagged {len(doomed_match_ids)} completed matches from unapproved leagues"
    )

    if doomed_match_ids:
        removed = state_db.delete_matches(list(doomed_match_ids))
        console.print(f"  Removed {removed} matches from state.db")

    # Rebuild the .pt batches from the surviving samples and re-chunk uniformly.
    batch_files = sorted(output_dir.glob("drafts_batch_*.pt"))
    surviving = [
        sample
        for batch_file in batch_files
        for sample in _load_samples(batch_file)
        if sample.match_id not in doomed_match_ids
    ]
    console.print(
        f"Retained {len(surviving)} matches across {len(batch_files)} batch files"
    )

    if not surviving:
        console.print("[bold yellow]No surviving matches; leaving batches untouched.[/bold yellow]")
        return

    for batch_file in batch_files:
        batch_file.unlink()

    dataset_builder = DatasetBuilder(config.output)
    for sample in surviving:
        dataset_builder.add(sample)
    dataset_builder.flush()

    ds_stats = dataset_builder.get_stats()
    console.print(
        f"[bold green]Re-chunked dataset into {ds_stats['batches_saved']} batches "
        f"-> {ds_stats['output_dir']}[/bold green]"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        sys.exit(130)

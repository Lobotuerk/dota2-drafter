"""Entry points for the Dota 2 draft ingestion pipelines."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from dota2drafter.api.opendota_client import OpenDotaClient
from dota2drafter.api.stratz_client import StratzClient
from dota2drafter.config import PipelineConfig
from dota2drafter.dataset.builder import DatasetBuilder
from dota2drafter.discovery.league_manifest import (
    approved_entries,
    load_manifest,
    manifest_path,
)
from dota2drafter.discovery.league_mapper import LeagueMapper
from dota2drafter.discovery.match_finder import MatchFinder
from dota2drafter.processor.draft_validator import DraftValidator
from dota2drafter.processor.hero_indexer import HeroIndexer
from dota2drafter.processor.tensor_transformer import TensorTransformer
from dota2drafter.state import StateDatabase

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


async def _process_match(
    match_id: str,
    stratz_client: StratzClient,
    opendota_client: OpenDotaClient,
    transformer: TensorTransformer,
    state_db: StateDatabase,
    dataset_builder: DatasetBuilder,
) -> str:
    """Fetch and process a single match using STRATZ first, with OpenDota fallback."""
    try:
        match_data = await stratz_client.fetch_match_details(match_id)

        if match_data is None:
            match_id_int = int(match_id)
            match_data = await opendota_client.fetch_match(match_id_int)

        if match_data is None:
            state_db.mark_failed(match_id, "No data from either STRATZ or OpenDota API")
            return "failed"

        source = "stratz" if match_data.get("draft") else "opendota"
        processed = transformer.transform(match_data, source=source)

        if processed is None:
            state_db.mark_invalid(match_id)
            return "invalid"

        radiant_win = match_data.get("radiantWin")
        if radiant_win is None:
            radiant_win = match_data.get("radiant_win")

        state_db.mark_completed(match_id, radiant_win)
        dataset_builder.add(processed)

        return "processed"

    except Exception as e:
        import aiohttp

        # Check if the error is temporary (rate limit, timeout, server errors)
        is_temporary = False

        # Check for aiohttp.ClientResponseError
        if isinstance(e, aiohttp.ClientResponseError):
            if e.status in (429, 500, 502, 503, 504, 520):
                is_temporary = True

        # Check for connection or timeout errors
        elif isinstance(e, (aiohttp.ClientConnectorError, asyncio.TimeoutError)):
            is_temporary = True

        if is_temporary:
            logger.warning(
                "Temporary error processing match %s (will remain pending for next run): %s",
                match_id,
                e,
            )
            state_db.mark_temp_failed(match_id)
            return "failed"

        # For actual permanent code/schema or hard errors, mark as failed in DB
        state_db.mark_failed(match_id, str(e))
        return "failed"


async def _build_hero_index(
    opendota_client: OpenDotaClient, stratz_client: StratzClient
) -> HeroIndexer:
    """Fetch the hero roster and build the indexer used by the transformer."""
    hero_indexer = HeroIndexer()

    console.print("\n[bold yellow]Step 1/1:[/bold yellow] Fetching hero roster...")
    heroes = []
    try:
        raw_heroes = await opendota_client.fetch_heroes()
        heroes = [
            {"id": h.get("id"), "name": h.get("name"), "playable": True}
            for h in raw_heroes
        ]
    except Exception as e:
        logger.warning("Failed to fetch heroes from OpenDota (%s). Trying STRATZ...", e)

    if not heroes:
        try:
            heroes = await stratz_client.fetch_heroes()
        except Exception as e:
            logger.error("Failed to fetch heroes from fallback STRATZ API: %s", e)
            raise RuntimeError(
                "Could not retrieve hero roster from either OpenDota or STRATZ APIs."
            ) from e

    hero_indexer.build_mapping(heroes)
    console.print(
        f"  [green]OK[/green] Hero mapping built: K={hero_indexer.get_contiguous_count()}"
    )
    return hero_indexer


async def run_discovery_pipeline(config: PipelineConfig) -> None:
    """Discover leagues and merge new entries into the reviewable manifest."""
    console.rule("[bold blue]Dota 2 Draft Discovery Pipeline[/bold blue]")

    stratz_client = StratzClient(config.stratz)
    opendota_client = OpenDotaClient(config.opendota)
    try:
        await _build_hero_index(opendota_client, stratz_client)

        # Step 2: League discovery
        console.print("\n[bold yellow]Step 2/2:[/bold yellow] Discovering leagues...")
        league_mapper = LeagueMapper(
            opendota_client, config, config.concurrency
        )
        discovered = await league_mapper.discover_leagues()
        console.print(f"  [green]OK[/green] Discovered {len(discovered)} leagues")

        # Merge newly discovered leagues into the human-reviewable manifest.
        manifest_file = manifest_path(config.output.directory)
        existing = load_manifest(manifest_file)
        existing_ids = {str(entry["id"]) for entry in existing}
        new_entries = []
        for league in discovered:
            if str(league["id"]) in existing_ids:
                continue
            existing_ids.add(str(league["id"]))
            new_entries.append(
                {
                    "id": league["id"],
                    "name": league["name"],
                    "tier": league["tier"],
                    "review": False,
                    "start_date": league["start_date"],
                    "end_date": league["end_date"],
                }
            )

        if new_entries:
            manifest = existing + new_entries
            _save_manifest(manifest_file, manifest)
            console.print(f"  Added {len(new_entries)} new leagues to {manifest_file}")
        else:
            console.print("  No new leagues to add to the manifest")
    finally:
        await stratz_client.close()


async def run_match_gather_pipeline(config: PipelineConfig) -> None:
    """Gather matches for approved leagues and build the dataset batches."""
    console.rule("[bold blue]Dota 2 Draft Match Gather Pipeline[/bold blue]")

    stratz_client = StratzClient(config.stratz)
    opendota_client = OpenDotaClient(config.opendota)
    try:
        state_db = StateDatabase(config.state.database_path)
        state_db.reset_temp_failures()
        hero_indexer = await _build_hero_index(opendota_client, stratz_client)
        validator = DraftValidator()
        transformer = TensorTransformer(hero_indexer, validator)
        dataset_builder = DatasetBuilder(config.output)

        # Load only the leagues approved for inclusion.
        approved = approved_entries(
            load_manifest(manifest_path(config.output.directory))
        )

        if approved:
            # Register approved leagues so match references stay consistent in state.db
            for league in approved:
                state_db.insert_league(
                    str(league["id"]), league.get("name", ""), league.get("tier", 2)
                )

            # Step 2: Match discovery
            console.print("\n[bold yellow]Step 2/3:[/bold yellow] Discovering matches...")
            match_finder = MatchFinder(
                stratz_client, opendota_client, state_db, config, config.concurrency
            )
            total_new = await match_finder.find_all_matches(approved)
            console.print(f"  [green]OK[/green] Registered {total_new} new matches")
        else:
            console.print(
                "\n[bold yellow]No approved leagues in the manifest. Skipping match "
                "discovery and proceeding to process existing pending matches.[/bold yellow]"
            )

        # Step 3: Fetch, process, and save
        console.print("\n[bold yellow]Step 3/3:[/bold yellow] Processing matches...")
        try:
            total_processed, total_invalid, total_failed = await _process_pending_matches(
                stratz_client, opendota_client, state_db, transformer, dataset_builder, config
            )
        finally:
            dataset_builder.flush()
            state_db.reset_temp_failures()

        # Print summary
        console.print("\n[bold blue]Pipeline Complete![/bold blue]")
        stats = state_db.get_stats()
        console.print(f"  Completed:  {stats.get('completed', 0)}")
        console.print(f"  Invalid:    {stats.get('invalid', 0)}")
        console.print(f"  Failed:     {stats.get('failed', 0)}")
        console.print(f"  Pending:    {stats.get('pending', 0)}")
        console.print(f"  Processed in this run: {total_processed}")
        ds_stats = dataset_builder.get_stats()
        console.print(f"  Batches:    {ds_stats['batches_saved']}")
        console.print(f"  Output:     {ds_stats['output_dir']}")
    finally:
        await stratz_client.close()


async def _process_pending_matches(
    stratz_client: StratzClient,
    opendota_client: OpenDotaClient,
    state_db: StateDatabase,
    transformer: TensorTransformer,
    dataset_builder: DatasetBuilder,
    config: PipelineConfig,
) -> tuple[int, int, int]:
    """Process all pending matches in bounded batches and report tallies."""
    total_processed = 0
    total_invalid = 0
    total_failed = 0
    batch_size = config.concurrency.max_workers

    while True:
        pending = state_db.get_pending_matches(limit=batch_size)
        if not pending:
            break

        tasks = [
            _process_match(
                match_id, stratz_client, opendota_client,
                transformer, state_db, dataset_builder,
            )
            for match_id, league_id in pending
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                total_failed += 1
            elif result == "invalid":
                total_invalid += 1
            elif result == "processed":
                total_processed += 1

        console.print(
            f"  Progress: {total_processed} processed, "
            f"{total_invalid} invalid, {total_failed} failed"
        )

    return total_processed, total_invalid, total_failed


def _save_manifest(path: Path, entries: list[dict]) -> None:
    """Write the league manifest to disk as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

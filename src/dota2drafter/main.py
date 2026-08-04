"""Main entry point for the Dota 2 draft ingestion pipeline."""

from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.logging import RichHandler

from dota2drafter.api.opendota_client import OpenDotaClient
from dota2drafter.api.stratz_client import StratzClient
from dota2drafter.config import PipelineConfig
from dota2drafter.dataset.builder import DatasetBuilder
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


async def run_pipeline(config: PipelineConfig) -> None:
    """Execute the full ingestion pipeline."""
    console.rule("[bold blue]Dota 2 Draft Ingestion Pipeline[/bold blue]")

    stratz_client = StratzClient(config.stratz)
    opendota_client = OpenDotaClient(config.opendota)
    try:
        state_db = StateDatabase(config.state.database_path)
        state_db.reset_temp_failures()
        hero_indexer = HeroIndexer()
        validator = DraftValidator()
        transformer = TensorTransformer(hero_indexer, validator)
        dataset_builder = DatasetBuilder(config.output)

        # Step 1: Hero mapping
        console.print("\n[bold yellow]Step 1/5:[/bold yellow] Fetching hero roster...")
        heroes = []
        try:
            raw_heroes = await opendota_client.fetch_heroes()
            heroes = []
            for h in raw_heroes:
                heroes.append({
                    "id": h.get("id"),
                    "name": h.get("name"),
                    "playable": True  # All heroes returned by OpenDota are playable
                })
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

        try:
            hero_indexer.build_mapping(heroes)
            console.print(
                f"  [green]OK[/green] Hero mapping built: K={hero_indexer.get_contiguous_count()}"
            )
        except Exception as e:
            logger.error("Failed to build hero mapping: %s", e)
            raise

        # Step 2: League discovery
        console.print("\n[bold yellow]Step 2/5:[/bold yellow] Discovering leagues...")
        league_mapper = LeagueMapper(
            stratz_client, opendota_client, state_db, config, config.concurrency
        )
        leagues = await league_mapper.discover_leagues()
        console.print(f"  [green]OK[/green] Found {len(leagues)} tier 1/2 leagues")

        if leagues:
            # Step 3: Match discovery
            console.print("\n[bold yellow]Step 3/5:[/bold yellow] Discovering matches...")
            match_finder = MatchFinder(
                stratz_client, opendota_client, state_db, config, config.concurrency
            )

            # Filter out leagues that have already ended and already exist in our local database
            ended_league_ids = state_db.get_ended_league_ids()
            active_leagues = [
                league for league in leagues if league["id"] not in ended_league_ids
            ]
            skipped_count = len(leagues) - len(active_leagues)
            console.print(
                f"  Querying {len(active_leagues)} active leagues "
                f"(skipped {skipped_count} already ended leagues)"
            )

            total_new = await match_finder.find_all_matches(active_leagues)
            console.print(f"  [green]OK[/green] Registered {total_new} new matches")
        else:
            console.print(
                "\n[bold yellow]No new leagues discovered. Skipping match discovery and "
                "proceeding to process existing pending matches.[/bold yellow]"
            )

        # Step 4 & 5: Fetch, process, and save
        console.print("\n[bold yellow]Step 4/5:[/bold yellow] Processing matches...")
        total_processed = 0
        total_invalid = 0
        total_failed = 0
        batch_size = config.concurrency.max_workers

        try:
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
        ds_stats = dataset_builder.get_stats()
        console.print(f"  Batches:    {ds_stats['batches_saved']}")
        console.print(f"  Output:     {ds_stats['output_dir']}")

    finally:
        await stratz_client.close()




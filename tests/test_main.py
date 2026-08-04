import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dota2drafter.config import PipelineConfig
from dota2drafter.discovery.league_mapper import LeagueMapper
from dota2drafter.discovery.match_finder import MatchFinder
from dota2drafter.main import _process_match, run_discovery_pipeline, run_match_gather_pipeline


@pytest.mark.asyncio
async def test_process_match_success():
    # Setup mocks
    stratz_client = AsyncMock()
    opendota_client = AsyncMock()
    transformer = MagicMock()
    state_db = MagicMock()
    dataset_builder = MagicMock()

    # Valid mock match details from STRATZ
    mock_stratz_match = {
        "id": "10001",
        "radiantWin": True,
        "draft": {"picksBans": []}
    }
    stratz_client.fetch_match_details.return_value = mock_stratz_match

    # Setup transformer mock
    mock_processed = MagicMock()
    transformer.transform.return_value = mock_processed

    # Run _process_match
    result = await _process_match(
        "10001",
        stratz_client,
        opendota_client,
        transformer,
        state_db,
        dataset_builder
    )

    assert result == "processed"
    stratz_client.fetch_match_details.assert_called_once_with("10001")
    opendota_client.fetch_match.assert_not_called()
    transformer.transform.assert_called_once_with(mock_stratz_match, source="stratz")
    state_db.mark_completed.assert_called_once_with("10001", True)
    dataset_builder.add.assert_called_once_with(mock_processed)


@pytest.mark.asyncio
async def test_process_match_opendota_fallback():
    # Setup mocks
    stratz_client = AsyncMock()
    opendota_client = AsyncMock()
    transformer = MagicMock()
    state_db = MagicMock()
    dataset_builder = MagicMock()

    # STRATZ returns None, fallback to OpenDota
    stratz_client.fetch_match_details.return_value = None
    mock_opendota_match = {
        "match_id": 10001,
        "radiant_win": False,
        "game_mode": 2,
        "picks_bans": []
    }
    opendota_client.fetch_match.return_value = mock_opendota_match

    mock_processed = MagicMock()
    transformer.transform.return_value = mock_processed

    result = await _process_match(
        "10001",
        stratz_client,
        opendota_client,
        transformer,
        state_db,
        dataset_builder
    )

    assert result == "processed"
    stratz_client.fetch_match_details.assert_called_once_with("10001")
    opendota_client.fetch_match.assert_called_once_with(10001)
    transformer.transform.assert_called_once_with(mock_opendota_match, source="opendota")
    state_db.mark_completed.assert_called_once_with("10001", False)
    dataset_builder.add.assert_called_once_with(mock_processed)


@pytest.mark.asyncio
async def test_process_match_validation_fail():
    # Setup mocks
    stratz_client = AsyncMock()
    opendota_client = AsyncMock()
    transformer = MagicMock()
    state_db = MagicMock()
    dataset_builder = MagicMock()

    mock_stratz_match = {
        "id": "10001",
        "draft": {"picksBans": []}
    }
    stratz_client.fetch_match_details.return_value = mock_stratz_match
    transformer.transform.return_value = None  # Validation/transform failed

    result = await _process_match(
        "10001",
        stratz_client,
        opendota_client,
        transformer,
        state_db,
        dataset_builder
    )

    assert result == "invalid"
    stratz_client.fetch_match_details.assert_called_once_with("10001")
    opendota_client.fetch_match.assert_not_called()
    state_db.mark_invalid.assert_called_once_with("10001")
    dataset_builder.add.assert_not_called()


@pytest.mark.asyncio
@patch("dota2drafter.main.StratzClient")
@patch("dota2drafter.main.OpenDotaClient")
@patch("dota2drafter.main.StateDatabase")
@patch("dota2drafter.main.MatchFinder")
@patch("dota2drafter.main.DatasetBuilder")
async def test_run_match_gather_pipeline_hero_fallback(
    mock_dataset_builder_cls,
    mock_match_finder_cls,
    mock_state_db_cls,
    mock_opendota_client_cls,
    mock_stratz_client_cls,
    tmp_path,
):
    # Setup mocks
    mock_stratz_client = mock_stratz_client_cls.return_value
    mock_stratz_client.fetch_heroes = AsyncMock(return_value=[
        {"id": 1, "name": "npc_dota_hero_antimage", "playable": True},
        {"id": 2, "name": "npc_dota_hero_axe", "playable": True},
    ])
    mock_stratz_client.close = AsyncMock()

    mock_opendota_client = mock_opendota_client_cls.return_value
    mock_opendota_client.fetch_heroes = AsyncMock(side_effect=Exception("OpenDota 429 Limit"))

    mock_state_db = mock_state_db_cls.return_value
    mock_state_db.get_ended_league_ids.return_value = set()
    mock_state_db.get_pending_matches.return_value = []
    mock_state_db.get_stats.return_value = {}

    mock_match_finder = mock_match_finder_cls.return_value
    mock_match_finder.find_all_matches = AsyncMock(return_value=0)

    mock_dataset_builder = mock_dataset_builder_cls.return_value
    mock_dataset_builder.get_stats.return_value = {"batches_saved": 0, "output_dir": "./data"}

    # Manifest with one approved league so match gathering proceeds
    manifest = tmp_path / "leagues.json"
    manifest.write_text(
        '[{"id": 19944, "name": "EPL Masters", "tier": 2, "review": true}]'
    )

    # Execute
    config = PipelineConfig()
    config.output.directory = str(tmp_path)
    await run_match_gather_pipeline(config)

    # Verifications
    mock_opendota_client.fetch_heroes.assert_called_once()
    mock_stratz_client.fetch_heroes.assert_called_once()
    mock_stratz_client.close.assert_called_once()


@pytest.mark.asyncio
@patch("dota2drafter.main.StratzClient")
@patch("dota2drafter.main.OpenDotaClient")
@patch("dota2drafter.main.LeagueMapper")
async def test_run_discovery_pipeline_merges_new_leagues(
    mock_league_mapper_cls,
    mock_opendota_client_cls,
    mock_stratz_client_cls,
    tmp_path,
):
    mock_stratz_client = mock_stratz_client_cls.return_value
    mock_stratz_client.fetch_heroes = AsyncMock(return_value=[
        {"id": 1, "name": "npc_dota_hero_antimage", "playable": True},
    ])
    mock_stratz_client.close = AsyncMock()

    mock_opendota_client = mock_opendota_client_cls.return_value
    mock_opendota_client.fetch_heroes = AsyncMock(return_value=[
        {"id": 1, "name": "npc_dota_hero_antimage", "playable": True},
    ])

    mock_league_mapper = mock_league_mapper_cls.return_value
    mock_league_mapper.discover_leagues = AsyncMock(return_value=[
        {"id": 19944, "name": "EPL Masters", "tier": 2,
         "start_date": "2026-05-10", "end_date": "2026-06-01"},
        {"id": 55555, "name": "Fun Cup", "tier": 3,
         "start_date": "2026-05-12", "end_date": "2026-05-30"},
    ])

    # Pre-existing manifest with a league already known
    manifest = tmp_path / "leagues.json"
    manifest.write_text(
        '[{"id": 19944, "name": "EPL Masters", "tier": 2, "review": true, '
        '"start_date": "2026-05-10", "end_date": "2026-06-01"}]'
    )

    config = PipelineConfig()
    config.output.directory = str(tmp_path)
    await run_discovery_pipeline(config)

    entries = json.loads(manifest.read_text())
    assert {entry["id"] for entry in entries} == {19944, 55555}
    new_entry = next(entry for entry in entries if entry["id"] == 55555)
    assert new_entry["review"] is False
    # Existing entry is untouched (not re-added with review reset)
    existing_entry = next(entry for entry in entries if entry["id"] == 19944)
    assert existing_entry["review"] is True


@pytest.mark.asyncio
async def test_league_mapper_opendota_discovery():
    # Setup mocks
    opendota_client = AsyncMock()
    opendota_client.fetch_leagues.return_value = [
        {"leagueid": 19944, "tier": "professional", "name": "EPL Masters 2026"}
    ]

    state_db = MagicMock()
    config = PipelineConfig()
    from dota2drafter.config import ConcurrencyConfig
    concurrency = ConcurrencyConfig()

    mapper = LeagueMapper(opendota_client, config, concurrency)

    # Run
    call_count = 0
    async def mock_pro_matches(less_than_id=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            cutoff_ts = int(datetime.fromisoformat(config.cutoff_date).timestamp())
            return [
                {
                    "leagueid": 19944,
                    "league_name": "EPL Masters 2026",
                    "start_time": cutoff_ts + 3600,
                    "match_id": 8906479441,
                }
            ]
        return []

    opendota_client.fetch_recent_pro_matches.side_effect = mock_pro_matches

    leagues = await mapper.discover_leagues()

    assert len(leagues) == 1
    assert leagues[0]["id"] == 19944
    assert leagues[0]["name"] == "EPL Masters 2026"
    assert leagues[0]["tier"] == 2
    assert leagues[0]["start_date"] is not None
    assert leagues[0]["end_date"] is not None


@pytest.mark.asyncio
async def test_match_finder_opendota_fallback():
    # Setup mocks
    stratz_client = AsyncMock()
    stratz_client.fetch_matches_by_league.side_effect = Exception("STRATZ error")

    opendota_client = AsyncMock()
    cutoff_ts = int(datetime.fromisoformat("2026-06-04").timestamp())
    opendota_client.fetch_league_matches.return_value = [
        {"match_id": 8906479441, "start_time": cutoff_ts + 3600}
    ]

    state_db = MagicMock()
    config = PipelineConfig()
    config.cutoff_date = "2026-06-04"
    from dota2drafter.config import ConcurrencyConfig
    concurrency = ConcurrencyConfig()

    finder = MatchFinder(stratz_client, opendota_client, state_db, config, concurrency)

    # Run
    matches = await finder.find_matches_for_league("19944")

    assert len(matches) == 1
    assert matches[0] == ("8906479441", "pending", "19944")
    state_db.upsert_matches.assert_called_once_with([("8906479441", "pending", "19944")])

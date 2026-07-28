import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from dota2drafter.config import PipelineConfig
from dota2drafter.main import _process_match, run_pipeline


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
    state_db.mark_invalid.assert_called_once_with("10001")
    dataset_builder.add.assert_not_called()

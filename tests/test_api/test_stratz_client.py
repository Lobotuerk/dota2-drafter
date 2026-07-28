"""Tests for StratzClient connection pooling and patch filtering."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from dota2drafter.api.stratz_client import StratzClient
from dota2drafter.config import StratzConfig


@pytest.mark.asyncio
async def test_stratz_client_creates_session_on_first_graphql_call():
    """Verify that _graphql creates a persistent session on first call."""
    config = StratzConfig(api_key="test-key")
    client = StratzClient(config)

    assert client._session is None

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": {"leagues": []}})
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_post = MagicMock(return_value=mock_response)
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = mock_post

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await client._graphql("query { test }")

    assert client._session is not None
    assert result == {"data": {"leagues": []}}


@pytest.mark.asyncio
async def test_stratz_client_reuses_session():
    """Verify that subsequent _graphql calls reuse the same session."""
    config = StratzConfig(api_key="test-key")
    client = StratzClient(config)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": {"leagues": []}})
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_post = MagicMock(return_value=mock_response)
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = mock_post

    with patch("aiohttp.ClientSession", return_value=mock_session) as mock_session_cls:
        await client._graphql("query { test1 }")
        await client._graphql("query { test2 }")

    assert mock_session_cls.call_count == 1


@pytest.mark.asyncio
async def test_stratz_client_close_clears_session():
    """Verify that close() clears the session."""
    config = StratzConfig(api_key="test-key")
    client = StratzClient(config)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": {"test": "ok"}})
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_post = MagicMock(return_value=mock_response)
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = mock_post
    mock_session.close = AsyncMock()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await client._graphql("query { test }")

    assert client._session is not None
    await client.close()
    assert client._session is None
    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_leagues_includes_patch_filtering():
    """Verify that fetch_leagues injects patchIds into the request."""
    config = StratzConfig(api_key="test-key")
    client = StratzClient(config)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": {"leagues": []}})
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_post = MagicMock(return_value=mock_response)
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = mock_post

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await client.fetch_leagues([1], "7.35")

    call_args = mock_session.post.call_args
    payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
    variables = payload.get("variables", {})
    request = variables.get("request", {})
    assert "patchIds" in request
    assert request["patchIds"] == ["7.35"]


@pytest.mark.asyncio
async def test_fetch_matches_by_league_includes_patch_filtering():
    """Verify that fetch_matches_by_league injects patchIds into the request."""
    config = StratzConfig(api_key="test-key")
    client = StratzClient(config)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={"data": {"league": {"matches": []}}}
    )
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_post = MagicMock(return_value=mock_response)
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = mock_post

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await client.fetch_matches_by_league("123", "7.35")

    call_args = mock_session.post.call_args
    payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
    variables = payload.get("variables", {})
    request = variables.get("request", {})
    assert "patchIds" in request
    assert request["patchIds"] == ["7.35"]


@pytest.mark.asyncio
async def test_fetch_leagues_without_patch():
    """Verify that fetch_leagues works without patch parameter."""
    config = StratzConfig(api_key="test-key")
    client = StratzClient(config)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": {"leagues": []}})
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_post = MagicMock(return_value=mock_response)
    mock_post.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = mock_post

    with patch("aiohttp.ClientSession", return_value=mock_session):
        await client.fetch_leagues([1], "")

    call_args = mock_session.post.call_args
    payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
    variables = payload.get("variables", {})
    request = variables.get("request", {})
    assert "patchIds" not in request

"""Async OpenDota REST API client for Dota 2 match data."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from dota2drafter.config import OpenDotaConfig

logger = logging.getLogger(__name__)


class OpenDotaClient:
    """Async REST client for the OpenDota API."""

    def __init__(self, config: OpenDotaConfig) -> None:
        self._config = config

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _get(self, endpoint: str) -> dict[str, Any]:
        """Execute a GET request with retry logic."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            url = f"{self._config.base_url}/{endpoint}"
            async with session.get(url) as resp:
                if resp.status == 429:
                    logger.warning("Rate limited by OpenDota API, waiting...")
                    await asyncio.sleep(self._config.retry_delay)
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=429,
                    )
                resp.raise_for_status()
                return await resp.json()

    async def fetch_match(self, match_id: int) -> dict[str, Any] | None:
        """Fetch match details by match ID."""
        try:
            return await self._get(f"matches/{match_id}")
        except Exception as e:
            logger.warning("Failed to fetch match %d: %s", match_id, e)
            return None

    async def fetch_heroes(self) -> list[dict[str, Any]]:
        """Fetch the active hero list."""
        return await self._get("heroes")

    async def fetch_league(self, league_id: int) -> dict[str, Any] | None:
        """Fetch league details by ID."""
        try:
            return await self._get(f"leagues/{league_id}")
        except Exception as e:
            logger.warning("Failed to fetch league %d: %s", league_id, e)
            return None

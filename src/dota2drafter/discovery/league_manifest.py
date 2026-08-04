"""Persistence for the human-reviewable league manifest (data/leagues.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def manifest_path(output_dir: str | Path) -> Path:
    """Return the path to the league manifest inside the output directory."""
    return Path(output_dir) / "leagues.json"


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load the league manifest, returning an empty list when absent."""
    manifest_file = Path(path)
    if not manifest_file.exists():
        return []
    with open(manifest_file, encoding="utf-8") as f:
        return json.load(f)


def reviewable(entry: dict[str, Any]) -> bool:
    """Return True when a manifest entry is flagged for inclusion in the dataset."""
    return bool(entry.get("review"))


def approved_entries(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter the manifest to entries whose review flag is set."""
    return [entry for entry in manifest if reviewable(entry)]

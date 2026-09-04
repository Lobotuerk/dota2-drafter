#!/usr/bin/env python3
"""Gather matches for approved leagues and build dataset batches.

Wraps the match gather pipeline, which reads ``data/leagues.json`` and only
processes leagues whose ``review`` flag is ``true``. Run this after
``01a_gather_leagues.py`` and after reviewing/cleaning the manifest.

Usage::

    python scripts/01b_gather_matches.py
    python scripts/01b_gather_matches.py custom_config.yaml
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console

from dota2drafter.config import load_config
from dota2drafter.main import run_match_gather_pipeline

console = Console()


def main() -> None:
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    try:
        asyncio.run(run_match_gather_pipeline(config))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()

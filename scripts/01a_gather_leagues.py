#!/usr/bin/env python3
"""Discover leagues and refresh the reviewable league manifest.

Wraps the discovery pipeline so leagues discovered after ``cutoff_date`` are
merged into ``data/leagues.json`` for a human review pass. Run this before
``01b_gather_matches.py``.

Usage::

    python scripts/01a_gather_leagues.py
    python scripts/01a_gather_leagues.py custom_config.yaml
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console

from dota2drafter.config import load_config
from dota2drafter.main import run_discovery_pipeline

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
        asyncio.run(run_discovery_pipeline(config))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()

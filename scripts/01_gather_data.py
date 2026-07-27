#!/usr/bin/env python3
"""Run the Dota 2 draft ingestion pipeline.

Wraps the ingestion pipeline from config.yaml, replacing the old
``dota2-drafter`` CLI entry point.

Usage::

    python scripts/01_gather_data.py
    python scripts/01_gather_data.py custom_config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

from dota2drafter.config import load_config
from dota2drafter.main import run_pipeline

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
        import asyncio

        asyncio.run(run_pipeline(config))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()

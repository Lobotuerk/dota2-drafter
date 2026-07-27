"""Tests for the standalone usage scripts under scripts/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.parametrize(
    "script_name",
    [
        "01_gather_data.py",
        "01b_build_comfort.py",
        "02_train_embeddings.py",
        "03_train_rgcn.py",
        "04_train_transformer.py",
    ],
)
def test_script_syntax_and_help(script_name: str) -> None:
    """Verify that each script is syntactically valid and handles arguments or helps correctly."""
    script_path = Path("scripts") / script_name
    assert script_path.exists(), f"{script_name} does not exist on disk"

    if script_name == "01_gather_data.py":
        # 01_gather_data.py doesn't have an argparse --help, so test running with a missing config
        result = subprocess.run(
            [sys.executable, str(script_path), "non_existent_config.yaml"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stdout or "Error" in result.stderr
    else:
        # Other scripts support --help
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout or "options:" in result.stdout

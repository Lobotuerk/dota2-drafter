"""Tests for the 01b_build_comfort.py script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest
import torch


def test_build_comfort_functional(tmp_path: Path) -> None:
    """Verify that 01b_build_comfort.py correctly builds and L2-normalizes the comfort matrix."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_file = tmp_path / "player_comfort.pt"

    # Create mock batch data
    # Match 1: Radiant wins (y = 1.0)
    # Radiant players: 101 (hero 5), 102 (hero 12), 0 (anonymous), 104 (hero 20), 105 (hero 3)
    # Dire players: 201 (hero 8), 202 (hero -1), 203 (hero 45), 204 (hero 6), 205 (hero 77)
    #
    # Match 2: Radiant loses (y = 0.0) -> Dire wins
    # Radiant players: 101 (hero 5), 103 (hero 1), 104 (hero 20), 105 (hero 3), 106 (hero 14)
    # Dire players: 201 (hero 8), 202 (hero 15), 203 (hero 45), 204 (hero 6), 205 (hero 77)
    #
    # Expected outcomes:
    # Player 101:
    # - Hero 5: Played in match 1 (radiant win -> +1) and match 2 (radiant loss -> -1) -> Net score = 0.
    # Player 102:
    # - Hero 12: Played in match 1 (radiant win -> +1) -> Net score = 1. L2 norm = 1. Normalized = 1.
    # Player 202:
    # - Hero -1: Unmapped, should be ignored.
    # - Hero 15: Played in match 2 (dire win -> +1) -> Net score = 1. L2 norm = 1. Normalized = 1.
    # Player 203:
    # - Hero 45: Played in match 1 (dire loss -> -1) and match 2 (dire win -> +1) -> Net score = 0.

    x_dummy = torch.randn(2, 24, 4)
    y_dummy = torch.tensor([1.0, 0.0])
    match_ids = ["match_1", "match_2"]

    radiant_players = [
        [101, 102, 0, 104, 105],
        [101, 103, 104, 105, 106],
    ]
    dire_players = [
        [201, 202, 203, 204, 205],
        [201, 202, 203, 204, 205],
    ]
    radiant_heroes = [
        [5, 12, -1, 20, 3],
        [5, 1, 20, 3, 14],
    ]
    dire_heroes = [
        [8, -1, 45, 6, 77],
        [8, 15, 45, 6, 77],
    ]

    mock_batch = {
        "x": x_dummy,
        "y": y_dummy,
        "match_ids": match_ids,
        "radiant_players": radiant_players,
        "dire_players": dire_players,
        "radiant_heroes": radiant_heroes,
        "dire_heroes": dire_heroes,
    }

    # Save mock batch
    torch.save(mock_batch, data_dir / "drafts_batch_00001.pt")

    # Run the script
    script_path = Path("scripts") / "01b_build_comfort.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data_dir",
            str(data_dir),
            "--output",
            str(output_file),
            "--vocab_size",
            "124",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Load results
    assert output_file.exists()
    comfort_map = torch.load(output_file, weights_only=True)

    # Player 102: only hero 12, radiant win (+1). Normalized vector should have 1.0 at index 12.
    assert 102 in comfort_map
    vec_102 = comfort_map[102]
    assert vec_102.shape == (124,)
    assert torch.isclose(vec_102[12], torch.tensor(1.0))
    # All other values for player 102 should be 0.0
    vec_102_other = vec_102.clone()
    vec_102_other[12] = 0.0
    assert torch.all(vec_102_other == 0.0)

    # Player 202: in match 1 hero -1 (ignored). In match 2 hero 15, dire win (+1).
    assert 202 in comfort_map
    vec_202 = comfort_map[202]
    assert vec_202.shape == (124,)
    assert torch.isclose(vec_202[15], torch.tensor(1.0))
    vec_202_other = vec_202.clone()
    vec_202_other[15] = 0.0
    assert torch.all(vec_202_other == 0.0)

    # Player 101: hero 5 played in win (+1) and loss (-1). Net = 0.
    # Player 101 also played: in Match 1 (win) but it was hero 5. In Match 2 (loss) hero 5.
    assert 101 in comfort_map
    vec_101 = comfort_map[101]
    assert torch.all(vec_101 == 0.0)

    # Anonymous player 0 should NOT be in the map
    assert 0 not in comfort_map

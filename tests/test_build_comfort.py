"""Tests for the 01c_build_comfort.py script."""

from __future__ import annotations

import subprocess
import sys
import math
from pathlib import Path
import pytest
import torch


def wilson_score(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.5
    p = wins / n
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2))
    return (center - spread) / denominator


def test_build_comfort_functional(tmp_path: Path) -> None:
    """Verify that 01c_build_comfort.py correctly builds the Hybrid Player Comfort Matrix."""
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
    script_path = Path("scripts") / "01c_build_comfort.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data_dir",
            str(data_dir),
            "--output",
            str(output_file),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Load results
    assert output_file.exists()
    comfort_map = torch.load(output_file, weights_only=True)

    vocab_size = 127  # default

    # Player 102: only hero 12, radiant win (+1). Total games = 1.
    assert 102 in comfort_map
    vec_102 = comfort_map[102]
    assert vec_102.shape == (vocab_size * 2,)
    # hero 12 -> 0-indexed as index 11
    assert torch.isclose(vec_102[11], torch.tensor(1.0))
    assert torch.isclose(vec_102[vocab_size + 11], torch.tensor(wilson_score(1, 1)))
    # Other affinities should be 0.0, other Wilson scores should be 0.5
    for i in range(vocab_size):
        if i != 11:
            assert vec_102[i] == 0.0
            assert vec_102[vocab_size + i] == 0.5

    # Player 202: in match 1 hero -1 (ignored). In match 2 hero 15, dire win (+1). Total games = 1.
    assert 202 in comfort_map
    vec_202 = comfort_map[202]
    assert vec_202.shape == (vocab_size * 2,)
    # hero 15 -> 0-indexed as index 14
    assert torch.isclose(vec_202[14], torch.tensor(1.0))
    assert torch.isclose(vec_202[vocab_size + 14], torch.tensor(wilson_score(1, 1)))
    for i in range(vocab_size):
        if i != 14:
            assert vec_202[i] == 0.0
            assert vec_202[vocab_size + i] == 0.5

    # Player 101: hero 5 played in win (+1) and loss (-1). Total games = 2.
    assert 101 in comfort_map
    vec_101 = comfort_map[101]
    assert vec_101.shape == (vocab_size * 2,)
    # hero 5 -> index 4. Affinity = 2/2 = 1.0. Wins = 1. Wilson score = wilson_score(1, 2)
    assert torch.isclose(vec_101[4], torch.tensor(1.0))
    assert torch.isclose(vec_101[vocab_size + 4], torch.tensor(wilson_score(1, 2)))
    for i in range(vocab_size):
        if i != 4:
            assert vec_101[i] == 0.0
            assert vec_101[vocab_size + i] == 0.5

    # Anonymous player 0 should NOT be in the map
    assert 0 not in comfort_map

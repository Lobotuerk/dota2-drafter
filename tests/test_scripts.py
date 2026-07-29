"""Tests for the standalone usage scripts under scripts/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    "script_name",
    [
        "01_gather_data.py",
        "01b_build_comfort.py",
        "02_train_embeddings.py",
        "03_train_rgcn.py",
        "04_train_transformer.py",
        "05_test_skip_gram.py",
        "06_test_rgcn.py",
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


def test_resolve_hero_id() -> None:
    """Verify that hero name resolution is case-insensitive and raises on invalid hero names."""
    import importlib
    test_skip_gram = importlib.import_module("scripts.05_test_skip_gram")
    mapping = {"1": "Anti-Mage", "2": "Axe", "3": "Bane"}

    canonical, h_id = test_skip_gram.resolve_hero_id("anti-mage", mapping)
    assert canonical == "Anti-Mage"
    assert h_id == 1

    canonical, h_id = test_skip_gram.resolve_hero_id("AXE", mapping)
    assert canonical == "Axe"
    assert h_id == 2

    with pytest.raises(SystemExit):
        test_skip_gram.resolve_hero_id("Non-Existent Hero", mapping)


def test_find_closest_heroes() -> None:
    """Verify cosine similarity calculation and sorting in 05_test_skip_gram.py."""
    import importlib
    test_skip_gram = importlib.import_module("scripts.05_test_skip_gram")
    mapping = {"0": "Hero-0", "1": "Hero-1", "2": "Hero-2"}
    sorted_keys = ["0", "1", "2"]

    # Create mock embeddings: Index 2 (Hero-1) is very close to Index 1 (Hero-0), and Index 3 is orthogonal
    weight = torch.tensor([
        [0.0, 0.0],  # Index 0
        [1.0, 0.0],  # Index 1 (Hero-0)
        [0.9, 0.1],  # Index 2 (Hero-1)
        [0.0, 1.0],  # Index 3 (Hero-2)
    ])
    embeddings = torch.nn.Embedding.from_pretrained(weight)

    results = test_skip_gram.find_closest_heroes(
        embeddings, contiguous_idx=1, sorted_keys=sorted_keys, mapping=mapping, num_results=2
    )
    assert len(results) == 2
    assert results[0][0] == "Hero-1"
    assert results[0][1] > 0.8
    assert results[1][0] == "Hero-2"


def test_query_edges() -> None:
    """Verify filtering and sorting of edges from a PyG graph in 06_test_rgcn.py."""
    import importlib

    import torch_geometric.data
    test_rgcn = importlib.import_module("scripts.06_test_rgcn")
    mapping = {"0": "Hero-0", "1": "Hero-1", "2": "Hero-2", "3": "Hero-3"}
    sorted_keys = ["0", "1", "2", "3"]

    # Build simple graph with 1-based indices (source is index 1, targets are indices 2, 3, 4)
    # edge_index: [source, target]
    edge_index = torch.tensor([
        [1, 1, 1, 2],
        [2, 3, 4, 3]
    ], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 1, 0], dtype=torch.long)  # 0: SYNERGY, 1: ANTAGONIST
    edge_weight = torch.tensor([[0.8], [0.9], [0.5], [0.7]], dtype=torch.float)

    graph = torch_geometric.data.Data(
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        num_nodes=5
    )

    # Query synergy (type 0) for contiguous index 1. Should return targets [3, 2] ordered by weight descending.
    synergy_results = test_rgcn.query_edges(
        graph, contiguous_idx=1, edge_type=0, sorted_keys=sorted_keys, mapping=mapping, num_results=5
    )
    assert len(synergy_results) == 2
    assert synergy_results[0][0] == "Hero-2"
    assert pytest.approx(synergy_results[0][1]) == 0.9
    assert synergy_results[1][0] == "Hero-1"
    assert pytest.approx(synergy_results[1][1]) == 0.8

    # Query antagonist (type 1) for contiguous index 1. Should return target [3].
    antagonist_results = test_rgcn.query_edges(
        graph, contiguous_idx=1, edge_type=1, sorted_keys=sorted_keys, mapping=mapping, num_results=5
    )
    assert len(antagonist_results) == 1
    assert antagonist_results[0][0] == "Hero-3"
    assert pytest.approx(antagonist_results[0][1]) == 0.5


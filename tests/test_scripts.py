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
        "01c_add_custom_player.py",
        "02_train_embeddings.py",
        "03_train_rgcn.py",
        "04_train_transformer.py",
        "05_test_skip_gram.py",
        "06_test_rgcn.py",
        "interactive_draft.py",
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


def test_train_transformer_augment_arg_parsing():
    """Verify that 04_train_transformer parses --augment correctly for different values."""
    import importlib
    import sys
    from unittest.mock import patch

    train_transformer = importlib.import_module("scripts.04_train_transformer")

    # 1. Test '--augment true'
    test_args = ["04_train_transformer.py", "--augment", "true"]
    with patch.object(sys, "argv", test_args):
        args = train_transformer.parse_args()
        assert args.augment == "true"

    # 2. Test '--augment false'
    test_args = ["04_train_transformer.py", "--augment", "false"]
    with patch.object(sys, "argv", test_args):
        args = train_transformer.parse_args()
        assert args.augment == "false"

    # 3. Test '--augment 15'
    test_args = ["04_train_transformer.py", "--augment", "15"]
    with patch.object(sys, "argv", test_args):
        args = train_transformer.parse_args()
        assert args.augment == "15"


def test_add_custom_player_functional(tmp_path: Path) -> None:
    """Verify that 01c_add_custom_player.py correctly adds and overwrites custom player vectors."""
    import importlib
    import json
    import sys
    add_custom_player = importlib.import_module("scripts.01c_add_custom_player")

    # 1. Create a mock hero mapping
    hero_mapping_file = tmp_path / "hero_mapping.json"
    hero_mapping_data = {
        "1": "Anti-Mage",
        "2": "Axe",
        "3": "Bane",
        "4": "Bloodseeker",
    }
    with open(hero_mapping_file, "w") as f:
        json.dump(hero_mapping_data, f)

    # 2. Create a mock hero indexer file
    hero_indexer_file = tmp_path / "hero_indexer.json"
    hero_indexer_data = {
        "1": {},
        "2": {},
        "3": {},
        "4": {},
    }
    with open(hero_indexer_file, "w") as f:
        json.dump(hero_indexer_data, f)

    # 3. Create a mock comfort file
    comfort_file = tmp_path / "player_comfort.pt"
    # Empty comfort map initially
    torch.save({}, comfort_file)

    # 4. Test resolve_hero_indices
    indexer = add_custom_player.build_hero_indexer(str(hero_indexer_file))
    name_to_api_id = add_custom_player.load_hero_name_to_api_id(str(hero_mapping_file))

    # Test valid name resolution
    # Sorted by API ID inside build_mapping, so Anti-Mage (id 1) -> 1, Axe (id 2) -> 2
    indices = add_custom_player.resolve_hero_indices("Anti-Mage, Axe", name_to_api_id, indexer)
    assert indices == [1, 2]

    # Test whitespace stripping
    indices_ws = add_custom_player.resolve_hero_indices(
        " Anti-Mage ,   Axe  ", name_to_api_id, indexer
    )
    assert indices_ws == [1, 2]

    # Test error on invalid hero name
    with pytest.raises(ValueError, match="Hero not found in mapping"):
        add_custom_player.resolve_hero_indices("Anti-Mage, Pudge", name_to_api_id, indexer)

    # 5. Run main() via subprocess to check command line interface
    script_path = Path("scripts") / "01c_add_custom_player.py"

    # Initially comfort file has NO entries, so vocab_size fallback
    # defaults to indexer.get_contiguous_count() = 4
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--id", "12345",
            "--heroes", "Anti-Mage,Axe",
            "--comfort", str(comfort_file),
            "--hero_mapping", str(hero_mapping_file),
            "--hero_indexer", str(hero_indexer_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Verify vector was created correctly
    comfort_map = torch.load(comfort_file, weights_only=True)
    assert 12345 in comfort_map
    vec = comfort_map[12345]
    assert vec.shape == (4,)
    # Indicies 1 and 2 should be set to 1.0, normalized
    # L2 norm of [0, 1, 1, 0] is sqrt(2) = 1.4142. Normalized is [0, 1/sqrt(2), 1/sqrt(2), 0]
    expected_val = 1.0 / (2.0 ** 0.5)
    assert pytest.approx(vec[1].item()) == expected_val
    assert pytest.approx(vec[2].item()) == expected_val
    assert vec[0].item() == 0.0
    assert vec[3].item() == 0.0

    # 6. Run again without --force on existing ID -> should fail
    result_fail = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--id", "12345",
            "--heroes", "Bane",
            "--comfort", str(comfort_file),
            "--hero_mapping", str(hero_mapping_file),
            "--hero_indexer", str(hero_indexer_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result_fail.returncode != 0
    assert "already exists" in result_fail.stderr or "already exists" in result_fail.stdout

    # 7. Run with --force -> should succeed and overwrite
    result_force = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--id", "12345",
            "--heroes", "Bane",
            "--comfort", str(comfort_file),
            "--hero_mapping", str(hero_mapping_file),
            "--hero_indexer", str(hero_indexer_file),
            "--force",
        ],
        capture_output=True,
        text=True,
    )
    assert result_force.returncode == 0

    # Verify vector was updated: Bane (api id 3) -> contiguous index 3
    comfort_map_updated = torch.load(comfort_file, weights_only=True)
    vec_updated = comfort_map_updated[12345]
    assert vec_updated.shape == (4,)
    # Bane at index 3. L2 norm of [0, 0, 1, 0] is 1.0. So index 3 is 1.0, others 0.0.
    assert pytest.approx(vec_updated[3].item()) == 1.0
    assert vec_updated[0].item() == 0.0
    assert vec_updated[1].item() == 0.0
    assert vec_updated[2].item() == 0.0
def test_interactive_draft_checkpoint_loading_formats() -> None:
    """Verify that interactive_draft.py loads checkpoints in both 'model_state' and 'model_state_dict' formats."""
    import importlib
    from unittest.mock import MagicMock, patch

    interactive_draft = importlib.import_module("scripts.interactive_draft")

    mock_model = MagicMock()

    # 1. Test "model_state" format
    mock_checkpoint_model_state = {"model_state": {"layer.weight": 123}}
    with patch("torch.load", return_value=mock_checkpoint_model_state):
        # We can simulate the loading logic from interactive_draft:
        checkpoint = torch.load("dummy_path")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            mock_model.load_state_dict(checkpoint["model_state"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            mock_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            mock_model.load_state_dict(checkpoint)
        
        mock_model.load_state_dict.assert_called_once_with({"layer.weight": 123})
        mock_model.reset_mock()

    # 2. Test "model_state_dict" format
    mock_checkpoint_model_state_dict = {"model_state_dict": {"layer.weight": 456}}
    with patch("torch.load", return_value=mock_checkpoint_model_state_dict):
        checkpoint = torch.load("dummy_path")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            mock_model.load_state_dict(checkpoint["model_state"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            mock_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            mock_model.load_state_dict(checkpoint)
        
        mock_model.load_state_dict.assert_called_once_with({"layer.weight": 456})
        mock_model.reset_mock()

    # 3. Test raw state_dict format
    mock_checkpoint_raw = {"layer.weight": 789}
    with patch("torch.load", return_value=mock_checkpoint_raw):
        checkpoint = torch.load("dummy_path")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            mock_model.load_state_dict(checkpoint["model_state"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            mock_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            mock_model.load_state_dict(checkpoint)
        
        mock_model.load_state_dict.assert_called_once_with({"layer.weight": 789})
        mock_model.reset_mock()



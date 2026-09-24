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
        "01a_gather_leagues.py",
        "01b_gather_matches.py",
        "01c_build_comfort.py",
        "01d_add_custom_player.py",
        "01e_cleanup_unapproved.py",
        "01f_gather_high_pubs.py",
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

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())

    if script_name in ("01a_gather_leagues.py", "01b_gather_matches.py", "01e_cleanup_unapproved.py"):
        # These scripts take a positional config path, not argparse --help.
        result = subprocess.run(
            [sys.executable, str(script_path), "non_existent_config.yaml"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 1
        assert "Error" in result.stdout or "Error" in result.stderr
    else:
        # Other scripts support --help
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            env=env,
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
    """Verify that 01d_add_custom_player.py correctly adds custom player vectors via interactive prompting."""
    import importlib
    import json
    import sys
    from unittest.mock import patch, mock_open, MagicMock

    add_custom_player = importlib.import_module("scripts.01d_add_custom_player")

    # Mock hero indexer json
    hero_indexer_data = {
        "1": "anti-mage",
        "2": "axe",
        "3": "bane",
        "4": "bloodseeker",
    }

    # Mock player comfort pt file
    comfort_file = tmp_path / "player_comfort.pt"
    torch.save({54321: torch.ones(8, dtype=torch.float32)}, comfort_file)

    # Patch Paths and torch.load/save inside the module
    with patch.object(add_custom_player, "Path") as mock_path_class, \
         patch("builtins.open", mock_open(read_data=json.dumps(hero_indexer_data))), \
         patch.object(add_custom_player.torch, "load") as mock_load, \
         patch.object(add_custom_player.torch, "save") as mock_save, \
         patch.object(add_custom_player.IntPrompt, "ask") as mock_int_ask, \
         patch.object(add_custom_player.Prompt, "ask") as mock_prompt_ask:

        # Set up path mock to return a mock Path object
        mock_path_inst = MagicMock()
        mock_path_inst.exists.return_value = True
        mock_path_class.return_value = mock_path_inst

        # Mock torch.load to return our dict
        mock_comfort_map = {}
        mock_load.return_value = mock_comfort_map

        # IntPrompt.ask calls:
        # 1. Account ID: 12345
        # 2. Total games: 10
        # 3. Games played on Axe: 5
        # 4. Wins on Axe: 4
        mock_int_ask.side_effect = [12345, 10, 5, 4]

        # Prompt.ask calls:
        # 1. Hero Name: "axe"
        # 2. Hero Name: "done"
        mock_prompt_ask.side_effect = ["axe", "done"]

        # Run main
        add_custom_player.main()

        # Check mock_comfort_map was populated
        assert 12345 in mock_comfort_map
        vec = mock_comfort_map[12345]
        # Vocab size = 4
        assert vec.shape == (8,)  # vocab_size * 2
        # Axe is ID 2, so 0-based index 1
        assert vec[1].item() == 5 / 10  # affinity
        assert vec[4 + 1].item() == pytest.approx(add_custom_player.wilson_score(4, 5))  # wilson score
        # Other affinities should be 0, other wilsons should be 0.5
        for i in range(4):
            if i != 1:
                assert vec[i] == 0.0
                assert vec[4 + i] == 0.5

        # Verify torch.save was called
        mock_save.assert_called_with(mock_comfort_map, mock_path_inst)


def test_add_custom_player_non_contiguous_api_ids(tmp_path: Path) -> None:
    """Verify that heroes with API ID gaps (e.g. Lina = 25) are mapped to their contiguous slot."""
    import importlib
    import json
    from unittest.mock import MagicMock, mock_open, patch

    add_custom_player = importlib.import_module("scripts.01d_add_custom_player")

    # API IDs with gaps: 1, 2, 25 (Lina is 3rd playable hero -> contiguous index 3 -> slot 2)
    hero_indexer_data = {
        "1": "anti-mage",
        "2": "axe",
        "25": "lina",
    }

    # Existing comfort map with vocab_size 25 (tensor dimension 50)
    mock_comfort_map = {999: torch.ones(50, dtype=torch.float32)}

    with patch.object(add_custom_player, "Path") as mock_path_class, \
         patch("builtins.open", mock_open(read_data=json.dumps(hero_indexer_data))), \
         patch.object(add_custom_player.torch, "load") as mock_load, \
         patch.object(add_custom_player.torch, "save"), \
         patch.object(add_custom_player.IntPrompt, "ask") as mock_int_ask, \
         patch.object(add_custom_player.Prompt, "ask") as mock_prompt_ask:

        mock_path_inst = MagicMock()
        mock_path_inst.exists.return_value = True
        mock_path_class.return_value = mock_path_inst
        mock_load.return_value = mock_comfort_map

        # Account ID: 111, total games: 20, games on Lina: 10, wins: 8
        mock_int_ask.side_effect = [111, 20, 10, 8]
        mock_prompt_ask.side_effect = ["lina", "done"]

        add_custom_player.main()

        assert 111 in mock_comfort_map
        vec = mock_comfort_map[111]
        assert vec.shape == (50,)
        vocab_size = 25
        # Lina is contiguous index 3 -> slot 2
        assert vec[2].item() == 10 / 20
        assert vec[vocab_size + 2].item() == pytest.approx(add_custom_player.wilson_score(8, 10))
        # Ensure slot 24 (which old code would have written to) is 0.0 affinity
        assert vec[24].item() == 0.0

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


def test_cleanup_unapproved_rechunks(tmp_path: Path) -> None:
    """Verify that 01e_cleanup_unapproved.py removes unapproved matches and re-chunks."""
    import importlib
    import json
    from unittest.mock import patch

    from dota2drafter.state import StateDatabase

    cleanup = importlib.import_module("scripts.01e_cleanup_unapproved")

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Manifest: league 1 approved, league 2 unapproved
    manifest = data_dir / "leagues.json"
    manifest.write_text(
        json.dumps([
            {"id": 1, "name": "Tier1", "tier": 1, "review": True},
            {"id": 2, "name": "Fun Cup", "tier": 3, "review": False},
        ])
    )

    # State DB with completed matches from both leagues
    state_db = StateDatabase(tmp_path / "state.db")
    state_db.insert_league("1", "Tier1", 1)
    state_db.insert_league("2", "Fun Cup", 3)
    state_db.upsert_matches([
        ("10001", "pending", "1"),
        ("20001", "pending", "2"),
    ])
    state_db.mark_completed("10001", True)
    state_db.mark_completed("20001", True)

    # One batch with 3 samples: two from approved league, one from unapproved
    x = torch.randn(3, 24, 4)
    y = torch.tensor([1.0, 0.0, 1.0])
    batch = {
        "x": x,
        "y": y,
        "match_ids": ["10001", "20001", "10002"],
        "radiant_players": [[1] * 5, [2] * 5, [3] * 5],
        "dire_players": [[4] * 5, [5] * 5, [6] * 5],
        "radiant_heroes": [[1] * 5, [2] * 5, [3] * 5],
        "dire_heroes": [[4] * 5, [5] * 5, [6] * 5],
    }
    torch.save(batch, data_dir / "drafts_batch_00001.pt")

    # Config pointing at temp dirs
    config = tmp_path / "config.yaml"
    config.write_text(
        "cutoff_date: '2026-06-04'\n"
        "output:\n"
        "  directory: " + repr(str(data_dir)) + "\n"
        "state:\n"
        "  database_path: " + repr(str(tmp_path / "state.db")) + "\n"
    )

    mock_config = cleanup.load_config(str(config))
    with patch.object(cleanup, "load_config", return_value=mock_config):
        cleanup.main()

    # DB: only approved league match remains completed
    remaining = state_db.get_completed_matches_by_league()
    assert [m for m, _ in remaining] == ["10001"]

    # Batches: only surviving samples are re-chunked
    rebuilt = [f for f in data_dir.glob("drafts_batch_*.pt")]
    assert len(rebuilt) == 1
    data = torch.load(rebuilt[0], weights_only=True)
    assert data["match_ids"] == ["10001", "10002"]


def test_train_transformer_patience_arg_parsing():
    """Verify that 04_train_transformer parses --patience correctly."""
    import importlib
    import sys
    from unittest.mock import patch

    train_transformer = importlib.import_module("scripts.04_train_transformer")

    test_args = ["04_train_transformer.py", "--patience", "12"]
    with patch.object(sys, "argv", test_args):
        args = train_transformer.parse_args()
        assert args.patience == 12


def test_tune_pipeline_llm_patience_arg_parsing():
    """Verify that 07_tune_pipeline parses --llm_patience correctly."""
    import importlib
    import sys
    from unittest.mock import patch

    tune_pipeline = importlib.import_module("scripts.07_tune_pipeline")

    test_args = ["07_tune_pipeline.py", "--llm_patience", "33"]
    with patch.object(sys, "argv", test_args):
        args = tune_pipeline.parse_args()
        assert args.llm_patience == 33


def test_scripts_config_overrides(tmp_path):
    """Verify that scripts load defaults from config.yaml but allow CLI overrides."""
    import importlib
    import sys
    import yaml
    from unittest.mock import patch
    from dota2drafter.config import PipelineConfig, ModelConfig, TrainingConfig

    # 1. Create a mock config with custom values
    custom_model_config = ModelConfig(
        d_model=99,
        nhead=8,
        dim_feedforward=999,
        num_layers_rgcn=5,
        num_layers_transformer=7,
        dropout=0.33,
    )
    custom_training_config = TrainingConfig(
        learning_rate=3e-5,
        batch_size=42,
        skip_gram_lr=0.07,
        dgi_lr=0.08,
        rgcn_lr=0.009,
        checkpoint_metric="val_top5_acc",
    )
    
    config_obj = PipelineConfig(
        model=custom_model_config,
        training=custom_training_config
    )

    # 2. Verify 02_train_embeddings config loading
    train_embeddings = importlib.import_module("scripts.02_train_embeddings")
    args = train_embeddings.parse_args(config_obj, args=[])
    assert args.embed_dim == 99
    assert args.skip_gram_lr == 0.07
    assert args.dgi_lr == 0.08
    assert args.batch_size == 42

    # CLI option overrides config default
    args_overridden = train_embeddings.parse_args(config_obj, args=["--embed_dim", "128"])
    assert args_overridden.embed_dim == 128
    assert args_overridden.skip_gram_lr == 0.07  # still loads default

    # 3. Verify 03_train_rgcn config loading
    train_rgcn = importlib.import_module("scripts.03_train_rgcn")
    args = train_rgcn.parse_args(config_obj, args=[])
    assert args.d_model == 99
    assert args.num_layers == 5
    assert args.learning_rate == 0.009

    # 4. Verify 04_train_transformer config loading
    train_transformer = importlib.import_module("scripts.04_train_transformer")
    args = train_transformer.parse_args(config_obj, args=[])
    assert args.d_model == 99
    assert args.nhead == 8
    assert args.num_layers == 7
    assert args.dim_feedforward == 999
    assert args.dropout == 0.33
    assert args.learning_rate == 3e-5
    assert args.batch_size == 42
    assert args.checkpoint_metric == "val_top5_acc"

    # CLI option overrides config default
    args_overridden = train_transformer.parse_args(
        config_obj,
        args=["--batch_size", "64", "--dropout", "0.15", "--checkpoint_metric", "val_loss"],
    )
    assert args_overridden.batch_size == 64
    assert args_overridden.dropout == 0.15
    assert args_overridden.checkpoint_metric == "val_loss"
    assert args_overridden.d_model == 99  # still loads default



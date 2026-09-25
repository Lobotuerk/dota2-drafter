import os
from pathlib import Path
import pytest
import yaml

from dota2drafter.config import load_config, PipelineConfig


def test_default_config():
    # PipelineConfig should initialize with sensible defaults
    config = PipelineConfig()
    assert config.cutoff_date == "2026-06-04"
    assert config.tiers == [1, 2]
    assert config.stratz.base_url == "https://api.stratz.com/v1"
    assert config.opendota.base_url == "https://api.opendota.com/api"


def test_load_config_from_file(tmp_path):
    config_data = {
        "cutoff_date": "2025-01-01",
        "tiers": [1],
        "stratz": {
            "api_key": "test_key",
            "base_url": "https://test.stratz.com"
        },
        "opendota": {
            "base_url": "https://test.opendota.com"
        },
        "concurrency": {
            "max_workers": 2,
            "max_connections_per_host": 2,
            "rate_limit_per_second": 1
        },
        "output": {
            "directory": str(tmp_path / "data"),
            "chunk_size": 10
        },
        "state": {
            "database_path": str(tmp_path / "state.db")
        }
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
        
    config = load_config(config_file)
    assert config.cutoff_date == "2025-01-01"
    assert config.tiers == [1]
    assert config.stratz.api_key == "test_key"
    assert config.stratz.base_url == "https://test.stratz.com"
    assert config.opendota.base_url == "https://test.opendota.com"
    assert config.concurrency.max_workers == 2
    assert config.output.directory == str(tmp_path / "data")
    assert config.state.database_path == str(tmp_path / "state.db")


def test_resolve_env_vars(tmp_path):
    os.environ["TEST_STRATZ_API_KEY"] = "env_secret_key"
    try:
        config_data = {
            "stratz": {
                "api_key": "${TEST_STRATZ_API_KEY}"
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
            
        config = load_config(config_file)
        assert config.stratz.api_key == "env_secret_key"
    finally:
        del os.environ["TEST_STRATZ_API_KEY"]


def test_load_config_calls_load_dotenv(tmp_path):
    from unittest.mock import patch
    
    config_data = {
        "stratz": {
            "api_key": "some_key"
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
        
    with patch("dota2drafter.config.load_dotenv") as mock_load_dotenv:
        load_config(config_file)
        mock_load_dotenv.assert_called_once()


def test_model_and_training_config_defaults():
    config = PipelineConfig()
    assert config.model.d_model == 64
    assert config.model.dim_feedforward == 128
    assert config.training.learning_rate == 1e-4
    assert config.training.batch_size == 16
    assert config.training.checkpoint_metric == "val_auc"
    assert config.training.stage == 1
    assert config.training.draft_sample_weight == 5.0
    assert config.training.pub_data_dir == "data"
    assert config.training.aw_tau_start == 0.15
    assert config.training.aw_tau_end == 0.08
    assert config.training.aw_tau_decay_epochs == 50
    assert config.training.aw_clip_min == 0.1
    assert config.training.aw_clip_max == 10.0


def test_load_model_and_training_from_file(tmp_path):
    config_data = {
        "model": {
            "d_model": 128,
            "dim_feedforward": 512,
            "dropout": 0.2
        },
        "training": {
            "learning_rate": 5e-5,
            "batch_size": 32,
            "skip_gram_lr": 0.02,
            "checkpoint_metric": "val_loss",
            "aw_tau_start": 0.20,
            "aw_tau_end": 0.05,
            "aw_tau_decay_epochs": 30,
            "aw_clip_min": 0.2,
            "aw_clip_max": 8.0,
            "stage": 2,
            "draft_sample_weight": 5.0,
            "pub_data_dir": "custom_pub_data",
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
        
    config = load_config(config_file)
    assert config.model.d_model == 128
    assert config.model.dim_feedforward == 512
    assert config.model.dropout == 0.2
    assert config.training.learning_rate == 5e-5
    assert config.training.batch_size == 32
    assert config.training.skip_gram_lr == 0.02
    assert config.training.checkpoint_metric == "val_loss"
    assert config.training.aw_tau_start == 0.20
    assert config.training.aw_tau_end == 0.05
    assert config.training.aw_tau_decay_epochs == 30
    assert config.training.aw_clip_min == 0.2
    assert config.training.aw_clip_max == 8.0
    assert config.training.stage == 2
    assert config.training.draft_sample_weight == 5.0
    assert config.training.pub_data_dir == "custom_pub_data"


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


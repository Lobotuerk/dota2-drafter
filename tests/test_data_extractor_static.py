"""Tests for DataExtractor static load_batches method."""

import pytest
import torch
from pathlib import Path
from dota2drafter.embeddings.data_extractor import DataExtractor


def test_load_batches_is_static():
    """Verify that load_batches is a static method."""
    assert isinstance(DataExtractor.__dict__["load_batches"], staticmethod)


def test_load_batches_without_instantiation(tmp_path: Path) -> None:
    """Verify that load_batches can be called without instantiating DataExtractor."""
    output_dir = tmp_path / "data"
    output_dir.mkdir()

    for i in range(3):
        x = torch.randn(10, 24, 3)
        y = torch.tensor([1.0] * 10)
        batch = {"x": x, "y": y, "match_ids": [str(i)]}
        torch.save(batch, output_dir / f"drafts_batch_{i + 1:05d}.pt")

    batches = DataExtractor.load_batches(output_dir)

    assert len(batches) == 3
    assert batches[0]["x"].shape == (10, 24, 3)


def test_load_batches_raises_on_missing_dir(tmp_path: Path) -> None:
    """Verify that load_batches raises FileNotFoundError for empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No batch files found"):
        DataExtractor.load_batches(empty_dir)

import pytest
import torch
from pathlib import Path
from dota2drafter.config import OutputConfig
from dota2drafter.processor.tensor_transformer import ProcessedMatch
from dota2drafter.dataset.builder import DatasetBuilder


def test_dataset_builder(tmp_path):
    config = OutputConfig(directory=str(tmp_path / "data"), chunk_size=2)
    builder = DatasetBuilder(config)
    
    # Create mock processed matches
    match1 = ProcessedMatch(
        x_tensor=torch.randn(24, 3),
        y_tensor=torch.tensor([1.0]),
        match_id="101",
        patch_id=21
    )
    match2 = ProcessedMatch(
        x_tensor=torch.randn(24, 3),
        y_tensor=torch.tensor([0.0]),
        match_id="102",
        patch_id=21
    )
    match3 = ProcessedMatch(
        x_tensor=torch.randn(24, 3),
        y_tensor=torch.tensor([1.0]),
        match_id="103",
        patch_id=21
    )
    
    # Add first match (chunk_size is 2, shouldn't flush yet)
    builder.add(match1)
    stats = builder.get_stats()
    assert stats["batches_saved"] == 0
    assert stats["buffer_size"] == 1
    
    # Add second match (should flush first batch)
    builder.add(match2)
    stats = builder.get_stats()
    assert stats["batches_saved"] == 1
    assert stats["buffer_size"] == 0
    
    # Add third match and manually flush
    builder.add(match3)
    builder.flush()
    stats = builder.get_stats()
    assert stats["batches_saved"] == 2
    assert stats["buffer_size"] == 0
    
    # Check saved files
    out_dir = Path(config.directory)
    file1 = out_dir / "drafts_batch_00001.pt"
    file2 = out_dir / "drafts_batch_00002.pt"
    assert file1.exists()
    assert file2.exists()
    
    # Load and verify contents
    batch1 = torch.load(file1)
    assert batch1["match_ids"] == ["101", "102"]
    assert batch1["x"].shape == (2, 24, 3)
    assert batch1["y"].shape == (2,)
    assert batch1["y"].tolist() == [1.0, 0.0]
    
    batch2 = torch.load(file2)
    assert batch2["match_ids"] == ["103"]
    assert batch2["x"].shape == (1, 24, 3)
    assert batch2["y"].shape == (1,)
    assert batch2["y"].tolist() == [1.0]


def test_dataset_builder_custom_prefix(tmp_path):
    config = OutputConfig(directory=str(tmp_path / "data"), chunk_size=2)
    builder = DatasetBuilder(config, prefix="games_batch_")

    match1 = ProcessedMatch(
        x_tensor=torch.randn(24, 4),
        y_tensor=torch.tensor([1.0]),
        match_id="pub_1",
        patch_id=22,
    )
    match2 = ProcessedMatch(
        x_tensor=torch.randn(24, 4),
        y_tensor=torch.tensor([0.0]),
        match_id="pub_2",
        patch_id=22,
    )

    builder.add(match1)
    builder.add(match2)

    out_dir = Path(config.directory)
    file1 = out_dir / "games_batch_00001.pt"
    assert file1.exists()

    # Verify continuing batch numbering with new builder instance
    builder2 = DatasetBuilder(config, prefix="games_batch_")
    match3 = ProcessedMatch(
        x_tensor=torch.randn(24, 4),
        y_tensor=torch.tensor([1.0]),
        match_id="pub_3",
        patch_id=22,
    )
    builder2.add(match3)
    builder2.flush()

    file2 = out_dir / "games_batch_00002.pt"
    assert file2.exists()


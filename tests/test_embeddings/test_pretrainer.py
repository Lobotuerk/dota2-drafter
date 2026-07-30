"""Tests for pretrainer module."""

from pathlib import Path

import torch
import torch.nn as nn

from dota2drafter.embeddings.pretrainer import load_frozen_embeddings, train_embeddings


def _create_mock_data_dir(tmp_path: Path, num_matches: int = 5) -> Path:
    """Create mock batch files for pretrainer testing."""
    output_dir = tmp_path / "data"
    output_dir.mkdir()

    for batch_idx in range(num_matches):
        steps = []
        for step in range(24):
            if step < 20:
                is_pick = 1.0
                team = 0.0 if step < 10 else 1.0
                hero = (step % 10) + 1
            else:
                is_pick = 0.0
                team = 0.0
                hero = (step % 10) + 1

            steps.append([is_pick, team, float(hero), float(step)])

        x = torch.tensor(steps, dtype=torch.float32)
        y = torch.tensor([1.0 if batch_idx % 2 == 0 else 0.0])

        batch = {"x": x, "y": y, "match_ids": [str(batch_idx)]}
        torch.save(batch, output_dir / f"drafts_batch_{batch_idx + 1:05d}.pt")

    return output_dir


def test_load_frozen_embeddings(tmp_path: Path) -> None:
    # Create mock embedding weights
    num_heroes = 20
    embed_dim = 32
    weights = torch.randn(num_heroes + 1, embed_dim)
    weights_path = tmp_path / "embeddings.pt"
    torch.save(weights, weights_path)

    embedding = load_frozen_embeddings(weights_path, embed_dim, num_heroes)

    assert isinstance(embedding, nn.Embedding)
    assert embedding.weight.shape == (num_heroes + 1, embed_dim)
    assert not embedding.weight.requires_grad


def test_load_frozen_embeddings_shape_mismatch(tmp_path: Path) -> None:
    weights = torch.randn(20, 32)
    weights_path = tmp_path / "embeddings.pt"
    torch.save(weights, weights_path)

    try:
        load_frozen_embeddings(weights_path, 64, 20)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "shape mismatch" in str(e)


def test_load_frozen_embeddings_invalid_type(tmp_path: Path) -> None:
    weights_path = tmp_path / "embeddings.pt"
    torch.save({"not": "a tensor"}, weights_path)

    try:
        load_frozen_embeddings(weights_path, 32, 20)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Expected a tensor" in str(e)


def test_train_embeddings(tmp_path: Path) -> None:
    data_dir = _create_mock_data_dir(tmp_path, num_matches=3)
    output_file = tmp_path / "final_embeddings.pt"

    result_path = train_embeddings(
        data_dir=data_dir,
        output_file=output_file,
        embed_dim=16,
        skip_gram_epochs=1,
        dgi_epochs=1,
        skip_gram_lr=0.01,
        dgi_lr=0.01,
        batch_size=16,
    )

    assert result_path.exists()
    assert output_file.exists()

    # Verify saved embeddings
    embeddings = torch.load(output_file, weights_only=True)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[1] == 16  # embed_dim


def test_train_embeddings_load_frozen(tmp_path: Path) -> None:
    data_dir = _create_mock_data_dir(tmp_path, num_matches=3)
    output_file = tmp_path / "final_embeddings.pt"

    train_embeddings(
        data_dir=data_dir,
        output_file=output_file,
        embed_dim=16,
        skip_gram_epochs=1,
        dgi_epochs=1,
        skip_gram_lr=0.01,
        dgi_lr=0.01,
        batch_size=16,
    )

    # Load as frozen module
    embedding = load_frozen_embeddings(output_file, embed_dim=16, num_heroes=127)
    assert not embedding.weight.requires_grad


def test_package_import(tmp_path: Path) -> None:
    """Test that the embeddings package exports the expected symbols."""
    from dota2drafter.embeddings import (
        DataExtractor,
        DGIModel,
        SkipGramModel,
        SkipGramPair,
        load_frozen_embeddings,
        train_embeddings,
    )

    assert DataExtractor is not None
    assert SkipGramPair is not None
    assert SkipGramModel is not None
    assert DGIModel is not None
    assert callable(train_embeddings)
    assert callable(load_frozen_embeddings)

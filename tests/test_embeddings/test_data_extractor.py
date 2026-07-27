"""Tests for data_extractor module."""

import torch
from pathlib import Path
from dota2drafter.embeddings.data_extractor import DataExtractor, SkipGramPair


def _create_mock_batches(tmp_path: Path, num_matches: int = 10) -> Path:
    """Create mock batch files for testing."""
    output_dir = tmp_path / "data"
    output_dir.mkdir()

    for batch_idx in range(num_matches):
        # Each match: 24 steps, first 20 are picks (10 per team), last 4 are bans
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

            steps.append([is_pick, team, float(hero)])

        x = torch.tensor(steps, dtype=torch.float32)
        y = torch.tensor([1.0 if batch_idx % 2 == 0 else 0.0])

        batch = {"x": x, "y": y, "match_ids": [str(batch_idx)]}
        torch.save(batch, output_dir / f"drafts_batch_{batch_idx + 1:05d}.pt")

    return output_dir


def test_data_extractor_load_batches(tmp_path: Path) -> None:
    extractor = DataExtractor(num_heroes=124)
    data_dir = _create_mock_batches(tmp_path)

    batches = extractor.load_batches(data_dir)
    assert len(batches) == 10
    assert batches[0]["x"].shape == (24, 3)
    assert batches[0]["y"].shape == (1,)


def test_data_extractor_skip_gram_pairs(tmp_path: Path) -> None:
    extractor = DataExtractor(num_heroes=124)
    data_dir = _create_mock_batches(tmp_path, num_matches=5)

    batches = extractor.load_batches(data_dir)
    pairs = extractor.extract_skip_gram_pairs(batches)

    assert len(pairs) > 0
    assert isinstance(pairs[0], SkipGramPair)
    # Each match has 10 picks (5 per team), generating pairs within each team
    # Each team of 5 heroes generates 5*4 = 20 pairs (center, context)
    # 5 matches * 2 teams * 20 pairs = 200 pairs minimum
    assert len(pairs) >= 100


def test_data_extractor_hero_graph(tmp_path: Path) -> None:
    extractor = DataExtractor(num_heroes=124)
    data_dir = _create_mock_batches(tmp_path, num_matches=10)

    batches = extractor.load_batches(data_dir)
    graph = extractor.build_hero_graph(batches)

    assert graph.num_nodes == 124
    assert graph.edge_index.shape[0] == 2
    # Should have synergy edges between co-picked heroes
    assert graph.edge_index.shape[1] > 0


def test_data_extractor_negative_sampling(tmp_path: Path) -> None:
    extractor = DataExtractor(num_heroes=20, negative_samples=5)
    data_dir = _create_mock_batches(tmp_path, num_matches=2)

    batches = extractor.load_batches(data_dir)
    pairs = extractor.extract_skip_gram_pairs(batches)

    # All negatives should be valid hero indices
    for pair in pairs:
        if pair.negative is not None:
            assert 1 <= pair.negative <= 20
            assert pair.negative != pair.center


def test_data_extractor_empty_dir(tmp_path: Path) -> None:
    extractor = DataExtractor(num_heroes=124)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    try:
        extractor.load_batches(empty_dir)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_skip_gram_pair_dataclass() -> None:
    pair = SkipGramPair(center=5, context=10)
    assert pair.center == 5
    assert pair.context == 10
    assert pair.negative is None

    pair_with_neg = SkipGramPair(center=5, context=10, negative=15)
    assert pair_with_neg.negative == 15

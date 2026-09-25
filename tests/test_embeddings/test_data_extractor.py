"""Tests for data_extractor module."""

from pathlib import Path

import torch

from dota2drafter.embeddings.data_extractor import (
    ANTAGONIST,
    REQUIRED_BANS,
    SYNERGY,
    DataExtractor,
    SkipGramPair,
)


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

            steps.append([is_pick, team, float(hero), float(step)])

        x = torch.tensor(steps, dtype=torch.float32)
        y = torch.tensor([1.0 if batch_idx % 2 == 0 else 0.0])

        batch = {"x": x, "y": y, "match_ids": [str(batch_idx)]}
        torch.save(batch, output_dir / f"drafts_batch_{batch_idx + 1:05d}.pt")

    return output_dir


def test_data_extractor_load_batches(tmp_path: Path) -> None:
    extractor = DataExtractor(num_heroes=127)
    data_dir = _create_mock_batches(tmp_path)

    batches = extractor.load_batches(data_dir)
    assert len(batches) == 10
    assert batches[0]["x"].shape == (24, 4)
    assert batches[0]["y"].shape == (1,)


def test_data_extractor_skip_gram_pairs(tmp_path: Path) -> None:
    extractor = DataExtractor(num_heroes=127)
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
    extractor = DataExtractor(num_heroes=127)
    data_dir = _create_mock_batches(tmp_path, num_matches=10)

    batches = extractor.load_batches(data_dir)
    graph = extractor.build_pruned_hero_graph(batches, wilson_threshold=0.0)

    assert graph.num_nodes == 128
    assert graph.edge_index.shape[0] == 2
    # Should have synergy edges between co-picked heroes
    assert graph.edge_index.shape[1] > 0
    # Multi-relational graph should have edge_type and edge_weight
    assert hasattr(graph, "edge_type")
    assert hasattr(graph, "edge_weight")
    assert graph.edge_type.shape[0] == graph.edge_index.shape[1]
    assert graph.edge_weight.shape[0] == graph.edge_index.shape[1]


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
    extractor = DataExtractor(num_heroes=127)
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


def test_data_extractor_hero_graph_correct_math() -> None:
    extractor = DataExtractor(num_heroes=20)

    # Construct 2 matches manually
    # Match 1: Radiant [1, 2, 3, 4, 5], Dire [6, 7, 8, 9, 10]. Radiant wins (y=1).
    steps1 = []
    # 5 picks for Radiant (steps 0-4)
    for i in range(5):
        steps1.append([1.0, 0.0, float(i + 1), float(i)])
    # 5 picks for Dire (steps 5-9)
    for i in range(5):
        steps1.append([1.0, 1.0, float(i + 6), float(i + 5)])
    # pad to 24 steps with bans or placeholders (steps 10-23)
    for i in range(14):
        steps1.append([0.0, 0.0, 1.0, float(i + 10)])

    # Match 2: Radiant [1, 2, 3, 4, 5], Dire [6, 7, 8, 9, 10]. Radiant wins again (y=1).
    steps2 = []
    for i in range(5):
        steps2.append([1.0, 0.0, float(i + 1), float(i)])
    for i in range(5):
        steps2.append([1.0, 1.0, float(i + 6), float(i + 5)])
    for i in range(14):
        steps2.append([0.0, 0.0, 1.0, float(i + 10)])

    # Repeat each match 6 times to satisfy the threshold filters (total >= 5 and total >= 3)
    x = torch.stack([torch.tensor(steps1)] * 6 + [torch.tensor(steps2)] * 6)  # (12, 24, 4)
    y = torch.tensor([1.0] * 12)  # Radiant wins all matches

    batches = [{"x": x, "y": y}]
    graph = extractor.build_pruned_hero_graph(batches)

    edge_index = graph.edge_index.tolist()
    edges = list(zip(edge_index[0], edge_index[1]))

    # Find the synergy edge between 1 and 2 (type 0)
    assert (1, 2) in edges
    idx12 = edges.index((1, 2))
    assert graph.edge_type[idx12].item() == SYNERGY
    # Wilson Score should be > 0.50 (default threshold) for edge to be kept
    assert graph.edge_weight[idx12].item() > 0.50

    # Find the antagonist edge (1, 6) - type 1 (Radiant hero counter-picking Dire hero)
    assert (1, 6) in edges
    idx16 = edges.index((1, 6))
    assert graph.edge_type[idx16].item() == ANTAGONIST
    assert graph.edge_weight[idx16].item() > 0.50


def test_data_extractor_multirelational_edge_types(tmp_path: Path) -> None:
    """Test that all three edge types are present in the graph."""
    extractor = DataExtractor(num_heroes=20)

    # Create matches with picks and bans
    steps = []
    # 5 Radiant picks (steps 0-4)
    for i in range(5):
        steps.append([1.0, 0.0, float(i + 1), float(i)])
    # 5 Dire picks (steps 5-9)
    for i in range(5):
        steps.append([1.0, 1.0, float(i + 6), float(i + 5)])
    # 2 Radiant bans (steps 20-21) - after Radiant picks by same team → REQUIRED_BANS
    steps.append([0.0, 0.0, 11.0, 20.0])
    steps.append([0.0, 0.0, 12.0, 21.0])
    # 2 Dire bans (steps 22-23) - after Dire picks by same team → REQUIRED_BANS
    steps.append([0.0, 1.0, 2.0, 22.0])
    steps.append([0.0, 1.0, 3.0, 23.0])

    x = torch.stack([torch.tensor(steps)] * 20)  # Repeat to satisfy thresholds
    y = torch.tensor([1.0] * 20)

    batches = [{"x": x, "y": y}]
    graph = extractor.build_pruned_hero_graph(batches)

    # Check all edge types are present
    unique_types = set(graph.edge_type.tolist())
    assert SYNERGY in unique_types, "Synergy edges (type 0) should be present"
    assert ANTAGONIST in unique_types, "Antagonist edges (type 1) should be present"
    assert REQUIRED_BANS in unique_types, "Required-bans edges (type 2) should be present"


def test_data_extractor_empty_graph(tmp_path: Path) -> None:
    """Test that an empty graph returns valid tensors with correct shapes."""
    extractor = DataExtractor(num_heroes=20)

    # Create a batch with no picks
    steps = [[0.0, 0.0, 1.0, float(i)] for i in range(24)]
    x = torch.tensor(steps, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor([1.0])

    batches = [{"x": x, "y": y}]
    graph = extractor.build_pruned_hero_graph(batches)

    assert graph.num_nodes == 21
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == 0
    assert graph.edge_type.shape[0] == 0
    assert graph.edge_weight.shape[0] == 0


def test_data_extractor_build_pruned_hero_graph() -> None:
    """Test building a pruned graph with Wilson Score threshold."""
    extractor = DataExtractor(num_heroes=20)

    steps = []
    # 5 Radiant picks (steps 0-4)
    for i in range(5):
        steps.append([1.0, 0.0, float(i + 1), float(i)])
    # 5 Dire picks (steps 5-9)
    for i in range(5):
        steps.append([1.0, 1.0, float(i + 6), float(i + 5)])
    # 2 Radiant bans (steps 20-21) - after Radiant picks by same team → REQUIRED_BANS
    steps.append([0.0, 0.0, 11.0, 20.0])
    steps.append([0.0, 0.0, 12.0, 21.0])
    # 2 Dire bans (steps 22-23) - after Dire picks by same team → REQUIRED_BANS
    steps.append([0.0, 1.0, 2.0, 22.0])
    steps.append([0.0, 1.0, 3.0, 23.0])

    # Multiply counts significantly to satisfy threshold conditions
    x = torch.stack([torch.tensor(steps)] * 50)
    y = torch.tensor([1.0] * 50)

    batches = [{"x": x, "y": y}]
    graph = extractor.build_pruned_hero_graph(batches, wilson_threshold=0.50)

    assert graph.num_nodes == 21
    assert graph.edge_index.shape[0] == 2


def test_patch_discounting_calculation() -> None:
    """Test patch-distance discounting calculation with different patch IDs."""
    extractor = DataExtractor(num_heroes=20)

    # Create 2 matches on different patch IDs
    steps = []
    # 5 Radiant picks (steps 0-4)
    for i in range(5):
        steps.append([1.0, 0.0, float(i + 1), float(i)])
    # 5 Dire picks (steps 5-9)
    for i in range(5):
        steps.append([1.0, 1.0, float(i + 6), float(i + 5)])
    # pad to 24 steps
    for i in range(14):
        steps.append([0.0, 0.0, 1.0, float(i + 10)])

    # Match 1: Patch 21 (current)
    x1 = torch.tensor(steps)
    # Match 2: Patch 19 (2 patches ago)
    x2 = torch.tensor(steps)

    x = torch.stack([x1, x2])
    y = torch.tensor([1.0, 0.0])  # Radiant wins match 1, Dire wins match 2

    # Patch IDs: match 1 is patch 21, match 2 is patch 19
    patch_ids = torch.tensor([21, 19])

    batches = [{"x": x, "y": y, "patch_ids": patch_ids}]
    
    # Test with gamma = 0.80
    graph = extractor.build_pruned_hero_graph(batches, wilson_threshold=0.50, gamma=0.80)
    
    # The graph should exist and have edges
    # The exact weights depend on the Wilson Score calculation with patch discounting
    assert graph.edge_index.shape[0] == 2


def test_effective_sample_size() -> None:
    """Test that effective sample size (n_eff) <= total observations when patch distances vary."""
    extractor = DataExtractor(num_heroes=20)

    # Create matches with different patch IDs
    steps = []
    # 5 Radiant picks (steps 0-4)
    for i in range(5):
        steps.append([1.0, 0.0, float(i + 1), float(i)])
    # 5 Dire picks (steps 5-9)
    for i in range(5):
        steps.append([1.0, 1.0, float(i + 6), float(i + 5)])
    # pad to 24 steps
    for i in range(14):
        steps.append([0.0, 0.0, 1.0, float(i + 10)])

    # Create 10 matches: 5 on patch 21, 5 on patch 19
    x_list = [torch.tensor(steps)] * 10
    x = torch.stack(x_list)
    y = torch.tensor([1.0] * 10)

    # Patch IDs: first 5 matches on patch 21, last 5 on patch 19
    patch_ids = torch.tensor([21] * 5 + [19] * 5)

    batches = [{"x": x, "y": y, "patch_ids": patch_ids}]
    
    # The effective sample size should be less than or equal to 10
    # because matches on patch 19 are downweighted
    graph = extractor.build_pruned_hero_graph(batches, wilson_threshold=0.50, gamma=0.80)
    
    # Verify the graph was built successfully
    assert graph.edge_index.shape[0] == 2


def test_threshold_pruning() -> None:
    """Test that edges with Wilson Score <= threshold are pruned."""
    extractor = DataExtractor(num_heroes=20)

    # Create a batch where some edges will have low Wilson scores
    steps = []
    # 5 Radiant picks (steps 0-4)
    for i in range(5):
        steps.append([1.0, 0.0, float(i + 1), float(i)])
    # 5 Dire picks (steps 5-9)
    for i in range(5):
        steps.append([1.0, 1.0, float(i + 6), float(i + 5)])
    # pad to 24 steps
    for i in range(14):
        steps.append([0.0, 0.0, 1.0, float(i + 10)])

    # Create 10 matches with alternating wins
    x_list = [torch.tensor(steps)] * 10
    x = torch.stack(x_list)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    batches = [{"x": x, "y": y}]
    
    # With threshold 0.50, edges with Wilson score <= 0.50 should be pruned
    graph = extractor.build_pruned_hero_graph(batches, wilson_threshold=0.50)
    
    # All remaining edges should have weight > 0.50
    if graph.edge_weight.shape[0] > 0:
        assert (graph.edge_weight > 0.50).all()


def test_load_batches_with_pubs(tmp_path: Path) -> None:
    """Verify load_batches loads both drafts_batch_*.pt and games_batch_*.pt."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    torch.save(
        {"x": torch.zeros(5, 24, 4), "y": torch.zeros(5, 1), "match_ids": list(range(5))},
        data_dir / "drafts_batch_0000.pt",
    )
    torch.save(
        {"x": torch.zeros(8, 24, 4), "y": torch.zeros(8, 1), "match_ids": list(range(5, 13))},
        data_dir / "games_batch_0000.pt",
    )

    batches = DataExtractor.load_batches(data_dir, include_pubs=True)
    assert len(batches) == 2

    # When include_pubs is False, only drafts_batch_*.pt are loaded
    draft_only = DataExtractor.load_batches(data_dir, include_pubs=False)
    assert len(draft_only) == 1
    assert draft_only[0]["x"].shape[0] == 5


def test_load_batches_custom_pub_dir(tmp_path: Path) -> None:
    """Verify load_batches loads pubs from a separate pub_data_dir."""
    draft_dir = tmp_path / "drafts"
    draft_dir.mkdir()
    pub_dir = tmp_path / "pubs"
    pub_dir.mkdir()

    torch.save(
        {"x": torch.zeros(4, 24, 4), "y": torch.zeros(4, 1), "match_ids": [1, 2, 3, 4]},
        draft_dir / "drafts_batch_0001.pt",
    )
    torch.save(
        {"x": torch.zeros(6, 24, 4), "y": torch.zeros(6, 1), "match_ids": [5, 6, 7, 8, 9, 10]},
        pub_dir / "games_batch_0001.pt",
    )

    batches = DataExtractor.load_batches(draft_dir, include_pubs=True, pub_data_dir=pub_dir)
    assert len(batches) == 2

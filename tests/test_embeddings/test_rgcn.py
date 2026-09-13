"""Tests for RGCN model."""

import torch
from pathlib import Path
from dota2drafter.embeddings.rgcn import HeroRGCN
from torch_geometric.data import Data


def _create_mock_graph(num_nodes: int = 10, num_edges: int = 20) -> Data:
    """Create a mock multi-relational graph for testing."""
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, 3, (num_edges,))
    edge_weight = torch.rand(num_edges, 1)
    return Data(
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
    )


def test_hero_rgcn_init() -> None:
    """Test HeroRGCN initialization."""
    frozen_embeddings = torch.randn(11, 64)  # 10 heroes + padding
    model = HeroRGCN(
        num_nodes=11,
        d_model=64,
        num_relations=3,
        frozen_embeddings=frozen_embeddings,
    )

    assert model.num_nodes == 11
    assert model.d_model == 64
    assert model.num_relations == 3
    assert model.embedding.num_embeddings == 11
    assert model.embedding.embedding_dim == 64
    assert model.embedding.weight.requires_grad is False


def test_hero_rgcn_forward() -> None:
    """Test HeroRGCN forward pass."""
    frozen_embeddings = torch.randn(11, 64)
    model = HeroRGCN(
        num_nodes=11,
        d_model=64,
        num_relations=3,
        frozen_embeddings=frozen_embeddings,
        num_layers=2,
    )

    graph = _create_mock_graph(num_nodes=11, num_edges=30)
    h_gnn = model(graph.edge_index, graph.edge_type)

    assert h_gnn.shape == (11, 64)


def test_hero_rgcn_single_layer() -> None:
    """Test HeroRGCN with a single layer."""
    frozen_embeddings = torch.randn(11, 64)
    model = HeroRGCN(
        num_nodes=11,
        d_model=64,
        num_relations=3,
        frozen_embeddings=frozen_embeddings,
        num_layers=1,
    )

    graph = _create_mock_graph(num_nodes=11, num_edges=30)
    h_gnn = model(graph.edge_index, graph.edge_type)

    assert h_gnn.shape == (11, 64)


def test_hero_rgcn_hidden_dim() -> None:
    """Test HeroRGCN with custom hidden dimension."""
    frozen_embeddings = torch.randn(11, 32)
    model = HeroRGCN(
        num_nodes=11,
        d_model=32,
        num_relations=3,
        frozen_embeddings=frozen_embeddings,
        hidden_dim=16,
        num_layers=2,
    )

    graph = _create_mock_graph(num_nodes=11, num_edges=30)
    h_gnn = model(graph.edge_index, graph.edge_type)

    assert h_gnn.shape == (11, 32)


def test_hero_rgcn_get_embeddings() -> None:
    """Test get_embeddings method."""
    frozen_embeddings = torch.randn(11, 64)
    model = HeroRGCN(
        num_nodes=11,
        d_model=64,
        num_relations=3,
        frozen_embeddings=frozen_embeddings,
    )

    graph = _create_mock_graph(num_nodes=11, num_edges=30)
    h_gnn = model.get_embeddings(graph)

    assert h_gnn.shape == (11, 64)


def test_hero_rgcn_save_load(tmp_path: Path) -> None:
    """Test saving and loading HeroRGCN model."""
    frozen_embeddings = torch.randn(11, 64)
    model = HeroRGCN(
        num_nodes=11,
        d_model=64,
        num_relations=3,
        frozen_embeddings=frozen_embeddings,
    )

    save_path = tmp_path / "rgcn.pt"
    model.save(save_path)

    loaded_model = HeroRGCN.load(
        path=save_path,
        frozen_embeddings=frozen_embeddings,
        d_model=64,
        num_relations=3,
    )

    graph = _create_mock_graph(num_nodes=11, num_edges=30)
    orig_out = model(graph.edge_index, graph.edge_type)
    loaded_out = loaded_model(graph.edge_index, graph.edge_type)

    assert torch.allclose(orig_out, loaded_out)


def test_hero_rgcn_save_load_multi_layer(tmp_path: Path) -> None:
    """Test saving and loading HeroRGCN model with custom multi-layer structures (e.g., 6 layers)."""
    frozen_embeddings = torch.randn(11, 64)
    model = HeroRGCN(
        num_nodes=11,
        d_model=64,
        num_relations=3,
        frozen_embeddings=frozen_embeddings,
        num_layers=6,
    )

    save_path = tmp_path / "rgcn_6layers.pt"
    model.save(save_path)

    loaded_model = HeroRGCN.load(
        path=save_path,
        frozen_embeddings=frozen_embeddings,
        d_model=64,
        num_relations=3,
    )

    assert len(loaded_model.rgcn_layers) == 6

    graph = _create_mock_graph(num_nodes=11, num_edges=30)
    orig_out = model(graph.edge_index, graph.edge_type)
    loaded_out = loaded_model(graph.edge_index, graph.edge_type)

    assert torch.allclose(orig_out, loaded_out)

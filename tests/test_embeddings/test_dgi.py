"""Tests for dgi module."""

import torch
from pathlib import Path
from dota2drafter.embeddings.dgi import DGIModel, DGIEncoder
from torch_geometric.data import Data


def _create_mock_graph(num_nodes: int = 20, num_edges: int = 50) -> Data:
    """Create a mock hero interaction graph."""
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(edge_index=edge_index, num_nodes=num_nodes)


def test_dgi_encoder_init() -> None:
    encoder = DGIEncoder(embed_dim=64)
    assert encoder.conv1.in_channels == 64
    assert encoder.conv1.out_channels == 64


def test_dgi_encoder_forward() -> None:
    encoder = DGIEncoder(embed_dim=32, hidden_dim=16)
    x = torch.randn(20, 32)
    edge_index = torch.randint(0, 20, (2, 50))

    out = encoder(x, edge_index)
    assert out.shape == (20, 32)


def test_dgi_model_init() -> None:
    model = DGIModel(embed_dim=64)
    assert model.encoder.conv1.in_channels == 64


def test_dgi_model_forward() -> None:
    model = DGIModel(embed_dim=32)
    x = torch.randn(20, 32)
    edge_index = torch.randint(0, 20, (2, 50))

    local, _, global_repr = model(x, edge_index)
    assert local.shape == (20, 32)
    assert global_repr.shape == (32,)





def test_dgi_loss_computation() -> None:
    model = DGIModel(embed_dim=32)

    pos_local = torch.randn(10, 32)
    pos_global = torch.randn(1, 32)
    neg_local = torch.randn(10, 32)

    loss = model.model.loss(pos_local, neg_local, pos_global)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_dgi_train_epoch() -> None:
    model = DGIModel(embed_dim=32)
    graph = _create_mock_graph(num_nodes=20, num_edges=50)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model.train_epoch(graph, optimizer)

    assert loss > 0
    assert loss < 30.0


def test_dgi_get_embeddings() -> None:
    model = DGIModel(embed_dim=32)
    graph = _create_mock_graph(num_nodes=20, num_edges=50)

    embeddings = model.get_embeddings(graph)
    assert embeddings.shape == (20, 32)





def test_dgi_corrupt() -> None:
    from dota2drafter.embeddings.dgi import corruption
    x = torch.randn(20, 32)
    edge_index = torch.randint(0, 20, (2, 50))
    corrupted, _ = corruption(x, edge_index)
    assert corrupted.shape == x.shape
    # Corrupted should be a permutation
    assert torch.allclose(torch.sort(corrupted, dim=0).values, torch.sort(x, dim=0).values)

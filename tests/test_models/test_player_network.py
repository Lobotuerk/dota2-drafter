"""Unit tests for PlayerComfortNetwork."""

import pytest
import torch

from dota2drafter.models.player_network import PlayerComfortNetwork


def test_player_comfort_network_forward():
    """Test forward pass shape and basic behavior."""
    d_model = 64
    input_dim = 10  # C features
    batch_size = 4

    model = PlayerComfortNetwork(input_dim=input_dim, d_model=d_model)

    # Input: (B, 10, C)
    player_comfort = torch.randn(batch_size, 10, input_dim)
    output = model(player_comfort)

    # Output: (B, 10, d_model)
    assert output.shape == (batch_size, 10, d_model)
    assert output.dtype == torch.float32


def test_player_comfort_network_different_input_dims():
    """Test with various input dimensions."""
    for input_dim in [5, 10, 20, 50]:
        d_model = 128
        batch_size = 2

        model = PlayerComfortNetwork(input_dim=input_dim, d_model=d_model)
        player_comfort = torch.randn(batch_size, 10, input_dim)
        output = model(player_comfort)

        assert output.shape == (batch_size, 10, d_model)


def test_player_comfort_network_hidden_dim():
    """Test with explicit hidden dimension."""
    d_model = 64
    hidden_dim = 32
    input_dim = 10
    batch_size = 1

    model = PlayerComfortNetwork(
        input_dim=input_dim,
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_layers=3,
    )

    player_comfort = torch.randn(batch_size, 10, input_dim)
    output = model(player_comfort)

    assert output.shape == (batch_size, 10, d_model)


def test_player_comfort_network_batch_sizes():
    """Test with various batch sizes."""
    d_model = 64
    input_dim = 10

    model = PlayerComfortNetwork(input_dim=input_dim, d_model=d_model)

    for batch_size in [1, 4, 16, 64]:
        player_comfort = torch.randn(batch_size, 10, input_dim)
        output = model(player_comfort)
        assert output.shape == (batch_size, 10, d_model)


def test_player_comfort_network_requires_grad():
    """Test that parameters require gradients."""
    d_model = 64
    input_dim = 10

    model = PlayerComfortNetwork(input_dim=input_dim, d_model=d_model)

    for param in model.parameters():
        assert param.requires_grad

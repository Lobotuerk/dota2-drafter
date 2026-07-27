"""Player comfort network - maps raw player comfort matrices to latent preference vectors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlayerComfortNetwork(nn.Module):
    """Maps raw player comfort/performance matrices into latent preference vectors.

    Architecture: Multi-layer perceptron (MLP) that maps C features to d_model.

    Inputs:  Player comfort tensor of shape (B, 10, C)
             where B is batch size, 10 is the number of players (5 Radiant, 5 Dire),
             and C represents historical metrics.
    Outputs: Latent preference vectors of shape (B, 10, d_model).
    """

    def __init__(self, input_dim: int, d_model: int, hidden_dim: int | None = None, num_layers: int = 2) -> None:
        """Initialize the PlayerComfortNetwork.

        Args:
            input_dim: C, number of input features per player (historical metrics).
            d_model: Transformer embedding dimension.
            hidden_dim: Hidden dimension for MLP layers. Defaults to d_model.
            num_layers: Number of MLP layers (2-3 recommended).
        """
        super().__init__()
        self.d_model = d_model

        if hidden_dim is None:
            hidden_dim = d_model

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i in range(num_layers):
            if i == num_layers - 1:
                # Final layer maps to d_model
                layers.append(nn.Linear(prev_dim, d_model))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
                    layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = d_model if i == num_layers - 1 else hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, player_comfort: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PlayerComfortNetwork.

        Args:
            player_comfort: Player comfort tensor of shape (B, 10, C).

        Returns:
            Latent preference vectors of shape (B, 10, d_model).
        """
        # player_comfort: (B, 10, C) -> (B, 10, d_model)
        output = self.mlp(player_comfort)
        return output

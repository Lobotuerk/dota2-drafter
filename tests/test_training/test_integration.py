"""Integration tests for end-to-end tensor flow from frozen H_GNN to final probability."""

import pytest
import torch

from dota2drafter.models.player_network import PlayerComfortNetwork
from dota2drafter.models.match_network import HierarchicalTransformer, MatchNetwork
from dota2drafter.training.transformer_trainer import (
    TransformerTrainer,
    TrainingConfig,
    PlayerComfortDataset,
    compute_metrics,
)


def test_end_to_end_tensor_flow():
    """Test complete tensor flow: h_gnn -> joint embedding -> transformer -> probability."""
    d_model = 64
    num_heroes = 120
    batch_size = 4
    player_input_dim = 10

    # Simulate frozen H_GNN from RGCN
    h_gnn = torch.randn(num_heroes + 1, d_model)

    # Create full match network
    model = MatchNetwork(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        num_heroes=num_heroes,
        player_input_dim=player_input_dim,
        h_gnn=h_gnn,
    )

    # Create draft sequence (B, 24, 4)
    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1
        x_draft[:, t, 0] = 1.0  # all picks for simplicity
        x_draft[:, t, 1] = t % 2
        x_draft[:, t, 3] = float(t)

    # Create player comfort tensors (B, 10, C)
    player_comfort = torch.randn(batch_size, 10, player_input_dim)

    # Forward pass through the full pipeline
    logits = model(x_draft, player_comfort)  # (B,)
    proba = model.predict_proba(x_draft, player_comfort)  # (B,)

    # Verify shapes
    assert logits.shape == (batch_size,)
    assert proba.shape == (batch_size,)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_player_network_to_match_network_interface():
    """Test that PlayerComfortNetwork output correctly interfaces with MatchNetwork."""
    d_model = 64
    num_heroes = 120
    batch_size = 4
    player_input_dim = 10

    h_gnn = torch.randn(num_heroes + 1, d_model)

    # Player Network
    player_net = PlayerComfortNetwork(
        input_dim=player_input_dim,
        d_model=d_model,
    )

    # Match Network (HierarchicalTransformer)
    match_net = HierarchicalTransformer(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        num_heroes=num_heroes,
        h_gnn=h_gnn,
    )

    # Player comfort input: (B, 10, C)
    player_comfort = torch.randn(batch_size, 10, player_input_dim)

    # Player Network forward: (B, 10, C) -> (B, 10, d_model)
    player_pref_vectors = player_net(player_comfort)
    assert player_pref_vectors.shape == (batch_size, 10, d_model)

    # Draft sequence: (B, 24, 4)
    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1
        x_draft[:, t, 0] = 1.0
        x_draft[:, t, 1] = t % 2
        x_draft[:, t, 3] = float(t)

    # Match Network forward: (B, 24, d_model) + (B, 10, d_model) -> (B,)
    logits = match_net(x_draft, player_pref_vectors)
    assert logits.shape == (batch_size,)


def test_trainer_dataset():
    """Test PlayerComfortDataset yields correct shapes."""
    batch_size = 8

    x_drafts = [torch.randn(24, 4) for _ in range(batch_size)]
    y_labels = [torch.tensor([1.0 if i % 2 == 0 else 0.0]) for i in range(batch_size)]

    radiant_players = [[1001 + i * 10 + j for j in range(5)] for i in range(batch_size)]
    dire_players = [[2001 + i * 10 + j for j in range(5)] for i in range(batch_size)]

    dataset = PlayerComfortDataset(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        player_input_dim=10,
    )

    assert len(dataset) == batch_size

    # Sample a single item
    x, player_comfort, y = dataset[0]
    assert x.shape == (24, 4)
    assert player_comfort.shape == (10, 10)
    assert y.shape == (1,)


def test_trainer_small_training_step():
    """Test a single training step with mock data."""
    d_model = 64
    num_heroes = 120
    batch_size = 4
    player_input_dim = 10

    h_gnn = torch.randn(num_heroes + 1, d_model)

    model = MatchNetwork(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        num_heroes=num_heroes,
        player_input_dim=player_input_dim,
        h_gnn=h_gnn,
    )

    config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=batch_size,
        device="cpu",
    )

    trainer = TransformerTrainer(model=model, train_config=config)

    # Create mock data
    x_drafts = []
    y_labels = []
    radiant_players = []
    dire_players = []

    for i in range(batch_size):
        x = torch.zeros(24, 4)
        for t in range(24):
            x[t, 2] = (t % num_heroes) + 1
            x[t, 0] = 1.0
            x[t, 1] = t % 2
            x[t, 3] = float(t)
        x_drafts.append(x)
        y_labels.append(torch.tensor([1.0 if i < batch_size // 2 else 0.0]))
        radiant_players.append([1000 + i * 10 + j for j in range(5)])
        dire_players.append([2000 + i * 10 + j for j in range(5)])

    # Run a single training step (we won't run full epochs)
    # Just test that the model can forward and backward
    model.train()
    x_batch = torch.stack(x_drafts)
    player_comfort = torch.randn(batch_size, 10, player_input_dim)
    y_batch = torch.stack(y_labels).squeeze(-1)

    logits = model(x_batch, player_comfort)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_batch)
    loss.backward()

    assert logits.shape == (batch_size,)
    assert loss.item() > 0


def test_compute_metrics():
    """Test metric computation."""
    predictions = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

    metrics = compute_metrics(predictions, targets)

    assert "bce" in metrics
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["bce"] >= 0


def test_full_training_loop_small():
    """Test a very small training loop with mock data."""
    d_model = 32
    num_heroes = 60
    batch_size = 8
    player_input_dim = 10

    h_gnn = torch.randn(num_heroes + 1, d_model)

    model = MatchNetwork(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        num_heroes=num_heroes,
        player_input_dim=player_input_dim,
        h_gnn=h_gnn,
    )

    config = TrainingConfig(
        learning_rate=1e-2,
        num_epochs=2,
        batch_size=batch_size,
        device="cpu",
        checkpoint_dir="./checkpoints_test",
        patience=10,
    )

    trainer = TransformerTrainer(model=model, train_config=config)

    # Create mock data
    x_drafts = []
    y_labels = []
    radiant_players = []
    dire_players = []

    for i in range(batch_size):
        x = torch.zeros(24, 4)
        for t in range(24):
            x[t, 2] = (t % num_heroes) + 1
            x[t, 0] = 1.0
            x[t, 1] = t % 2
            x[t, 3] = float(t)
        x_drafts.append(x)
        y_labels.append(torch.tensor([1.0 if i < batch_size // 2 else 0.0]))
        radiant_players.append([1000 + i * 10 + j for j in range(5)])
        dire_players.append([2000 + i * 10 + j for j in range(5)])

    metrics = trainer.train(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
    )

    assert len(metrics.train_losses) > 0
    assert len(metrics.val_losses) > 0
    assert metrics.best_epoch >= 0
    assert metrics.best_val_loss < float("inf")

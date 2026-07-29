"""Unit and integration tests for the Transformer improvements.

Includes tests for MLM, prefix training, permutations, and label smoothing.
"""

import os
import shutil

import torch

from dota2drafter.models.match_network import HierarchicalTransformer, MatchNetwork
from dota2drafter.training.transformer_trainer import (
    PlayerComfortDataset,
    TrainingConfig,
    TransformerTrainer,
    apply_prefix_truncation,
    augment_draft_permutations,
)


def test_mlm_mode_forward_pass():
    """Test HierarchicalTransformer and MatchNetwork forward passes in MLM mode."""
    d_model = 64
    num_heroes = 120
    batch_size = 4
    player_input_dim = 10

    h_gnn = torch.randn(num_heroes + 1, d_model)

    # 1. Test HierarchicalTransformer
    model_ht = HierarchicalTransformer(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        num_heroes=num_heroes,
        h_gnn=h_gnn,
    )

    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1
        x_draft[:, t, 0] = 1.0
        x_draft[:, t, 1] = t % 2
        x_draft[:, t, 3] = float(t)

    player_pref_vectors = torch.randn(batch_size, 10, d_model)

    # MLM mode forward: should return (B, 24, num_heroes + 1)
    mlm_logits_ht = model_ht(x_draft, player_pref_vectors, mlm_mode=True)
    assert mlm_logits_ht.shape == (batch_size, 24, num_heroes + 1)

    # Win prediction mode forward: should return (B,)
    win_logits_ht = model_ht(x_draft, player_pref_vectors, mlm_mode=False)
    assert win_logits_ht.shape == (batch_size,)

    # 2. Test MatchNetwork
    model_mn = MatchNetwork(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        num_heroes=num_heroes,
        player_input_dim=player_input_dim,
        h_gnn=h_gnn,
    )

    player_comfort = torch.randn(batch_size, 10, player_input_dim)

    # MLM mode forward
    mlm_logits_mn = model_mn(x_draft, player_comfort, mlm_mode=True)
    assert mlm_logits_mn.shape == (batch_size, 24, num_heroes + 1)

    # Win prediction mode forward
    win_logits_mn = model_mn(x_draft, player_comfort, mlm_mode=False)
    assert win_logits_mn.shape == (batch_size,)


def test_apply_prefix_truncation():
    """Test the apply_prefix_truncation helper function zeroes out indices correctly."""
    num_heroes = 120
    player_input_dim = 10

    x_draft = torch.zeros(24, 4)
    for t in range(24):
        x_draft[t, 0] = 1.0  # is_pick
        x_draft[t, 1] = float(t % 2)  # team
        x_draft[t, 2] = float((t % num_heroes) + 1)  # hero index
        x_draft[t, 3] = float(t)  # step

    y_label = torch.tensor([1.0])
    radiant_players = [101, 102, 103, 104, 105]
    dire_players = [201, 202, 203, 204, 205]
    player_comfort_map = {
        uid: torch.ones(player_input_dim) for uid in radiant_players + dire_players
    }

    samples = apply_prefix_truncation(
        x_draft=x_draft,
        y_label=y_label,
        radiant_players=radiant_players,
        dire_players=dire_players,
        player_comfort_map=player_comfort_map,
        player_input_dim=player_input_dim,
    )

    # Should return exactly 4 samples for truncation points [6, 12, 18, 24]
    assert len(samples) == 4

    truncation_points = [6, 12, 18, 24]
    for i, t in enumerate(truncation_points):
        x_truncated, player_comfort, y = samples[i]
        assert x_truncated.shape == (24, 4)
        assert player_comfort.shape == (10, player_input_dim)
        assert torch.equal(y, y_label)

        # Verify zeroing out beyond truncation point
        if t < 24:
            # Everything from index t to 23 should be 0.0
            assert torch.all(x_truncated[t:, :] == 0.0)
            # Before t should remain intact
            assert torch.all(x_truncated[:t, :] == x_draft[:t, :])
        else:
            # Complete draft: everything should remain intact
            assert torch.equal(x_truncated, x_draft)


def test_augment_draft_permutations():
    """Test that augment_draft_permutations generates valid swaps within the same phase."""
    num_heroes = 120
    player_input_dim = 10

    x_draft = torch.zeros(24, 4)
    for t in range(24):
        x_draft[t, 0] = 1.0
        x_draft[t, 1] = float(t % 2)
        x_draft[t, 2] = float((t % num_heroes) + 1)
        x_draft[t, 3] = float(t)

    y_label = torch.tensor([0.0])
    radiant_players = [101, 102, 103, 104, 105]
    dire_players = [201, 202, 203, 204, 205]

    samples = augment_draft_permutations(
        x_draft=x_draft,
        y_label=y_label,
        radiant_players=radiant_players,
        dire_players=dire_players,
        player_input_dim=player_input_dim,
    )

    # Should return 1 to 2 random valid permutations
    assert len(samples) in [1, 2]

    for x_permuted, player_comfort, y in samples:
        assert x_permuted.shape == (24, 4)
        assert player_comfort.shape == (10, player_input_dim)
        assert torch.equal(y, y_label)


def test_player_comfort_dataset_augmentation():
    """Test PlayerComfortDataset with and without augmentation flag."""
    num_samples = 3
    num_heroes = 120
    player_input_dim = 10

    x_drafts = []
    y_labels = []
    radiant_players = []
    dire_players = []

    for i in range(num_samples):
        x = torch.zeros(24, 4)
        for t in range(24):
            x[t, 2] = float((t % num_heroes) + 1)
            x[t, 0] = 1.0
            x[t, 1] = float(t % 2)
            x[t, 3] = float(t)
        x_drafts.append(x)
        y_labels.append(torch.tensor([1.0 if i % 2 == 0 else 0.0]))
        radiant_players.append([1000 + i * 10 + j for j in range(5)])
        dire_players.append([2000 + i * 10 + j for j in range(5)])

    # 1. Without Augmentation
    dataset_no_aug = PlayerComfortDataset(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        player_input_dim=player_input_dim,
        augment=False,
    )
    assert len(dataset_no_aug) == num_samples
    x, comfort, y = dataset_no_aug[0]
    assert x.shape == (24, 4)
    assert comfort.shape == (10, player_input_dim)

    # 2. With Augmentation (multiplies size by 10)
    dataset_with_aug = PlayerComfortDataset(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        player_input_dim=player_input_dim,
        augment=True,
    )
    assert len(dataset_with_aug) == num_samples * 10

    # Test retrieving various indices
    for i in range(len(dataset_with_aug)):
        x_item, comfort_item, y_item = dataset_with_aug[i]
        assert x_item.shape == (24, 4)
        assert comfort_item.shape == (10, player_input_dim)
        assert y_item.dim() == 1


def test_mlm_pre_training_loop():
    """Test MLM pre-training loop executes successfully."""
    d_model = 32
    num_heroes = 60
    batch_size = 4
    player_input_dim = 10
    checkpoint_dir = "./checkpoints_mlm_test"

    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

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
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=batch_size,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        patience=5,
    )

    trainer = TransformerTrainer(model=model, train_config=config)

    # Create mock data
    x_drafts = []
    y_labels = []
    radiant_players = []
    dire_players = []

    for i in range(8):  # 8 samples
        x = torch.zeros(24, 4)
        for t in range(24):
            x[t, 2] = float((t % num_heroes) + 1)
            x[t, 0] = 1.0
            x[t, 1] = float(t % 2)
            x[t, 3] = float(t)
        x_drafts.append(x)
        y_labels.append(torch.tensor([1.0 if i < 4 else 0.0]))
        radiant_players.append([1000 + i * 10 + j for j in range(5)])
        dire_players.append([2000 + i * 10 + j for j in range(5)])

    # Execute MLM pre-training (1 epoch)
    mlm_metrics = trainer.mlm_train(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        num_epochs=1,
        mlm_probability=0.15,
    )

    assert len(mlm_metrics.train_losses) == 1
    assert mlm_metrics.train_losses[0] > 0
    assert os.path.exists(os.path.join(checkpoint_dir, "mlm_pretrained.pt"))

    # Cleanup test checkpoint dir
    shutil.rmtree(checkpoint_dir, ignore_errors=True)


def test_label_smoothing_training():
    """Test regular training with label smoothing executes successfully."""
    d_model = 32
    num_heroes = 60
    batch_size = 4
    player_input_dim = 10
    checkpoint_dir = "./checkpoints_smooth_test"

    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

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
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=batch_size,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        patience=5,
        label_smoothing_eps=0.15,
    )

    trainer = TransformerTrainer(model=model, train_config=config)

    # Create mock data
    x_drafts = []
    y_labels = []
    radiant_players = []
    dire_players = []

    for i in range(8):  # 8 samples
        x = torch.zeros(24, 4)
        for t in range(24):
            x[t, 2] = float((t % num_heroes) + 1)
            x[t, 0] = 1.0
            x[t, 1] = float(t % 2)
            x[t, 3] = float(t)
        x_drafts.append(x)
        y_labels.append(torch.tensor([1.0 if i < 4 else 0.0]))
        radiant_players.append([1000 + i * 10 + j for j in range(5)])
        dire_players.append([2000 + i * 10 + j for j in range(5)])

    # Execute regular training
    metrics = trainer.train(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
    )

    assert len(metrics.train_losses) == 1
    assert metrics.train_losses[0] > 0

    # Cleanup test checkpoint dir
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

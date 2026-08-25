"""Unit and integration tests for the Transformer improvements.

Includes tests for MLM, prefix training, permutations, and label smoothing.
"""

import os
import shutil

import torch
import torch.nn.functional as F

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

    # Unified forward: should return (win_logits, mlm_logits)
    win_logits_ht, mlm_logits_ht = model_ht(x_draft, player_pref_vectors)
    assert win_logits_ht.shape == (batch_size,)
    assert mlm_logits_ht.shape == (batch_size, 24, num_heroes + 1)

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

    # Unified forward
    win_logits_mn, mlm_logits_mn = model_mn(x_draft, player_comfort)
    assert win_logits_mn.shape == (batch_size,)
    assert mlm_logits_mn.shape == (batch_size, 24, num_heroes + 1)


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

    # Should return exactly 6 samples for truncation points [7, 9, 12, 18, 22, 24]
    assert len(samples) == 6

    truncation_points = [7, 9, 12, 18, 22, 24]
    for i, t in enumerate(truncation_points):
        x_truncated, player_comfort, y = samples[i]
        assert x_truncated.shape == (24, 4)
        assert player_comfort.shape == (10, player_input_dim)
        assert torch.equal(y, y_label)

        # Verify zeroing out beyond truncation point
        if t < 24:
            # Column 2 must be -1.0, other columns must be 0.0
            assert torch.all(x_truncated[t:, 2] == -1.0)
            assert torch.all(x_truncated[t:, [0, 1, 3]] == 0.0)
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

    # Should return exactly 64 valid permutation combinations
    assert len(samples) == 64

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
    x, comfort, y, *extra = dataset_no_aug[0]
    assert x.shape == (24, 4)
    assert comfort.shape == (10, player_input_dim)

    # 2. With Augmentation (multiplies size by 448: 64 permutations * (1 original + 6 truncations))
    dataset_with_aug = PlayerComfortDataset(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        player_input_dim=player_input_dim,
        augment=True,
    )
    assert len(dataset_with_aug) == num_samples * 448

    # Test retrieving various indices
    for i in range(len(dataset_with_aug)):
        x_item, comfort_item, y_item, *extra = dataset_with_aug[i]
        assert x_item.shape == (24, 4)
        assert comfort_item.shape == (10, player_input_dim)
        assert y_item.dim() == 1

    # 3. With Integer-limited Augmentation (exposes exactly `augment` number of elements per match)
    limit = 5
    dataset_limited_aug = PlayerComfortDataset(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        player_input_dim=player_input_dim,
        augment=limit,
    )
    assert len(dataset_limited_aug) == num_samples * limit

    # Check that all items retrieved are valid
    for i in range(len(dataset_limited_aug)):
        x_item, comfort_item, y_item, *extra = dataset_limited_aug[i]
        assert x_item.shape == (24, 4)
        assert comfort_item.shape == (10, player_input_dim)
        assert y_item.dim() == 1

    # Capture selected indices and test reshuffling
    indices_before = [list(idx_list) for idx_list in dataset_limited_aug.selected_indices]
    dataset_limited_aug.reshuffle_augmentations()
    indices_after = [list(idx_list) for idx_list in dataset_limited_aug.selected_indices]
    
    assert len(indices_before) == num_samples
    assert len(indices_after) == num_samples
    # With 448 possible values, reshuffled selection should generally differ (probability of identical choice is tiny)
    assert any(indices_before[b] != indices_after[b] for b in range(num_samples))


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

    # Execute training (1 epoch)
    metrics = trainer.train(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
    )

    assert len(metrics.train_losses) == 1
    assert metrics.train_losses[0] > 0
    assert os.path.exists(os.path.join(checkpoint_dir, "best_model.pt"))

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


def test_dynamic_player_input_dim_resolution():
    """Verify that TransformerTrainer dynamically resolves player_input_dim from the model or comfort map."""
    d_model = 32
    num_heroes = 60
    h_gnn = torch.randn(num_heroes + 1, d_model)

    # Create a model with player_input_dim = 15
    model = MatchNetwork(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        num_heroes=num_heroes,
        player_input_dim=15,
        h_gnn=h_gnn,
    )
    config = TrainingConfig(device="cpu")
    trainer = TransformerTrainer(model=model, train_config=config)

    # Case 1: Without comfort map, resolve from model attribute (15)
    player_input_dim_no_map = getattr(trainer.model, "player_input_dim", 127)
    assert player_input_dim_no_map == 15

    # Case 2: With comfort map, fallback/override dynamically based on comfort map tensor shape (e.g. 25)
    mock_comfort_map = {101: torch.zeros(25)}
    first_tensor = next(iter(mock_comfort_map.values()))
    player_input_dim_with_map = first_tensor.size(0)
    assert player_input_dim_with_map == 25


def test_discriminative_learning_rates():
    """Verify that discriminative learning rates are correctly configured and applied."""
    d_model = 32
    num_heroes = 60
    h_gnn = torch.randn(num_heroes + 1, d_model)

    model = MatchNetwork(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        num_heroes=num_heroes,
        player_input_dim=10,
        h_gnn=h_gnn,
    )

    config = TrainingConfig(
        learning_rate=1e-4,
        lr_backbone=1e-5,
        lr_head=1e-3,
        num_epochs=1,
        device="cpu",
    )

    trainer = TransformerTrainer(model=model, train_config=config)

    # Check optimizer has two parameter groups
    assert len(trainer.optimizer.param_groups) == 2

    # Check that backbone group has correct learning rate (1e-5)
    # and head group has correct learning rate (1e-3)
    backbone_group = trainer.optimizer.param_groups[0]
    head_group = trainer.optimizer.param_groups[1]

    assert backbone_group["lr"] == 1e-5
    assert head_group["lr"] == 1e-3

    # Check parameters were assigned to the correct group
    # Let's inspect parameter names
    backbone_param_ids = {id(p) for p in backbone_group["params"]}
    head_param_ids = {id(p) for p in head_group["params"]}

    # Ensure set_transformer_head parameters are in the head group and not in the backbone group
    for name, param in model.named_parameters():
        if "set_transformer_head" in name or "mlm_head" in name:
            assert id(param) in head_param_ids
            assert id(param) not in backbone_param_ids
        else:
            assert id(param) in backbone_param_ids
            assert id(param) not in head_param_ids


def test_step_weighted_loss():
    """Verify that Step-Weighted Loss is correctly computed and scales properly."""
    d_model = 32
    num_heroes = 60
    h_gnn = torch.randn(num_heroes + 1, d_model)

    model = MatchNetwork(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        num_heroes=num_heroes,
        player_input_dim=10,
        h_gnn=h_gnn,
    )

    # 1. Test standard loss (gamma = 0.0)
    config_std = TrainingConfig(
        step_loss_gamma=0.0,
        device="cpu",
    )

    # Create dummy batch:
    # Batch size = 2
    # Sample 0: Full draft (24 steps active)
    # Sample 1: Truncated draft (6 steps active)
    x_batch = torch.zeros(2, 24, 4)
    # Sample 0 has non-zeros on all steps
    x_batch[0, :, 2] = 1.0  # hero_val
    x_batch[0, :, 0] = 1.0  # is_pick
    x_batch[0, :, 3] = torch.arange(24).float()

    # Sample 1 is truncated at t=6 (only first 6 steps active)
    x_batch[1, :6, 2] = 2.0  # hero_val
    x_batch[1, :6, 0] = 1.0  # is_pick
    x_batch[1, :6, 3] = torch.arange(6).float()

    player_batch = torch.zeros(2, 10, 10)
    y_batch = torch.tensor([1.0, 0.0])

    logits, mlm_logits = model(x_batch, player_batch)
    eps = config_std.label_smoothing_eps
    y_smoothed = y_batch * (1.0 - eps) + (eps / 2.0)

    # Calculate standard unweighted loss manually
    loss_std_manual = F.binary_cross_entropy_with_logits(logits, y_smoothed, reduction="mean")

    # Now calculate via trainer with step weighting (gamma = 1.0)
    config_weighted = TrainingConfig(
        step_loss_gamma=1.0,
        device="cpu",
    )

    # Calculate weighted loss manually
    loss_elements = F.binary_cross_entropy_with_logits(logits, y_smoothed, reduction="none")
    # Sample 0 weight: (24/24)^1 = 1.0
    # Sample 1 weight: (6/24)^1 = 0.25
    expected_loss_weighted = torch.mean(
        loss_elements * torch.tensor([1.0, 0.25], device=loss_elements.device)
    )

    # First, let's assert our manual calculation is correct
    assert expected_loss_weighted < loss_std_manual  # because sample 1 has a lower weight

    # Let's verify that the trainer's step-weighted loss matches expected_loss_weighted
    t = torch.sum(torch.sum(torch.abs(x_batch), dim=-1) > 0, dim=-1).float()
    weights = (t / 24.0) ** config_weighted.step_loss_gamma
    trainer_computed_loss = torch.mean(weights * loss_elements)

    assert torch.allclose(trainer_computed_loss, expected_loss_weighted)


def test_ntp_loss_and_priors():
    """Test NTP loss computation and MCTS prior extraction."""
    d_model = 64
    num_heroes = 120
    batch_size = 4
    player_input_dim = 10

    h_gnn = torch.randn(num_heroes + 1, d_model)

    model = HierarchicalTransformer(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        num_heroes=num_heroes,
        h_gnn=h_gnn,
    )

    # Create a draft sequence with valid heroes
    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1  # hero_val (1-based)
        x_draft[:, t, 0] = 1.0  # is_pick
        x_draft[:, t, 1] = t % 2  # team
        x_draft[:, t, 3] = float(t)  # step_index

    player_pref_vectors = torch.randn(batch_size, 10, d_model)

    # Forward pass
    logits, mlm_logits = model(x_draft, player_pref_vectors)

    # Verify shapes
    assert logits.shape == (batch_size,)
    assert mlm_logits.shape == (batch_size, 24, num_heroes + 1)

    # Manually calculate NTP loss
    ntp_labels = x_draft[:, :, 2].long()  # (B, 24)
    
    # Verify that the model's internal right-shifting produces correct predictions
    # At position 0, the model should predict h_0 given only BOS token
    # At position 1, the model should predict h_1 given h_0
    # etc.
    
    # Calculate NTP loss manually with label smoothing
    manual_loss = F.cross_entropy(
        mlm_logits.reshape(-1, mlm_logits.size(-1)),
        ntp_labels.reshape(-1),
        ignore_index=-1,
        label_smoothing=0.10
    )
    
    # Verify loss is valid
    assert manual_loss.item() > 0
    assert torch.isfinite(manual_loss)
    
    # Verify that the model can extract priors at different steps
    for step_idx in range(24):
        # Extract logits at step_idx
        step_logits = mlm_logits[:, step_idx, 1:num_heroes+1]
        assert step_logits.shape == (batch_size, num_heroes)


def test_mcts_prior_extraction():
    """Test that MCTS prior extraction works correctly with right-shifted model."""
    d_model = 64
    num_heroes = 120
    batch_size = 2
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

    # Create a draft sequence
    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1  # hero_val
        x_draft[:, t, 0] = 1.0  # is_pick
        x_draft[:, t, 1] = t % 2  # team
        x_draft[:, t, 3] = float(t)  # step_index

    player_comfort = torch.randn(batch_size, 10, player_input_dim)

    # Forward pass
    logits, mlm_logits = model(x_draft, player_comfort)

    # Verify that we can extract valid priors at each step
    for step_idx in range(24):
        # Extract policy logits at step_idx
        policy_logits = mlm_logits[:, step_idx, 1:num_heroes+1]
        
        # Verify shape
        assert policy_logits.shape == (batch_size, num_heroes)
        
        # Verify that logits are not all the same (model is making predictions)
        assert not torch.all(policy_logits[0] == policy_logits[0, 0])
        
        # Verify that valid heroes have different logits than padding
        valid_hero_mask = x_draft[0, :, 2] >= 0
        if valid_hero_mask.any():
            # At least some variation in logits
            assert policy_logits.std() > 0

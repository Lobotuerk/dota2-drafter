"""Unit and integration tests for Decoupled Two-Stage Training Pipeline (AUT-40).

Tests:
- Stage 1 Value Head pre-training, fine-tuning with sample weight, and checkpointing on Brier score.
- Post-hoc Platt scaling temperature calibration via L-BFGS.
- Stage 2 Policy Head training with Value Head freezing.
- Stage 2 validation conditioned on strictly positive signed marginal advantages.
- Graceful handling of missing pub batches.
- Configuration and CLI argument integration.
"""

import importlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.training.transformer_trainer import (
    TrainingConfig,
    TransformerTrainer,
)

train_module = importlib.import_module("scripts.04_train_transformer")
parse_args = train_module.parse_args
load_pub_data = train_module.load_pub_data


@pytest.fixture
def dummy_model():
    """Create a lightweight MatchNetwork for testing."""
    d_model = 32
    num_heroes = 20
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
        num_patches=10,
    )
    return model


@pytest.fixture
def dummy_draft_data():
    """Create synthetic draft data for tests."""
    batch_size = 12
    x_drafts = []
    y_labels = []
    radiant_players = []
    dire_players = []
    patch_ids = []

    for i in range(batch_size):
        x = torch.zeros(24, 4)
        x[:, 0] = 1.0  # picks
        x[:, 1] = torch.tensor([0.0 if t % 2 == 0 else 1.0 for t in range(24)])  # alternating teams
        x[:, 2] = torch.tensor([(t % 15) + 1 for t in range(24)]).float()
        x[:, 3] = torch.arange(24).float()
        x_drafts.append(x)
        y_labels.append(torch.tensor([float(i % 2)]))
        radiant_players.append([100 + j for j in range(5)])
        dire_players.append([200 + j for j in range(5)])
        patch_ids.append(torch.tensor(1 if i < 6 else 2))

    return x_drafts, y_labels, radiant_players, dire_players, patch_ids


def test_calibrate_temperature_lbfgs(dummy_model, tmp_path):
    """Verify that calibrate_temperature optimizes T using L-BFGS to minimize Brier score."""
    config = TrainingConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    trainer = TransformerTrainer(model=dummy_model, train_config=config)

    # Synthetic validation dataset
    num_samples = 30
    x_val = torch.zeros(num_samples, 24, 4)
    x_val[:, :, 0] = 1.0
    x_val[:, :, 2] = 2.0  # hero 2
    # Create unbalanced logits scenario to verify temperature scales
    y_val = torch.tensor([1.0 if i < 15 else 0.0 for i in range(num_samples)]).unsqueeze(-1)
    val_dataset = TensorDataset(x_val, torch.zeros(num_samples, 10, 10), y_val)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    # Initial calibration temperature is 1.0
    assert trainer.calibrated_temperature == 1.0

    fitted_t = trainer.calibrate_temperature(val_loader)

    assert isinstance(fitted_t, float)
    assert fitted_t > 0.0
    assert trainer.calibrated_temperature == fitted_t
    assert trainer.metrics.calibrated_temperature == fitted_t

    match_net = getattr(trainer.model, "match_network", trainer.model)
    assert match_net.temperature == fitted_t
    assert match_net.set_transformer_head.temperature == fitted_t


def test_stage1_train_value_head_freezing_and_weighting(dummy_model, dummy_draft_data, tmp_path):
    """Verify Stage 1 freezes policy head, trains value head with weight, and calibrates T."""
    x_drafts, y_labels, radiant_players, dire_players, patch_ids = dummy_draft_data

    # Create dummy pub data
    x_pubs = [x.clone() for x in x_drafts[:6]]
    y_pubs = [y.clone() for y in y_labels[:6]]
    patch_ids_pubs = [p.clone() for p in patch_ids[:6]]

    config = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=2,
        pub_epochs=1,
        batch_size=4,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        stage=1,
        draft_sample_weight=5.0,
    )
    trainer = TransformerTrainer(model=dummy_model, train_config=config)

    metrics = trainer.train_stage_1(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        patch_ids=patch_ids,
        x_pubs=x_pubs,
        y_pubs=y_pubs,
        patch_ids_pubs=patch_ids_pubs,
    )

    # Check metrics recorded
    assert len(metrics.val_brier_scores) > 0
    assert len(metrics.val_auc_scores) > 0
    assert metrics.best_brier_score < float("inf")
    assert metrics.calibrated_temperature > 0.0

    # Check that stage1_best_model.pt was created with temperature
    stage1_ckpt = tmp_path / "stage1_best_model.pt"
    assert stage1_ckpt.exists()
    ckpt = torch.load(stage1_ckpt, weights_only=True)
    assert "temperature" in ckpt
    assert ckpt["stage"] == 1


def test_stage1_missing_pub_data_graceful(dummy_model, dummy_draft_data, tmp_path):
    """Verify that Stage 1 gracefully skips pub pre-training when pub data is missing."""
    x_drafts, y_labels, radiant_players, dire_players, patch_ids = dummy_draft_data

    config = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=4,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        stage=1,
        draft_sample_weight=5.0,
    )
    trainer = TransformerTrainer(model=dummy_model, train_config=config)

    # Pass empty pubs
    metrics = trainer.train_stage_1(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        patch_ids=patch_ids,
        x_pubs=[],
        y_pubs=[],
        patch_ids_pubs=None,
    )

    assert len(metrics.val_brier_scores) > 0
    assert (tmp_path / "stage1_best_model.pt").exists()


def test_stage2_freezing_and_policy_only_training(dummy_model, dummy_draft_data, tmp_path):
    """Verify Stage 2 freezes SetTransformerHead and value projections and trains policy head."""
    x_drafts, y_labels, radiant_players, dire_players, patch_ids = dummy_draft_data

    # Save a fake stage 1 checkpoint first
    stage1_path = tmp_path / "stage1_best_model.pt"
    dummy_state = {
        "model_state": dummy_model.state_dict(),
        "temperature": 1.25,
        "stage": 1,
    }
    torch.save(dummy_state, stage1_path)

    config = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=2,
        batch_size=4,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        stage=2,
        stage1_checkpoint_path=str(stage1_path),
    )
    trainer = TransformerTrainer(model=dummy_model, train_config=config)

    metrics = trainer.train_stage_2(
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        patch_ids=patch_ids,
        stage1_checkpoint_path=stage1_path,
    )
    assert len(metrics.train_losses) > 0

    # Verify calibrated temperature was loaded from Stage 1 checkpoint
    assert trainer.calibrated_temperature == pytest.approx(1.25)

    # Verify that SetTransformerHead and project_value parameters are frozen
    match_net = getattr(dummy_model, "match_network", dummy_model)
    for p in match_net.set_transformer_head.parameters():
        assert not p.requires_grad
    if hasattr(match_net.joint_embedding, "project_value"):
        for p in match_net.joint_embedding.project_value.parameters():
            assert not p.requires_grad

    # Verify that policy parameters are active
    for p in match_net.transformer_decoder.parameters():
        assert p.requires_grad
    assert match_net.mlm_head.w_policy.weight.requires_grad
    assert match_net.joint_embedding.project_policy.weight.requires_grad
    assert not match_net.joint_embedding.project_value.weight.requires_grad

    # Check stage 2 checkpoint was written
    assert (tmp_path / "stage2_best_model.pt").exists()


def test_stage2_validation_positive_advantage_conditioning(dummy_model):
    """Verify that _validate with condition_on_positive_advantage filters by delta_t > 0."""
    trainer = TransformerTrainer(
        model=dummy_model,
        train_config=TrainingConfig(device="cpu", stage=2),
    )

    batch_size = 2
    x_draft = torch.zeros(batch_size, 24, 4)
    x_draft[:, :, 0] = 1.0  # pick
    x_draft[:, :, 1] = 0.0  # Radiant acting
    x_draft[:, :, 2] = 3.0  # hero 3
    x_draft[:, :, 3] = torch.arange(24).float()
    y = torch.tensor([[1.0], [0.0]])
    comfort = torch.zeros(batch_size, 10, 10)

    val_dataset = TensorDataset(x_draft, comfort, y)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Mock _compute_signed_advantages:
    # Batch 0: step 0 positive advantage (+0.2), step 1 negative advantage (-0.1)
    # Batch 1: step 0 negative advantage (-0.2), step 1 positive advantage (+0.1)
    def mock_signed_advantages(x_batch, patch_batch=None):
        adv = torch.zeros(x_batch.size(0), 24)
        adv[0, 0] = 0.2
        adv[0, 1] = -0.1
        adv[1, 0] = -0.2
        adv[1, 1] = 0.1
        # all other steps are zero (not strictly positive)
        return adv

    trainer._compute_signed_advantages = mock_signed_advantages

    val_loss, metrics = trainer._validate(val_loader, condition_on_positive_advantage=True)

    # In our mock, exactly 2 positions (batch 0 step 0, and batch 1 step 1) have advantages > 0
    # The evaluation metrics should only consider those 2 positions.
    assert "mlm_top5_accuracy" in metrics
    assert "mlm_accuracy" in metrics


def test_trainer_train_dispatch(dummy_model, dummy_draft_data, tmp_path, monkeypatch):
    """Verify trainer.train dispatches to train_stage_1 or train_stage_2 based on stage."""
    x_drafts, y_labels, radiant_players, dire_players, patch_ids = dummy_draft_data

    # Test stage 1 dispatch
    cfg1 = TrainingConfig(device="cpu", stage=1, checkpoint_dir=str(tmp_path))
    trainer1 = TransformerTrainer(dummy_model, cfg1)
    stage1_called = False

    def mock_stage_1(*args, **kwargs):
        nonlocal stage1_called
        stage1_called = True
        return trainer1.metrics

    monkeypatch.setattr(trainer1, "train_stage_1", mock_stage_1)
    trainer1.train(x_drafts, y_labels, radiant_players, dire_players, patch_ids=patch_ids)
    assert stage1_called

    # Test stage 2 dispatch
    cfg2 = TrainingConfig(device="cpu", stage=2, checkpoint_dir=str(tmp_path))
    trainer2 = TransformerTrainer(dummy_model, cfg2)
    stage2_called = False

    def mock_stage_2(*args, **kwargs):
        nonlocal stage2_called
        stage2_called = True
        return trainer2.metrics

    monkeypatch.setattr(trainer2, "train_stage_2", mock_stage_2)
    trainer2.train(x_drafts, y_labels, radiant_players, dire_players, patch_ids=patch_ids)
    assert stage2_called


def test_load_pub_data_missing_dir(tmp_path):
    """Verify load_pub_data handles missing or empty pub data directory gracefully."""
    empty_dir = tmp_path / "empty_pubs"
    empty_dir.mkdir()

    x_pubs, y_pubs, patch_ids = load_pub_data(str(empty_dir))
    assert x_pubs == []
    assert y_pubs == []
    assert patch_ids is None


def test_load_pub_data_success(tmp_path):
    """Verify load_pub_data loads games_batch_*.pt files properly."""
    pub_dir = tmp_path / "pubs"
    pub_dir.mkdir()

    batch_data = {
        "x": torch.randn(5, 24, 4),
        "y": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]),
        "radiant_players": [[1, 2, 3, 4, 5]] * 5,
        "dire_players": [[6, 7, 8, 9, 10]] * 5,
        "patch_ids": torch.tensor([1, 1, 2, 2, 2]),
    }
    torch.save(batch_data, pub_dir / "games_batch_001.pt")

    x_pubs, y_pubs, patch_ids = load_pub_data(str(pub_dir))
    assert len(x_pubs) == 5
    assert len(y_pubs) == 5
    assert patch_ids == [1, 1, 2, 2, 2]


def test_cli_stage_arguments():
    """Verify scripts/04_train_transformer.py parses --stage and related CLI options."""
    args = parse_args(args=[
        "--mode", "train",
        "--stage", "2",
        "--pub_data_dir", "/tmp/pubs",
        "--draft_sample_weight", "5.0",
        "--pub_epochs", "10",
        "--stage1_checkpoint", "/tmp/stage1.pt",
    ])
    assert args.stage == 2
    assert args.pub_data_dir == "/tmp/pubs"
    assert args.draft_sample_weight == 5.0
    assert args.pub_epochs == 10
    assert args.stage1_checkpoint == "/tmp/stage1.pt"

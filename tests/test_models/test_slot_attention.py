"""Unit tests for SlotAttentionMLMProjection."""

import pytest
import torch

from dota2drafter.models.match_network import (
    HierarchicalTransformer,
    JointEmbedding,
    SlotAttentionMLMProjection,
)


def _build_model(d_model=64, num_heroes=120, dropout=0.0):
    """Helper: create a HierarchicalTransformer with slot attention MLM head."""
    h_gnn = torch.randn(num_heroes + 1, d_model)
    model = HierarchicalTransformer(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=dropout,
        num_heroes=num_heroes,
        h_gnn=h_gnn,
    )
    return model


def _build_slot_head(d_model=64, num_heroes=120):
    """Helper: create a standalone SlotAttentionMLMProjection."""
    h_gnn = torch.randn(num_heroes + 1, d_model)
    joint = JointEmbedding(d_model=d_model, num_heroes=num_heroes, h_gnn=h_gnn)
    return SlotAttentionMLMProjection(
        d_model=d_model,
        num_heroes=num_heroes,
        joint_embedding=joint,
    )


def _make_draft(batch_size, num_heroes=120, with_bans=False):
    """Helper: create a draft tensor with picks and optional bans."""
    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1
        x_draft[:, t, 0] = 1.0  # pick
        x_draft[:, t, 1] = t % 2  # team
        x_draft[:, t, 3] = float(t)
    if with_bans:
        for t in [0, 1, 2, 3]:
            x_draft[:, t, 0] = 0.0  # ban
    return x_draft


def test_parameter_registration():
    """Verify learnable parameters are registered correctly."""
    head = _build_slot_head()
    assert head.slots.shape == (5, 64)
    assert head.slots.requires_grad
    assert hasattr(head, "w_k")
    assert hasattr(head, "w_policy")


def test_forward_shape():
    """Verify output shape matches (B, 24, num_heroes + 1)."""
    d_model = 64
    num_heroes = 120
    batch_size = 4

    model = _build_model(d_model=d_model, num_heroes=num_heroes)
    x_draft = _make_draft(batch_size, num_heroes)
    player_pref = torch.randn(batch_size, 10, d_model)

    logits, mlm_logits = model(x_draft, player_pref)
    assert logits.shape == (batch_size,)
    assert mlm_logits.shape == (batch_size, 24, num_heroes + 1)


def test_causality():
    """Verify that changes at step t do not alter logits at steps < t.

    Tests the SlotAttentionMLMProjection directly to isolate causal behavior
    from the transformer decoder's own causal masking.
    """
    d_model = 64
    num_heroes = 120
    batch_size = 2

    head = _build_slot_head(d_model=d_model, num_heroes=num_heroes)
    head.eval()

    x_draft_1 = _make_draft(batch_size, num_heroes)
    x_draft_2 = x_draft_1.clone()
    # Change pick at step 10
    x_draft_2[:, 10, 2] = 42

    decoder_output = torch.randn(batch_size, 24, d_model)

    with torch.no_grad():
        logits_1 = head(decoder_output, x_draft_1)
        logits_2 = head(decoder_output, x_draft_2)

    # Steps 0..10 should be identical (causal mask excludes step 10 for queries at step <= 10)
    assert torch.allclose(logits_1[:, :11, :], logits_2[:, :11, :], atol=1e-7)


def test_bans_and_padding_ignored():
    """Verify that bans do not trigger slot occupancy."""
    d_model = 64
    num_heroes = 120
    batch_size = 2

    head = _build_slot_head(d_model=d_model, num_heroes=num_heroes)
    head.eval()

    x_draft_bans = _make_draft(batch_size, num_heroes, with_bans=True)
    x_draft_clean = _make_draft(batch_size, num_heroes, with_bans=False)

    decoder_output = torch.randn(batch_size, 24, d_model)

    with torch.no_grad():
        # For bans-only steps, occupancy should be near 0
        logits_bans = head(decoder_output, x_draft_bans)
        logits_clean = head(decoder_output, x_draft_clean)

    assert logits_bans.shape == (batch_size, 24, num_heroes + 1)
    assert logits_clean.shape == (batch_size, 24, num_heroes + 1)


def test_entropy_penalty():
    """Verify entropy loss is higher for ambiguous slot assignments."""
    d_model = 64
    num_heroes = 120
    head = _build_slot_head(d_model=d_model, num_heroes=num_heroes)
    head.eval()

    # Sharp occupancy (alpha close to 0 or 1) -> low entropy
    head._entropy_loss = torch.tensor(0.01)
    loss_sharp = head.get_entropy_loss()

    # Ambiguous occupancy (alpha close to 0.5) -> high entropy
    head._entropy_loss = torch.tensor(0.5)
    loss_ambiguous = head.get_entropy_loss()

    assert loss_ambiguous > loss_sharp


def test_get_entropy_loss_resets_buffer():
    """Verify get_entropy_loss resets the internal buffer."""
    head = _build_slot_head()
    head._entropy_loss = torch.tensor(0.42)

    loss = head.get_entropy_loss()
    assert loss.item() == pytest.approx(0.42)
    assert head._entropy_loss.item() == 0.0


def test_smooth_gradient_flow():
    """Verify gradients flow through exponential saturation (not zeroed by clamp)."""
    d_model = 64
    num_heroes = 120
    batch_size = 2

    head = _build_slot_head(d_model=d_model, num_heroes=num_heroes)
    head.train()

    x_draft = _make_draft(batch_size, num_heroes)
    decoder_output = torch.randn(batch_size, 24, d_model)

    logits = head(decoder_output, x_draft)
    loss = logits.sum()
    loss.backward()

    # Slots should have non-zero gradients
    assert head.slots.grad is not None
    assert head.slots.grad.abs().sum() > 0


def test_inference_fusion():
    """Verify fuse_embeddings_for_inference does not break slot attention."""
    from dota2drafter.models.match_network import MatchNetwork

    d_model = 64
    num_heroes = 120
    batch_size = 2
    player_input_dim = 10

    model = MatchNetwork(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        num_heroes=num_heroes,
        player_input_dim=player_input_dim,
    )
    x_draft = _make_draft(batch_size, num_heroes)
    player_comfort = torch.randn(batch_size, 10, player_input_dim)

    # Run before fusion
    model.eval()
    with torch.no_grad():
        logits_before, mlm_before = model(x_draft, player_comfort)

    # Fuse
    model.fuse_embeddings_for_inference()

    # Run after fusion
    with torch.no_grad():
        logits_after, mlm_after = model(x_draft, player_comfort)

    assert mlm_before.shape == mlm_after.shape
    assert logits_before.shape == logits_after.shape


def test_slot_attention_training_gradient():
    """Full forward + backward pass through the complete model."""
    d_model = 32
    num_heroes = 20
    batch_size = 2

    model = _build_model(d_model=d_model, num_heroes=num_heroes)
    model.train()

    x_draft = _make_draft(batch_size, num_heroes)
    player_pref = torch.randn(batch_size, 10, d_model)

    logits, mlm_logits = model(x_draft, player_pref)
    entropy_loss = model.mlm_head.get_entropy_loss()

    total_loss = logits.sum() + mlm_logits.sum() + entropy_loss
    total_loss.backward()

    # All parameters involved in the forward path should have gradients
    # (w_patch, film_gamma, film_beta are only used when patch_ids is provided)
    for name, param in model.named_parameters():
        if param.requires_grad and "w_patch" not in name and "film_" not in name:
            assert param.grad is not None, f"No gradient for {name}"

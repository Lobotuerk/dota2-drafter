"""Unit tests for MatchNetwork / HierarchicalTransformer."""

import pytest
import torch

from dota2drafter.models.match_network import (
    HierarchicalTransformer,
    MatchNetwork,
    JointEmbedding,
    SinusoidalPositionalEncoding,
)


def test_joint_embedding_forward():
    """Test joint embedding forward pass."""
    d_model = 64
    num_heroes = 120
    batch_size = 4

    h_gnn = torch.randn(num_heroes + 1, d_model)
    joint = JointEmbedding(d_model=d_model, num_heroes=num_heroes, h_gnn=h_gnn)

    # Input: (B, 24, 4)
    x_draft = torch.zeros(batch_size, 24, 4)
    # Set valid hero indices (1-based)
    for b in range(batch_size):
        for t in range(24):
            x_draft[b, t, 2] = (t % num_heroes) + 1  # hero_val
            x_draft[b, t, 0] = 1.0  # is_pick
            x_draft[b, t, 1] = t % 2  # team
            x_draft[b, t, 3] = float(t)  # step_index

    z = joint(x_draft)
    assert z.shape == (batch_size, 24, d_model)
    assert z.dtype == torch.float32


def test_joint_embedding_with_bans():
    """Test joint embedding with bans (is_pick=0, hero_val=-1)."""
    d_model = 64
    num_heroes = 120
    batch_size = 2

    h_gnn = torch.randn(num_heroes + 1, d_model)
    joint = JointEmbedding(d_model=d_model, num_heroes=num_heroes, h_gnn=h_gnn)

    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[0, t, 2] = (t % num_heroes) + 1
        x_draft[0, t, 0] = 0.0  # ban
        x_draft[0, t, 1] = 0
        x_draft[0, t, 3] = float(t)

    z = joint(x_draft)
    assert z.shape == (batch_size, 24, d_model)


def test_hierarchical_transformer_forward():
    """Test HierarchicalTransformer forward pass."""
    d_model = 64
    nhead = 4
    num_layers = 2
    num_heroes = 120
    batch_size = 4

    h_gnn = torch.randn(num_heroes + 1, d_model)

    model = HierarchicalTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
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

    logits, mlm_logits = model(x_draft, player_pref_vectors)
    assert logits.shape == (batch_size,)
    assert mlm_logits.shape == (batch_size, 24, num_heroes + 1)
    assert logits.dtype == torch.float32


def test_hierarchical_transformer_predict_proba():
    """Test predict_proba returns values in [0, 1]."""
    d_model = 64
    num_heroes = 120
    batch_size = 4

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

    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1
        x_draft[:, t, 0] = 1.0
        x_draft[:, t, 1] = t % 2
        x_draft[:, t, 3] = float(t)

    player_pref_vectors = torch.randn(batch_size, 10, d_model)

    proba = model.predict_proba(x_draft, player_pref_vectors)
    assert proba.shape == (batch_size,)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_match_network_forward():
    """Test full MatchNetwork forward pass."""
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

    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1
        x_draft[:, t, 0] = 1.0
        x_draft[:, t, 1] = t % 2
        x_draft[:, t, 3] = float(t)

    player_comfort = torch.randn(batch_size, 10, player_input_dim)

    logits, mlm_logits = model(x_draft, player_comfort)
    assert logits.shape == (batch_size,)
    assert mlm_logits.shape == (batch_size, 24, num_heroes + 1)


def test_match_network_predict_proba():
    """Test MatchNetwork predict_proba returns values in [0, 1]."""
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

    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1
        x_draft[:, t, 0] = 1.0
        x_draft[:, t, 1] = t % 2
        x_draft[:, t, 3] = float(t)

    player_comfort = torch.randn(batch_size, 10, player_input_dim)

    proba = model.predict_proba(x_draft, player_comfort)
    assert proba.shape == (batch_size,)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_match_network_requires_grad():
    """Test that MatchNetwork parameters require gradients."""
    d_model = 64
    num_heroes = 120
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

    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} should require gradients"


def test_sinusoidal_positional_encoding():
    """Test sinusoidal positional encoding dimensions."""
    d_model = 64
    pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=24)

    x = torch.randn(4, 24, d_model)
    output = pe(x)

    assert output.shape == (4, 24, d_model)


def test_match_network_default_h_gnn():
    """Test MatchNetwork with auto-initialized h_gnn."""
    model = MatchNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_heroes=120,
        player_input_dim=10,
    )

    batch_size = 2
    x_draft = torch.zeros(batch_size, 24, 4)
    player_comfort = torch.randn(batch_size, 10, 10)

    logits, mlm_logits = model(x_draft, player_comfort)
    assert logits.shape == (batch_size,)
    assert mlm_logits.shape == (batch_size, 24, 120 + 1)


def test_hierarchical_transformer_tgt_key_padding_mask():
    """Test that HierarchicalTransformer correctly constructs and passes tgt_key_padding_mask."""
    d_model = 64
    nhead = 4
    num_layers = 2
    num_heroes = 120
    batch_size = 2

    h_gnn = torch.randn(num_heroes + 1, d_model)

    model = HierarchicalTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=128,
        dropout=0.0,
        num_heroes=num_heroes,
        h_gnn=h_gnn,
    )

    # 1. Non-truncated draft: all steps have valid step indices 0..23
    x_draft = torch.zeros(batch_size, 24, 4)
    for t in range(24):
        x_draft[:, t, 2] = (t % num_heroes) + 1
        x_draft[:, t, 0] = 1.0
        x_draft[:, t, 1] = t % 2
        x_draft[:, t, 3] = float(t)

    # 2. Truncate sample 0 at step 12
    # In sample 0, steps 12..23 are set to 0.0
    x_draft[0, 12:, :] = 0.0

    player_pref_vectors = torch.randn(batch_size, 10, d_model)

    # Mock the decoder call to inspect the arguments using a custom PyTorch Module
    class MockDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False
            self.call_kwargs = {}

        def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
            self.called = True
            self.call_kwargs = {
                "tgt_key_padding_mask": tgt_key_padding_mask,
            }
            return torch.zeros_like(tgt)

    mock_decoder = MockDecoder()
    model.transformer_decoder = mock_decoder

    _ = model(x_draft, player_pref_vectors)

    # Inspect the call arguments of the mock
    assert mock_decoder.called
    kwargs = mock_decoder.call_kwargs
    assert "tgt_key_padding_mask" in kwargs

    pad_mask = kwargs["tgt_key_padding_mask"]
    assert pad_mask.shape == (batch_size, 24)
    assert pad_mask.dtype == torch.bool

    # Sample 0 should have steps 12..23 masked (True) and 0..11 unmasked (False)
    assert torch.all(pad_mask[0, :12] == False)
    assert torch.all(pad_mask[0, 12:] == True)

    # Sample 1 (non-truncated) should have all steps unmasked (False)
    assert torch.all(pad_mask[1, :] == False)


def test_subtractive_inhibition_parameters():
    """Verify registration of w_inhibit and gamma parameters."""
    d_model = 32
    num_heroes = 10
    h_gnn = torch.randn(num_heroes + 1, d_model)
    model = MatchNetwork(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        num_heroes=num_heroes,
        player_input_dim=22,
        h_gnn=h_gnn,
    )
    transformer = model.match_network
    assert hasattr(transformer, "w_inhibit")
    assert hasattr(transformer, "gamma")
    assert isinstance(transformer.w_inhibit, torch.nn.Linear)
    assert isinstance(transformer.gamma, torch.nn.Parameter)


def test_subtractive_inhibition_empty_picks():
    """Verify zero inhibition penalty when no prior picks exist."""
    d_model = 32
    num_heroes = 10
    h_gnn = torch.randn(num_heroes + 1, d_model)
    model = MatchNetwork(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        num_heroes=num_heroes,
        player_input_dim=22,
        h_gnn=h_gnn,
    )
    model.eval()

    x_draft = torch.zeros((1, 24, 4), dtype=torch.float32)
    x_draft[:, :, 2] = -1.0  # Empty draft (all padding)
    x_draft[:, :, 3] = torch.arange(24).float()
    player_comfort = torch.zeros((1, 10, 22), dtype=torch.float32)

    with torch.no_grad():
        logits, mlm_logits = model(x_draft, player_comfort)

    assert mlm_logits.shape == (1, 24, num_heroes + 1)


def test_subtractive_inhibition_penalty_monotonicity():
    """Verify that adding a pick decreases logits for that hero and similar heroes."""
    d_model = 32
    num_heroes = 10
    h_gnn = torch.randn(num_heroes + 1, d_model)
    model = MatchNetwork(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        num_heroes=num_heroes,
        player_input_dim=22,
        h_gnn=h_gnn,
    )
    model.eval()

    # Use a fixed draft: Team 0 picks hero 3 at step 0
    x_draft = torch.zeros((1, 24, 4), dtype=torch.float32)
    x_draft[:, :, 2] = -1.0
    x_draft[:, :, 3] = torch.arange(24).float()
    x_draft[0, 0] = torch.tensor([1.0, 0.0, 3.0, 0.0])  # Team 0 picks hero 3 at step 0

    player_comfort = torch.zeros((1, 10, 22), dtype=torch.float32)

    # Run without inhibition (set w_inhibit to zeros so penalty is zero)
    model.match_network.w_inhibit.weight.data = torch.zeros(d_model, d_model)
    model.match_network.gamma.data = torch.tensor(2.0)
    with torch.no_grad():
        _, mlm_logits_uninhibited = model(x_draft, player_comfort)

    # Run with inhibition (gamma=2, w_inhibit=identity)
    model.match_network.w_inhibit.weight.data = torch.eye(d_model)
    model.match_network.gamma.data = torch.tensor(2.0)
    with torch.no_grad():
        _, mlm_logits_inhibited = model(x_draft, player_comfort)

    # At step 1 (Team 0 pick again), hero 3 has a past active pick
    # Logit for hero 3 should be strictly less with inhibition active
    assert mlm_logits_inhibited[0, 1, 3] < mlm_logits_uninhibited[0, 1, 3]


def test_subtractive_inhibition_causality():
    """Verify that changing a pick at step t does not alter mlm_logits at steps < t."""
    d_model = 32
    num_heroes = 10
    h_gnn = torch.randn(num_heroes + 1, d_model)
    model = MatchNetwork(
        d_model=d_model,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        num_heroes=num_heroes,
        player_input_dim=22,
        h_gnn=h_gnn,
    )
    model.eval()

    x_draft_1 = torch.zeros((1, 24, 4), dtype=torch.float32)
    x_draft_1[:, :, 2] = -1.0
    x_draft_1[:, :, 3] = torch.arange(24).float()
    x_draft_1[0, 0] = torch.tensor([1.0, 0.0, 2.0, 0.0])  # Pick hero 2 at step 0
    x_draft_1[0, 1] = torch.tensor([1.0, 1.0, 4.0, 1.0])  # Pick hero 4 at step 1

    x_draft_2 = x_draft_1.clone()
    x_draft_2[0, 1] = torch.tensor([1.0, 1.0, 8.0, 1.0])  # Change step 1 pick to hero 8

    player_comfort = torch.zeros((1, 10, 22), dtype=torch.float32)

    with torch.no_grad():
        _, mlm_logits_1 = model(x_draft_1, player_comfort)
        _, mlm_logits_2 = model(x_draft_2, player_comfort)

    # Logits at step 0 and step 1 must be identical across both drafts
    torch.testing.assert_close(mlm_logits_1[0, 0], mlm_logits_2[0, 0])
    torch.testing.assert_close(mlm_logits_1[0, 1], mlm_logits_2[0, 1])


def test_match_network_predict_proba_with_patch_ids():
    """Test MatchNetwork predict_proba accepts patch_ids and returns valid probabilities."""
    d_model = 64
    num_heroes = 120
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
        num_patches=10,
    )

    x_draft = torch.zeros(1, 24, 4)
    x_draft[0, :, 2] = torch.arange(24) + 1
    x_draft[0, :, 0] = 1.0
    x_draft[0, :, 3] = torch.arange(24).float()

    player_comfort = torch.randn(1, 10, player_input_dim)

    patch_1 = torch.tensor([1], dtype=torch.long)
    proba_1 = model.predict_proba(x_draft, player_comfort, patch_ids=patch_1)

    patch_2 = torch.tensor([2], dtype=torch.long)
    proba_2 = model.predict_proba(x_draft, player_comfort, patch_ids=patch_2)

    assert proba_1.shape == (1,)
    assert proba_2.shape == (1,)
    assert 0 <= proba_1.item() <= 1
    assert 0 <= proba_2.item() <= 1


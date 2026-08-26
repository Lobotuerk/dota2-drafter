import pytest
import torch

from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.processor.hero_indexer import HeroIndexer


@pytest.fixture
def hero_indexer():
    indexer = HeroIndexer()
    # Build mapping for 120 heroes (from 1 to 120) so they map to contiguous indices 1..120
    indexer.build_mapping([{"id": i, "playable": True} for i in range(1, 121)])
    return indexer


@pytest.fixture
def patch_id():
    return torch.zeros(1, dtype=torch.long)


@pytest.fixture
def model():
    return MatchNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_heroes=120,
        player_input_dim=127,
    )


@torch.no_grad()
def test_slot_confusion(model, patch_id, hero_indexer):
    model.eval()
    mlm_head = model.match_network.mlm_head
    device = next(model.parameters()).device

    slark_idx = hero_indexer.map_hero_id(93)  # Pos 1 Slark
    cm_idx = hero_indexer.map_hero_id(5)      # Pos 5 Crystal Maiden

    assert slark_idx is not None
    assert cm_idx is not None

    # Draft A: Pick Slark at step 7
    draft_a = torch.zeros((1, 24, 4), device=device)
    draft_a[:, :, 2] = -1.0
    draft_a[0, 7] = torch.tensor([1.0, 0.0, float(slark_idx), 7.0])  # Pick Slark at step 7

    # Draft B: Pick Crystal Maiden at step 7
    draft_b = torch.zeros((1, 24, 4), device=device)
    draft_b[:, :, 2] = -1.0
    draft_b[0, 7] = torch.tensor([1.0, 0.0, float(cm_idx), 7.0])  # Pick CM at step 7

    dummy_ht = torch.zeros((1, 24, model.d_model), device=device)

    # Run forward to compute alpha inside mlm_head
    _ = mlm_head(dummy_ht, draft_a, patch_id)
    assert mlm_head.raw_occupancy is not None
    alpha_slark = (1.0 - torch.exp(-mlm_head.raw_occupancy)).squeeze()  # Inspect alpha at step 8

    _ = mlm_head(dummy_ht, draft_b, patch_id)
    assert mlm_head.raw_occupancy is not None
    alpha_cm = (1.0 - torch.exp(-mlm_head.raw_occupancy)).squeeze()

    print("\n--- Slot Vector Confusion Diagnostics ---")
    print("Slot Occupancies (alpha) after picking Slark (Pos 1) at Step 7:")
    print(alpha_slark[8].cpu().numpy().round(3))
    print("Slot Occupancies (alpha) after picking CM (Pos 5) at Step 7:")
    print(alpha_cm[8].cpu().numpy().round(3))

    # Assert shape is (24, 5)
    assert alpha_slark.shape == (24, 5)
    assert alpha_cm.shape == (24, 5)

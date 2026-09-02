from pathlib import Path

import pytest
import torch

from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.processor.hero_indexer import HeroIndexer


@pytest.fixture
def hero_indexer():
    indexer = HeroIndexer()
    # Build mapping for 120 heroes (from 1 to 120) so they map to contiguous indices 1..120
    indexer_path = Path("data") / "hero_indexer.json"
    assert indexer_path.exists(), f"Hero indexer missing at {indexer_path.resolve()}"
    import json
    with open(indexer_path, "r") as f:
        hero_data = json.load(f)
    # Reconstruct hero list from mapping
    heroes = [{"id": int(api_id), "playable": True} for api_id in hero_data.keys()]
    indexer.build_mapping(heroes)
    return indexer


@pytest.fixture
def patch_id():
    return torch.zeros(1, dtype=torch.long)


@pytest.fixture
def model(hero_indexer):
    d_model = 64
    num_heroes = 127
    player_input_dim = 310

    # Load pre-trained RGCN hero embeddings or fallback to random
    rgcn_path = Path("models/rgcn.pt")
    assert rgcn_path.exists(), f"Checkpoint missing at {rgcn_path.resolve()}"
    h_gnn = torch.load(rgcn_path, weights_only=True)
    h_gnn = h_gnn.get("embedding.weight")
    assert h_gnn.shape == (num_heroes + 1, d_model)

    model = MatchNetwork(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_heroes=num_heroes,
        player_input_dim=player_input_dim,
        h_gnn=h_gnn,
    )

    # Load trained checkpoint if available; fallback to eval mode.
    checkpoint_path = Path("checkpoints/best_model.pt")
    assert checkpoint_path.exists(), f"Checkpoint missing at {checkpoint_path.resolve()}"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=True)

    model.eval()
    return model


@torch.no_grad()
def test_slot_confusion(model, patch_id, hero_indexer):
    model.eval()
    mlm_head = model.match_network.mlm_head
    device = next(model.parameters()).device

    slark_idx = hero_indexer.map_hero_id(93)  # Pos 1 Slark
    cm_idx = hero_indexer.map_hero_id(5)      # Pos 5 Crystal Maiden
    dw_idx = hero_indexer.map_hero_id(123) # Pos 4 Dark willow
    invoker_idx = hero_indexer.map_hero_id(74) # Pos 2 Invoker
    centaur_idx = hero_indexer.map_hero_id(96) # Pos 3 Centaur

    assert slark_idx is not None
    assert cm_idx is not None
    assert dw_idx is not None
    assert invoker_idx is not None
    assert centaur_idx is not None


    # Draft A: Pick Slark at step 7
    draft_a = torch.zeros((1, 24, 4), device=device)
    draft_a[:, :, 2] = -1.0
    draft_a[0, 7] = torch.tensor([1.0, 0.0, float(slark_idx), 7.0])  # Pick Slark at step 7

    # Draft B: Pick Crystal Maiden at step 7
    draft_b = torch.zeros((1, 24, 4), device=device)
    draft_b[:, :, 2] = -1.0
    draft_b[0, 7] = torch.tensor([1.0, 0.0, float(cm_idx), 7.0])  # Pick CM at step 7

    # Draft C: Pick Dark Willow at step 7
    draft_c = torch.zeros((1, 24, 4), device=device)
    draft_c[:, :, 2] = -1.0
    draft_c[0, 7] = torch.tensor([1.0, 0.0, float(dw_idx), 7.0])  # Pick KOTL at step 7

    # Draft D: Pick Invoker at step 7
    draft_d = torch.zeros((1, 24, 4), device=device)
    draft_d[:, :, 2] = -1.0
    draft_d[0, 7] = torch.tensor([1.0, 0.0, float(invoker_idx), 7.0])  # Pick invoker at step 7

    # Draft E: Pick Centaur at step 7
    draft_e = torch.zeros((1, 24, 4), device=device)
    draft_e[:, :, 2] = -1.0
    draft_e[0, 7] = torch.tensor([1.0, 0.0, float(centaur_idx), 7.0])  # Pick Centaur at step 7

    dummy_ht = torch.zeros((1, 24, model.d_model), device=device)

    # Run forward to compute alpha inside mlm_head
    _ = mlm_head(dummy_ht, draft_a, patch_id)
    assert mlm_head.raw_occupancy is not None
    alpha_slark = (1.0 - torch.exp(-mlm_head.raw_occupancy)).squeeze()  # Inspect alpha at step 8

    _ = mlm_head(dummy_ht, draft_b, patch_id)
    assert mlm_head.raw_occupancy is not None
    alpha_cm = (1.0 - torch.exp(-mlm_head.raw_occupancy)).squeeze()

    _ = mlm_head(dummy_ht, draft_c, patch_id)
    assert mlm_head.raw_occupancy is not None
    alpha_dw = (1.0 - torch.exp(-mlm_head.raw_occupancy)).squeeze()

    _ = mlm_head(dummy_ht, draft_d, patch_id)
    assert mlm_head.raw_occupancy is not None
    alpha_invoker = (1.0 - torch.exp(-mlm_head.raw_occupancy)).squeeze()

    _ = mlm_head(dummy_ht, draft_e, patch_id)
    assert mlm_head.raw_occupancy is not None
    alpha_centaur = (1.0 - torch.exp(-mlm_head.raw_occupancy)).squeeze()

    print("\n--- Slot Vector Confusion Diagnostics ---")
    print("Slot Occupancies (alpha) after picking Slark (Pos 1) at Step 7:")
    print(alpha_slark[8].cpu().numpy().round(3))
    print("Slot Occupancies (alpha) after picking CM (Pos 5) at Step 7:")
    print(alpha_cm[8].cpu().numpy().round(3))
    print("Slot Occupancies (alpha) after picking DW (Pos 4) at Step 7:")
    print(alpha_dw[8].cpu().numpy().round(3))
    print("Slot Occupancies (alpha) after picking Invoker (Pos 2) at Step 7:")
    print(alpha_invoker[8].cpu().numpy().round(3))
    print("Slot Occupancies (alpha) after picking Centaur (Pos 3) at Step 7:")
    print(alpha_centaur[8].cpu().numpy().round(3))

    # Assert shape is (24, 5)
    assert alpha_slark.shape == (24, 5)
    assert alpha_cm.shape == (24, 5)
    assert alpha_dw.shape == (24, 5)
    assert alpha_invoker.shape == (24, 5)
    assert alpha_centaur.shape == (24, 5)


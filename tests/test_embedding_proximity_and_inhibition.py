import pytest
import torch
import torch.nn.functional as f_api

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


@pytest.fixture
def x_draft():
    # shape: (1, 24, 4)
    x = torch.zeros(1, 24, 4)
    for t in range(24):
        x[0, t, 3] = float(t)
        if t <= 8:
            x[0, t, 0] = 1.0  # is_pick
            x[0, t, 1] = float(t % 2)  # team
            x[0, t, 2] = float(t + 1)  # valid hero_val (1-based index)
        else:
            x[0, t, 0] = 0.0
            x[0, t, 1] = 0.0
            x[0, t, 2] = -1.0  # padded
    return x


@torch.no_grad()
def test_embedding_proximity_and_inhibition(model, x_draft, patch_id, hero_indexer):
    model.eval()
    device = x_draft.device

    all_indices = torch.arange(
        model.match_network.num_heroes + 1, device=device
    ).unsqueeze(0)
    e_hero = model.match_network.joint_embedding.get_pure_hero_embeddings(
        all_indices, patch_id
    ).squeeze(0)
    e_norm = f_api.normalize(e_hero, p=2, dim=-1)

    slark_idx = hero_indexer.map_hero_id(93)
    sf_idx = hero_indexer.map_hero_id(11)
    ta_idx = hero_indexer.map_hero_id(46)
    lina_idx = hero_indexer.map_hero_id(25)
    cm_idx = hero_indexer.map_hero_id(5)

    assert slark_idx is not None
    assert sf_idx is not None
    assert ta_idx is not None
    assert lina_idx is not None
    assert cm_idx is not None

    # 1. Cosine Similarity in H_GNN
    sim_slark_sf = torch.dot(e_norm[slark_idx], e_norm[sf_idx]).item()
    sim_sf_ta = torch.dot(e_norm[sf_idx], e_norm[ta_idx]).item()
    sim_sf_lina = torch.dot(e_norm[sf_idx], e_norm[lina_idx]).item()
    sim_sf_cm = torch.dot(e_norm[sf_idx], e_norm[cm_idx]).item()

    print("\n--- H_GNN Cosine Similarity Diagnostics ---")
    print(f"Sim(Slark Pos1, SF Pos2): {sim_slark_sf:.3f}")
    print(f"Sim(SF Pos2, TA Pos2):    {sim_sf_ta:.3f}")
    print(f"Sim(SF Pos2, Lina Pos2):  {sim_sf_lina:.3f}")
    print(f"Sim(SF Pos2, CM Pos5):    {sim_sf_cm:.3f}")

    # 2. Subtractive Penalty Inspection (AUT-28)
    dummy_comfort = torch.zeros((1, 10, model.player_input_dim), device=device)
    _, mlm_logits = model(x_draft, dummy_comfort, patch_ids=patch_id)

    # Extract penalty applied to TA, Lina, vs CM at step 8
    penalty = model.match_network.inhibition_penalty.squeeze(0)  # Shape (24, num_heroes + 1)
    step_t = 8

    penalty_ta = penalty[step_t, ta_idx].item()
    penalty_lina = penalty[step_t, lina_idx].item()
    penalty_cm = penalty[step_t, cm_idx].item()

    print("\n--- Subtractive Penalty Diagnostics ---")
    print(f"Subtractive Penalty for TA:   {penalty_ta:.3f}")
    print(f"Subtractive Penalty for Lina: {penalty_lina:.3f}")
    print(f"Subtractive Penalty for CM:   {penalty_cm:.3f}")

    assert penalty.shape == (24, model.match_network.num_heroes + 1)

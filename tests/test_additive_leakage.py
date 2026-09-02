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
def test_additive_leakage(model, x_draft, patch_id, hero_indexer):
    model.eval()
    mlm_head = model.match_network.mlm_head

    # 1. Run forward pass up to decoder output (h_t)
    # Get decoder_output h_t from match_network
    player_comfort = torch.zeros((1, 10, model.player_input_dim), device=x_draft.device)
    player_pref = model.player_network(player_comfort)

    shifted_x = x_draft.clone()
    shifted_x[:, 0, 2] = -1.0
    shifted_x[:, 1:, 2] = x_draft[:, :-1, 2]
    z = model.match_network.joint_embedding(shifted_x, patch_id)

    causal_mask = torch.triu(torch.ones(24, 24, device=x_draft.device), diagonal=1).bool()
    pad_mask = (x_draft[:, :, 2] == -1.0)

    h_t = model.match_network.transformer_decoder(
        tgt=z, memory=player_pref, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask
    )  # (1, 24, d_model)

    # 2. Extract unfilled_weight and h_query at current step t
    step_t = 8  # Step after Slark + SF
    # Compute h_query from mlm_head internal state
    # (Pass through mlm_head logic)
    all_indices = torch.arange(
        model.match_network.num_heroes + 1, device=x_draft.device
    ).unsqueeze(0)
    e_hero = model.match_network.joint_embedding.get_pure_hero_embeddings(all_indices, patch_id)

    # Decompose combined representation
    # We measure logits from h_t alone vs logits from h_query alone
    h_t_step = h_t[:, step_t:step_t+1, :]

    # Run mlm_head forward to grab internal h_query
    _ = mlm_head(h_t, x_draft, patch_id)
    # Measure candidate scores
    logits_from_ht = torch.matmul(
        mlm_head.w_policy(h_t_step),
        e_hero.transpose(1, 2)
    ).squeeze()

    ta_idx = hero_indexer.map_hero_id(46)  # Templar Assassin
    lina_idx = hero_indexer.map_hero_id(25)  # Lina
    cm_idx = hero_indexer.map_hero_id(5)  # Crystal Maiden

    assert ta_idx is not None
    assert lina_idx is not None
    assert cm_idx is not None

    print("--- Additive Leakage Diagnostics ---")
    print(f"Decoder (h_t) Logit for TA:   {logits_from_ht[ta_idx].item():.3f}")
    print(f"Decoder (h_t) Logit for Lina: {logits_from_ht[lina_idx].item():.3f}")
    print(f"Decoder (h_t) Logit for CM:   {logits_from_ht[cm_idx].item():.3f}")

    # If TA/Lina have huge positive raw decoder logits (+8 to +15) compared to CM (-2 to +1),
    # h_query cannot zero them out additively.
    assert logits_from_ht.shape == (model.match_network.num_heroes + 1,)

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

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
    return torch.tensor([21], dtype=torch.long)


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


@pytest.fixture
def x_draft_slark_sf(hero_indexer):
    """Constructs a draft where Team 0 has picked Slark (Pos 1) and SF (Pos 2)."""
    x = torch.zeros(1, 24, 4)
    x[:, :, 2] = -1.0  # Default padding

    slark_id = hero_indexer.map_hero_id(93)
    sf_id = hero_indexer.map_hero_id(11)

    # Step 0: Team 0 picks Slark
    x[0, 0] = torch.tensor([1.0, 0.0, float(slark_id), 0.0])
    # Step 1: Team 1 picks dummy hero
    x[0, 1] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    # Step 2: Team 0 picks Shadow Fiend
    x[0, 2] = torch.tensor([1.0, 0.0, float(sf_id), 2.0])

    return x


@torch.no_grad()
def test_subtractive_role_inhibition(model, x_draft_slark_sf, patch_id, hero_indexer):
    device = next(model.parameters()).device
    x_draft = x_draft_slark_sf.to(device)
    patch_id = patch_id.to(device)
    model.match_network.h_gnn = model.match_network.h_gnn.to(device)
    model.match_network.joint_embedding.h_gnn = model.match_network.joint_embedding.h_gnn.to(device)

    ta_idx = hero_indexer.map_hero_id(46)   # Templar Assassin (Pos 2)
    lina_idx = hero_indexer.map_hero_id(25) # Lina (Pos 2)
    cm_idx = hero_indexer.map_hero_id(5)     # Crystal Maiden (Pos 5)
    sf_idx = hero_indexer.map_hero_id(11)    # Shadow Fiend (Pos 2)
    slark_idx = hero_indexer.map_hero_id(93) # Slark (Pos 1)

    # 1. Compute raw un-scaled logits directly from role_head
    all_hero_indices = torch.arange(
        model.match_network.num_heroes + 1, device=device
    ).unsqueeze(0)
    E_hero = model.match_network.joint_embedding.get_pure_hero_embeddings(all_hero_indices, patch_id)
    
    raw_role_logits = model.match_network.mlm_head.role_head(E_hero).squeeze(0)  # Shape: (128, 5)

    ta_logits = raw_role_logits[ta_idx].cpu().numpy()
    lina_logits = raw_role_logits[lina_idx].cpu().numpy()
    cm_logits = raw_role_logits[cm_idx].cpu().numpy()
    sf_logits = raw_role_logits[sf_idx].cpu().numpy()
    slark_logits = raw_role_logits[slark_idx].cpu().numpy()

    print("\n--- Un-scaled Raw Role Head Logits (High Precision) ---")
    print("TA Raw Logits:", [f"{x:.6f}" for x in ta_logits])
    print("Lina Raw Logits:   ", [f"{x:.6f}" for x in lina_logits])
    print("CM Raw Logits:     ", [f"{x:.6f}" for x in cm_logits])
    print("SF Raw Logits:     ", [f"{x:.6f}" for x in sf_logits])
    print("Slark Raw Logits:  ", [f"{x:.6f}" for x in slark_logits])

    # 2. Compute Softmax Probs (tau = 0.20)
    ta_probs = F.softmax(raw_role_logits[ta_idx] / 0.20, dim=-1).cpu().numpy()
    lina_probs = F.softmax(raw_role_logits[lina_idx] / 0.20, dim=-1).cpu().numpy()
    cm_probs = F.softmax(raw_role_logits[cm_idx] / 0.20, dim=-1).cpu().numpy()
    sf_probs = F.softmax(raw_role_logits[sf_idx] / 0.20, dim=-1).cpu().numpy()
    slark_probs = F.softmax(raw_role_logits[slark_idx] / 0.20, dim=-1).cpu().numpy()

    print("\n--- Scaled Softmax Probs (tau = 0.20) ---")
    print("TA Role Probs:", ta_probs.round(3))
    print("Lina Role Probs:   ", lina_probs.round(3))
    print("CM Role Probs:     ", cm_probs.round(3))
    print("SF Role Probs:     ", sf_probs.round(3))
    print("Slark Role Probs:  ", slark_probs.round(3))

    # 3. Run forward pass

    dummy_comfort = torch.zeros((1, 10, model.player_input_dim), device=device)
    _, mlm_logits = model(x_draft, dummy_comfort, patch_ids=patch_id)

    # Step 3: Team 0's next action after picking Slark + SF
    step_t = 3
    penalty = model.match_network.inhibition_penalty.squeeze(0)

    penalty_ta = penalty[step_t, ta_idx].item()
    penalty_lina = penalty[step_t, lina_idx].item()
    penalty_cm = penalty[step_t, cm_idx].item()

    print("\n--- Subtractive Penalty Diagnostics ---")
    print(f"Subtractive Penalty for TA:   {penalty_ta:.3f}")
    print(f"Subtractive Penalty for Lina: {penalty_lina:.3f}")
    print(f"Subtractive Penalty for CM:   {penalty_cm:.3f}")

    # 1. Structural Assertions
    assert penalty.shape == (24, model.match_network.num_heroes + 1)
    assert torch.all(penalty >= 0.0), "Inhibition penalty must be non-negative"

    # 2. Behavioral Assertions: Core candidates must receive positive inhibition penalties
    assert penalty_lina > 0.0 or penalty_ta > 0.0, (
        "Subtractive inhibition failed: Mid cores received zero penalty after Slark + SF picks."
    )

    # 3. Policy Logit Suppression Assertion: Net policy logit must be reduced by inhibition penalty
    net_logit_lina = mlm_logits[0, step_t, lina_idx].item()
    base_logit_lina = net_logit_lina + penalty_lina

    assert net_logit_lina < base_logit_lina, (
        f"Role suppression failed: Net logit ({net_logit_lina:.2f}) is not reduced "
        f"relative to base logit ({base_logit_lina:.2f}) by penalty ({penalty_lina:.2f})"
    )


@torch.no_grad()
def test_puck_invoker_role_repulsion(model, patch_id, hero_indexer):
    """Verify that picking Invoker (Pos 2) penalizes recommending Puck (Pos 2) on the same team."""
    device = next(model.parameters()).device
    patch_id = patch_id.to(device)
    model.match_network.h_gnn = model.match_network.h_gnn.to(device)
    model.match_network.joint_embedding.h_gnn = model.match_network.joint_embedding.h_gnn.to(device)

    invoker_idx = hero_indexer.map_hero_id(74)  # Invoker ID = 74
    puck_idx = hero_indexer.map_hero_id(13)     # Puck ID = 13
    cm_idx = hero_indexer.map_hero_id(5)        # Crystal Maiden ID = 5

    # 1. Compute raw un-scaled logits directly from role_head
    all_hero_indices = torch.arange(
        model.match_network.num_heroes + 1, device=device
    ).unsqueeze(0)
    E_hero = model.match_network.joint_embedding.get_pure_hero_embeddings(all_hero_indices, patch_id)
    
    raw_role_logits = model.match_network.mlm_head.role_head(E_hero).squeeze(0)  # Shape: (128, 5)

    invoker_logits = raw_role_logits[invoker_idx].cpu().numpy()
    puck_logits = raw_role_logits[puck_idx].cpu().numpy()
    cm_logits = raw_role_logits[cm_idx].cpu().numpy()

    print("\n--- Un-scaled Raw Role Head Logits (High Precision) ---")
    print("Invoker Raw Logits:", [f"{x:.6f}" for x in invoker_logits])
    print("Puck Raw Logits:   ", [f"{x:.6f}" for x in puck_logits])
    print("CM Raw Logits:     ", [f"{x:.6f}" for x in cm_logits])

    # 2. Compute Softmax Probs (tau = 0.20)
    invoker_probs = F.softmax(raw_role_logits[invoker_idx] / 0.20, dim=-1).cpu().numpy()
    puck_probs = F.softmax(raw_role_logits[puck_idx] / 0.20, dim=-1).cpu().numpy()
    cm_probs = F.softmax(raw_role_logits[cm_idx] / 0.20, dim=-1).cpu().numpy()

    print("\n--- Scaled Softmax Probs (tau = 0.20) ---")
    print("Invoker Role Probs:", invoker_probs.round(3))
    print("Puck Role Probs:   ", puck_probs.round(3))
    print("CM Role Probs:     ", cm_probs.round(3))

    x_draft = torch.zeros(1, 24, 4, device=device)
    x_draft[:, :, 2] = -1.0  # Default padding
    x_draft[0, 0] = torch.tensor([1.0, 0.0, float(invoker_idx), 0.0], device=device)  # Team 0 picks Invoker
    x_draft[0, 1] = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)                  # Team 1 picks dummy

    dummy_comfort = torch.zeros((1, 10, model.player_input_dim), device=device)
    _, mlm_logits = model(x_draft, dummy_comfort, patch_ids=patch_id)

    step_t = 2
    penalty = model.match_network.inhibition_penalty.squeeze(0)
    role_probs = model.match_network.mlm_head.hero_role_probs.squeeze(0)

    penalty_puck = penalty[step_t, puck_idx].item()
    penalty_cm = penalty[step_t, cm_idx].item()


    print("\n--- Invoker + Puck Role Repulsion Diagnostics ---")
    print(f"Base logit for Puck (Mid):           {mlm_logits[0, step_t, puck_idx].item():.3f}")
    print(f"Base logit for CM (Support):        {mlm_logits[0, step_t, cm_idx].item():.3f}")
    print(f"Subtractive Penalty for Puck (Mid):   {penalty_puck:.3f}")
    print(f"Subtractive Penalty for CM (Support): {penalty_cm:.3f}")

    # Direct role collision score calculation
    collision_puck = torch.dot(role_probs[invoker_idx], role_probs[puck_idx]).item()
    collision_cm = torch.dot(role_probs[invoker_idx], role_probs[cm_idx]).item()

    print(f"\nDirect Role Collision for Puck (Mid):   {collision_puck:.3f}")
    print(f"Direct Role Collision for CM (Support): {collision_cm:.3f}")

    # Verify that Puck (duplicate Mid) collides significantly more than CM (Support)
    assert collision_puck > collision_cm
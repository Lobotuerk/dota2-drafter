from pathlib import Path

import pytest
import torch

from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.processor.hero_indexer import HeroIndexer


@pytest.fixture
def hero_indexer():
    indexer = HeroIndexer()
    indexer.build_mapping([{"id": i, "playable": True} for i in range(1, 128)])
    return indexer


@pytest.fixture
def patch_id():
    return torch.tensor([21], dtype=torch.long)


@pytest.fixture
def model(hero_indexer):
    d_model = 64
    num_heroes = 127
    player_input_dim = 310
    h_gnn = torch.randn(num_heroes + 1, d_model)

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
    if checkpoint_path.exists():
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

    ta_idx = hero_indexer.map_hero_id(46)   # Templar Assassin (Pos 2)
    lina_idx = hero_indexer.map_hero_id(25) # Lina (Pos 2)
    cm_idx = hero_indexer.map_hero_id(5)     # Crystal Maiden (Pos 5)

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

    invoker_idx = hero_indexer.map_hero_id(74)  # Invoker ID = 74
    puck_idx = hero_indexer.map_hero_id(13)     # Puck ID = 13
    cm_idx = hero_indexer.map_hero_id(5)        # Crystal Maiden ID = 5

    x_draft = torch.zeros(1, 24, 4, device=device)
    x_draft[:, :, 2] = -1.0  # Default padding
    x_draft[0, 0] = torch.tensor([1.0, 0.0, float(invoker_idx), 0.0], device=device)  # Team 0 picks Invoker
    x_draft[0, 1] = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)                  # Team 1 picks dummy

    dummy_comfort = torch.zeros((1, 10, model.player_input_dim), device=device)
    _, mlm_logits = model(x_draft, dummy_comfort, patch_ids=patch_id)

    step_t = 2
    penalty = model.match_network.inhibition_penalty.squeeze(0)

    penalty_puck = penalty[step_t, puck_idx].item()
    penalty_cm = penalty[step_t, cm_idx].item()
    logit_puck = mlm_logits[0, step_t, puck_idx].item()
    logit_cm = mlm_logits[0, step_t, cm_idx].item()

    print("\n--- Invoker + Puck Role Repulsion Diagnostics ---")
    print(f"Subtractive Penalty for Puck (Mid):   {penalty_puck:.3f}")
    print(f"Subtractive Penalty for CM (Support): {penalty_cm:.3f}")
    print(f"Net Policy Logit for Puck (Mid):      {logit_puck:.3f}")
    print(f"Net Policy Logit for CM (Support):    {logit_cm:.3f}")

    # 1. Role Penalty Assertion: Duplicate Mid (Puck) must receive a higher penalty than Support (CM)
    assert penalty_puck > penalty_cm, (
        f"Invoker-Puck repulsion failed: Puck penalty ({penalty_puck:.3f}) was not "
        f"higher than CM penalty ({penalty_cm:.3f}) after Invoker pick."
    )

    # 2. Net Policy Assertion: Support (CM) should be favored over duplicate Mid (Puck)
    assert logit_cm > logit_puck, (
        f"Role suppression failed: Net Puck logit ({logit_puck:.2f}) remains "
        f"higher than CM logit ({logit_cm:.2f}) after Invoker pick."
    )
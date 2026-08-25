# Technical Specification: AUT-28 - Subtractive Inhibition

## Overview
This specification replaces the purely additive linear projection in `HierarchicalTransformer.mlm_head` with a subtractive role-inhibition head. When evaluating a potential draft pick/ban at step $t$, the new mechanism computes an active-team-conditioned inhibition vector from their prior picks, evaluates its similarity to all draft candidates using the pure hero embedding table, and subtracts a non-negative penalty directly from the policy logits.

This implements a soft-inhibition prior: selecting a hero of a certain type/role strictly decreases the policy logits for all candidate heroes sharing similar embeddings (e.g., similar roles or playstyles), encouraging balanced team compositions.

## 1. Architectural Changes
**File**: `src/dota2drafter/models/match_network.py`

### 1.1 `HierarchicalTransformer.__init__`
We will introduce two new learnable components in `HierarchicalTransformer.__init__` that govern the subtractive role-inhibition mechanism:
- **`self.w_inhibit`**: An `nn.Linear(d_model, d_model, bias=False)` projection layer. This maps the accumulated pick representation ($v_{\text{active}}$) to a role-inhibition vector in the same latent space.
- **`self.gamma`**: An `nn.Parameter` initialized to `1.0` (global learnable scalar scaling factor for the inhibition penalty).

### 1.2 `HierarchicalTransformer.forward`
We will modify the forward pass to apply the subtractive penalty to `mlm_logits` computed at step $t$:
1. **Active Team Past Picks Extraction (Vectorized)**:
   - Identify the active team at each step $t$ using `x_draft[:, :, 1]` (shape: `(B, 24)`).
   - Construct a boolean mask `past_active_picks_mask` of shape `(B, 24, 24)` where `past_active_picks_mask[b, t, s]` is `True` if and only if:
     - $s < t$ (causal constraint: only consider past choices).
     - The step $s$ is a pick: `x_draft[b, s, 0] == 1.0`.
     - The team at step $s$ matches the active team at step $t$: `x_draft[b, s, 1] == x_draft[b, t, 1]`.
     - The hero index is valid: `x_draft[b, s, 2] >= 0.0`.
   - Convert `past_active_picks_mask` to a float weight tensor `mask_weight` of shape `(B, 24, 24)`.

2. **Embedding Averaging ($v_{\text{active}}$)**:
   - Extract the pure hero embeddings of the draft sequence:
     `draft_hero_embeds = self.joint_embedding.get_pure_hero_embeddings(x_draft[:, :, 2].long(), patch_ids)` (shape: `(B, 24, d_model)`).
   - Compute the sum of past active pick embeddings using batch matrix multiplication:
     `sum_embeds = torch.bmm(mask_weight, draft_hero_embeds)` (shape: `(B, 24, d_model)`).
   - Count the number of active picks up to step $t$:
     `num_picks = mask_weight.sum(dim=2, keepdim=True)` (shape: `(B, 24, 1)`).
   - Compute $v_{\text{active}} \in \mathbb{R}^{B \times 24 \times d_{\text{model}}}$:
     `v_active = sum_embeds / torch.clamp(num_picks, min=1.0)`.

3. **Inhibition Signal Generation & Projection**:
   - Project the accumulated pick representation:
     `v_inhibit = self.w_inhibit(v_active)` (shape: `(B, 24, d_model)`).
   - Dynamically reconstruct the pure candidate hero embedding table $E_{\text{hero}} \in \mathbb{R}^{B \times (K+1) \times d_{\text{model}}}$ for all $K+1$ candidate indices:
     `all_hero_indices = torch.arange(self.num_heroes + 1, device=x_draft.device).unsqueeze(0).repeat(batch_size, 1)`
     `E_hero = self.joint_embedding.get_pure_hero_embeddings(all_hero_indices, patch_ids)` (shape: `(B, num_heroes + 1, d_model)`).
   - Compute the raw similarity / penalty logits:
     `penalty_logits = torch.bmm(v_inhibit, E_hero.transpose(1, 2))` (shape: `(B, 24, num_heroes + 1)`).

4. **Penalty Subtraction**:
   - Compute the non-negative penalty term using `ReLU` and scale it by `self.gamma`:
     `inhibition_enabled = (num_picks > 0).float()`
     `inhibition_penalty = self.gamma * torch.relu(penalty_logits) * inhibition_enabled`
   - Subtract this penalty directly from the linear projection's logits:
     `mlm_logits = mlm_logits - inhibition_penalty`

## 2. Downstream Compatibility
The modified signature of the forward pass remains completely unchanged:
- `logits, mlm_logits = model(x_draft, player_pref_vectors, patch_ids)`
All training scripts (`transformer_trainer.py`, `scripts/04_train_transformer.py`), search routines (`state.py`), and evaluation code will function without any modifications, keeping complexity low and preventing system regressions.

## 3. Edge Cases & Robustness
1. **Empty Active Team Picks**: At step $t=0$, or any step before the active team has made a pick, `num_picks` will be `0.0`. Clamping the division denominator prevents `NaN`s, and `inhibition_enabled` ensures the penalty is exactly zero, fulfilling the requirement to "skip inhibition" when a team has no picks yet.
2. **Padding Tensors**: Step indices equal to `0.0` (padding) are already masked out in the cross-entropy loss computation downstream. Our masking logic relies on `x_draft[:, :, 2] >= 0.0`, which correctly filters out padding slots (hero value `0.0` is valid, but padding/masked hero IDs are `-1.0`).
3. **Causal Masking**: The mask only considers past steps ($s < t$). A pick made at step $t$ or later cannot affect the active team's pick average $v_{\text{active}}$ at step $t$, maintaining causal ordering and preventing information leakage.

## 4. Testing Strategy
A new unit test suite MUST be added in `tests/test_models/test_match_network.py` containing the following tests:
1. **`test_subtractive_inhibition_parameters`**:
   - Initialize `HierarchicalTransformer` and verify that `self.w_inhibit` and `self.gamma` are registered as learnable parameters.
2. **`test_subtractive_inhibition_empty_picks`**:
   - Verify that at step 0 (or when no picks have been made for the active team), the computed `inhibition_penalty` is exactly `0.0`, and the `mlm_logits` are identical to the raw `mlm_head` outputs.
3. **`test_subtractive_inhibition_penalty_monotonicity`**:
   - Design a draft sequence where team 0 picks hero $H$ at step 0. Evaluate step 1 (where team 0 acts again, or team 1 acts).
   - Check that the penalty for candidate $H$ (and candidates similar to $H$) is positive, strictly decreasing their MLM logits.
4. **`test_subtractive_inhibition_causality`**:
   - Verify that altering a pick at step $t$ has zero impact on the `mlm_logits` computed at steps $< t$.

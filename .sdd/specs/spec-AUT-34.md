# Technical Specification: AUT-34 - Inverted Causal Mask in SlotAttentionMLMProjection

## Overview
This specification addresses a critical logic bug in the slot attention query network of the Dota 2 Draft Match Network. Specifically, the causal masking logic in `SlotAttentionMLMProjection.forward` uses the inverted comparison operator `>` instead of `<`. This incorrectly masks past sequence steps while exposing future draft picks during slot occupancy calculation, defeating the causality of the autoregressive next-token prediction module.

We will correct this mask comparison to ensure proper step causality, audit other parts of the network, and strengthen existing test coverage to detect mathematical leakages across time steps.

---

## 1. File Structure Changes
**Files to Modify:**
- `src/dota2drafter/models/match_network.py`
- `tests/test_models/test_slot_attention.py`

**Files to Create/Delete:**
- None

---

## 2. Interfaces & Signatures

### `SlotAttentionMLMProjection.forward`
* **File:** `src/dota2drafter/models/match_network.py`
* **Signature:**
  ```python
  def forward(
      self,
      decoder_output: torch.Tensor,
      x_draft: torch.Tensor,
      patch_ids: torch.Tensor | None = None,
  ) -> torch.Tensor
  ```
* **Modification Details:**
  On line 415, the causal active team masking evaluates sequence index `s` against target index `t`.
  - **Current Line:**
    ```python
    causal_mask = step_indices.unsqueeze(0) > step_indices.unsqueeze(1)  # s < t
    ```
  - **Proposed Fix:**
    ```python
    causal_mask = step_indices.unsqueeze(0) < step_indices.unsqueeze(1)  # s < t
    ```
  This aligns it exactly with the correct implementation found in `HierarchicalTransformer.forward` on line 616.

---

## 3. Edge Cases & Safety Invariants

### 1. Draft Padding and Non-Picks
- The draft tensor contains padded/empty elements (hero ID = `-1.0`) and non-pick steps (bans, etc.).
- These are correctly excluded from slot occupancy attention calculations by the element-wise intersection in `past_active_picks_mask`:
  ```python
  past_active_picks_mask = causal_mask.unsqueeze(0) & same_team & is_pick_s & valid_hero_s
  ```
  This is completely safe and requires no modification.

### 2. First Target Step (`t = 0`)
- When `t = 0`, no past picks exist.
- Under the corrected mask `s < t` (`step_indices.unsqueeze(0) < step_indices.unsqueeze(1)`):
  - Row `t = 0` evaluates to all `False` across the sequence dimension `s`.
  - Thus `past_active_picks_mask[..., 0, :]` is entirely `False`.
  - The masked attention logits for slot occupancy softmax will be correctly filled with `-1e9`.
  - The resulting `unfilled_weight` is `exp(0.0) = 1.0` (all slots fully unfilled), which perfectly mirrors the physical starting state of the draft.

---

## 4. Testing & Verification Strategy

### Root Cause of Test Blindness
The existing `test_causality()` in `tests/test_models/test_slot_attention.py` was blind to this bug because:
1. `_make_draft` populates the entire sequence of 24 steps with active picks.
2. For `t = 8`, the inverted mask `s > t` evaluated steps `[10, 12, 14, 16, 18, 20, 22]` as active, leaking step 10 into step 8 evaluation.
3. The learnable parameters (`slots`) are scaled down by `0.02` during initialization. Since `decoder_output` has standard Gaussian variance (~1.0), the effect of `h_query` (derived from slots) on the final logits is minuscule. The actual logit shift when changing step 10 was `~0.000195`.
4. The test used a relaxed tolerance of `atol=1e-3` (`torch.allclose(..., atol=1e-3)`), allowing the mathematical leak to pass unnoticed.

### Strengthened Test Plan
The `SDD-Implementer` must update `test_causality` to enforce absolute time-step isolation:
- **Tighten Tolerance:** Change the verification assert to use an exact comparison or extremely tight tolerance of `atol=1e-7` (or exact equality):
  ```python
  assert torch.allclose(logits_1[:, :11, :], logits_2[:, :11, :], atol=1e-7)
  ```
  Since any future step should be perfectly masked out with `-1e9`, its contribution to the softmax must be mathematically zero. Changing future steps will result in a difference of exactly `0.0` for past logits, which easily satisfies `atol=1e-7`.
- **Pre-verification:** Running the modified test on the current code must fail, confirming it can reliably catch future leaks.
- **Post-verification:** Once the fix is applied, the test must pass cleanly.

---

## 5. Execution Plan
1. **Branch Checkout:** Verify we are working on the feature branch `sdd/feature-AUT-34`.
2. **Apply the Mask Fix:** Edit line 415 of `src/dota2drafter/models/match_network.py` to change `>` to `<`.
3. **Audit Training and Execution Scripts:** Confirm `scripts/04_train_transformer.py` has no separate mask comparisons.
4. **Apply Test Verification Changes:** Update `tests/test_models/test_slot_attention.py` as detailed above.
5. **Run Test Suite:** Run `pytest tests/test_models/test_slot_attention.py` to verify the fix works and all tests pass perfectly.

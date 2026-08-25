# Technical Specification: AUT-25 - Causal Next-Token Prediction (Revised with Internal Alignment)

## Overview
Transitions the draft sequence policy learning objective from Masked Language Modeling (MLM), which randomly masks 15% of heroes and predicts them, to 100% Causal Next-Token Prediction (NTP). To completely eliminate structural metadata mismatches inside the win probability output head (such as pairing right-shifted hero IDs with unshifted `is_pick`/`team` metadata), the NTP right-shifting operation is encapsulated **internally** within `HierarchicalTransformer.forward()`. This maintains clean, unshifted draft tensors for all external trainer, search, and MCTS code.

---

## 1. Architectural & Sequence Mapping Analysis
The draft sequence features 24 actions indexed $t \in [0, 23]$, represented in PyTorch tensors as shape `(B, 24, 4)` where feature index 2 represents `hero_val` (with `-1.0` as the empty/padding token value).

* **Causal Masking:** The `HierarchicalTransformer` natively implements a strict causal attention mask (`causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()`). The output representation at index $t$ only depends on steps $0 \dots t$. No transformer attention layers need modification.
* **Encapsulated Internal Right-Shift (Step 0 Resolution):**
  To ensure the model learns valid first picks/bans at step 0 ($h_0$) and avoids fatal alignment mismatches inside MCTS, we construct `shifted_x_draft` **internally** within the `HierarchicalTransformer.forward()` by shifting the `hero_val` feature (index 2) right by 1 slot:
  * **Slot 0:** `hero_val` is set to `-1.0` (empty draft prompt for step 0 / BOS).
  * **Slots 1...23:** `hero_val` receives the values from steps $0...22$ ($h_0 \dots h_{22}$).
  * **Metadata Alignment Preservation:** Other features in `shifted_x_draft` (`is_pick`, `team`, `step_index`) remain at their original positions (0 to 23). This guarantees that the Set Transformer or masked global average pooling layers can access clean, unshifted `x_draft` features for exact team/hero/action metadata alignments, avoiding any fatal misalignment bugs.
  * **Sequence Alignment:** 
    * The causal transformer output at sequence position `j` sees a prefix of inputs at positions `0...j`. The `hero_val` inputs at those positions are `[-1.0, h_0, h_1, ... h_{j-1}]`.
    * Thus, the prediction output `mlm_logits[:, j, :]` has causal access to exactly $h_0 \dots h_{j-1}$ and is perfectly aligned to predict target hero $h_j$ at position `j`.

---

## 2. Loss & Optimization Formulation
1. **NTP Cross-Entropy Loss with Internal Alignment:**
   * Pass the clean, unshifted `x_batch` directly to the model:
     `logits, mlm_logits = self.model(x_batch, player_batch, patch_ids=patch_batch)` where `mlm_logits` has shape `(B, 24, num_heroes + 1)`.
   * **Targets:** All 24 steps: `ntp_labels = x_batch[:, :, 2].long()` (shape `(B, 24)`).
   * **Uniform Step Weighting:** Weight loss uniformly across all 24 prediction steps.
   * **Label Smoothing:** Apply Label Smoothing ($\epsilon = 0.10$) directly inside PyTorch's `F.cross_entropy`.
   ```python
   policy_loss = torch.nn.functional.cross_entropy(
       mlm_logits.reshape(-1, mlm_logits.size(-1)),
       ntp_labels.reshape(-1),
       ignore_index=-1,
       label_smoothing=0.10,
   )
   ```
2. **Loss Integration ($\text{PolicyWeight} = 1.0$):**
   * Combine win prediction (Value) loss and NTP (Policy) loss using:
     $$\text{Loss}_{\text{total}} = \text{Loss}_{\text{value}} + 1.0 \times \text{Loss}_{\text{policy}}$$
   * The policy loss multiplier is increased from `0.5` to `1.0`.

---

## 3. Codebase Modifications

### A. Model Architecture Logic (`src/dota2drafter/models/match_network.py`)
* Modify `HierarchicalTransformer.forward` to execute internal right-shifting for the causal policy decoder query embedding sequence:
  ```python
  # 1. Internal Right-Shift for Causal Policy Decoder (NTP Alignment)
  shifted_x_draft = x_draft.clone()
  shifted_x_draft[:, 0, 2] = -1.0           # Step 0 BOS prompt
  shifted_x_draft[:, 1:, 2] = x_draft[:, :-1, 2]  # Right-shift hero IDs

  # Compute joint embeddings using shifted input for causal sequence modeling
  z = self.joint_embedding(shifted_x_draft, patch_ids)  # (B, 24, d_model)
  ```
* For win prediction global average pooling, mask the output using the clean, unshifted `x_draft[:, :, 2]`:
  ```python
  # Win prediction mode: masked global average pooling (uses clean x_draft for exact alignment)
  hero_vals = x_draft[:, :, 2]
  valid_mask = hero_vals != -1.0
  valid_mask = valid_mask.unsqueeze(-1).float()  # (B, 24, 1)

  masked_output = decoder_output * valid_mask  # (B, 24, d_model)
  sum_mask = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).squeeze(-1)  # (B, 1)
  pooled = masked_output.sum(dim=1) / sum_mask  # (B, d_model)

  logits = self.output_head(pooled).squeeze(-1)  # (B,)
  ```

### B. Trainer Logic (`src/dota2drafter/training/transformer_trainer.py`)
* **Remove 15% Masking:** Remove all occurrences of random masking (`torch.rand(...) < 0.15`) in both the training loop (`train`) and validation loop (`_validate`). The trainer passes the clean, unshifted `x_batch` directly to the model.
* **Loss Alignment in Train Loop:**
  * Clean up the mask setup. Simply pass `x_batch` as-is.
  * Compute `policy_loss` via PyTorch's cross entropy with label smoothing $\epsilon=0.10$ over all 24 sequence positions, targeting `ntp_labels = x_batch[:, :, 2].long()`.
  * Compute `total_loss = loss + 1.0 * policy_loss`.
* **Validation Metric Computation (`_validate`):**
  * Pass clean `x_batch` directly to the model.
  * Define NTP target labels: `ntp_labels = x_batch[:, :, 2].long()`.
  * Define validation step mask: `valid_mask = ntp_labels != -1` (evaluates real picks/bans).
  * Compute NTP Top-1 accuracy over valid steps:
    ```python
    preds = mlm_logits.argmax(dim=-1)
    mlm_correct = (preds[valid_mask] == ntp_labels[valid_mask]).sum().item()
    ```
  * Compute NTP Top-5 accuracy over valid steps:
    ```python
    top5_preds = mlm_logits.topk(k=5, dim=-1).indices
    expanded_labels = ntp_labels[valid_mask].unsqueeze(-1)
    mlm_top5_correct = (top5_preds[valid_mask] == expanded_labels).any(dim=-1).sum().item()
    mlm_total = valid_mask.sum().item()
    ```
  * Maintain backward-compatible metrics keys `mlm_accuracy` and `mlm_top5_accuracy` in the returned metrics dictionary.

### C. MCTS Policy Priors (`src/dota2drafter/search/state.py`)
* **Clean Inference Mechanism:**
  During rollouts, when retrieving policy priors at draft step `step_idx` (representing the length of the applied actions: `len(s.actions)`):
  * Pass `base_batch` directly (unshifted) to the model.
  * Since right-shifting is encapsulated inside the model's forward pass, extracting the prior simplifies directly to querying `step_idx` position of the logits:
    ```python
    policy_logits[i] = mlm_logits[i, step_idx, 1:K+1]
    ```

---

## 4. Pre-existing Bug Fixes & Stability
To ensure production readiness, the implementing developer MUST resolve the following pre-existing bugs currently active on `main`:
1. **`ProcessedMatch` missing `patch_id` default:** In `src/dota2drafter/processor/tensor_transformer.py`, add `patch_id: int = 0` as a default parameter for `ProcessedMatch` to resolve failures in tests that instantiate it without `patch_id` (e.g. `test_dataset_builder.py`).
2. **Model output tuple unpacking:** Several unit tests (e.g. `tests/test_models/test_match_network.py` and `tests/test_training/test_integration.py`) call `model(...)` and expect a single tensor output `logits` but receive a tuple `(logits, mlm_logits)`. Update all such tests to unpack the tuple cleanly (`logits, _ = model(...)`).
3. **`TrainingMetrics` class attribute:** Add `best_roc_auc: float = 0.0` or adjust assertion check inside `tests/test_training/test_integration.py` to fix missing attribute errors.

---

## 5. Verification & Test Plan
1. **New NTP Policy Unit Test:** Add `test_ntp_loss_and_priors` inside `tests/test_training/test_transformer_improvements.py` to:
   * Generate an unmasked draft sequence.
   * Forward pass through `HierarchicalTransformer` and manually calculate NTP Cross-Entropy loss over all 24 steps with targets `ntp_labels = x_batch[:, :, 2].long()`.
   * Assert label smoothing is $\epsilon=0.10$ and ignore_index is $-1$.
2. **MCTS Priors Test:** Add assertions inside `tests/test_search.py` to:
   * Verify that `evaluate_batch` correctly extracts the priority logit at position `step_idx` and matches unshifted inference with shifted training.
3. **Execution Check:** Run `python -m pytest` across the test suites to ensure 100% test completion and verify all improvements.

---

## 6. Godot MCP Visual Playtesting Playbook (Not Applicable)
This change is purely algorithmic and mathematical, located entirely inside the ML modeling, training, and search logic layers. There are no UI components, Godot layout coordinate updates, or visual elements.
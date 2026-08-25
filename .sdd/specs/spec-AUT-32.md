# Technical Specification - AUT-32: Fragile Transformer Sequence Masking

## 1. File Structure Changes
* **Modify:** `src/dota2drafter/models/match_network.py`
* **Modify:** `tests/test_models/test_match_network.py`
* **Modify:** `src/dota2drafter/search/state.py` (Optional / Recommended for alignment of dummy padding steps)

## 2. Interfaces & Signatures
There are no changes to external APIs, function signatures, or model architecture parameters. The signature of `HierarchicalTransformer.forward` remains exactly the same:
```python
def forward(
    self,
    x_draft: torch.Tensor,
    player_pref_vectors: torch.Tensor,
    patch_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

## 3. Implementation Details & Rationale
Currently, the `pad_mask` is derived in `HierarchicalTransformer.forward` using:
```python
pad_mask = (x_draft[:, :, 3] == 0.0)
if pad_mask.size(1) > 0:
    pad_mask[:, 0] = False
```
This conflates valid draft steps at step 0 (which has `step_index == 0.0`) with padded/ignored slots, necessitating a fragile override `pad_mask[:, 0] = False`. If step 0 is genuinely unmade (e.g., at the start of an evaluation), this override erroneously unmasks empty/invalid data.

### Proposed Fix:
1. **Derive pad_mask via Hero Index Sentinel:**
   Replace the `step_index` check with the `hero_val` sentinel (`-1.0`) at column index 2 of `x_draft`:
   ```python
   pad_mask = (x_draft[:, :, 2] == -1.0)
   ```
   This is highly robust because:
   - A step with an unmade or padded hero will have its hero value (column 2) set to `-1.0`.
   - A step with a selected hero will have `hero_val >= 0.0`, regardless of whether it is step 0 or any other step.
   - We completely eliminate the need for the fragile `pad_mask[:, 0] = False` override, perfectly aligning with the "Error Handling" design philosophy of making misuse difficult and defining away invalid states.

2. **Align rollout padding sentinel `_ZERO_STEP`:**
   In `src/dota2drafter/search/state.py`, update `_ZERO_STEP` to have `-1.0` as its hero index (column 2) to match the expected unmade sentinel:
   ```python
   _ZERO_STEP = torch.tensor([0.0, 0.0, -1.0, 0.0], dtype=torch.float32)
   ```

## 4. Edge Cases
* **Step 0 of a valid draft is unmasked:** If a draft has a valid pick at step 0 (e.g., `hero_val >= 0`), `x_draft[:, 0, 2] == -1.0` is `False`, so it will NOT be masked. This is the correct behavior.
* **Step 0 of an unmade draft is masked:** If the draft is completely empty and step 0 is unmade (`hero_val == -1.0`), `x_draft[:, 0, 2] == -1.0` is `True`, so step 0 will be correctly masked.
* **Batch with mixed lengths:** Each batch sequence will be masked independently based on whether each step contains a valid hero or the `-1.0` sentinel.
* **Device mismatch:** The generated `pad_mask` must be explicitly moved to the device of the target tensor `tgt` via `pad_mask.to(tgt.device)`.

## 5. Testing Strategy
We must update the existing unit tests and add new ones to verify the correct masking behavior under all conditions.

### Test Updates in `tests/test_models/test_match_network.py`:
In `test_hierarchical_transformer_tgt_key_padding_mask()`:
Update the truncation of sample 0 so that truncated steps have `hero_val = -1.0` (column 2):
```python
    # 2. Truncate sample 0 at step 12
    # In sample 0, steps 12..23 are set to 0.0, and hero_val (column 2) is set to -1.0
    x_draft[0, 12:, :] = 0.0
    x_draft[0, 12:, 2] = -1.0
```

### New Test Cases to Add in `tests/test_models/test_match_network.py`:
Create a new test function `test_hierarchical_transformer_step0_unmasking()`:
1. **Case A: Genuinely empty draft (B=1, seq_len=24, all hero indices are -1.0).**
   - Assert that the derived `pad_mask` has `True` (masked) at all indices, including step 0.
2. **Case B: Valid draft starting with step 0 made (hero_val >= 0), and step 1..23 unmade (-1.0).**
   - Assert that `pad_mask` is `False` at index 0 and `True` at indices 1..23.

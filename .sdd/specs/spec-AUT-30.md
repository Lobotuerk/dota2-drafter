# Technical Specification: AUT-30 - Truncation Padding Index Corruption

## Overview
During transformer training and MCTS rollout evaluation, truncated draft sequences and padding steps are zero-padded using a value of `0.0`. However, in the (24, 4) draft sequence representation, column 2 represents the Hero ID. A value of `0.0` is treated as a valid hero ID (Hero 0) rather than a padding indicator by `JointEmbedding` and downstream metric evaluation. 

During validation, the `_validate()` loop filters out valid tokens using `valid_mask = ntp_labels != -1`. Because padded steps are assigned `0.0` rather than `-1.0`, they are treated as real draft picks, forcing the model to predict Hero index 0, corrupting MLM Top-1/Top-5 accuracy metrics, and skewing predictions.

This specification corrects the sequence padding across all three padding locations by explicitly assigning the `-1.0` sentinel to column 2 (Hero ID) for padded steps.

---

## 1. File Structure Changes
The implementation will modify the following existing files:
- **`src/dota2drafter/training/transformer_trainer.py`**: Update prefix truncation and dataset augmentation padding.
- **`src/dota2drafter/search/state.py`**: Update search rollout padding step representation.
- **`tests/test_training/test_transformer_improvements.py`**: Update unit tests to verify `-1.0` sentinel padding for prefix truncation.
- **`tests/test_search.py`**: Add a unit test to verify the `_ZERO_STEP` sentinel value.

---

## 2. Detailed Implementation Design

### A. Prefix Truncation Helper Update
**File**: `src/dota2drafter/training/transformer_trainer.py`
**Function**: `apply_prefix_truncation()`

The sequence truncation zeroing logic currently sets the entire step slice to `0.0`:
```python
        if t < 24:
            x_truncated[t:, :] = 0.0
```
This will be updated to:
```python
        if t < 24:
            x_truncated[t:, :] = 0.0
            x_truncated[t:, 2] = -1.0  # Assign explicit hero padding sentinel
```

### B. PlayerComfortDataset Augmentation Update
**File**: `src/dota2drafter/training/transformer_trainer.py`
**Class**: `PlayerComfortDataset` (inside `__init__`)

The precomputed augmentations block currently zero-pads the truncated portion:
```python
                    for t in truncation_points:
                        x_truncated = x_permuted.clone()
                        x_truncated[t:, :] = 0.0
                        match_augmentations.append((x_truncated, player_comfort, y_label, patch_id))
```
This will be updated to:
```python
                    for t in truncation_points:
                        x_truncated = x_permuted.clone()
                        x_truncated[t:, :] = 0.0
                        x_truncated[t:, 2] = -1.0  # Assign explicit hero padding sentinel
                        match_augmentations.append((x_truncated, player_comfort, y_label, patch_id))
```

### C. MCTS Rollout Zero Step Update
**File**: `src/dota2drafter/search/state.py`
**Variable**: `_ZERO_STEP`

The dummy step variable for rollout padding currently initializes all columns to `0.0`:
```python
_ZERO_STEP = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
```
This will be updated to:
```python
_ZERO_STEP = torch.tensor([0.0, 0.0, -1.0, 0.0], dtype=torch.float32)  # Fix rollout padding
```

---

## 3. Interfaces & Signatures
No changes to external interfaces, function signatures, or classes. The modifications strictly correct internal tensor-padding values.

---

## 4. Edge Cases & Safety
- **Column Consistency**: Column 2 represents Hero ID consistently across the workspace. We must ensure no other index is offset during slice assignment.
- **Sentinel Robustness**: The `-1.0` sentinel is safely routed to the dedicated padding embedding (index 127) within `JointEmbedding` and is automatically excluded from cross-entropy loss and metric evaluation via `ntp_labels != -1` masking. No new masking logic or downstream evaluation updates are necessary.

---

## 5. Testing & Verification Strategy

### Unit Test Updates (`tests/test_training/test_transformer_improvements.py`)
Update the existing `test_apply_prefix_truncation()` unit test to verify that the prefix truncation assigns `-1.0` to column 2 of padded steps, while keeping other columns zeroed.
- **Replace**:
  ```python
  # Everything from index t to 23 should be 0.0
  assert torch.all(x_truncated[t:, :] == 0.0)
  ```
- **With**:
  ```python
  # Column 2 must be -1.0, other columns must be 0.0
  assert torch.all(x_truncated[t:, 2] == -1.0)
  assert torch.all(x_truncated[t:, [0, 1, 3]] == 0.0)
  ```

### New Search State Unit Test (`tests/test_search.py`)
Add a new unit test `test_zero_step_padding_value()` to verify that the `_ZERO_STEP` tensor contains the proper sentinel value.
```python
def test_zero_step_padding_value():
    """Verify that _ZERO_STEP has hero ID set to -1.0 and other elements set to 0.0."""
    from dota2drafter.search.state import _ZERO_STEP
    assert _ZERO_STEP[2] == -1.0
    assert _ZERO_STEP[0] == 0.0
    assert _ZERO_STEP[1] == 0.0
    assert _ZERO_STEP[3] == 0.0
```

---

## Execution Plan
The Coder (`SDD-Implementer`) will:
1. Apply the padding corrections in `transformer_trainer.py` and `state.py`.
2. Update and execute the test suite (via `pytest tests/test_training/test_transformer_improvements.py` and `pytest tests/test_search.py`) to confirm correctness and prevent regressions.

# Technical Specification: AUT-33 - Omitted Patch Conditioning During Inference

## Overview
During model inference in `scripts/04_train_transformer.py` (predict mode), `model.predict_proba(x_draft, player_comfort)` is invoked without passing `patch_ids`. This bypasses the FiLM modulation layers (`gamma` and `beta` embeddings) in `JointEmbedding`, leading to a critical feature distribution mismatch between training (where FiLM conditioning is active) and inference (where it is bypassed).

Additionally, a critical backend bug was discovered during the dependency/test audit: a `NameError: name 'batch_size' is not defined` exists in `HierarchicalTransformer.forward` (line 641 in `src/dota2drafter/models/match_network.py`), which causes all model forward passes and tests to fail.

This specification addresses both the prediction mode patch conditioning omission and the `NameError` to restore model correctness and clean test passes.

---

## 1. File Structure Changes
The implementation will modify the following existing files:
- **`scripts/04_train_transformer.py`** (prediction loop)
- **`src/dota2drafter/models/match_network.py`** (`HierarchicalTransformer.forward` block)

No files will be created or deleted.

---

## 2. Interfaces & Signatures

### A. Match Network / Hierarchical Transformer Bugfix
**File**: `src/dota2drafter/models/match_network.py`
Inside `HierarchicalTransformer.forward`:
- **Context**: The `batch_size` variable is used on line 641 to expand `all_hero_indices`:
  ```python
  all_hero_indices = torch.arange(
      self.num_heroes + 1, device=x_draft.device
  ).unsqueeze(0).expand(batch_size, -1)
  ```
  However, `batch_size` is never defined in this method scope.
- **Change**: Define `batch_size` at the beginning of the `forward` function using `x_draft.shape[0]`.
  ```python
  def forward(
      self,
      x_draft: torch.Tensor,
      player_pref_vectors: torch.Tensor,
      patch_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
      """Forward pass through the Match Network."""
      batch_size = x_draft.shape[0]
      ...
  ```

### B. Prediction Script Fix
**File**: `scripts/04_train_transformer.py`
Inside the prediction block of `main()` (under `elif args.mode == "predict"`):
- **Context**: `patch_ids` are loaded from `load_data(args.data_dir)` but never passed into `model.predict_proba`.
- **Change**: Extract the corresponding patch ID for match `i` from the loaded `patch_ids` list and pass it as a tensor of shape `(1,)` to `predict_proba`.
  ```python
  prob = model.predict_proba(x_draft, player_comfort)
  ```
  should be changed to:
  ```python
  # Build player comfort tensor for this match
  ...
  
  match_patch_id = None
  if patch_ids is not None:
      match_patch_id = torch.tensor([patch_ids[i]], dtype=torch.long, device=device)

  prob = model.predict_proba(x_draft, player_comfort, patch_ids=match_patch_id)
  ```

---

## 3. Edge Cases

### A. Missing Patch IDs in Inference Data
If `patch_ids` is `None` (for example, if dataset batches lack patch metadata), `match_patch_id` must remain `None`. The models and `predict_proba` method are designed to handle optional `patch_ids` gracefully via the default `patch_ids=None` condition.

### B. Batch Size Bound & Device Mapping
- During prediction, inference is run sample-by-sample (batch size = 1). The patch tensor is thus wrapped as `[patch_ids[i]]` to yield a `(1,)` shape, ensuring shape alignment with the batch size.
- The `match_patch_id` tensor must be instantiated on the active `device` to prevent execution device mismatch errors.

---

## 4. Testing Strategy

### A. Unit & Integration Test Suite Verification
Fixing the `NameError` in `match_network.py` will restore the behavior of the full test suite.
The Implementer MUST run:
```bash
PYTHONPATH=src pytest
```
All 22 failing tests in `tests/test_models/test_match_network.py` and `tests/test_training/test_integration.py` must pass successfully.

### B. Add a Dedicated Test Case for Inference Conditioning
The Implementer MUST add a new test case in `tests/test_models/test_match_network.py` to assert that:
1. `predict_proba` returns valid probabilities when `patch_ids` is passed.
2. Passing different `patch_ids` results in different win probabilities due to FiLM modulation.

**Example Test Structure**:
```python
def test_match_network_predict_proba_with_patch_ids():
    """Test MatchNetwork predict_proba modulates output based on patch_ids."""
    d_model = 64
    num_heroes = 120
    player_input_dim = 10
    h_gnn = torch.randn(num_heroes + 1, d_model)

    model = MatchNetwork(
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        num_heroes=num_heroes,
        player_input_dim=player_input_dim,
        h_gnn=h_gnn,
        num_patches=10,
    )

    x_draft = torch.zeros(1, 24, 4)
    # Populate sequence
    x_draft[0, :, 2] = torch.arange(24) + 1
    x_draft[0, :, 0] = 1.0
    x_draft[0, :, 3] = torch.arange(24).float()

    player_comfort = torch.randn(1, 10, player_input_dim)

    # Predict with patch 1
    patch_1 = torch.tensor([1], dtype=torch.long)
    proba_1 = model.predict_proba(x_draft, player_comfort, patch_ids=patch_1)

    # Predict with patch 2
    patch_2 = torch.tensor([2], dtype=torch.long)
    proba_2 = model.predict_proba(x_draft, player_comfort, patch_ids=patch_2)

    assert proba_1.shape == (1,)
    assert proba_2.shape == (1,)
    # Verify outputs are valid probabilities
    assert 0 <= proba_1.item() <= 1
    assert 0 <= proba_2.item() <= 1
    # Verify different patch IDs modulate the outputs
    assert not torch.allclose(proba_1, proba_2)
```

### C. Validation of the Training/Prediction Script
The Implementer MUST run:
```bash
python scripts/04_train_transformer.py --mode predict
```
This script must run to completion, loaded checkpoint successfully, evaluate the 5 prediction matches, and log output like:
```
Running prediction on X matches...
  Match 0: win probability = X.XXXX
  Match 1: win probability = X.XXXX
  ...
```

---

## Execution Plan
The `SDD-Implementer` will:
1. Checkout the feature branch `sdd/feature-AUT-33`.
2. Apply the surgical fixes to `src/dota2drafter/models/match_network.py` and `scripts/04_train_transformer.py`.
3. Add the integration test case and run the test suite to ensure 100% green pass.
4. Run the prediction script to visually/behaviorally confirm the fix.

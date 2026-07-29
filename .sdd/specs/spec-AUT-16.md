### 📋 Technical Specification

#### 1. Overview
This specification addresses two codebase improvements:
1. Fixing a performance bottleneck in `PlayerComfortDataset` by pre-computing dataset augmentations during `__init__` rather than generating them dynamically on the fly in `__getitem__`.
2. Resolving a mathematical conflict in `HeroRGCN` by removing `nn.LayerNorm` layers, which are counterproductive due to the immediate application of L2 normalization.

#### 2. `PlayerComfortDataset` Refactoring
**File:** `src/dota2drafter/training/transformer_trainer.py`

**Changes:**
- Modify `PlayerComfortDataset.__init__` to pre-compute and store all samples in a flat list `self.samples` (a list of tuples: `(x_draft, player_comfort, y_label)`).
- If `augment=False`, populate `self.samples` by iterating over `self.x_drafts` and computing the basic sample (using `self._build_player_comfort(idx)`).
- If `augment=True`, iterate over all base drafts (`for base_idx in range(len(self.x_drafts)):`). For each base draft:
  - Call `augment_draft_permutations(...)` to generate up to 64 permutations.
  - Loop `perm_idx` from 0 to 63:
    - If `perm_idx < len(perm_samples)`, extract `x_permuted`, `player_comfort`, and `y_label` from `perm_samples[perm_idx]`.
      - Append `(x_permuted, player_comfort, y_label)` to `self.samples` (for `trunc_idx == 0`).
      - For each truncation point `t` in `[6, 8, 11, 17, 21, 23]`, clone `x_permuted`, set `x_truncated[t:, :] = 0.0`, and append `(x_truncated, player_comfort, y_label)` to `self.samples`.
    - If `perm_idx >= len(perm_samples)`, retrieve the basic sample using `self._get_basic_sample(base_idx)` and append it 7 times to `self.samples` to maintain the 448 length multiplier per base match.
- Modify `__len__` to simply return `len(self.samples)`.
- Modify `__getitem__` to simply return `self.samples[idx]`.
- Remove `_get_augmented_sample` method entirely, as its logic is now housed in `__init__`.
- The `_get_basic_sample` and `_build_player_comfort` methods can remain as helpers used during `__init__`. Note: Make sure `self.radiant_players`, `self.dire_players`, etc., are assigned before calling these helpers.

#### 3. `HeroRGCN` Refactoring
**File:** `src/dota2drafter/embeddings/rgcn.py`

**Changes:**
- **In `__init__`**:
  - Remove the initialization of `ln_layers`. Do not append `nn.LayerNorm` to any lists.
  - Remove the assignment `self.ln_layers = nn.ModuleList(ln_layers)`.
- **In `forward`**:
  - Remove the line `h = self.ln_layers[i](h)`.
  - The order of operations inside the `for i, rgcn_layer in enumerate(self.rgcn_layers):` loop should be:
    1. `h_out = rgcn_layer(h, edge_index, edge_type)`
    2. Residual connection: `h = h_out + h_in` (if dimensions match), else `h = h_out`
    3. Activation: `h = self.activation(h)` (only if `i < len(self.rgcn_layers) - 1`)
    4. L2 Normalization: `h = F.normalize(h, p=2, dim=1)`
- **In `load`**:
  - Since backward compatibility is not required, no changes are needed for handling older state dicts with `ln_layers`. A strict load is acceptable.

#### 4. Testing & Validation
- No new regression tests are required.
- The implementer must ensure the project's existing tests pass (`pytest tests/test_training/test_transformer_improvements.py` and `pytest tests/test_embeddings/test_rgcn.py`). 
- If existing tests mock or assert the presence of `ln_layers` or its parameters, they should be updated to remove those assertions.
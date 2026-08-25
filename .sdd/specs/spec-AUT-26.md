# Technical Specification: AUT-26 - MatchNetwork Optimization: PMA Weight Overloading Fix and Dead Code Removal

## Overview
Optimizes `MatchNetwork`'s `SetTransformerHead` by fixing a parameter collision/weight overloading issue and removes dead code (`set_attention_block` in `JointEmbedding`) in `match_network.py`.

## 1. Dead Code Removal
**Files**: `src/dota2drafter/models/match_network.py`
- In `JointEmbedding` class, remove the dead method `set_attention_block` (originally lines 222–233):
  ```python
  def set_attention_block(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
      """Set Attention Block (SAB) - permutation-equivariant self-attention without positional encodings."""
      attn = nn.MultiheadAttention(self.d_model, num_heads=4, batch_first=True, dropout=0.1)
      return attn(x, x, x, key_padding_mask=key_padding_mask)[0]
  ```
- This method is never invoked in the codebase and instantiates `nn.MultiheadAttention` inside the forward pass, creating untrained/random weights on every execution. Removing it simplifies the class and cleans up the architecture.

## 2. PMA Weight Overloading Fix in SetTransformerHead
**Files**: `src/dota2drafter/models/match_network.py`
- **Background**: Currently, in `SetTransformerHead`, the intra-team synergy block `self.sab` (which performs self-attention among 5 hero embeddings) is also reused to perform Pooling by Multihead Attention (PMA) on `seed_r` and `seed_d` (which projects a single seed token over the 5 cross-attended team representations). This forces `self.sab`'s query, key, and value linear projection weights ($W_q, W_k, W_v$) to serve two completely incompatible mathematical and semantic roles, leading to a parameter collision and severe optimization bottleneck.
- **Initialization Changes**:
  In `SetTransformerHead.__init__` (around line 65), instantiate two dedicated Multihead Attention blocks specifically for PMA pooling:
  ```python
  # Dedicated PMA modules for Radiant and Dire sets
  self.pma_r = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
  self.pma_d = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
  ```
- **Forward Pass Changes**:
  In `SetTransformerHead.forward` (around lines 159–160), replace the `self.sab` calls for Pooling by Multihead Attention with the new dedicated PMA modules:
  ```python
  v_r, _ = self.pma_r(seed_r, r_cross, r_cross, key_padding_mask=safe_r_pad_mask)
  v_d, _ = self.pma_d(seed_d, d_cross, d_cross, key_padding_mask=safe_d_pad_mask)
  ```

## 3. Verification Plan
**Files**: `tests/test_models/test_match_network.py`
- The `SDD-Implementer` must verify that all 10 existing unit tests pass successfully by running:
  ```bash
  pytest tests/test_models/test_match_network.py
  ```
- This ensures that there are no shape mismatches or runtime exceptions introduced by using dedicated PMA multihead attention modules.

## Execution Plan
The `SDD-Implementer` will checkout the `sdd/feature-AUT-26` branch, apply the changes in `src/dota2drafter/models/match_network.py`, run `pytest` to verify the execution of tests, and push the verified changes.

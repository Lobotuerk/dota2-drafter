# Technical Specification: Win Probability Calculation Overhaul (DeepSets)

## 1. Overview
The current Value Head computes win probabilities by taking a global average pool across all 24 draft sequence steps (including both picks and bans) and passing the pooled vector into a 2-layer MLP. This approach fails to capture complex non-linear hero synergies and inter-team counters, and it inappropriately incorporates the presence of banned heroes.

This specification details the replacement of the existing pooled `output_head` with a **Set Transformer Head**. This new parallel pathway evaluates the outcome of a draft by treating the teams (Radiant and Dire) as disjoint sets of selected heroes. By using permutation-invariant set attention (SAB, Cross-Set Attention, and PMA), the Set Transformer accurately models cross-team counters and team synergies independent of the draft order, greatly simplifying the MCTS search space manifold.

## 2. Architecture Changes

### 2.1. Feature Representation (`JointEmbedding`)
The Set Transformer requires pure hero representations modulated by patch context, without sequence step, action type, or team positional encodings.
- Add a new method to `JointEmbedding` (in `src/dota2drafter/models/match_network.py`):
  `def get_pure_hero_embeddings(self, hero_indices: torch.Tensor, patch_ids: torch.Tensor | None = None) -> torch.Tensor`
  This method will:
  1. Retrieve `self.h_gnn[hero_indices]`.
  2. Apply `self.project(...)`.
  3. Apply FiLM conditioning if `patch_ids` is provided (using `self.w_patch`, `self.film_gamma`, and `self.film_beta`).
  4. Return the resulting pure hero representations `(B, seq_len, d_model)`.

### 2.2. Set Transformer Head (`SetTransformerHead`)
Introduce a new PyTorch module `SetTransformerHead` inside `match_network.py` to replace the `output_head`.
- **Inputs**: Pure hero embeddings `e_h`, raw draft sequences `x_draft` (to extract team and pick/ban flags), and `d_model`.
- **Set Construction**:
  - Filter `x_draft` to extract only valid picks (`action == 1` and `hero_val != -1.0`).
  - Separate into Radiant ($R$, where `team == 0`) and Dire ($D$, where `team == 1`).
  - Pack $R$ and $D$ into padded tensors of shape `(B, 5, d_model)` with corresponding boolean `key_padding_mask` tensors of shape `(B, 5)`. 
  *(Missing hero slots—such as in partial draft states—must be zero-padded and their padding masks set to `True` so attention mechanisms ignore them).*
- **Set Attention Block (SAB)**:
  - Utilize standard PyTorch `nn.MultiheadAttention` with `batch_first=True`.
  - Compute internal team synergies:
    - $R_{syn} = \text{MHA}(Q=R, K=R, V=R, \text{key\_padding\_mask}=R_{mask})$
    - $D_{syn} = \text{MHA}(Q=D, K=D, V=D, \text{key\_padding\_mask}=D_{mask})$
- **Cross-Set Attention**:
  - Compute counter-picks / cross-team dynamics symmetrically:
    - $R_{cross} = \text{MHA}(Q=R_{syn}, K=D_{syn}, V=D_{syn}, \text{key\_padding\_mask}=D_{mask})$
    - $D_{cross} = \text{MHA}(Q=D_{syn}, K=R_{syn}, V=R_{syn}, \text{key\_padding\_mask}=R_{mask})$
- **Pooling by Multihead Attention (PMA)**:
  - Define two learnable parameter seeds `S_R` and `S_D` (each of shape `(1, 1, d_model)`).
  - Pool variable-sized sets into fixed 1D representations:
    - $v_R = \text{MHA}(Q=S_R.expand(B, 1, d_model), K=R_{cross}, V=R_{cross}, \text{key\_padding\_mask}=R_{mask})$
    - $v_D = \text{MHA}(Q=S_D.expand(B, 1, d_model), K=D_{cross}, V=D_{cross}, \text{key\_padding\_mask}=D_{mask})$
  - Output shape of $v_R$ and $v_D$ becomes `(B, 1, d_model)`. Flatten these to `(B, d_model)`.
- **Value MLP**:
  - Concatenate the pooled team vectors: $[v_R \,||\, v_D]$ of shape `(B, 2 * d_model)`.
  - Pass through an MLP:
    - `Linear(2 * d_model, dim_feedforward)`
    - `ReLU()`
    - `Dropout(dropout)`
    - `Linear(dim_feedforward, 1)` -> outputs raw un-sigmoid logits of shape `(B,)`.

### 2.3. Hierarchical Transformer Updates
- **Initialization**: Remove the old `self.output_head = nn.Sequential(...)` in `HierarchicalTransformer.__init__`. Instantiate `self.set_transformer_head = SetTransformerHead(...)` instead.
- **Forward Pass**:
  - The causal Transformer decoder computes `mlm_logits` as usual.
  - Call `self.joint_embedding.get_pure_hero_embeddings(...)` on `x_draft[:, :, 2]`.
  - Pass the pure hero embeddings alongside `x_draft` into `self.set_transformer_head` to compute the win probability logits.
  - Return `(logits, mlm_logits)`.

## 3. Training Modifications
- **Loss Rebalancing**: The Set Transformer completely separates value estimation from sequence modeling, providing a more robust signal. Thus, we will re-weight the auxiliary task.
- Modify `src/dota2drafter/training/transformer_trainer.py` in the `train()` loop:
  - Locate `total_loss = loss + (0.5 * mlm_loss)` around line 663.
  - Change to `total_loss = loss + (1.0 * mlm_loss)`.
- Since the intra-phase permutation augmentation is orthogonal and mathematically consistent with the new permutation-invariant Value Head, the `augment = True` settings in the configuration remain unchanged.

## 4. Inference Considerations
- Standard PyTorch `nn.MultiheadAttention` operates with very low latency. Because the Set Transformer evaluates sets of maximum size 10 (5 Radiant + 5 Dire), overhead is negligible compared to causal sequence generation.
- Ensure `predict_proba` correctly passes the un-normalized logits through `torch.sigmoid()` as it currently does.

## 5. Summary of Files Affected
- `src/dota2drafter/models/match_network.py`:
  - Update `JointEmbedding` to add pure hero extraction logic.
  - Add `SetTransformerHead` module.
  - Update `HierarchicalTransformer` to swap out the pooling logic and wire in the new head.
- `src/dota2drafter/training/transformer_trainer.py`:
  - Update loss weight multiplier from `0.5` to `1.0`.
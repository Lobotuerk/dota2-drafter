# Technical Specification: Transformer Improvements (AUT-15)

## 1. Overview
This specification details the implementation of four key data and modeling improvements to the Transformer pipeline:
1. Masked Draft Modeling (MLM) Pre-training
2. Multi-Prefix Sequence Crop (Prefix Training)
3. Intra-Phase Draft Permutations (Domain Augmentation)
4. Label Smoothing for Outcome Noise

These changes will reside primarily in `src/dota2drafter/models/match_network.py` and `src/dota2drafter/training/transformer_trainer.py`, with dataset augmentation logic introduced to the preprocessing pipeline or `PlayerComfortDataset`.

## 2. Masked Draft Modeling (BERT-Style MLM)
### 2.1 Model Changes (`match_network.py`)
- **Add MLM Head:** In `HierarchicalTransformer.__init__`, add `self.mlm_head = nn.Linear(d_model, num_heroes + 1)`.
- **Forward Signature Update:** Add a flag `mlm_mode: bool = False` to `forward()`.
  - If `mlm_mode` is True: bypass the masked global average pooling. Pass the full decoder output `(B, 24, d_model)` to `self.mlm_head` to get `(B, 24, num_heroes + 1)` logits and return them.
  - If False: execute the existing win probability path (pooling + output MLP head).

### 2.2 Pre-training Loop (`transformer_trainer.py`)
- **New Trainer / Mode:** Create an `MLMTrainer` class (or add an `mlm_train` method to `TransformerTrainer`).
- **Masking Logic:** 
  - Create a masking collator or augment `__getitem__`.
  - With 15 percent probability per valid hero action, replace `x_draft[..., 2]` (hero index) with `0` (the `[MASK]` token).
  - Track the ground truth unmasked indices to compute the standard Cross-Entropy loss.
- **Two-Stage Training Flow:**
  - Stage 1: Pre-train the model using MLM loss on the entire dataset.
  - Stage 2: Freeze the Transformer layers (and `h_gnn`/joint embedding). Train only `output_head` for win prediction using the smoothed BCE loss.

## 3. Multi-Prefix Sequence Crop (Prefix Training)
### 3.1 Dataset Augmentation
- **Where to Apply:** Add an augmentation utility called from `PlayerComfortDataset.__init__` or dynamically within `__getitem__`.
- **Truncation Points:** Extract partial drafts truncated at steps 6 (Phase 1 finish), 12 (Phase 2 mid), 18 (Phase 2 finish), and 24 (completed).
- **Truncation Method:** For a truncation at step `t`:
  - Keep steps `0` to `t-1` intact.
  - Zero out steps `t` to `23` (hero index 0, action type 0, etc.) to represent an incomplete state.
  - The final target `y` remains the actual match outcome.
- **Impact:** Multiplies the effective sequence dataset size by 4x.

## 4. Intra-Phase Draft Permutations (Domain Augmentation)
### 4.1 Permutation Logic
- Captains Mode phases define sets of consecutive picks/bans for the same team that are functionally order-invariant.
- **Implementation:** Create an `augment_draft_permutations(x_draft)` function.
  - Define phase boundaries.
  - Identify independent swaps (e.g., Radiant Ban 1 and Radiant Ban 2 in Phase 1).
  - Generate 1 to 2 random valid permutations per original match.
  - Append these augmented tensors to the dataset lists before feeding them to the DataLoader.

## 5. Label Smoothing for Outcome Noise
### 5.1 Training Adjustment (`transformer_trainer.py`)
- **Parameterization:** Add `label_smoothing_eps: float = 0.15` to `TrainingConfig`.
- **Target Smoothing:** In the `TransformerTrainer.train` training loop, prior to calculating the loss:
  `eps = self.config.label_smoothing_eps`
  `y_batch_smoothed = y_batch * (1.0 - eps) + (eps / 2.0)`
  `loss = self.criterion(logits, y_batch_smoothed)`
- **Validation:** Retain original binary targets (0 or 1) for computing validation ROC-AUC and Accuracy to ensure metrics reflect real performance accurately.

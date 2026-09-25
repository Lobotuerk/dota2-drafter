# Technical Specification: Decoupled Two-Stage Training Pipeline

## 1. Overview
This specification outlines the decoupling of the `MatchNetwork` training into a two-stage process. Stage 1 focuses on pre-training and calibrating the Value Head (SetTransformerHead) using both high-MMR pub games and draft games. Stage 2 freezes the Value Head and trains the Policy Head (Hierarchical causal transformer) using Advantage-Weighted Masked Language Modeling (AW-MLM), evaluating metrics on positive advantage steps.

## 2. Architecture & Design

### Stage 1: Value Head Pre-training and Calibration
- **Data Loading:**
  - Introduce loading support for high-MMR pub games (`games_batch_*.pt`) alongside draft games (`drafts_batch_*.pt`).
- **Sequential Training:**
  - **Phase 1 (Pre-train):** Train the `SetTransformerHead` and its projections in `JointEmbedding` on the high-MMR pub dataset. Optimize using standard Binary Cross-Entropy (BCE) loss.
  - **Phase 2 (Fine-tune):** Fine-tune the weights on draft games. Apply a sample weight multiplier of 5.0 to draft games during loss computation.
  - **FiLM Embedding:** Ensure `patch_ids` are passed through the FiLM embedding layer within `JointEmbedding` during both phases (this is currently implemented via `get_pure_hero_embeddings`, but must be strictly maintained in the isolated Value Head pass).
- **Metrics & Early Stopping:**
  - Track both ROC-AUC and Brier Score at each validation epoch.
  - Early stopping will monitor patience on minimizing the validation Brier Score, ensuring ROC-AUC does not significantly degrade.
- **Post-Hoc Calibration:**
  - After fine-tuning converges, freeze the network and fit a single scalar temperature parameter (Platt scaling) using L-BFGS on the validation logits to directly minimize the validation Brier Score (`F.mse_loss(torch.sigmoid(logits / T), targets)`). Store this calibrated temperature for use in estimating advantage during Stage 2.

### Stage 2: Policy Head Training
- **Freezing:**
  - Freeze the entirety of the `SetTransformerHead` and its corresponding value projection paths in `JointEmbedding`.
- **Training:**
  - Train the `HierarchicalTransformer` and `SlotAttentionMLMProjection` (Policy Head) strictly on the draft games trajectories.
  - Utilize Advantage-Weighted Masked Language Modeling (AW-MLM) for the policy cross-entropy loss, using the frozen, calibrated Value Head to estimate step-wise advantage.
- **Validation & Checkpointing:**
  - Redefine the validation tracking for top-5 match accuracy: it must be conditioned on positive advantage.
  - Specifically, compute the signed marginal advantage `\delta_t = \sigma(t) \cdot (V(s_t) - V(s_{t-1}))` at each step inside `_validate()`, where `\sigma(t) = +1.0` for Radiant steps and `-1.0` for Dire steps.
  - Only evaluate next-token prediction accuracy and top-5 accuracy on draft steps where this signed estimated advantage is strictly positive (`\delta_t > 0`).
  - Checkpointing criteria will converge on this positive-advantage-conditioned top-5 match accuracy.

## 3. Implementation Steps

1. **Configuration Updates (`src/dota2drafter/config.py` & `config.yaml`):**
   - Add pipeline stages configuration (`stage: 1` vs `stage: 2`).
   - Add sample weight configuration for fine-tuning (`draft_sample_weight: 5.0`).
   - Add paths for pub games dataset directory.

2. **Trainer Refactoring (`src/dota2drafter/training/transformer_trainer.py`):**
   - Separate `total_loss` computation based on the active stage.
   - **Stage 1 Logic:**
     - Implement sequential `train_value_head()`:
       - Loop 1: Train on `pubs_loader` (BCE Loss).
       - Loop 2: Train on `drafts_loader` (BCE Loss * 5.0).
       - Save best checkpoint based on `val_brier_score` rather than `val_auc` or `val_loss`.
     - Implement `calibrate_temperature()`: Setup an `nn.Parameter` initialized to 1.0, optimize via `torch.optim.LBFGS` against validation labels using MSE loss on the sigmoid outputs. Attach the learned `temperature` to the model state.
   - **Stage 2 Logic:**
     - Load Stage 1 checkpoint.
     - Set `requires_grad = False` for `SetTransformerHead` and value projections in `JointEmbedding`.
     - Execute standard AW-MLM loop calculating policy losses only.
   - **Validation Update (`_validate`):**
     - In Stage 2, calculate step-wise advantages using the frozen Value Head and the calibrated temperature.
     - Apply mask `valid_mask & (advantage > 0)` before summing hits for top-1 and top-5 accuracy metrics.

3. **Data Loading (`scripts/04_train_transformer.py`):**
   - Add CLI args for `--stage`, `--pub_data_dir`.
   - Conditionally load pub games (`games_batch_*.pt`) and construct datasets based on the requested stage. Ensure phase transition is handled cleanly.

## 4. Design Philosophy Considerations
- **High Modular Depth:** The trainer class should expose distinct `train_stage_1()` and `train_stage_2()` methods rather than overloading a single loop with complex branching and boolean flags.
- **Clean Exception Paths:** Handle missing pub game batches gracefully (log warning and proceed/abort clearly).
- **Interface Segregation:** The `MatchNetwork` interface remains clean; stage behaviors are strictly managed by the `TransformerTrainer` leveraging PyTorch's `requires_grad` flags.

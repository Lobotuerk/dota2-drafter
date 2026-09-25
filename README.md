# Dota 2 Draft Ingestion Pipeline

A Python-based data ingestion pipeline that fetches, validates, and transforms Dota 2 Captains Mode draft sequences into PyTorch-ready tensor datasets, followed by a state-of-the-art AlphaZero-style multi-stage neural network training pipeline.

## Table of Contents

- [Setup](#setup)
- [MCTS Library Installation](#mcts-library-installation)
- [Pipeline Overview](#pipeline-overview)
- [Stage 1: Data Gathering](#stage-1-data-gathering)
- [Stage 2: Hero Embeddings](#stage-2-hero-embeddings)
- [Stage 3: RGCN Training](#stage-3-rgcn-training)
- [Stage 4: Transformer Training](#stage-4-transformer-training)
- [Stage 5: Interactive Draft (MCTS)](#stage-5-interactive-draft-mcts)
- [Hyperparameter Tuning (Optuna)](#hyperparameter-tuning-optuna)

---

## Setup

### Install dependencies

```bash
pip install -e .
```

### Run the test suite

```bash
pytest tests/
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env to set STRATZ_API_KEY
```

Obtain a STRATZ API key from [https://www.stratz.com/account/api](https://www.stratz.com/account/api).

---

## MCTS Library Installation

This project uses the `pymcts` C++ library (via pybind11) for high-performance Monte Carlo Tree Search. The library must be installed from source before running the interactive draft tool.

### Install from source

```bash
# Clone the MCTS library (if not already cloned)
git clone https://github.com/Lobotuerk/MonteCarloTreeSearch.git
cd MonteCarloTreeSearch

# Install pymcts in development mode (editable)
pip install --no-build-isolation -e .
```

---

## Pipeline Overview

The architecture utilizes a state-of-the-art approach to Dota 2 drafting:

1. **Data Gathering** — Dual-API fetch (STRATZ/OpenDota) with automatic Historical Patch Tagging for pro matches, plus high-MMR (Immortal) Ranked All Pick matches.
2. **Hero Embeddings (Skip-Gram + DGI)** — Unsupervised deep graph infomax trained across pro and high-MMR pub matches to map the spatial topology of all 124+ heroes.
3. **Hero Embeddings (RGCN)** — Multi-relational GCN with a deep Link Prediction Decoder to encode synergies, counters, and required bans with patch-weighted Wilson Score pruning.
4. **Transformer Training (Two-Stage Decoupled Pipeline)**:
   - **Stage 1 (Value Head)**: Pre-trains permutation-invariant `SetTransformerHead` on high-MMR pub games, fine-tunes on pro draft games with sample weighting, and calibrates temperature $T$ (Platt scaling) via L-BFGS to minimize Brier score.
   - **Stage 2 (Policy Head)**: Freezes the calibrated Value Head and trains the sequential `TransformerDecoder` + `mlm_head` via Advantage-Weighted Masked Language Modeling (AW-MLM), evaluating on positive advantage drafting decisions ($\delta_t > 0$).
5. **MCTS Inference** — Real-time interactive drafting using PyMCTS, utilizing the MLM Policy Head for instant $O(B)$ PUCT priors and the calibrated SetTransformer Value Head for leaf evaluation.

---

## Stage 1: Data Gathering

Runs the ingestion pipeline to fetch pro and pub matches and produce PyTorch batch files.

```bash
# 1. Discover pro leagues
python scripts/01a_gather_leagues.py

# 2. (Optional) Edit data/leagues.json and set review=false for leagues you want to reject.

# 3. Gather pro draft matches from OpenDota/Stratz (generates data/hero_indexer.json & drafts_batch_*.pt)
python scripts/01b_gather_matches.py

# 4. Gather high-MMR pub matches (Ranked All Pick, Immortal rank; generates games_batch_*.pt)
python scripts/01f_gather_high_pubs.py \
    --config config.yaml \
    --limit 10000 \
    --min_rank 80 \
    --region europe \
    --output_dir data

# 5. Clean unapproved leagues and mathematically re-chunk the dataset uniformly
python scripts/01e_cleanup_unapproved.py

# 6. Build player comfort vectors based on pro tournament matches
python scripts/01c_build_comfort.py
```

**Configuration:** See `config.yaml` to adjust the `cutoff_date` (determines how far back in patch history the scraper goes) and `chunk_size`.

---

## Stage 2: Hero Embeddings

Trains Skip-Gram + DGI unsupervised hero embeddings. By default, high-MMR pub games (`games_batch_*.pt`) are included alongside pro drafts (`--include_pubs`), ensuring dense co-occurrence data for all 124+ heroes.

```bash
python scripts/02_train_embeddings.py \
    --mode train \
    --data_dir data \
    --pub_data_dir data \
    --include_pubs \
    --dgi_epochs 300 \
    --skip_gram_epochs 10 \
    --dgi_lr 10e-3
```

---

## Stage 3: RGCN Training

Trains the Relational GNN over the multi-relational hero graph using a 3-layer deep MLP Link Prediction Decoder. High-MMR pub matches supply the necessary statistical sample size to clear the Wilson Score threshold for synergies and antagonist matchups.

```bash
python scripts/03_train_rgcn.py \
    --mode train \
    --data_dir data \
    --pub_data_dir data \
    --include_pubs \
    --frozen_embeddings_path models/skip_gram_dgi.pt \
    --output_file models/rgcn.pt \
    --d_model 64 \
    --rgcn_epochs 500 \
    --learning_rate 3e-4 \
    --wilson_threshold 0.52
```

---

## Stage 4: Transformer Training (Two-Stage Decoupled Pipeline)

Transformer training decouples the Value and Policy heads into two sequential stages:

### Step 4a: Stage 1 — Value Head Pre-training, Fine-Tuning & Calibration

Trains the `SetTransformerHead` win-probability estimator:
- **Phase 1 (Pub Pre-training)**: Learns permutation-invariant team compositions on high-MMR pub games (`games_batch_*.pt`). Automatically tracks validation Brier score and rolls back to the best pub weights before fine-tuning.
- **Phase 2 (Draft Fine-Tuning)**: Fine-tunes on pro tournament drafts (`drafts_batch_*.pt`) using sample weighting (`--draft_sample_weight 5.0`).
- **Temperature Calibration**: Optimizes temperature scalar $T$ using L-BFGS to minimize validation Brier score (Platt scaling).
- **Checkpoints**: Saved to `checkpoints/stage1_best_model.pt` and `checkpoints/best_model.pt`.

```bash
python scripts/04_train_transformer.py \
    --mode train \
    --stage 1 \
    --data_dir data \
    --pub_data_dir data \
    --rgcn_path models/rgcn.pt \
    --comfort_path data/player_comfort.pt \
    --checkpoint_dir checkpoints \
    --device cuda \
    --num_heroes 127 \
    --pub_epochs 20 \
    --num_epochs 30 \
    --draft_sample_weight 5.0 \
    --batch_size 64 \
    --dropout 0.1 \
    --dim_feedforward 128 \
    --lr_head 3e-4 \
    --wilson_threshold 0.52
```

### Step 4b: Stage 2 — Policy Head Advantage-Weighted MLM (AW-MLM)

Trains the `TransformerDecoder` and `SlotAttentionMLMProjection` policy heads:
- **Frozen Value Head**: Freezes `SetTransformerHead` and value projection weights from Stage 1.
- **Advantage-Weighted MLM**: Weights draft sequence tokens using signed advantage $w_t = \exp\left(\frac{\sigma(t) \cdot (V(s_t) - V(s_{t-1}))}{\tau}\right)$ from the calibrated Value Head.
- **Policy Validation**: Tracks and checkpoints on Top-5 accuracy conditioned strictly on positive advantage drafting decisions ($\delta_t > 0$).
- **Checkpoints**: Saves `checkpoints/stage2_best_model.pt` and updates `checkpoints/best_model.pt`.

```bash
python scripts/04_train_transformer.py \
    --mode train \
    --stage 2 \
    --data_dir data \
    --rgcn_path models/rgcn.pt \
    --comfort_path data/player_comfort.pt \
    --stage1_checkpoint checkpoints/stage1_best_model.pt \
    --checkpoint_dir checkpoints \
    --device cuda \
    --num_heroes 127 \
    --num_epochs 50 \
    --batch_size 64 \
    --dropout 0.1 \
    --dim_feedforward 128 \
    --lr_backbone 1e-4 \
    --lr_head 3e-4 \
    --wilson_threshold 0.52
```

---

## Stage 5: Interactive Draft (MCTS)

Boot up the real-time CLI assistant to guide you through a draft. The MCTS engine evaluates thousands of sequences per second by using the Transformer policy (MLM) head for highly contextualized priors, scaled progressively by temperature.

```bash
python scripts/interactive_draft.py     --num_heroes 127     --max_iterations 15000     --max_seconds 60     --device cuda     --top_n 10     --max_candidates 5
```

---

## Hyperparameter Tuning (Optuna)

To find the absolute best set of parameters for this AI architecture, you can use the multi-objective tuning script. Because the embedding size (`d_model`) dictates both the spatial embedding dimensions and downstream transformer dimensions, this script wraps all three training stages (**Skip-Gram + DGI pre-training**, **RGCN training**, and **Transformer training**) into a single, end-to-end objective function.

It utilizes Optuna's multi-objective search (NSGA-II) to find the Pareto front across two conflicting targets:
1. **Top-5 MLM Drafting Policy Accuracy** (the model's capacity to recommend the best contextual picks/bans).
2. **Win-rate ROC-AUC score** (the accuracy of win probability predictions).

### Installation

Ensure `optuna` is installed (it is already registered as a project dependency):
```bash
pip install optuna
```

### Usage

Execute the unified tuning pipeline:
```bash
python scripts/07_tune_pipeline.py \
    --data_dir data \
    --pub_data_dir data \
    --comfort_path data/player_comfort.pt \
    --include_pubs \
    --two_stage \
    --n_trials 20 \
    --skip_gram_epochs 5 \
    --dgi_epochs 10 \
    --rgcn_epochs 10 \
    --pub_epochs 10 \
    --transformer_epochs 20 \
    --draft_sample_weight 5.0
```

*Note: You can pass custom epochs or trials using the command line arguments to balance search depth with your available compute budget. When `--two_stage` is enabled (default), Optuna jointly optimizes the Pareto front of Stage 1 Value Head ROC-AUC / Brier score and Stage 2 Policy Head AW-MLM Top-5 accuracy.*

### Remote Storage & Parallelization (Optional)

You can persist study results to a SQLite database. This allows you to safely interrupt the study and resume it later, or run multiple parallel tuning workers concurrently:

```bash
# Save to a local database
python scripts/07_tune_pipeline.py --storage sqlite:///optuna_study.db
```

To monitor the search and visualize the Pareto front / hyperparameter importance in real-time:
```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna_study.db
```

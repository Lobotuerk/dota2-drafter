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

1. **Data Gathering** — Dual-API fetch (STRATZ/OpenDota) with automatic Historical Patch Tagging.
2. **Hero Embeddings (Skip-Gram + DGI)** — Unsupervised deep graph infomax to map the spatial topology of heroes.
3. **Hero Embeddings (RGCN)** — Multi-relational GCN with a deep Link Prediction Decoder to encode synergies, counters, and required bans.
4. **Transformer Training** — A Two-Headed AlphaZero-style Transformer (predicting Win-Probability via a SetTransformer and Policy via Slot-Attentive MLM) utilizing FiLM Contextual Patch Embeddings.
5. **MCTS Inference** — Real-time interactive drafting using PyMCTS, utilizing the MLM Policy Head for instant O(B) PUCT priors.

---

## Stage 1: Data Gathering

Runs the ingestion pipeline to fetch matches and produce PyTorch batch files.

```bash
# 1. Discover leagues
python scripts/01a_gather_leagues.py

# 2. (Optional) Edit data/leagues.json and set review=false for leagues you want to reject.

# 3. Gather matches from OpenDota/Stratz
python scripts/01b_gather_matches.py

# 4. Clean unapproved leagues and mathematically re-chunk the dataset uniformly
python scripts/01e_cleanup_unapproved.py

# 5. Build player comfort vectors based on the downloaded matches
python scripts/01c_build_comfort.py
```

**Configuration:** See `config.yaml` to adjust the `cutoff_date` (determines how far back in patch history the scraper goes) and `chunk_size`.

---

## Stage 2: Hero Embeddings

Trains Skip-Gram + DGI unsupervised hero embeddings from draft co-occurrence data.

```bash
python scripts/02_train_embeddings.py --mode train --dgi_epochs 300 --skip_gram_epochs 10 --dgi_lr 10e-3
```

---

## Stage 3: RGCN Training

Trains the Relational GNN over the multi-relational hero graph using a 3-layer deep MLP Link Prediction Decoder.

```bash
python scripts/03_train_rgcn.py --mode train --data_dir data --frozen_embeddings_path models/skip_gram_dgi.pt --output_file models/rgcn.pt --d_model 64 --rgcn_epochs 500 --learning_rate 3e-4 --wilson_threshold 0.52
```

---

## Stage 4: Transformer Training

Trains the Two-Headed Hierarchical Sequence Transformer. The script automatically isolates the absolute latest patch in your dataset for the Validation Split, ensuring your `Val AUC` accurately reflects generalizability to the current meta.

*Note: MLM training is now performed in parallel with Value training (AlphaZero-style), so no `--mlm_epochs` flag is needed.*

```bash
python scripts/04_train_transformer.py     --mode train     --data_dir data     --rgcn_path models/rgcn.pt     --comfort_path data/player_comfort.pt     --checkpoint_dir checkpoints     --device cuda     --num_heroes 127     --num_epochs 150     --batch_size 64     --dropout 0.1     --dim_feedforward 128     --lr_backbone 1e-4     --lr_head 3e-4     --augment 5 --wilson_threshold 0.52
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
    --comfort_path data/player_comfort.pt \
    --n_trials 20 \
    --skip_gram_epochs 5 \
    --dgi_epochs 10 \
    --rgcn_epochs 10 \
    --transformer_epochs 20
```

*Note: You can pass custom epochs or trials using the command line arguments to balance search depth with your available compute budget.*

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

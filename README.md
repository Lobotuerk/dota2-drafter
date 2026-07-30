# Dota 2 Draft Ingestion Pipeline

A Python-based data ingestion pipeline that fetches, validates, and transforms Dota 2 Captains Mode draft sequences into PyTorch-ready tensor datasets, followed by a multi-stage embedding and match prediction training pipeline.

## Table of Contents

- [Setup](#setup)
- [Pipeline Overview](#pipeline-overview)
- [Stage 1: Data Gathering](#stage-1-data-gathering)
- [Stage 1b: Build Comfort Data](#stage-1b-build-comfort-data)
- [Stage 2: Hero Embeddings](#stage-2-hero-embeddings)
- [Stage 3: RGCN Training](#stage-3-rgcn-training)
- [Stage 4: Transformer Training](#stage-4-transformer-training)
- [Detailed Configuration](#detailed-configuration)

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

## Pipeline Overview

The pipeline consists of four sequential stages:

1. **Data Gathering** — Fetch Dota 2 matches from STRATZ/OpenDota APIs, validate drafts, produce `.pt` tensor batches.
2. **Hero Embeddings** — Train Skip-Gram + DGI unsupervised hero embeddings from draft co-occurrence data.
3. **RGCN Training** — Train a Relational Graph Convolutional Network over the multi-relational hero graph.
4. **Transformer Training** — Train the Hierarchical Sequence Transformer for match win-probability prediction.

Each stage has a standalone script under `scripts/`. Run them in order.

---

## Stage 1: Data Gathering

Runs the ingestion pipeline to fetch matches and produce PyTorch batch files.

```bash
python scripts/01_gather_data.py
python scripts/01_gather_data.py custom_config.yaml
```

Reads configuration from `config.yaml`. Output is saved to `./data/` as `drafts_batch_*.pt` files.

**Configuration:** See `src/dota2drafter/README.md` for full parameter tables.

---

## Stage 1b: Build Comfort Data

Builds historical player comfort data (required before transformer training).

```bash
python scripts/01b_build_comfort.py --data_dir data --output data/player_comfort.pt
python scripts/01b_build_comfort.py --data_dir data --output data/player_comfort.pt --dim 10 --random
```

---

## Stage 2: Hero Embeddings

Trains Skip-Gram + DGI hero embeddings.

```bash
# Train
python scripts/02_train_embeddings.py --mode train \
    --data_dir data \
    --output_file models/skip_gram_dgi.pt \
    --dgi_epochs 100 \
    --skip_gram_epochs 10 \
    --embed_dim 64 \
    --dgi_lr 5e-4

# Predict (print embedding for a specific hero)
python scripts/02_train_embeddings.py --mode predict \
    --output_file models/skip_gram_dgi.pt --hero_id 1
```

**Configuration:** See `src/dota2drafter/embeddings/README.md` for full parameter tables.

---

## Stage 3: RGCN Training

Trains the Relational GNN over the multi-relational hero graph.

```bash
# Train
python scripts/03_train_rgcn.py --mode train  \
    --data_dir data  \
    --frozen_embeddings_path models/skip_gram_dgi.pt  \
    --output_file models/rgcn.pt \
    --d_model 64 \
    --rgcn_epochs 275 \
    --learning_rate 1.5e-3

# Predict (extract structural hero embeddings)
python scripts/03_train_rgcn.py --mode predict \
    --data_dir data \
    --frozen_embeddings_path models/skip_gram_dgi.pt \
    --rgcn_path models/rgcn.pt
```

**Configuration:** See `src/dota2drafter/embeddings/README.md` for full parameter tables.

---

## Stage 4: Transformer Training

Trains the Hierarchical Sequence Transformer for match prediction.

```bash
# Train (requires data from stages 1, 1b, and 3)
python scripts/04_train_transformer.py --mode train  \
    --data_dir data  \
    --rgcn_path models/rgcn.pt \
    --comfort_path data/player_comfort.pt \
    --checkpoint_dir checkpoints \
    --device cuda \
    --dropout 0.3 \
    --learning_rate 1e-4 \
    --lr_backbone 1e-5 \
    --lr_head 1e-3 \
    --step_loss_gamma 1.0 \
    --label_smoothing_eps 0.6 \
    --mlm_epochs 10 \
    --num_heroes 127 \
    --d_model 64 \
    --dim_feedforward 128 \
    --augment True

# Predict (requires a trained checkpoint)
python scripts/04_train_transformer.py --mode predict \
    --data_dir data \
    --rgcn_path models/rgcn.pt \
    --comfort_path data/player_comfort.pt \
    --checkpoint_dir checkpoints
```

**Configuration:** See `src/dota2drafter/training/README.md` for full parameter tables.

---

## Detailed Configuration

Full parameter tables for each module are available in the module READMEs:

- **Ingestion config:** `src/dota2drafter/README.md`
- **Embeddings & RGCN:** `src/dota2drafter/embeddings/README.md`
- **Match Network & Training:** `src/dota2drafter/training/README.md`

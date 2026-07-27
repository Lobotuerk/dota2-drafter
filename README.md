# Dota 2 Draft Ingestion Pipeline

A Python-based data ingestion pipeline that fetches, validates, and transforms Dota 2 Captains Mode draft sequences into PyTorch-ready tensor datasets, followed by a multi-stage embedding and match prediction training pipeline.

## Table of Contents

- [Setup & Verification](#setup--verification)
- [Data Ingestion](#data-ingestion)
- [Embedding Pre-training (Skip-Gram + DGI)](#embedding-pre-training-skip-gram--dgi)
- [RGCN Training](#rgcn-training)
- [Match Network Training (Transformer)](#match-network-training-transformer)

---

## Setup & Verification

### Install dependencies

```bash
pip install -e .
```

### Run the test suite

Verify your environment is correctly set up:

```bash
pytest tests/
```

---

## Data Ingestion

The first stage fetches Dota 2 matches from the STRATZ and OpenDota APIs, validates draft sequences, and produces PyTorch tensor batches (`.pt` files).

### Environment Setup

```bash
cp .env.example .env
# Edit .env to set STRATZ_API_KEY
```

Obtain a STRATZ API key from [https://www.stratz.com/account/api](https://www.stratz.com/account/api).

### Configuration (`config.yaml`)

The pipeline is configured via `config.yaml`. Key settings:

| Setting | Description | Example |
|---|---|---|
| `patch` | Dota patch version to filter matches | `"7.35"` |
| `tiers` | Tournament tiers to include | `[1, 2]` |
| `stratz.api_key` | STRATZ API key (read from env var) | `"${STRATZ_API_KEY}"` |
| `stratz.max_retries` | Retry attempts for API failures | `5` |
| `stratz.retry_delay` | Delay between retries (seconds) | `2.0` |
| `output.directory` | Where `.pt` batch files are saved | `"./data"` |
| `output.chunk_size` | Matches per `.pt` file | `1000` |
| `state.database_path` | SQLite database for tracking progress | `"./state.db"` |

### Execution

```bash
dota2-drafter config.yaml
```

This reads the config, discovers leagues and matches, processes draft sequences, and saves the results as `.pt` files in the configured output directory (default `./data`).

**Output format** — each `.pt` file contains a dictionary with:

- `x`: `(N, 24, 3)` tensor of draft sequences
- `y`: `(N,)` tensor of `radiant_win` labels
- `match_ids`: list of match IDs for traceability
- `radiant_players`: list of Radiant player account ID lists
- `dire_players`: list of Dire player account ID lists

---

## Embedding Pre-training (Skip-Gram + DGI)

This stage trains node embeddings for heroes in two steps:

1. **Skip-Gram** — learns co-occurrence-based embeddings from draft sequences.
2. **DGI (Deep Graph Infomax)** — learns structural embeddings using the Skip-Gram features as initial node representations.

### Python API

```python
from dota2drafter.embeddings.pretrainer import train_embeddings

train_embeddings(
    data_dir="data",          # Directory containing .pt batches from ingestion
    output_file="models/skip_gram_dgi.pt",  # Path to save final embeddings
    embed_dim=64,             # Dimension of the embedding space
    skip_gram_epochs=10,      # Number of Skip-Gram training epochs
    dgi_epochs=20,            # Number of DGI training epochs
    learning_rate=1e-2,       # Learning rate for both optimizers
    batch_size=256,           # Batch size for Skip-Gram DataLoader
    device="cpu",             # Device to train on (auto-detected if None)
)
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | *(required)* | Directory containing draft batch `.pt` files |
| `output_file` | *(required)* | Path to save the final embedding weights |
| `embed_dim` | `64` | Dimension of the embedding space |
| `skip_gram_epochs` | `10` | Number of Skip-Gram training epochs |
| `dgi_epochs` | `20` | Number of DGI training epochs |
| `learning_rate` | `1e-2` | Learning rate for both Skip-Gram and DGI optimizers |
| `batch_size` | `256` | Batch size for the Skip-Gram DataLoader |
| `device` | `None` (auto) | Device to train on (`"cpu"` or `"cuda"`) |

The function returns the path to the saved embedding weights. The output tensor is padded at index 0 (heroes are 1-indexed) and can be loaded directly as frozen embeddings for downstream stages.

---

## RGCN Training

This stage trains a Relational Graph Convolutional Network (HeroRGCN) over the multi-relational hero graph using the frozen DGI embeddings as initial node features. The result is relation-aware structural embeddings.

### Python API

```python
from dota2drafter.embeddings.train_rgcn import train_rgcn

train_rgcn(
    data_dir="data",                    # Directory containing .pt batches from ingestion
    frozen_embeddings_path="models/skip_gram_dgi.pt",  # Path to DGI embeddings from previous stage
    output_file="models/rgcn.pt",       # Path to save the trained RGCN model
    d_model=64,                         # Embedding dimension
    num_relations=3,                    # Number of edge types in the hero graph
    rgcn_epochs=20,                     # Number of RGCN training epochs
    learning_rate=1e-2,                 # Learning rate for the optimizer
    num_layers=2,                       # Number of RGCN layers (1-2 recommended)
    device="cpu",                       # Device to train on (auto-detected if None)
    hidden_dim=None,                    # Hidden dimension (defaults to d_model)
)
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | *(required)* | Directory containing draft batch `.pt` files |
| `frozen_embeddings_path` | *(required)* | Path to saved DGI/Skip-Gram embeddings (`.pt` tensor) |
| `output_file` | *(required)* | Path to save the trained RGCN model weights |
| `d_model` | `64` | Embedding dimension |
| `num_relations` | `3` | Number of edge types in the multi-relational hero graph |
| `rgcn_epochs` | `20` | Number of RGCN training epochs |
| `learning_rate` | `1e-2` | Learning rate for the optimizer |
| `num_layers` | `2` | Number of RGCN layers (1-2 recommended) |
| `device` | `None` (auto) | Device to train on (`"cpu"` or `"cuda"`) |
| `hidden_dim` | `None` (== d_model) | Hidden dimension for RGCN layers |

The function trains the HeroRGCN by maximizing mutual information between node embeddings and the global graph summary, then saves the trained model weights.

---

## Match Network Training (Transformer)

The final stage trains the downstream match prediction model, which combines a HierarchicalTransformer with a PlayerComfortNetwork for win-probability prediction.

Since there is no dedicated CLI entry-point for the dataset loader combined with transformer training, use the following self-contained script to assemble the components:

```python
import torch
from pathlib import Path

from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.training.transformer_trainer import TransformerTrainer, TrainingConfig

# 1. Load batched data from ingestion
x_drafts, y_labels, radiant_players, dire_players = [], [], [], []
for pt_file in Path("data").glob("*.pt"):
    batch = torch.load(pt_file)
    for i in range(len(batch["x"])):
        x_drafts.append(batch["x"][i])
        y_labels.append(batch["y"][i])
        radiant_players.append(batch["radiant_players"][i])
        dire_players.append(batch["dire_players"][i])

# 2. Load trained RGCN embeddings
h_gnn = torch.load("models/rgcn.pt")

# 3. Initialize the Match Network
model = MatchNetwork(
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    num_heroes=120,
    player_input_dim=10,
    h_gnn=h_gnn
)

# 4. Configure and train
config = TrainingConfig(
    learning_rate=1e-3,
    num_epochs=50,
    batch_size=64,
    device="cpu"
)

trainer = TransformerTrainer(model, config)
metrics = trainer.train(
    x_drafts=x_drafts,
    y_labels=y_labels,
    radiant_players=radiant_players,
    dire_players=dire_players
)
```

### Configuration

**MatchNetwork parameters:**

| Parameter | Default | Description |
|---|---|---|
| `d_model` | `128` | Transformer embedding dimension |
| `nhead` | `4` | Number of attention heads |
| `num_layers` | `4` | Number of TransformerDecoderLayer blocks |
| `dim_feedforward` | `256` | Feedforward dimension in decoder layers |
| `dropout` | `0.1` | Dropout rate |
| `num_heroes` | `120` | Number of heroes K |
| `player_input_dim` | `10` | Number of input features per player comfort vector |
| `h_gnn` | `None` | Frozen RGCN hero embeddings of shape `(K+1, d_model)` |

**TrainingConfig parameters:**

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `1e-3` | Learning rate for the Adam optimizer |
| `num_epochs` | `50` | Maximum number of training epochs |
| `batch_size` | `64` | Batch size for training and validation |
| `val_split` | `0.2` | Fraction of data reserved for validation |
| `device` | `"cpu"` | Device to train on (`"cpu"` or `"cuda"`) |
| `checkpoint_dir` | `"./checkpoints"` | Directory for saving best model checkpoints |
| `patience` | `10` | Early stopping patience |
| `min_delta` | `1e-4` | Minimum change to qualify as an improvement |

The trainer automatically splits data into train/validation sets, runs the training loop with early stopping, and saves the best model checkpoint. The returned `TrainingMetrics` object contains full training and validation history (losses, accuracies, ROC-AUC scores).

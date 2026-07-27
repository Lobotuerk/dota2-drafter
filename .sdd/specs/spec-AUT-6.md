# Technical Specification: Training Pipeline Instructions (AUT-6)

## 1. Overview
The goal is to update the repository `README.md` to document the end-to-end pipeline of the **dota2-drafter** project. As established by the Orchestrator, the pipeline consists of four main stages:
1. Data Ingestion
2. Embedding Pre-training (Skip-Gram + DGI)
3. Relational Graph Convolutional Network (RGCN) Training
4. Match Network (Transformer) Training

The implementation must focus strictly on documenting these stages, detailing Python API usages for training steps, describing configurations, and providing verification instructions. Prediction and inference usage are explicitly out of scope.

## 2. README Changes

The `README.md` must be expanded to include clear instructions and code snippets for each stage. It is recommended to use the following structure.

### Section: 1. Setup & Verification
Provide instructions on how to install dependencies and run the tests to ensure the environment is correctly set up.

```bash
# Install the package
pip install -e .

# Run the test suite to verify the environment
pytest tests/
```

### Section: 2. Data Ingestion
Document the `dota2-drafter` CLI pipeline used for data ingestion.

1. **Environment Setup**:
   ```bash
   cp .env.example .env
   # Edit .env to set STRATZ_API_KEY
   ```
2. **Configuration (`config.yaml`)**: Explain how to configure the ingestion pipeline (e.g., tweaking `patch`, `tiers`, and `output.directory`).
3. **Execution**:
   ```bash
   dota2-drafter config.yaml
   ```
   *Note: Explain that this will output tensor batches (defaulting to the `./data` directory).*

### Section: 3. Embedding Pre-training (Skip-Gram + DGI)
Document the Python API method to train the initial node features (Skip-Gram embeddings followed by DGI structural embeddings). Include a clear configuration snippet:

```python
from dota2drafter.embeddings.pretrainer import train_embeddings

train_embeddings(
    data_dir="data",  # Directory containing .pt batches from ingestion
    output_file="models/skip_gram_dgi.pt",
    embed_dim=64,
    skip_gram_epochs=10,
    dgi_epochs=20,
    learning_rate=1e-2,
    batch_size=256
)
```

### Section: 4. RGCN Training
Document the Python API method to train the HeroRGCN on the multi-relational hero graph using the frozen embeddings from the previous stage.

```python
from dota2drafter.embeddings.train_rgcn import train_rgcn

train_rgcn(
    data_dir="data",
    frozen_embeddings_path="models/skip_gram_dgi.pt",
    output_file="models/rgcn.pt",
    d_model=64,
    num_relations=3,
    rgcn_epochs=20,
    learning_rate=1e-2,
    num_layers=2
)
```

### Section: 5. Match Network Training (Transformer)
Document the Python API approach to train the downstream HierarchicalTransformer and PlayerComfortNetwork.
Since there is no dedicated CLI entry-point for the dataset loader combined with transformer training, provide a self-contained Python script snippet demonstrating how to assemble the components:

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

## 3. Implementation Steps for SDD-Implementer
1. Rewrite `README.md` replacing its current incomplete instructions with the comprehensive guide outlined above.
2. Ensure markdown formatting, code snippets, and explanations are clear and accessible to developers.
3. Validate that you do not include inference or prediction steps, as they are out of scope.

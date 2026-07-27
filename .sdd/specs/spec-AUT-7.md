# Technical Specification: Usage Scripts (AUT-7)

## Overview
This specification details the creation of standalone Python usage scripts for the `dota2-drafter` pipeline, the removal of the confusing `dota2-drafter` CLI entry point, and the restructuring of documentation into a minimal root README with detailed parameter tables moved to module-specific READMEs.

## 1. CLI Entry Point Removal
- **`pyproject.toml`**: Remove the `[project.scripts]` section that defines `dota2-drafter = "dota2drafter.main:main"`.
- **`src/dota2drafter/main.py`**: Remove the `main()` function and the `if __name__ == "__main__":` block to prevent it from being used as a CLI directly.

## 2. Usage Scripts (`scripts/`)
All scripts will use `argparse` to support configuration and, where applicable, a `--mode` flag (either `train` or `predict`). 

### `scripts/01_gather_data.py`
- **Purpose**: Runs the ingestion pipeline (replacing the old CLI).
- **Arguments**: 
  - `config` (positional): Path to the YAML configuration file (default: `config.yaml`).
- **Implementation**: Imports `run_pipeline` and `load_config` from `dota2drafter.main` and `dota2drafter.config`, sets up `asyncio.run(run_pipeline(config))`.

### `scripts/01b_build_comfort.py`
- **Purpose**: A mock/stub script to fulfill the ingestion pipeline requirement for building historical player comfort data before transformer training.
- **Arguments**:
  - `--data_dir`: Directory with `.pt` match batches.
  - `--output`: Path to save the comfort map (default: `data/player_comfort.pt`).
  - `--dim`: Player input dimension (default: 10).
- **Implementation**: Iterates through all match batches, extracts unique account IDs from `radiant_players` and `dire_players`, and assigns a zero-initialized (or randomly initialized) tensor of shape `(dim,)` to each player. Saves the resulting dictionary (`dict[int, torch.Tensor]`) to disk.

### `scripts/02_train_embeddings.py`
- **Purpose**: Trains or predicts with Unsupervised Hero Embeddings (Skip-Gram + DGI).
- **Arguments**:
  - `--mode`: `train` or `predict` (default: `train`).
  - `--data_dir`: Directory with `.pt` match batches.
  - `--output_file`: Model save/load path.
  - Model hyperparams: `--embed_dim`, `--skip_gram_epochs`, `--dgi_epochs`, etc.
- **Implementation**: 
  - **Train**: Calls `train_embeddings(...)` from `dota2drafter.embeddings.pretrainer`.
  - **Predict**: Calls `load_frozen_embeddings(...)`, and prints the embedding for a specific Hero ID.

### `scripts/03_train_rgcn.py`
- **Purpose**: Trains or predicts with the Relational GNN.
- **Arguments**:
  - `--mode`: `train` or `predict` (default: `train`).
  - `--data_dir`: Directory with `.pt` match batches.
  - `--frozen_embeddings_path`: Path to embeddings from stage 02.
  - `--output_file`: Model save/load path.
  - Model hyperparams: `--d_model`, `--rgcn_epochs`, etc.
- **Implementation**:
  - **Train**: Calls `train_rgcn(...)` from `dota2drafter.embeddings.train_rgcn`.
  - **Predict**: Calls `load_rgcn_embeddings(...)`, builds the hero graph using `DataExtractor`, and performs a forward pass to extract structural hero embeddings.

### `scripts/04_train_transformer.py`
- **Purpose**: Trains or predicts with the Hierarchical Sequence Transformer.
- **Arguments**:
  - `--mode`: `train` or `predict` (default: `train`).
  - `--data_dir`: Directory with `.pt` match batches.
  - `--rgcn_path`: Path to RGCN weights.
  - `--comfort_path`: Path to `player_comfort.pt`.
  - `--checkpoint_dir`: Directory for model checkpoints.
  - Model hyperparams: `--d_model`, `--num_epochs`, etc.
- **Implementation**:
  - **Train**: Loads all `.pt` batches, the RGCN embeddings (`h_gnn`), and the comfort map. Initializes `MatchNetwork` and `TransformerTrainer`, then calls `trainer.train()`.
  - **Predict**: Loads the `MatchNetwork` state from the best checkpoint, loads a single match from a batch, and calls `model.predict_proba(x_draft, player_comfort)` to output the win probability.

## 3. Documentation Restructuring
The root `README.md` currently contains inline Python scripts and detailed parameter tables.

- **Root `README.md`**: Simplified to only describe the 5 pipeline stages and provide the exact bash commands to run the newly created scripts (e.g., `python scripts/02_train_embeddings.py --mode train ...`).
- **Module READMEs**: The detailed parameter tables will be moved to:
  - `src/dota2drafter/README.md` (Ingestion config parameters)
  - `src/dota2drafter/embeddings/README.md` (Embeddings & RGCN parameters)
  - `src/dota2drafter/training/README.md` (Match Network & TrainingConfig parameters)

## Design Philosophy Constraints
- **Deep Modules**: The usage scripts simply act as shallow CLI wrappers that delegate immediately to the existing deep pipeline functions (`run_pipeline`, `train_embeddings`, `TransformerTrainer`).
- **Clean Exception Paths**: Scripts will catch `FileNotFoundError` for missing data or configs and print user-friendly messages rather than deep stack traces.
- **Legacy Isolation**: We are only removing `main()` from `main.py` and `pyproject.toml`, leaving the core logic of the ingestion pipeline entirely intact.
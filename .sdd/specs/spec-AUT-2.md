### 📋 Technical Specification

#### 1. Overview
The objective is to generate baseline structural hero embeddings prior to supervised training of the downstream Graph-Augmented Transformer. We will implement a pre-training module that uses Skip-Gram clustering on historical 5-hero compositions to establish semantic clusters. These embeddings will then be used as node features in a Deep Graph Infomax (DGI) model operating over a hero interaction graph (encoding synergies and counters) to inject structural topology. The resulting embeddings will be exported as a frozen module.

#### 2. Architecture & Modules
We will introduce a new package `src/dota2drafter/embeddings/` to house the pre-training pipeline.

**2.1. Dataset and Graph Extraction (`src/dota2drafter/embeddings/data_extractor.py`)**
- **Purpose:** Parse raw `(24, 3)` match tensors from the ingestion pipeline to generate training data.
- **Mechanics:**
  - **Skip-Gram Pairs:** For each match, extract the 5 Radiant picks and 5 Dire picks. Generate positive `(center, context)` pairs from within each team. Generate negative pairs by randomly sampling heroes outside the team.
  - **Hero Interaction Graph:** Aggregate empirical win rates across all matches. 
    - *Synergy Edges:* Co-pick win rates between heroes $i$ and $j$.
    - *Opposition Edges:* Head-to-head win rates of hero $i$ against hero $j$.
    - The graph will be represented as a PyTorch Geometric (PyG) `Data` object (`edge_index`, `edge_attr`).

**2.2. Skip-Gram Model (`src/dota2drafter/embeddings/skip_gram.py`)**
- **Purpose:** Learn semantic vector representations based purely on composition co-occurrence.
- **Design:** A simple `nn.Module` using two `nn.Embedding` layers (target and context) optimized via Binary Cross Entropy with Negative Sampling.

**2.3. Deep Graph Infomax (DGI) Model (`src/dota2drafter/embeddings/dgi.py`)**
- **Purpose:** Learn structural embeddings by maximizing mutual information between local node representations and a global graph summary.
- **Design:**
  - **Encoder:** A PyG `GCNConv` or `GATConv` network. Initial node features are the trained Skip-Gram embeddings.
  - **Corruption:** Row-wise permutation of the input node features to generate negative graph samples.
  - **Model:** Wraps the encoder via `torch_geometric.nn.DeepGraphInfomax`.

**2.4. Orchestration Module (`src/dota2drafter/embeddings/pretrainer.py`)**
- **Purpose:** Expose a clean, importable API to train and consume the embeddings.
- **Interface:**
  - `train_embeddings(data_dir: Path, output_file: Path, embed_dim: int)`: End-to-end pipeline executor.
  - `load_frozen_embeddings(weights_path: Path, embed_dim: int, num_heroes: int) -> nn.Embedding`: Factory method returning a frozen `nn.Embedding` module populated with the DGI output weights, ready to be injected into the Custom Graph-Augmented Transformer.

#### 3. Dependencies
- Add `torch-geometric` to `pyproject.toml` to support DGI and graph convolution operations.

#### 4. Design Philosophy Considerations
- **High Modular Depth:** The `data_extractor` abstracts away the iteration and mathematical logic to compute graphs from raw `.pt` tensors. The downstream model simply calls `load_frozen_embeddings()`.
- **Clean Exception Paths:** The `load_frozen_embeddings` method will validate the shape of the loaded state dictionary against the expected `embed_dim` and `num_heroes` before initializing the frozen module.

#### 5. Visual/UI Impact
- No visual or UI elements are associated with this pipeline task.

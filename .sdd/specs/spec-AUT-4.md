# Technical Specification: Relational Graph Neural Network Stage

## 1. Overview
This specification outlines the design for integrating a Relational Graph Convolutional Network (RGCN) into the Dota 2 drafting pipeline. The RGCN acts as a downstream processor that consumes frozen hero embeddings (produced by the existing Skip-Gram and DGI pipelines) and processes them over a multi-relational hero graph. This yields a relation-aware structural hero embedding matrix, H_GNN.

## 2. Multi-Relational Graph Construction
The current `DataExtractor.build_hero_graph` in `src/dota2drafter/embeddings/data_extractor.py` will be refactored to produce a multi-relational graph compatible with PyTorch Geometric's `RGCNConv` (yielding `edge_index`, `edge_type`, and `edge_weight`).

### 2.1 Relation Types (R)
1. **r_syn (Synergy - Edge Type 0)**
   - **Definition:** Co-Picked-Radiant / Co-Picked-Dire.
   - **Criteria:** Connects hero i and hero j (undirected) if they are picked on the same team.
   - **Weight:** Co-pick win rate.
2. **r_ant (Antagonist - Edge Type 1)**
   - **Definition:** Mechanical counter-picks.
   - **Criteria:** Connects hero i to hero j if hero i is picked by the winning team and hero j is picked by the losing team.
   - **Weight:** Head-to-head win rate of hero i against hero j.
3. **r_ban (Banned-Against - Edge Type 2)**
   - **Definition:** Systematic banning correlations.
   - **Criteria:** Connects hero i to hero j if hero i is picked by Team A and hero j is banned by Team B. This captures the tactical threat that hero j poses to hero i.
   - **Weight:** Co-occurrence frequency (normalized count).

### 2.2 Extraction Logic Modifications
- Read `draft[draft[:, 0] == 0.0]` to extract ban steps.
- Extract `radiant_bans` and `dire_bans` in addition to picks.
- Populate `ban_counts[(pick_h, ban_h)]`.
- Return a `Data` object containing:
  - `edge_index`: Tensor of shape `[2, num_edges]`
  - `edge_type`: Tensor of shape `[num_edges]` containing 0, 1, or 2.
  - `edge_weight`: Tensor of shape `[num_edges, 1]` with win-rates and frequencies.

## 3. RGCN Model Architecture
Create `src/dota2drafter/embeddings/rgcn.py`:
- Implement a `HeroRGCN` `nn.Module` using `torch_geometric.nn.RGCNConv`.
- **Inputs:** 
  - `num_nodes`: Number of heroes K+1.
  - `d_model`: Embedding dimension (e.g., 64).
  - `num_relations`: 3.
  - `frozen_embeddings`: Pre-trained tensor from the DGI stage.
- **Layers:**
  - An `nn.Embedding` initialized with `frozen_embeddings` (`requires_grad=False` to act as frozen features).
  - 1-2 layers of `RGCNConv(d_model, d_model, num_relations=3, num_bases=None)`.
  - Non-linear activation (e.g., `LeakyReLU`) between layers.
- **Output:** The updated embedding matrix H_GNN of shape `[K, d_model]`.

## 4. Pipeline Integration
Create `src/dota2drafter/embeddings/train_rgcn.py` (Separate Stage):
- Since this is a separate stage from `pretrainer.py`, create a standalone module.
- **Workflow:**
  1. Load batches via `DataExtractor`.
  2. Build the relational graph `Data` object with 3 edge types.
  3. Load the frozen embeddings from the DGI/Skip-Gram stage.
  4. Instantiate the `HeroRGCN` model.
  5. Provide a clear entrypoint to yield H_GNN for downstream models.

## 5. Design Philosophy Compliance
- **Deep Modules:** The `HeroRGCN` encapsulates all relational message-passing complexity. Downstream consumers simply call `rgcn(edge_index, edge_type)` and receive enriched embeddings.
- **Clear Interfaces:** `build_hero_graph()` will maintain its simple signature `(batches: list) -> Data`, but the `Data` object will have richer internal attributes (`edge_type`, `edge_weight`).

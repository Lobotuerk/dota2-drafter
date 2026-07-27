### 📋 Technical Specification

## 1. Overview
The Hierarchical Sequence Transformer Stage is the final component of the draft prediction architecture. It integrates frozen hero embeddings (`H_GNN`) from the RGCN, draft action sequences, and player preference vectors to predict match outcomes. The system is split into two swappable modules: the `PlayerComfortNetwork` (Player Network) and the `MatchNetwork` (Hierarchical Transformer).

## 2. Architecture & Modules

### 2.1. PlayerComfortNetwork (`src/dota2drafter/models/player_network.py`)
- **Responsibility**: Maps raw player comfort/performance matrices (historical stats per hero) into latent preference vectors `player_pref_vectors`.
- **Inputs**: Player comfort tensor of shape `(B, 10, C)` where B is batch size, 10 is the number of players (5 Radiant, 5 Dire), and C represents historical metrics (e.g., games played, win rate per hero).
- **Architecture**: A multi-layer perceptron (MLP) mapping `C` features to `d_model` (the transformer embedding dimension).
- **Outputs**: Latent preference vectors of shape `(B, 10, d_model)`.

### 2.2. Match Network / HierarchicalTransformer (`src/dota2drafter/models/match_network.py`)
- **Responsibility**: Models the sequence of draft actions and cross-attends with player preference vectors to compute win probability.
- **Inputs**: 
  - `x_draft`: Draft sequence tensor `(B, 24, 4)` containing $(h_t, p_t, c_t, o_t)$ for each step.
  - `player_pref_vectors`: `(B, 10, d_model)` from the Player Network.
  - `H_GNN`: Frozen RGCN hero embeddings `(K+1, d_model)`.
- **Joint Embedding**: Maps $a_t = (h_t, p_t, c_t, o_t)$ into $z_t$.
  - $h_t$: Hero index. Looks up `H_GNN[h_t]`, passes through `Project` linear layer.
  - $p_t$: Action type (Ban=0, Pick=1). Uses `W_type` embedding matrix.
  - $c_t$: Team side (Radiant=0, Dire=1). Uses `W_team` embedding matrix.
  - $o_t$: Absolute draft step (0-23). Uses positional encoding (sinusoidal or learned).
  - The joint embedding $z_t$ is the sum of these projections.
- **Transformer Body**: 
  - Stacked `TransformerDecoderLayer` or equivalent blocks. 
  - Self-attention on the draft sequence $Z$ to capture intra-team synergy and inter-team opposition.
  - Cross-attention where `query` is the draft sequence and `key`/`value` are `player_pref_vectors`, allowing the draft to align with player capabilities.
- **Output Head**: Executes masked global average pooling over the Transformer output sequence (or uses a prepended CLS token), followed by an MLP and Sigmoid activation to predict `P(Y=1 | S_24)`.

### 2.3. Data Pipeline Extension (`src/dota2drafter/processor/tensor_transformer.py`)
- **Updates to `ProcessedMatch`**:
  - Add `radiant_players: list[int]` and `dire_players: list[int]` containing `account_id`s.
  - Change `x_tensor` shape from `(24, 3)` to `(24, 4)` to include absolute order $o_t$: `[hero_val, is_pick, team, step_index]`.
- **Transformation Logic**:
  - Extract `players[].account_id` from STRATZ and OpenDota payloads.
  - Provide a mechanism (e.g., in a PyTorch `Dataset`) to dynamically look up the historical player matrices given the `account_id`s.

## 3. Training Loop & Validation (`src/dota2drafter/training/transformer_trainer.py`)
- **Components**:
  - DataLoader yielding `(x_draft, player_matrices, y)`.
  - Training loop utilizing `BCEWithLogitsLoss`.
  - Validation loop evaluating performance on a hold-out set.
  - Metrics tracking: Accuracy, ROC-AUC, BCE.
  - Checkpointing logic to save the best model weights per epoch.

## 4. Implementation Steps
1. **Extend Data Pipeline**: Modify `tensor_transformer.py` to parse players and explicitly output `(h_t, p_t, c_t, o_t)`.
2. **Implement Player Network**: Create `player_network.py` with the `PlayerComfortNetwork` class.
3. **Implement Match Network**: Create `match_network.py` containing the sequence embedding layer, cross-attention transformer stack, and output head.
4. **Implement Training Loop**: Create `transformer_trainer.py` for training logic, metric computation, and checkpointing.
5. **Testing**: 
   - Add unit tests for `tensor_transformer.py` extensions.
   - Add unit tests for `player_network.py` and `match_network.py` forward passes.
   - Add integration tests verifying end-to-end tensor flow from frozen `H_GNN` to final probability.

## 5. Design Philosophy Conformance
- **Deep Modules**: The Transformer and Player networks have simple interfaces (`forward(draft_seq, players)`) but encapsulate complex cross-attention and projection mechanisms.
- **Independent Modules**: `PlayerComfortNetwork` and `MatchNetwork` are physically separated, keeping interfaces clean and allowing future experimentation with player representation without altering the sequence modeling.

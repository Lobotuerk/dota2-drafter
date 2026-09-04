# Dota 2 Drafter: Deep Learning Architecture

This document details the mathematical and structural design of the Dota 2 Drafter AI. The system is heavily inspired by DeepMind's **AlphaZero**, utilizing a two-headed Transformer architecture, multi-relational graph embeddings, and a highly optimized Monte Carlo Tree Search (MCTS) engine to navigate the incredibly noisy and stochastic environment of competitive Dota 2.

---

## 1. Graph-Based Hero Representation (The "Eyes")

Before the Transformer can evaluate a draft, it needs to understand *what a hero is*. Instead of initializing random embeddings, the system pre-trains structural hero representations using a pipeline of Unsupervised and Self-Supervised Graph Neural Networks (GNNs).

### 1.1. Skip-Gram Base
The pipeline extracts sliding windows of draft co-occurrences (heroes drafted together) and trains a standard Skip-Gram model to map out the basic Euclidean proximity of heroes.

### 1.2. Deep Graph Infomax (DGI)
To understand the broader topological graph of Dota 2 (roles and clusters), the Skip-Gram embeddings are passed into a DGI model.
* **Architecture:** Uses a Graph Convolutional Network (GCN) encoder.
* **Oversmoothing Prevention:** Dota's hero graph is incredibly dense. To prevent features from "oversmoothing" (collapsing into a single identical point), the GCN uses an expanded hidden bottleneck (`embed_dim * 2`), `LeakyReLU(0.1)` activations, Dropout, and **Residual (Skip) Connections**.
* **Objective:** Maximizes Mutual Information between local patch representations and a global graph summary via a Bilinear Discriminator.

### 1.3. Relational Graph Convolutional Network (RGCN)
The DGI embeddings are then fed into an RGCN operating over a pruned, **Multi-Relational Graph** containing three edge types: *Synergies*, *Antagonists*, and *Required Bans*.
* **Directed Decoder:** Traditional decoders (like DistMult) are symmetric. Because Dota has directed relationships (Hero A counters Hero B does not mean Hero B counters Hero A), the RGCN uses a deep 3-layer MLP Link Prediction Decoder (`[3*d_model -> 2*d_model -> d_model -> 1]`).
* **Output:** Produces the final frozen `$H_{GNN}$` matrix, capturing complex directed relationships, which is passed to the Transformer.

---

## 2. Contextual Sequence Modeling (The "Brain")

The core of the system is the `MatchNetwork`, a Hierarchical Sequence Transformer that reads the draft sequentially.

### 2.1. The Joint Embedding ($z_t$)
Every step of the 24-action Captains Mode draft is embedded into a joint space before entering the Transformer:
$$ z_t = \gamma_{patch} \cdot (Project(H_{GNN}[h_t]) + W_{type}(p_t) + W_{team}(c_t) + PE(o_t)) + \beta_{patch} $$
* **$H_{GNN}$:** The frozen RGCN hero embedding.
* **$W_{type}$:** Action type (Pick vs. Ban).
* **$W_{team}$:** Team side (Radiant vs. Dire).
* **$PE$:** Sinusoidal absolute positional encoding (Step 0 through 23).
* **$\gamma_{patch}, \beta_{patch}$ (FiLM Contextual Patch Embeddings):** The system uses **Feature-wise Linear Modulation (FiLM)** instead of additive patch embeddings. The patch embedding generates two vectors — a scaling factor $\gamma$ and a shifting factor $\beta$ — that dynamically modulate the joint embedding based on the historical meta (e.g., Patch 7.37 vs 7.41e). This allows the network to train on hundreds of thousands of historical matches to learn the "universal mechanics" of Dota, while isolating meta-specific balance shifts through learned affine transformations rather than simple addition.

### 2.2. Player Comfort Network (PCN)
The model dynamically factors in the human element. The PCN takes a $(10, C)$ historical win/loss differential matrix for the players in the lobby and maps it to a $(10, d\_model)$ preference vector, which is cross-attended with the draft sequence in the Transformer.

### 2.3. Causal Masking
The `HierarchicalTransformer` utilizes strict Causal Masking (`tgt_mask=causal_mask`). This prevents the self-attention mechanism from looking at future picks/bans when evaluating the current step, which is absolutely mandatory for real-time sequential evaluation during MCTS.

---

## 3. Two-Headed AlphaZero Training

The Transformer is trained using a parallel dual-objective setup to combat Task Over-Specialization and representation rigidity.

### 3.1. The Value Head (Win-Probability)
* **Goal:** Predict the final outcome of the match.
* **Mechanism:** Uses a **SetTransformerHead** that operates over the set of drafted heroes. The head first groups Radiant and Dire picks independently, processing each set through Set Attention Blocks (SAB) and Pooling by Multihead Attention (PMA) to produce order-invariant set representations. It then performs explicit cross-attention between the Radiant and Dire sets (`r2d_attn`, `d2r_attn`) to model compositional synergies and antagonist counters — capturing, for example, how a specific Radiant lineup counters a particular Dire composition. The concatenated cross-attended representation is routed through a final MLP to output a sigmoid scalar predicting the Radiant Win Probability.
* **Optimization:** Evaluated using `BCEWithLogitsLoss`, combined with aggressive label smoothing (`eps=0.15`) and heavy weight decay (`0.1`) to prevent memorization of a noisy, high-variance dataset.

### 3.2. The Policy Head (Masked Language Modeling)
* **Goal:** Predict what a professional team is most likely to do at a given step.
* **Mechanism:** Features a **Slot-Attentive Policy Head** (`SlotAttentionMLMProjection`) that goes beyond simple linear projection. During training, 15% of the draft sequence is masked out (`hero_val = -1`). The head computes cosine similarities between the policy projection and pure hero embeddings, scaled by a learnable temperature parameter (`tau`) to produce sharp, calibrated logits over the hero vocabulary.
* **Self-Supervised Role Prediction:** A sub-network computes 5-slot role distributions (e.g., Carry, Mid, Offlane, Support, Hard Support) and applies expected collision and composition penalties to prevent drafting conflicting roles. This anchors the model's understanding of functional Dota 2 team drafts — not just which heroes are statistically likely, but which heroes fulfill complementary roles within a coherent strategy.
* **Parallel Loss:** The network is optimized on a combined loss function: `Loss = ValueLoss + (0.5 * PolicyLoss)`. This anchors the network, preventing it from collapsing into a naive win-predictor by forcing its internal representations to always understand standard drafting grammar.

---

## 4. Monte Carlo Tree Search (The "Engine")

At inference time, the CLI uses a highly-optimized C++ PyMCTS engine to traverse the massive branching tree of possible drafts.

### 4.1. $O(B)$ PUCT Priors via the Policy Head
Instead of exhaustively calculating the win probability of all 127 child nodes to determine the exploration priors (which requires massive $O(B \times K)$ matrix multiplications), the MCTS agent leverages the **Policy (MLM) Head**.
By passing the current state through the MLM head with an empty slot, the network natively outputs the exact conditional probabilities of all 127 heroes simultaneously. This $O(B)$ operation rockets the MCTS throughput to over `25,000 iters/sec`.

### 4.2. Additive Comfort Scaling
To integrate player specific hero pools into the exploration strategy:
$$ P'(a|s) = Softmax(PolicyLogits(a|s) + ComfortWeight(a)) $$
Using an additive shift rather than a multiplicative scalar ensures that anonymous drafts (where `ComfortWeight` is 0) do not flatten the distribution into a uniform random guess.

### 4.3. Progressive Temperature Scaling
Early in the draft, the branching factor is immense. To optimize search depth:
* **Phase 1 (Steps 0-7):** $Temperature = 0.1$. The priors are sharpened extremely aggressively, forcing the MCTS to only search the absolute mathematically optimal meta heroes.
* **Phase 2 (Steps 8-15):** $Temperature = 0.5$. Allows moderate exploration of synergies and flex picks.
* **Phase 3 (Steps 16-23):** $Temperature = 1.0$. Unscaled distribution, allowing the agent to evaluate wild cheese-picks and hyper-specific counter-picks to close out the draft.

### 4.4. Chunked Leaf Evaluation
To evaluate leaf nodes across massive parallel MCTS searches, the batched tensor sequences are chunked (e.g., `4096` sequences per pass). This keeps GPU utilization at exactly 100% without triggering VRAM saturation or memory-bound latency spikes.

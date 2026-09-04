### 📋 Technical Specification

#### Objective
Update `Architecture.md` and `README.md` to accurately reflect the current, highly-sophisticated implementation details of the Dota 2 Drafter AI models. The current documentation is outdated and oversimplifies several critical mechanisms (e.g., Value Head, Policy Head, and Patch Embeddings), which have evolved into `SetTransformerHead`, `SlotAttentionMLMProjection`, and `FiLM` architectures respectively.

#### Discrepancies Identified
1. **Value Head (Win-Probability)**:
   - *Current Docs*: States the system applies a simple "masked global average pooling over all valid draft steps".
   - *Actual Implementation*: Uses a `SetTransformerHead` that groups Radiant and Dire picks independently using Set Attention Blocks (SAB) and Pooling by Multihead Attention (PMA). It then performs explicit cross-attention between the radiant and dire sets (`r2d_attn`, `d2r_attn`) to model compositional synergies and antagonist counters before routing through a final MLP.
2. **Policy Head (Masked Language Modeling)**:
   - *Current Docs*: Claims the `mlm_head` is a standard linear projection back up to the hero vocabulary size.
   - *Actual Implementation*: Features a `SlotAttentionMLMProjection` (Slot-Attentive Policy Head) with Self-Supervised Role Prediction. It computes cosine similarities between the policy projection and pure hero embeddings, scaled by a learnable temperature (`tau`). Crucially, it incorporates a role-prediction sub-network that computes 5-slot role distributions and applies expected collision and composition penalties to prevent drafting conflicting roles.
3. **Joint Embedding ($z_t$) Patch Integration**:
   - *Current Docs*: Suggests patch embeddings are purely additive: `$ z_t = ... + W_{patch}(patch\_id) $`.
   - *Actual Implementation*: Utilizes Feature-wise Linear Modulation (FiLM). The contextual patch embedding generates `gamma` and `beta` scaling factors that modulate the joint embedding: `$ z_t = \gamma \cdot z_t + \beta $`.

#### Proposed Implementation

**1. Update `Architecture.md`**
- **Section 2.1. The Joint Embedding ($z_t$)**:
  - Update the mathematical formula: 
    `$ z_t = \gamma_{patch} \cdot (Project(H_{GNN}[h_t]) + W_{type}(p_t) + W_{team}(c_t) + PE(o_t)) + \beta_{patch} $`
  - Explain the use of **FiLM (Feature-wise Linear Modulation)** for Contextual Patch Embeddings, detailing how it dynamically scales and shifts the network's hidden state based on historical meta.
- **Section 3.1. The Value Head (Win-Probability)**:
  - Replace "masked global average pooling" with **SetTransformerHead**.
  - Detail the architectural mechanism: It explicitly separates Radiant and Dire hero sets using Set Attention Blocks and Pooling by Multihead Attention. The sets are then cross-attended against each other to accurately evaluate drafted compositions and counters before passing the concatenated representation to the MLP.
- **Section 3.2. The Policy Head (Masked Language Modeling)**:
  - Replace the simple projection description with the **Slot-Attentive Policy Head**.
  - Detail the mechanism: Uses normalized cosine similarity scaled by a dynamically learned temperature (`tau`). Mention the **Self-Supervised Role Prediction** sub-module, which models 5-slot role distributions (e.g., Carry, Mid, Offlane) to apply structural collision and composition penalties, anchoring the model's understanding of functional Dota 2 team drafts.

**2. Update `README.md`**
- **Pipeline Overview - Step 4**:
  - Update the bullet point to read: "A Two-Headed AlphaZero-style Transformer (predicting Win-Probability via a SetTransformer and Policy via Slot-Attentive MLM) utilizing FiLM Contextual Patch Embeddings."

#### Steps for Implementer
1. Directly modify `Architecture.md`, rewriting sections 2.1, 3.1, and 3.2 with the updated mechanisms and formulas detailed above. Preserve all formatting and markdown structure.
2. Modify `README.md` Pipeline Overview Step 4 to include the updated `SetTransformer`, `Slot-Attentive MLM`, and `FiLM` terminology.
3. Validate that the markdown renders correctly and submit the changes.
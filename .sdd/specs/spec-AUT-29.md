# Technical Specification: AUT-29 - Unfilled Slot-Attentive Policy Projection

## Overview
Replaces direct `mlm_logits` projection from the transformer decoder output with a custom, lightweight **Slot-Attentive Policy Projection** mechanism. This architecture introduces 5 learnable slot vectors representing roles/positions (Pos 1–5). A custom Slot Attention module computes occupancy weights $\alpha \in [0, 1]^5$ for these slots based on the active team's past picks. The policy logits are then computed by querying only the remaining unfilled slots ($1 - \alpha$), making it architecturally impossible for already-filled roles to receive high logits in subsequent drafting steps.

---

## 1. File Structure Changes

**Modified Files**:
- `src/dota2drafter/models/match_network.py`
  - Implement the custom `SlotAttentionMLMProjection` class.
  - Update the `HierarchicalTransformer.__init__` method to replace the linear `self.mlm_head = nn.Linear(d_model, num_heroes + 1)` with `self.mlm_head = SlotAttentionMLMProjection(...)`.
  - Update `HierarchicalTransformer.forward` to invoke `self.mlm_head` with additional inputs (`x_draft` and `patch_ids`).
- `src/dota2drafter/training/transformer_trainer.py`
  - Retrieve and accumulate the slot occupancy entropy loss from `self.model.match_network.mlm_head.get_entropy_loss()` inside the training loop.
  - Retrieve and add the same entropy loss to validation metrics inside the `_validate` method.

**Created Files**:
- `tests/test_models/test_slot_attention.py`
  - Add comprehensive unit tests verifying parameter registration, shape constraints, causal masking correctness, entropy regularizer functionality, and role-based logit inhibition behavior.

---

## 2. Interfaces & Signatures

### A. Custom Slot Attention Module
Inside `src/dota2drafter/models/match_network.py`, define the new custom module:

```python
class SlotAttentionMLMProjection(nn.Module):
    """Custom Slot-Attentive Policy Head for positional constraint modeling in drafting."""

    def __init__(
        self,
        d_model: int,
        num_heroes: int,
        joint_embedding: nn.Module,
        entropy_lambda: float = 0.01,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        """Initialize the Slot-Attentive Policy Head.

        Args:
            d_model: Dimension of the transformer embeddings (d_model).
            num_heroes: Number of heroes in the vocabulary (num_heroes + 1 total size).
            joint_embedding: Reference to the JointEmbedding module to query candidate/pick embeddings.
            entropy_lambda: Scaling coefficient for the slot occupancy entropy regularization.
            dropout: Dropout rate applied to slot assignment attention maps.
            temperature: Scaling temperature for softmax inside slot assignment.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heroes = num_heroes
        self.joint_embedding = joint_embedding
        self.entropy_lambda = entropy_lambda
        self.temperature = temperature

        # 5 learnable slot vectors S of shape (5, d_model)
        self.slots = nn.Parameter(torch.randn(5, d_model) * 0.02)

        # Key projection for active team picks
        self.w_k = nn.Linear(d_model, d_model, bias=False)

        # Output projection for query representations
        self.w_policy = nn.Linear(d_model, d_model)

        # Dropout for attention assignments
        self.attn_dropout = nn.Dropout(p=dropout)

        # Store slot entropy loss per forward pass
        self.register_buffer("_entropy_loss", torch.tensor(0.0))

    def forward(
        self,
        decoder_output: torch.Tensor,
        x_draft: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute next-action policy logits using slot attention over unfilled slots.

        Args:
            decoder_output: Output from the transformer decoder, shape (B, 24, d_model).
            x_draft: Draft sequence tensor of shape (B, 24, 4).
            patch_ids: Optional patch ID tensor of shape (B,).

        Returns:
            Policy logits of shape (B, 24, num_heroes + 1).
        """
```

Add helper method to retrieve and reset entropy loss:
```python
    def get_entropy_loss(self) -> torch.Tensor:
        """Retrieve and reset the accumulated slot assignment entropy loss.

        Returns:
            Scalar tensor representing the entropy penalty for this batch.
        """
        loss = self._entropy_loss
        self._entropy_loss = torch.tensor(0.0, device=self.slots.device)
        return loss
```

### B. HierarchicalTransformer Updates
Update class initialization and forward signature:
```python
# In HierarchicalTransformer.__init__:
self.mlm_head = SlotAttentionMLMProjection(
    d_model=d_model,
    num_heroes=num_heroes,
    joint_embedding=self.joint_embedding,
    dropout=dropout,
)

# In HierarchicalTransformer.forward:
mlm_logits = self.mlm_head(decoder_output, x_draft, patch_ids)
```

---

## 3. Mathematical & Algorithmic Design

### A. Causal Active Team Masking
Compute slot occupancy across all 24 steps in a parallel tensor pass using a causal active team pick mask:
1. **Sequence Indices**: `step_indices = torch.arange(24, device=x_draft.device)`
2. **Causal Mask (s < t)**: `causal_mask = step_indices.unsqueeze(0) > step_indices.unsqueeze(1)` (shape `(24, 24)`)
3. **Picks & Validity Mask**: 
   - `is_pick_s = (x_draft[:, :, 0] == 1.0).unsqueeze(1)` (shape `(B, 1, 24)`)
   - `valid_hero_s = (x_draft[:, :, 2] >= 0.0).unsqueeze(1)` (shape `(B, 1, 24)`)
4. **Active Team Agreement**:
   - `team_t = x_draft[:, :, 1].unsqueeze(2)` (shape `(B, 24, 1)`)
   - `team_s = x_draft[:, :, 1].unsqueeze(1)` (shape `(B, 1, 24)`)
   - `same_team = (team_t == team_s)` (shape `(B, 24, 24)`)
5. **Causal Active Picks Mask**:
   - `past_active_picks_mask = causal_mask.unsqueeze(0) & same_team & is_pick_s & valid_hero_s` (shape `(B, 24, 24)`)

### B. Slot Assignment & Occupancy
1. **Pick Embeddings**: Extract pure hero embeddings for all draft actions:
   `E = self.joint_embedding.get_pure_hero_embeddings(x_draft[:, :, 2].long(), patch_ids)` (shape `(B, 24, d_model)`)
2. **Project Keys**: `K = self.w_k(E)` (shape `(B, 24, d_model)`)
3. **Cross-Attention Alignment**: Compute scaled dot products between slots and pick keys:
   `attn_logits = torch.matmul(self.slots, K.transpose(1, 2)) / math.sqrt(self.d_model)` (shape `(B, 5, 24)`)
   `attn_logits = attn_logits.unsqueeze(1).expand(-1, 24, -1, -1)` (shape `(B, 24, 5, 24)`)
4. **Mask Non-Picks**: Set attention logits for inactive/non-pick/future steps to a large negative scalar:
   `mask = past_active_picks_mask.unsqueeze(2)` (shape `(B, 24, 1, 24)`)
   `masked_logits = attn_logits.masked_fill(~mask, -1e9)`
5. **Softmax Normalization**: Assign picks to slots via softmax across slots (dim=2):
   `attn_weights = F.softmax(masked_logits / self.temperature, dim=2)` (shape `(B, 24, 5, 24)`)
   `attn_weights = self.attn_dropout(attn_weights)`
6. **Slot Occupancy Sum**: Sum slot assignments over all picks to compute occupancy weights $\alpha$:
   `alpha = (attn_weights * mask.float()).sum(dim=-1)` (shape `(B, 24, 5)`)
   `alpha = torch.clamp(alpha, 0.0, 1.0)` (guarantees $\alpha \in [0, 1]^5$)

### C. Entropy Regularization Loss
To force slot occupancies to clearly differentiate (converging to 0 or 1), penalize binary entropy:
$$L_{\text{entropy}} = -\lambda \cdot \text{mean} \left( \alpha \log(\alpha + \epsilon) + (1 - \alpha) \log(1 - \alpha + \epsilon) \right)$$
where $\epsilon = 1\text{e}-6$ for numerical stability. Store this scalar in `self._entropy_loss`.

### D. Logits Prediction
1. **Policy Query vector**: Compute $h_{\text{query}}$ as the weighted sum of unfilled slots:
   `h_query = torch.matmul(1.0 - alpha, self.slots)` (shape `(B, 24, d_model)`)
2. **Output Projection**: Project query vectors:
   `h_projected = self.w_policy(h_query)` (shape `(B, 24, d_model)`)
3. **Candidate Hero Embeddings**: Retrieve pure embeddings for all possible hero candidates:
   `all_hero_indices = torch.arange(self.num_heroes + 1, device=x_draft.device).unsqueeze(0).expand(B, -1)`
   `E_hero = self.joint_embedding.get_pure_hero_embeddings(all_hero_indices, patch_ids)` (shape `(B, num_heroes + 1, d_model)`)
4. **Dot Product Logits**: Multiply query and hero tensors:
   `logits = torch.bmm(h_projected, E_hero.transpose(1, 2))` (shape `(B, 24, num_heroes + 1)`)

---

## 4. Edge Cases & Safeguards

- **Zero Active Picks (Step 0 / No past picks)**:
  If the active team has zero prior picks, the `past_active_picks_mask` is completely `False`. `attn_weights * mask.float()` becomes `0.0`, resulting in $\alpha = 0.0$ for all slots. The unfilled slot weight $1 - \alpha$ is exactly $1.0$. The query vector becomes the clean sum of all slot vectors: $h_{\text{query}} = \sum s_k$. This is completely numerically stable and mathematically correct.
- **Bans & Padding Slots**:
  Bans (`x_draft[:, :, 0] == 0.0`) and invalid/padding slots (`x_draft[:, :, 2] < 0.0`) are completely ignored and masked out of key projections. They do not occupy slots.
- **Entropy Loss NaN Safety**:
  Use `alpha + 1e-6` and `(1.0 - alpha) + 1e-6` inside `torch.log` to prevent `NaN` gradients when occupancies reach exactly `0.0` or `1.0`.

---

## 5. Testing Strategy

Add a new test suite in `tests/test_models/test_slot_attention.py`:
1. **Module Construction**: Verify that `SlotAttentionMLMProjection` properly registers learnable parameters: `slots` (shape `5 x d_model`), `w_k`, `w_policy`.
2. **Batch Forward Shape**: Confirm that the forward pass output has shape `(B, 24, num_heroes + 1)`.
3. **Causality Verification**: Verify that changes in picks/actions at step $t$ do not alter policy logits at steps $< t$.
4. **Ignored Actions (Bans & Padding)**: Verify that draft logs consisting solely of bans or invalid steps do not trigger slot occupancy (i.e. $\alpha$ remains exactly `0.0` and `1.0 - \alpha` remains exactly `1.0`).
5. **Entropy Penalty correctness**: Confirm that completely intermediate/undecided slots (e.g. $\alpha = 0.5$) produce higher entropy loss than sharp slots (e.g. $\alpha = 0$ or $\alpha = 1$).
6. **Inference Fusion Safety**: Confirm that `fuse_embeddings_for_inference` does not break slot attention, and works correctly when candidate embeddings are fetched.

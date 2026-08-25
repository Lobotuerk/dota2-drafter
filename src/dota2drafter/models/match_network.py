"""Match network - hierarchical transformer for draft sequence modeling."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for absolute draft positions (0-23)."""

    def __init__(self, d_model: int, max_len: int = 24) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length (24 for full draft).
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (B, seq_len, d_model).

        Returns:
            Tensor with positional encoding added, shape (B, seq_len, d_model).
        """
        return x + self.pe[:, : x.size(1), :]


class SetTransformerHead(nn.Module):
    """Set Transformer Head for permutation-invariant win probability estimation.

    Replaces the pooled output_head with a Set Transformer that models
    team synergies (SAB), cross-team counters (cross-set attention), and
    variable-length set pooling (PMA).
    """

    def __init__(self, d_model: int = 128, dim_feedforward: int = 256, dropout: float = 0.1) -> None:
        """Initialize the Set Transformer Head.

        Args:
            d_model: Transformer embedding dimension.
            dim_feedforward: Feedforward dimension in the value MLP.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model

        # SAB: intra-team synergy attention
        self.sab = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        # Dedicated PMA modules for Radiant and Dire sets
        self.pma_r = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.pma_d = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        # Cross-set attention (both directions)
        self.r2d_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.d2r_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        # PMA seeds (k=1 per team)
        self.seed_r = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.seed_d = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Value MLP
        self.value_mlp = nn.Sequential(
            nn.Linear(2 * d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(
        self,
        hero_embeddings: torch.Tensor,
        x_draft: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the Set Transformer Head.

        Args:
            hero_embeddings: Pure hero embeddings from JointEmbedding, shape (B, seq_len, d_model).
            x_draft: Raw draft sequence tensor of shape (B, seq_len, 4).

        Returns:
            Win probability logits, shape (B,).
        """
        batch_size = x_draft.size(0)

        # --- Vectorized Set Construction ---
        action_mask = x_draft[:, :, 0] == 1.0
        hero_valid = x_draft[:, :, 2] >= 0.0
        pick_mask = action_mask & hero_valid
        team_mask = x_draft[:, :, 1] == 0.0

        r_mask = pick_mask & team_mask
        d_mask = pick_mask & (~team_mask)

        # Zero-pad to max 5 heroes per team
        max_heroes = 5

        # Radiant team set (Vectorized)
        r_cumsum = torch.cumsum(r_mask.long(), dim=-1)
        r_pick_idx = torch.where(r_mask, r_cumsum, torch.tensor(0, device=r_mask.device))
        r_valid_pick = (r_pick_idx >= 1) & (r_pick_idx <= max_heroes)
        rb_coords, rt_coords = torch.where(r_valid_pick)
        rdest_coords = r_pick_idx[rb_coords, rt_coords] - 1

        r_embeds = torch.zeros(batch_size, max_heroes, self.d_model, device=hero_embeddings.device)
        r_embeds[rb_coords, rdest_coords] = hero_embeddings[rb_coords, rt_coords]

        r_pad_mask = torch.ones(batch_size, max_heroes, dtype=torch.bool, device=hero_embeddings.device)
        r_pad_mask[rb_coords, rdest_coords] = False

        # Dire team set (Vectorized)
        d_cumsum = torch.cumsum(d_mask.long(), dim=-1)
        d_pick_idx = torch.where(d_mask, d_cumsum, torch.tensor(0, device=d_mask.device))
        d_valid_pick = (d_pick_idx >= 1) & (d_pick_idx <= max_heroes)
        db_coords, dt_coords = torch.where(d_valid_pick)
        ddest_coords = d_pick_idx[db_coords, dt_coords] - 1

        d_embeds = torch.zeros(batch_size, max_heroes, self.d_model, device=hero_embeddings.device)
        d_embeds[db_coords, ddest_coords] = hero_embeddings[db_coords, dt_coords]

        d_pad_mask = torch.ones(batch_size, max_heroes, dtype=torch.bool, device=hero_embeddings.device)
        d_pad_mask[db_coords, ddest_coords] = False

        # --- NaN Guard for empty team sets ---
        empty_r = r_pad_mask.all(dim=-1, keepdim=True)  # (B, 1)
        empty_d = d_pad_mask.all(dim=-1, keepdim=True)  # (B, 1)

        safe_r_pad_mask = r_pad_mask.clone()
        safe_r_pad_mask[empty_r.squeeze(-1), 0] = False

        safe_d_pad_mask = d_pad_mask.clone()
        safe_d_pad_mask[empty_d.squeeze(-1), 0] = False

        # --- Set Attention Block (SAB) ---
        r_syn, _ = self.sab(r_embeds, r_embeds, r_embeds, key_padding_mask=safe_r_pad_mask)
        d_syn, _ = self.sab(d_embeds, d_embeds, d_embeds, key_padding_mask=safe_d_pad_mask)

        # --- Cross-Set Attention (symmetric) ---
        r_cross, _ = self.r2d_attn(r_syn, d_syn, d_syn, key_padding_mask=safe_d_pad_mask)
        d_cross, _ = self.d2r_attn(d_syn, r_syn, r_syn, key_padding_mask=safe_r_pad_mask)

        # --- Pooling by Multihead Attention (PMA) ---
        seed_r = self.seed_r.expand(batch_size, 1, self.d_model)
        seed_d = self.seed_d.expand(batch_size, 1, self.d_model)

        v_r, _ = self.pma_r(seed_r, r_cross, r_cross, key_padding_mask=safe_r_pad_mask)
        v_d, _ = self.pma_d(seed_d, d_cross, d_cross, key_padding_mask=safe_d_pad_mask)

        v_r = v_r.squeeze(1)  # (B, d_model)
        v_d = v_d.squeeze(1)  # (B, d_model)

        # Zero-out the pooled representations for empty teams to avoid gradients/noise from safe-masked slot
        v_r = v_r * (~empty_r).float()
        v_d = v_d * (~empty_d).float()

        # --- Value MLP ---
        concat = torch.cat([v_r, v_d], dim=-1)  # (B, 2 * d_model)
        logits = self.value_mlp(concat).squeeze(-1)  # (B,)

        return logits


class JointEmbedding(nn.Module):
    """Computes the joint embedding z_t for each draft action a_t = (h_t, p_t, c_t, o_t).

    z_t = Project(H_GNN[h_t]) + W_type e(p_t) + W_team e(c_t) + PE(o_t)
    """

    def __init__(self, d_model: int, num_heroes: int, h_gnn: torch.Tensor, num_patches: int = 30) -> None:
        """Initialize JointEmbedding.

        Args:
            d_model: Transformer embedding dimension.
            num_heroes: Number of heroes K (for embedding matrix sizing).
            h_gnn: Frozen RGCN hero embeddings of shape (K+1, d_model).
            num_patches: Number of unique patches for patch embedding.
        """
        super().__init__()
        self.d_model = d_model

        # Frozen hero embeddings from RGCN
        self.register_buffer("h_gnn", h_gnn)

        # Project: linear layer for hero embeddings
        self.project = nn.Linear(d_model, d_model)

        # Action type embedding (Ban=0, Pick=1)
        self.w_type = nn.Embedding(2, d_model)

        # Team side embedding (Radiant=0, Dire=1)
        self.w_team = nn.Embedding(2, d_model)

        # Patch embedding
        self.w_patch = nn.Embedding(num_patches, d_model)

        # FiLM parameters: Linear layers to generate gamma (scale) and beta (shift)
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)

        # Initialize gamma near 1 and beta near 0 for identity transformation start
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=24)

    def get_pure_hero_embeddings(
        self, hero_indices: torch.Tensor, patch_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract pure hero embeddings without positional/step/action/team tokens.

        Args:
            hero_indices: Hero index tensor of shape (B, seq_len).
            patch_ids: Optional patch ID tensor of shape (B,) for FiLM conditioning.

        Returns:
            Pure hero embeddings of shape (B, seq_len, d_model).
        """
        valid_heroes = (hero_indices >= 0)
        clamped_indices = hero_indices.clamp(min=0).long()
        hero_embeds = self.h_gnn[clamped_indices]  # (B, seq_len, d_model)
        hero_embeds = hero_embeds * valid_heroes.unsqueeze(-1).float() # Zero-out invalid slots
        hero_projected = self.project(hero_embeds)  # (B, seq_len, d_model)

        if patch_ids is not None:
            patch_ids_clamped = torch.clamp(patch_ids, 0, self.w_patch.num_embeddings - 1)
            e_patch = self.w_patch(patch_ids_clamped)  # (B, d_model)
            gamma = self.film_gamma(e_patch).unsqueeze(1)  # (B, 1, d_model)
            beta = self.film_beta(e_patch).unsqueeze(1)  # (B, 1, d_model)
            hero_projected = gamma * hero_projected + beta

        return hero_projected

    @torch.no_grad()
    def fuse_embeddings_for_inference(self):
        """Pre-computes the linear projection to speed up MCTS.
        
        This replaces the project layer with an Identity function and
        pre-multiplies the h_gnn buffer. Call this once before starting MCTS.
        """
        if isinstance(self.project, nn.Identity):
            return  # Already fused
            
        # Compute projection once for all K heroes and replace the raw h_gnn buffer
        fused_h_gnn = self.project(self.h_gnn)
        
        # We must delete the old buffer first before re-registering it with new shape/data
        del self.h_gnn
        self.register_buffer("h_gnn", fused_h_gnn)
        
        # Replace the linear layer with an Identity function to make it a no-op
        self.project = nn.Identity()

    def forward(
        self,
        x_draft: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint embeddings for the draft sequence.

        Args:
            x_draft: Draft sequence tensor of shape (B, 24, 4),
                     where each step is [hero_val, is_pick, team, step_index].
            patch_ids: Patch ID tensor of shape (B,).

        Returns:
            Joint embedding tensor z of shape (B, 24, d_model).
        """
        batch_size, seq_len, _ = x_draft.shape

        # Extract components from x_draft
        hero_indices = x_draft[:, :, 2].long()  # hero_val -> hero indices
        action_types = x_draft[:, :, 0].long()  # is_pick -> action type
        teams = x_draft[:, :, 1].long()  # team -> team side
        step_indices = x_draft[:, :, 3].long()  # step_index -> absolute position

        # Hero embeddings: lookup H_GNN[hero_val], then project
        # hero_indices: (B, 24) -> (B, 24, d_model)
        # self.h_gnn is a registered buffer, natively aligned with module device
        valid_heroes = (hero_indices >= 0)
        clamped_indices = hero_indices.clamp(min=0)
        hero_embeds = self.h_gnn[clamped_indices]  # (B, 24, d_model)
        hero_embeds = hero_embeds * valid_heroes.unsqueeze(-1).float() # Zero-out invalid slots
        hero_projected = self.project(hero_embeds)  # (B, 24, d_model)

        # Action type embeddings
        # Clamp to valid range in case of -1 (ban/invalid)
        action_types_clamped = torch.clamp(action_types, 0, 1)
        type_embeds = self.w_type(action_types_clamped)  # (B, 24, d_model)

        # Team side embeddings
        teams_clamped = torch.clamp(teams, 0, 1)
        team_embeds = self.w_team(teams_clamped)  # (B, 24, d_model)

        # Positional encoding using step index dynamically mapping to actual steps
        step_indices_clamped = torch.clamp(step_indices, 0, 23)
        
        # self.pos_enc.pe has shape (1, 24, d_model)
        # We need to gather the correct PE for each step_index in the batch
        # step_indices_clamped has shape (B, 24)
        
        # Expand PE to match batch size: (B, 24, d_model)
        pe_expanded = self.pos_enc.pe[:, :24, :].expand(batch_size, -1, -1)
        
        # Gather the specific PEs along the sequence dimension based on step_indices
        pos_embeds = torch.gather(
            pe_expanded, 
            dim=1, 
            index=step_indices_clamped.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        # Joint embedding: sum of all components
        z = hero_projected + type_embeds + team_embeds + pos_embeds
        
        # Apply FiLM conditioning if patch_ids is provided
        if patch_ids is not None:
            # 1. Get base patch embedding (e_patch)
            patch_ids_clamped = torch.clamp(patch_ids, 0, self.w_patch.num_embeddings - 1)
            e_patch = self.w_patch(patch_ids_clamped)  # (B, d_model)
            
            # 2. Compute affine parameters and unsqueeze for sequence broadcasting
            gamma = self.film_gamma(e_patch).unsqueeze(1)  # (B, 1, d_model)
            beta = self.film_beta(e_patch).unsqueeze(1)    # (B, 1, d_model)
            
            # 3. Apply FiLM modulation: gamma * z + beta
            z = gamma * z + beta

        return z


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
            d_model: Dimension of the transformer embeddings.
            num_heroes: Number of heroes in the vocabulary (num_heroes + 1 total size).
            joint_embedding: Reference to the JointEmbedding module to
                query candidate/pick embeddings.
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

        # 5 learnable slot vectors representing Pos 1-5
        self.slots = nn.Parameter(torch.randn(5, d_model) * 0.02)

        # Key projection for active team picks
        self.w_k = nn.Linear(d_model, d_model, bias=False)

        # Output projection and normalization
        self.w_policy = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(p=dropout)

        # Entropy loss buffer
        self.register_buffer("_entropy_loss", torch.tensor(0.0))

    def forward(
        self,
        decoder_output: torch.Tensor,
        x_draft: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute next-action policy logits using slot attention over unfilled slots.

        Args:
            decoder_output: Output from the transformer decoder (h_t), shape (B, 24, d_model).
            x_draft: Draft sequence tensor of shape (B, 24, 4).
            patch_ids: Optional patch ID tensor of shape (B,).

        Returns:
            Policy logits of shape (B, 24, num_heroes + 1).
        """
        batch_size, seq_len, _ = x_draft.shape
        device = x_draft.device

        # 1. Causal Active Team Masking (B, 24, 24)
        step_indices = torch.arange(seq_len, device=device)
        causal_mask = step_indices.unsqueeze(0) < step_indices.unsqueeze(1)  # s < t

        is_pick_s = (x_draft[:, :, 0] == 1.0).unsqueeze(1)
        valid_hero_s = (x_draft[:, :, 2] >= 0.0).unsqueeze(1)

        team_t = x_draft[:, :, 1].unsqueeze(2)
        team_s = x_draft[:, :, 1].unsqueeze(1)
        same_team = (team_t == team_s)

        past_active_picks_mask = causal_mask.unsqueeze(0) & same_team & is_pick_s & valid_hero_s

        # 2. Key Projection & Cross-Attention (B, 24, 5, 24)
        E = self.joint_embedding.get_pure_hero_embeddings(x_draft[:, :, 2].long(), patch_ids)
        K = self.w_k(E)  # (B, 24, d_model)

        attn_logits = (
            torch.matmul(self.slots, K.transpose(1, 2)) / math.sqrt(self.d_model)
        )  # (B, 5, 24)
        attn_logits = attn_logits.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (B, 24, 5, 24)

        mask = past_active_picks_mask.unsqueeze(2)  # (B, 24, 1, 24)
        masked_logits = attn_logits.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(masked_logits / self.temperature, dim=2)
        attn_weights = self.attn_dropout(attn_weights)

        # 3. Smooth Differentiable Occupancy & Unfilled Weight
        raw_occupancy = (attn_weights * mask.float()).sum(dim=-1)  # (B, 24, 5)
        unfilled_weight = torch.exp(-raw_occupancy)  # (1 - alpha) in (0, 1]

        # 4. Entropy Regularization
        alpha = 1.0 - unfilled_weight
        eps = 1e-6
        entropy = -(
            alpha * torch.log(alpha + eps) + (1.0 - alpha) * torch.log(1.0 - alpha + eps)
        )
        self._entropy_loss = self.entropy_lambda * entropy.mean()

        # 5. Query Vector Construction & Logit Projection
        h_query = torch.matmul(unfilled_weight, self.slots)  # (B, 24, d_model)
        h_combined = self.layer_norm(decoder_output + h_query)
        h_projected = self.w_policy(h_combined)  # (B, 24, d_model)

        all_hero_indices = (
            torch.arange(self.num_heroes + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        E_hero = self.joint_embedding.get_pure_hero_embeddings(all_hero_indices, patch_ids)

        logits = torch.bmm(h_projected, E_hero.transpose(1, 2))  # (B, 24, num_heroes + 1)
        return logits

    def get_entropy_loss(self) -> torch.Tensor:
        """Retrieve and reset the accumulated slot assignment entropy loss.

        Returns:
            Scalar tensor representing the entropy penalty for this batch.
        """
        loss = self._entropy_loss.clone()
        self._entropy_loss.zero_()
        return loss


class HierarchicalTransformer(nn.Module):
    """Hierarchical Transformer (Match Network) for draft sequence modeling.

    Models the sequence of draft actions and cross-attends with player
    preference vectors to compute win probability.

    Architecture:
    - Joint embedding layer: maps a_t = (h_t, p_t, c_t, o_t) to z_t
    - Transformer body: stacked decoder layers with self-attention + cross-attention
    - Output head: masked global average pooling + MLP + sigmoid
    - MLM head: predicts masked hero identities for pre-training
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_heroes: int = 120,
        h_gnn: Optional[torch.Tensor] = None,
        num_patches: int = 30,
    ) -> None:
        """Initialize the HierarchicalTransformer.

        Args:
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of stacked TransformerDecoderLayer blocks.
            dim_feedforward: Feedforward dimension in decoder layers.
            dropout: Dropout rate.
            num_heroes: Number of heroes K (for embedding matrix sizing).
            h_gnn: Frozen RGCN hero embeddings of shape (K+1, d_model).
                   If None, will be initialized randomly.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heroes = num_heroes

        # Hero embeddings if not provided
        if h_gnn is None:
            h_gnn = torch.randn(num_heroes + 1, d_model)
        self.register_buffer("h_gnn", h_gnn)

        # Joint embedding layer
        self.joint_embedding = JointEmbedding(d_model, num_heroes, self.h_gnn)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Set Transformer head for permutation-invariant value estimation
        self.set_transformer_head = SetTransformerHead(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Slot-Attentive MLM head for positional constraint modeling
        self.mlm_head = SlotAttentionMLMProjection(
            d_model=d_model,
            num_heroes=num_heroes,
            joint_embedding=self.joint_embedding,
            dropout=dropout,
        )

        # Subtractive Role-Inhibition components
        self.w_inhibit = nn.Linear(d_model, d_model, bias=False)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x_draft: torch.Tensor,
        player_pref_vectors: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Match Network.

        Args:
            x_draft: Draft sequence tensor of shape (B, 24, 4).
            player_pref_vectors: Player preference vectors from PlayerComfortNetwork,
                                 shape (B, 10, d_model).
            patch_ids: Patch ID tensor of shape (B,).

        Returns:
            Tuple of:
                - Win probability scalar per sample, shape (B,).
                - MLM logits per step, shape (B, 24, num_heroes + 1).
        """
        batch_size = x_draft.shape[0]

        # 1. Internal Right-Shift for Causal Policy Decoder (NTP Alignment)
        shifted_x_draft = x_draft.clone()
        shifted_x_draft[:, 0, 2] = -1.0           # Step 0 BOS prompt
        shifted_x_draft[:, 1:, 2] = x_draft[:, :-1, 2]  # Right-shift hero IDs

        # Compute joint embeddings using shifted input for causal sequence modeling
        z = self.joint_embedding(shifted_x_draft, patch_ids)  # (B, 24, d_model)

        # Prepare for transformer decoder:
        # query = draft sequence (z), key/value = player preference vectors
        # TransformerDecoder expects (batch, seq, feature) with batch_first=True
        tgt = z  # (B, 24, d_model)
        memory = player_pref_vectors  # (B, 10, d_model)

        # Create a causal mask for the draft sequence to preserve ordering
        seq_len = z.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=z.device), diagonal=1)
        causal_mask = causal_mask.bool()

        # Create a pad mask where True indicates padded/ignored tokens.
        # A token is padded if its hero index is -1.0 (no hero selected).
        pad_mask = (x_draft[:, :, 2] == -1.0)
        pad_mask = pad_mask.to(tgt.device)

        # Run through transformer decoder
        # tgt_mask: (seq_len, seq_len) for self-attention within target
        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )  # (B, 24, d_model)

        # Policy Head: Causal Next-Token Prediction Logits via Slot Attention
        mlm_logits = self.mlm_head(decoder_output, x_draft, patch_ids)

        # Subtractive Role-Inhibition: compute penalty from active team's past picks
        seq_len = x_draft.size(1)

        # 1. Causal Active Team Masking (B, 24, 24)
        step_indices = torch.arange(seq_len, device=x_draft.device)
        causal_mask = step_indices.unsqueeze(0) < step_indices.unsqueeze(1)  # s < t

        is_pick_s = (x_draft[:, :, 0] == 1.0).unsqueeze(1)       # (B, 1, 24)
        valid_hero_s = (x_draft[:, :, 2] >= 0.0).unsqueeze(1)    # (B, 1, 24)

        team_t = x_draft[:, :, 1].unsqueeze(2)  # (B, 24, 1)
        team_s = x_draft[:, :, 1].unsqueeze(1)  # (B, 1, 24)
        same_team = (team_t == team_s)           # (B, 24, 24)

        past_active_picks_mask = causal_mask.unsqueeze(0) & same_team & is_pick_s & valid_hero_s
        mask_weight = past_active_picks_mask.float()  # (B, 24, 24)

        # 2. Extract and Average Active Pick Embeddings (v_active)
        draft_hero_embeds = self.joint_embedding.get_pure_hero_embeddings(
            x_draft[:, :, 2].long(), patch_ids
        )  # (B, 24, d_model)
        sum_embeds = torch.bmm(mask_weight, draft_hero_embeds)     # (B, 24, d_model)
        num_picks = mask_weight.sum(dim=2, keepdim=True)            # (B, 24, 1)
        v_active = sum_embeds / torch.clamp(num_picks, min=1.0)     # (B, 24, d_model)

        # 3. Generate Candidate Inhibition Penalties
        v_inhibit = self.w_inhibit(v_active)  # (B, 24, d_model)
        all_hero_indices = torch.arange(
            self.num_heroes + 1, device=x_draft.device
        ).unsqueeze(0).expand(x_draft.size(0), -1)
        e_hero = self.joint_embedding.get_pure_hero_embeddings(
            all_hero_indices, patch_ids
        )  # (B, K+1, d_model)
        penalty_logits = torch.bmm(v_inhibit, e_hero.transpose(1, 2))  # (B, 24, K+1)

        # 4. Apply Subtractive Penalty to Policy Logits
        inhibition_enabled = (num_picks > 0).float()
        effective_gamma = F.softplus(self.gamma)
        inhibition_penalty = effective_gamma * torch.relu(penalty_logits) * inhibition_enabled
        mlm_logits = mlm_logits - inhibition_penalty

        # Win prediction mode: Set Transformer Head (uses CLEAN, unshifted x_draft for exact hero/team alignment)
        hero_indices = x_draft[:, :, 2]
        pure_hero_embeds = self.joint_embedding.get_pure_hero_embeddings(hero_indices, patch_ids)
        logits = self.set_transformer_head(pure_hero_embeds, x_draft)

        return logits, mlm_logits

    def predict_proba(
        self,
        x_draft: torch.Tensor,
        player_pref_vectors: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute win probability using sigmoid on logits.

        Args:
            x_draft: Draft sequence tensor of shape (B, 24, 4).
            player_pref_vectors: Player preference vectors, shape (B, 10, d_model).
            patch_ids: Patch ID tensor of shape (B,).

        Returns:
            Win probability, shape (B,).
        """
        logits, _ = self.forward(x_draft, player_pref_vectors, patch_ids=patch_ids)
        return torch.sigmoid(logits)


class MatchNetwork(nn.Module):
    """Top-level Match Network that wraps PlayerComfortNetwork + HierarchicalTransformer.

    Provides a unified interface: forward(draft_seq, player_matrices, h_gnn) -> probability.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_heroes: int = 120,
        player_input_dim: int = 127,
        h_gnn: Optional[torch.Tensor] = None,
        num_patches: int = 30,
    ) -> None:
        """Initialize the Match Network.

        Args:
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of stacked TransformerDecoderLayer blocks.
            dim_feedforward: Feedforward dimension in decoder layers.
            dropout: Dropout rate.
            num_heroes: Number of heroes K.
            player_input_dim: C, number of input features per player comfort vector (default: 127, matching total hero count).
            h_gnn: Frozen RGCN hero embeddings of shape (K+1, d_model).
        """
        super().__init__()
        self.d_model = d_model
        self.player_input_dim = player_input_dim

        # Player Network
        from dota2drafter.models.player_network import PlayerComfortNetwork as _PCN

        self.player_network = _PCN(
            input_dim=player_input_dim,
            d_model=d_model,
        )

        # Match Network (Hierarchical Transformer)
        self.match_network = HierarchicalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_heroes=num_heroes,
            h_gnn=h_gnn,
            num_patches=num_patches,
        )

    def forward(
        self,
        x_draft: torch.Tensor,
        player_comfort: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the full Match Network.

        Args:
            x_draft: Draft sequence tensor of shape (B, 24, 4).
            player_comfort: Player comfort tensor of shape (B, 10, C).
            patch_ids: Patch ID tensor of shape (B,).

        Returns:
            Tuple of:
                - Win probability logits, shape (B,).
                - MLM logits, shape (B, 24, num_heroes + 1).
        """
        player_pref_vectors = self.player_network(player_comfort)
        logits, mlm_logits = self.match_network(x_draft, player_pref_vectors, patch_ids=patch_ids)
        return logits, mlm_logits

    def predict_proba(
        self,
        x_draft: torch.Tensor,
        player_comfort: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute win probability.

        Args:
            x_draft: Draft sequence tensor of shape (B, 24, 4).
            player_comfort: Player comfort tensor of shape (B, 10, C).
            patch_ids: Patch ID tensor of shape (B,).

        Returns:
            Win probability, shape (B,).
        """
        logits, _ = self.forward(x_draft, player_comfort, patch_ids=patch_ids)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def fuse_embeddings_for_inference(self):
        """Pre-computes the linear projection to speed up MCTS."""
        self.match_network.joint_embedding.fuse_embeddings_for_inference()

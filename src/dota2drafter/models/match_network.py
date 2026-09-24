"""Match network - hierarchical transformer for draft sequence modeling."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 24) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SetTransformerHead(nn.Module):
    def __init__(self, d_model: int = 128, dim_feedforward: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model

        # Self-Attention Blocks (SABs) per channel
        self.sab_pick = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.sab_ban = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        # Cross-Team Attention: pick2dir, dir2pick, ban2dir, dir2ban
        self.r2d_pick = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.d2r_pick = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.r2d_ban = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.d2r_ban = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        # Cross-Channel Attention: picks attend to bans, bans attend to picks
        self.pick2ban = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.ban2pick = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        # PMA Poolers per channel
        self.pma_r_pick = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.pma_d_pick = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.pma_r_ban = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.pma_d_ban = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        # Seeds per channel
        self.seed_r_pick = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.seed_d_pick = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.seed_r_ban = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.seed_d_ban = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Value MLP: input dim expands from 2*d_model to 4*d_model
        self.value_mlp = nn.Sequential(
            nn.Linear(4 * d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

    @staticmethod
    def _extract_set(mask: torch.Tensor, embeddings: torch.Tensor, max_items: int) -> torch.Tensor:
        """Extract dynamically-sized set into padded tensor using vectorized scatter.

        Args:
            mask: Boolean tensor of shape (batch_size, seq_len) indicating valid items.
            embeddings: Tensor of shape (batch_size, seq_len, d_model) with embeddings.
            max_items: Maximum number of items to pad to.

        Returns:
            Padded tensor of shape (batch_size, max_items, d_model).
        """
        batch_size = embeddings.size(0)
        seq_len = embeddings.size(1)
        d_model = embeddings.size(2)
        device = embeddings.device

        # cumsum: (B, S) gives 1-based position of each valid item within its batch
        cumsum = mask.long().cumsum(dim=-1)  # (B, S)
        valid = (cumsum > 0) & (cumsum <= max_items)  # (B, S)

        # scatter_idx: (B, S) where valid positions have their target position, invalid get 0
        scatter_idx = torch.where(valid, cumsum - 1, torch.zeros(1, dtype=torch.long, device=device))

        # Expand for scatter: (B, S, 1) -> (B, S, d_model)
        scatter_idx_expanded = scatter_idx.unsqueeze(-1).expand(-1, -1, d_model)

        # Zero out invalid embeddings to prevent them from overwriting valid data
        embeddings_masked = embeddings * valid.unsqueeze(-1).float()

        # Scatter into result tensor
        result = torch.zeros(batch_size, max_items, d_model, device=device)
        result.scatter_(1, scatter_idx_expanded, embeddings_masked)

        return result

    def _make_pad_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create a key_padding_mask for attention from a boolean selection mask.

        Args:
            mask: Boolean tensor of shape (batch_size, max_items) indicating valid items.
                  True means the item is valid (not padding).

        Returns:
            key_padding_mask of shape (batch_size, max_items) where True means padding (to be ignored).
        """
        # Pad mask: True means ignore (padding), False means keep
        return ~mask

    def _get_set_pad_masks(
        self,
        action_mask: torch.Tensor,
        ban_mask: torch.Tensor,
        team: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract set masks and create pad masks for all 4 sets.

        Returns dict with 'r_pick', 'd_pick', 'r_ban', 'd_ban' pad masks
        of shape (batch_size, max_items) where True means padding (to be ignored).
        """
        batch_size = action_mask.size(0)
        max_picks = 5
        max_bans = 7

        r_pick_mask = action_mask & (team == 0.0)
        d_pick_mask = action_mask & (team == 1.0)
        r_ban_mask = ban_mask & (team == 0.0)
        d_ban_mask = ban_mask & (team == 1.0)

        # Count valid items per set per batch
        r_pick_count = r_pick_mask.sum(dim=-1, keepdim=True)  # (B, 1)
        d_pick_count = d_pick_mask.sum(dim=-1, keepdim=True)
        r_ban_count = r_ban_mask.sum(dim=-1, keepdim=True)
        d_ban_count = d_ban_mask.sum(dim=-1, keepdim=True)

        # Build pad masks: True means ignore (padding)
        # For each set, items beyond the count are padding
        r_pick_pad_mask = self._make_set_pad_mask(r_pick_mask, r_pick_count, max_picks)
        d_pick_pad_mask = self._make_set_pad_mask(d_pick_mask, d_pick_count, max_picks)
        r_ban_pad_mask = self._make_set_pad_mask(r_ban_mask, r_ban_count, max_bans)
        d_ban_pad_mask = self._make_set_pad_mask(d_ban_mask, d_ban_count, max_bans)

        return {
            "r_pick": r_pick_pad_mask,
            "d_pick": d_pick_pad_mask,
            "r_ban": r_ban_pad_mask,
            "d_ban": d_ban_pad_mask,
            "r_pick_valid": r_pick_mask,
            "d_pick_valid": d_pick_mask,
            "r_ban_valid": r_ban_mask,
            "d_ban_valid": d_ban_mask,
        }

    @staticmethod
    def _make_set_pad_mask(
        valid_mask: torch.Tensor,
        count: torch.Tensor,
        max_items: int,
    ) -> torch.Tensor:
        """Create pad mask from a valid mask and count using vectorized operations.

        Args:
            valid_mask: Boolean (batch_size, seq_len) indicating valid items.
            count: (batch_size, 1) number of valid items per batch.
            max_items: Maximum items per set.

        Returns:
            Boolean (batch_size, max_items) pad mask where True means padding.
        """
        batch_size = valid_mask.size(0)
        device = valid_mask.device
        # Create index: (batch_size, 1) positions 0..max_items-1
        positions = torch.arange(max_items, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, max_items)
        # Pad mask is True where position >= count
        pad_mask = positions >= count  # (B, max_items)
        return pad_mask

    def forward(
        self,
        hero_embeddings: torch.Tensor,
        x_draft: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_draft.size(0)

        # --- Step 1: Extract masks ---
        # Column 0: action_mask (1.0=pick, 0.0=ban)
        # Column 1: team (0.0=Radiant, 1.0=Dire)
        # Column 2: hero_idx (1-based, -1.0 for padding)
        action_mask = (x_draft[:, :, 0] == 1.0) & (x_draft[:, :, 2] >= 0.0)  # Fix 2: filter unpadded hero tokens
        ban_mask = (x_draft[:, :, 0] == 0.0) & (x_draft[:, :, 2] >= 0.0)
        team = x_draft[:, :, 1]  # (B, seq_len)

        # Get pad masks for all sets
        set_masks = self._get_set_pad_masks(action_mask, ban_mask, team)
        r_pick_valid = set_masks["r_pick_valid"]
        d_pick_valid = set_masks["d_pick_valid"]
        r_ban_valid = set_masks["r_ban_valid"]
        d_ban_valid = set_masks["d_ban_valid"]

        # Extract 4 sets using vectorized scatter
        rp_embeds = self._extract_set(r_pick_valid, hero_embeddings, max_items=5)
        dp_embeds = self._extract_set(d_pick_valid, hero_embeddings, max_items=5)
        rb_embeds = self._extract_set(r_ban_valid, hero_embeddings, max_items=7)
        db_embeds = self._extract_set(d_ban_valid, hero_embeddings, max_items=7)

        # Build pad masks for attention layers
        rp_pad_mask = set_masks["r_pick"]  # (B, 5) True=padding
        dp_pad_mask = set_masks["d_pick"]
        rb_pad_mask = set_masks["r_ban"]  # (B, 7) True=padding
        db_pad_mask = set_masks["d_ban"]

        # Expand pad masks for cross-attention (need to match key dimension)
        # For cross-team attention, the key padding mask should match the key set size
        # For cross-channel, the key padding mask should match the concatenated set size
        all_bans_pad_mask = torch.cat([rb_pad_mask, db_pad_mask], dim=-1)  # (B, 14)
        all_picks_pad_mask = torch.cat([rp_pad_mask, dp_pad_mask], dim=-1)  # (B, 10)

        # Handle empty sets: if a batch has no items in a set, the pad mask
        # will be all True. We need to ensure at least one position is valid
        # to prevent NaN in softmax normalization.
        def _safe_pad_mask(pad_mask: torch.Tensor) -> torch.Tensor:
            """Ensure at least one position is valid per batch to prevent NaN."""
            empty = pad_mask.all(dim=-1, keepdim=True)  # (B, 1) True=all padding
            safe_pad_mask = pad_mask.clone()
            # For batches where all items are padding, set first position to valid
            batch_indices = torch.arange(pad_mask.size(0), device=pad_mask.device)
            safe_pad_mask[empty.squeeze(-1), 0] = False
            return safe_pad_mask

        rp_pad_mask = _safe_pad_mask(set_masks["r_pick"])
        dp_pad_mask = _safe_pad_mask(set_masks["d_pick"])
        rb_pad_mask = _safe_pad_mask(set_masks["r_ban"])
        db_pad_mask = _safe_pad_mask(set_masks["d_ban"])
        all_bans_pad_mask = _safe_pad_mask(torch.cat([rb_pad_mask, db_pad_mask], dim=-1))
        all_picks_pad_mask = _safe_pad_mask(torch.cat([rp_pad_mask, dp_pad_mask], dim=-1))

        # Expand seeds to batch dimension
        seed_rp = self.seed_r_pick.expand(batch_size, 1, self.d_model)
        seed_dp = self.seed_d_pick.expand(batch_size, 1, self.d_model)
        seed_rb = self.seed_r_ban.expand(batch_size, 1, self.d_model)
        seed_db = self.seed_d_ban.expand(batch_size, 1, self.d_model)

        # --- Step 2: Self-Attention on each set ---
        rp_syn, _ = self.sab_pick(rp_embeds, rp_embeds, rp_embeds, key_padding_mask=rp_pad_mask)
        dp_syn, _ = self.sab_pick(dp_embeds, dp_embeds, dp_embeds, key_padding_mask=dp_pad_mask)
        rb_syn, _ = self.sab_ban(rb_embeds, rb_embeds, rb_embeds, key_padding_mask=rb_pad_mask)
        db_syn, _ = self.sab_ban(db_embeds, db_embeds, db_embeds, key_padding_mask=db_pad_mask)

        # --- Step 3: Cross-Team Attention ---
        # r2d_pick: Radiant picks attend to Dire picks
        rp_cross, _ = self.r2d_pick(rp_syn, dp_syn, dp_syn, key_padding_mask=dp_pad_mask)
        # d2r_pick: Dire picks attend to Radiant picks
        dp_cross, _ = self.d2r_pick(dp_syn, rp_syn, rp_syn, key_padding_mask=rp_pad_mask)
        # r2d_ban: Radiant bans attend to Dire bans
        rb_cross, _ = self.r2d_ban(rb_syn, db_syn, db_syn, key_padding_mask=db_pad_mask)
        # d2r_ban: Dire bans attend to Radiant bans
        db_cross, _ = self.d2r_ban(db_syn, rb_syn, rb_syn, key_padding_mask=rb_pad_mask)

        # --- Step 4: Cross-Channel Attention ---
        # Concatenate cross-attention outputs for cross-channel queries
        all_bans = torch.cat([rb_cross, db_cross], dim=1)  # (B, 14, d_model)
        all_picks = torch.cat([rp_cross, dp_cross], dim=1)  # (B, 10, d_model)
        # picks query all bans, bans query all picks
        rp_out, _ = self.pick2ban(rp_cross, all_bans, all_bans, key_padding_mask=all_bans_pad_mask)
        dp_out, _ = self.pick2ban(dp_cross, all_bans, all_bans, key_padding_mask=all_bans_pad_mask)
        rb_out, _ = self.ban2pick(rb_cross, all_picks, all_picks, key_padding_mask=all_picks_pad_mask)
        db_out, _ = self.ban2pick(db_cross, all_picks, all_picks, key_padding_mask=all_picks_pad_mask)

        # --- Step 5: PMA Pooling ---
        v_rp, _ = self.pma_r_pick(seed_rp, rp_out, rp_out, key_padding_mask=rp_pad_mask)
        v_dp, _ = self.pma_d_pick(seed_dp, dp_out, dp_out, key_padding_mask=dp_pad_mask)
        v_rb, _ = self.pma_r_ban(seed_rb, rb_out, rb_out, key_padding_mask=rb_pad_mask)
        v_db, _ = self.pma_d_ban(seed_db, db_out, db_out, key_padding_mask=db_pad_mask)

        v_rp = v_rp.squeeze(1)  # (B, d_model)
        v_dp = v_dp.squeeze(1)
        v_rb = v_rb.squeeze(1)
        v_db = v_db.squeeze(1)

        # --- Step 6: Value MLP ---
        concat = torch.cat([v_rp, v_dp, v_rb, v_db], dim=-1)  # (B, 4*d_model)
        logits = self.value_mlp(concat).squeeze(-1)  # (B,)

        return logits


class JointEmbedding(nn.Module):
    def __init__(self, d_model: int, num_heroes: int, h_gnn: torch.Tensor, num_patches: int = 30) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("h_gnn", h_gnn)

        self.project_policy = nn.Linear(d_model, d_model)
        self.project_value = nn.Linear(d_model, d_model)
        self.w_type = nn.Embedding(2, d_model)
        self.w_team = nn.Embedding(2, d_model)
        self.w_patch = nn.Embedding(num_patches, d_model)

        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)

        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=24)

    def get_pure_hero_embeddings(
        self, hero_indices: torch.Tensor, patch_ids: torch.Tensor | None = None, for_value: bool = False
    ) -> torch.Tensor:
        device = hero_indices.device
        valid_heroes = (hero_indices >= 0)
        clamped_indices = hero_indices.clamp(min=0).long()

        project_layer = self.project_value if for_value else self.project_policy

        if isinstance(project_layer, nn.Identity):
            h_gnn = (self.h_gnn_value if for_value else self.h_gnn_policy).to(device)
            hero_embeds = h_gnn[clamped_indices]
            hero_embeds = hero_embeds * valid_heroes.unsqueeze(-1).float()
            hero_projected = hero_embeds
        else:
            h_gnn = self.h_gnn.to(device)
            hero_embeds = h_gnn[clamped_indices]
            hero_embeds = hero_embeds * valid_heroes.unsqueeze(-1).float()
            hero_projected = project_layer(hero_embeds)

        if patch_ids is not None:
            patch_ids_clamped = torch.clamp(patch_ids.to(device), 0, self.w_patch.num_embeddings - 1)
            e_patch = self.w_patch(patch_ids_clamped)
            gamma = self.film_gamma(e_patch).unsqueeze(1)
            beta = self.film_beta(e_patch).unsqueeze(1)
            hero_projected = gamma * hero_projected + beta

        return hero_projected

    @torch.no_grad()
    def fuse_embeddings_for_inference(self):
        if isinstance(self.project_policy, nn.Identity) and isinstance(self.project_value, nn.Identity):
            return
        
        # Fuse policy embeddings
        fused_h_gnn_policy = self.project_policy(self.h_gnn)
        self.register_buffer("h_gnn_policy", fused_h_gnn_policy)
        self.project_policy = nn.Identity()

        # Fuse value embeddings
        fused_h_gnn_value = self.project_value(self.h_gnn)
        self.register_buffer("h_gnn_value", fused_h_gnn_value)
        self.project_value = nn.Identity()

        # Delete original to save memory
        del self.h_gnn

    def forward(
        self,
        x_draft: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x_draft.shape
        device = x_draft.device

        hero_indices = x_draft[:, :, 2].long()
        action_types = x_draft[:, :, 0].long()
        teams = x_draft[:, :, 1].long()
        step_indices = x_draft[:, :, 3].long()

        valid_heroes = (hero_indices >= 0)
        clamped_indices = hero_indices.clamp(min=0)

        if isinstance(self.project_policy, nn.Identity):
            h_gnn = self.h_gnn_policy.to(device)
            hero_embeds = h_gnn[clamped_indices]
            hero_embeds = hero_embeds * valid_heroes.unsqueeze(-1).float()
            hero_projected = hero_embeds
        else:
            h_gnn = self.h_gnn.to(device)
            hero_embeds = h_gnn[clamped_indices]
            hero_embeds = hero_embeds * valid_heroes.unsqueeze(-1).float()
            hero_projected = self.project_policy(hero_embeds)

        action_types_clamped = torch.clamp(action_types, 0, 1)
        type_embeds = self.w_type(action_types_clamped)

        teams_clamped = torch.clamp(teams, 0, 1)
        team_embeds = self.w_team(teams_clamped)

        step_indices_clamped = torch.clamp(step_indices, 0, 23)
        pe_expanded = self.pos_enc.pe[:, :24, :].to(device).expand(batch_size, -1, -1)
        pos_embeds = torch.gather(
            pe_expanded,
            dim=1,
            index=step_indices_clamped.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        z = hero_projected + type_embeds + team_embeds + pos_embeds

        if patch_ids is not None:
            patch_ids_clamped = torch.clamp(patch_ids.to(device), 0, self.w_patch.num_embeddings - 1)
            e_patch = self.w_patch(patch_ids_clamped)
            gamma = self.film_gamma(e_patch).unsqueeze(1)
            beta = self.film_beta(e_patch).unsqueeze(1)
            z = gamma * z + beta

        return z


class SlotAttentionMLMProjection(nn.Module):
    """Slot-Attentive Policy Head with Self-Supervised Role Prediction."""

    def __init__(
        self,
        d_model: int,
        num_heroes: int,
        joint_embedding: nn.Module,
        entropy_lambda: float = 0.10,
        dropout: float = 0.1,
        sharpness_lambda: float = 0.35,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heroes = num_heroes
        self.joint_embedding = joint_embedding
        self.entropy_lambda = entropy_lambda
        self.sharpness_lambda = sharpness_lambda

        self.w_policy = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(p=dropout)

        # Self-Supervised Role Predictor Head (d_model -> 5 roles)
        self.role_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 5)
        )

        self.tau = nn.Parameter(torch.tensor(4.0))

        self.register_buffer("_entropy_loss", torch.tensor(0.0))
        self.raw_occupancy: torch.Tensor | None = None
        self.hero_role_probs: torch.Tensor | None = None

    def forward(
        self,
        decoder_output: torch.Tensor,
        x_draft: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x_draft.shape
        device = x_draft.device

        step_indices = torch.arange(seq_len, device=device)
        causal_mask = step_indices.unsqueeze(0) < step_indices.unsqueeze(1)

        is_pick_s = (x_draft[:, :, 0] == 1.0).unsqueeze(1)
        valid_hero_s = (x_draft[:, :, 2] >= 0.0).unsqueeze(1)

        team_t = x_draft[:, :, 1].unsqueeze(2)
        team_s = x_draft[:, :, 1].unsqueeze(1)
        same_team = (team_t == team_s)

        past_active_picks_mask = causal_mask.unsqueeze(0) & same_team & is_pick_s & valid_hero_s

        # 1. Hero Encodings & 5-Slot Softmax Distributions
        all_hero_indices = (
            torch.arange(self.num_heroes + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        E_hero = self.joint_embedding.get_pure_hero_embeddings(all_hero_indices, patch_ids, for_value=False)

        # Compute 5-slot role vectors r_h using role_head
        role_logits = self.role_head(E_hero)
        hero_role_probs = F.softmax(role_logits / 0.20, dim=-1)
        # Zero out low-confidence tail probabilities (e.g. < 5% noise bleed)
        threshold = 0.05
        hero_role_probs = torch.where(hero_role_probs >= threshold, hero_role_probs, torch.zeros_like(hero_role_probs))

        # Re-normalize remaining probability mass
        hero_role_probs = F.normalize(hero_role_probs, p=1, dim=-1)
        self.hero_role_probs = hero_role_probs

        # 2. Gather the 5-dim role probability for each hero in the draft sequence
        # x_draft[:, :, 2] has shape [batch_size, seq_len].
        # These are hero indices. We clamp them to the range [0, num_heroes] for safe gathering.
        draft_hero_indices = torch.clamp(x_draft[:, :, 2].long(), min=0, max=self.num_heroes)
        
        # Gather the role probability distribution for each drafted hero
        drafted_hero_role_probs = torch.gather(
            hero_role_probs,
            dim=1,
            index=draft_hero_indices.unsqueeze(-1).expand(-1, -1, 5)
        ) # [batch_size, seq_len, 5]

        # 3. Sum the role probabilities over past active picks to get raw occupancy
        # past_active_picks_mask: [batch_size, seq_len, seq_len]
        mask_float = past_active_picks_mask.to(dtype=drafted_hero_role_probs.dtype)
        
        # [batch_size, seq_len, seq_len] @ [batch_size, seq_len, 5] -> [batch_size, seq_len, 5]
        raw_occupancy = torch.bmm(mask_float, drafted_hero_role_probs)
        self.raw_occupancy = raw_occupancy
        
        unfilled_weight = torch.exp(-raw_occupancy) # [batch_size, seq_len, 5]

        # 4. Sharpness Regularization (Ortho loss is removed since self.slots is deleted)
        eps = 1e-6
        role_entropy = -(hero_role_probs * torch.log(hero_role_probs + eps)).sum(dim=-1).mean()

        self._entropy_loss = self.sharpness_lambda * role_entropy

        # 5. Candidate Gating & Logits
        # bmm calculation:
        # unfilled_weight: [batch_size, seq_len, 5]
        # hero_role_probs: [batch_size, num_heroes + 1, 5]
        # bmm(unfilled_weight, hero_role_probs.transpose(1, 2)) -> [batch_size, seq_len, num_heroes + 1]
        g = torch.bmm(unfilled_weight, hero_role_probs.transpose(1, 2))

        norm_w_policy = F.normalize(self.w_policy(decoder_output), p=2, dim=-1)
        norm_E_hero = F.normalize(E_hero, p=2, dim=-1)
        cos_sim_policy = torch.bmm(norm_w_policy, norm_E_hero.transpose(1, 2))
        tau = torch.clamp(self.tau, min=0.1, max=4.0)
        base_logits = tau * cos_sim_policy

        eps = 1e-6
        gate_impact = 3.0 * torch.log(g + eps)
        logits = base_logits + gate_impact

        return logits

    def get_entropy_loss(self) -> torch.Tensor:
        loss = self._entropy_loss.clone()
        self._entropy_loss.zero_()
        return loss


class HierarchicalTransformer(nn.Module):
    """Hierarchical Transformer (Match Network) for draft sequence modeling."""

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
        super().__init__()
        self.d_model = d_model
        self.num_heroes = num_heroes

        if h_gnn is None:
            h_gnn = torch.randn(num_heroes + 1, d_model)
        self.register_buffer("h_gnn", h_gnn)

        self.joint_embedding = JointEmbedding(d_model, num_heroes, self.h_gnn)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.set_transformer_head = SetTransformerHead(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.mlm_head = SlotAttentionMLMProjection(
            d_model=d_model,
            num_heroes=num_heroes,
            joint_embedding=self.joint_embedding,
            dropout=dropout,
            entropy_lambda=0.10,
        )

        self.register_buffer("_collision_loss", torch.tensor(0.0))
        self.register_buffer("_composition_loss", torch.tensor(0.0))

    def forward(
        self,
        x_draft: torch.Tensor,
        player_pref_vectors: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_draft.shape[0]

        shifted_x_draft = x_draft.clone()
        shifted_x_draft[:, 0, 2] = -1.0
        shifted_x_draft[:, 1:, 2] = x_draft[:, :-1, 2]

        z = self.joint_embedding(shifted_x_draft, patch_ids)

        tgt = z
        memory = player_pref_vectors

        seq_len = z.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=z.device), diagonal=1).bool()

        pad_mask = (x_draft[:, :, 2] == -1.0).to(tgt.device)

        if pad_mask.all():
            pad_mask[0, :] = False

        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )

        mlm_logits = self.mlm_head(decoder_output, x_draft, patch_ids)

        # --- Self-Supervised Composition & Collision Loss Computation ---
        hero_role_probs = self.mlm_head.hero_role_probs  # (B, K+1, 5)

        step_indices = torch.arange(seq_len, device=x_draft.device)
        causal_mask = step_indices.unsqueeze(0) < step_indices.unsqueeze(1)

        is_pick_s = (x_draft[:, :, 0] == 1.0)
        valid_hero_s = (x_draft[:, :, 2] >= 0.0)

        # 1. Draft Composition Loss: Sum of 5 team picks' roles must equal [1,1,1,1,1]
        hero_indices_seq = x_draft[:, :, 2].clamp(min=0).long()
        seq_role_probs = torch.gather(
            hero_role_probs,
            dim=1,
            index=hero_indices_seq.unsqueeze(-1).expand(-1, -1, 5)
        )  # (B, 24, 5)

        team0_picks_mask = (x_draft[:, :, 1] == 0.0) & is_pick_s & valid_hero_s
        team1_picks_mask = (x_draft[:, :, 1] == 1.0) & is_pick_s & valid_hero_s

        team0_sum = (seq_role_probs * team0_picks_mask.unsqueeze(-1).float()).sum(dim=1)  # (B, 5)
        team1_sum = (seq_role_probs * team1_picks_mask.unsqueeze(-1).float()).sum(dim=1)  # (B, 5)

        target_ones = torch.ones_like(team0_sum)

        # Count active picks per team in each sample
        t0_count = team0_picks_mask.float().sum(dim=1, keepdim=True) # (B, 1)
        t1_count = team1_picks_mask.float().sum(dim=1, keepdim=True) # (B, 1)

        # Only enforce [1,1,1,1,1] composition when a team has completed all 5 picks
        mask_t0 = (t0_count == 5.0).float()
        mask_t1 = (t1_count == 5.0).float()

        loss_t0 = (F.mse_loss(team0_sum, target_ones, reduction="none") * mask_t0).sum() / torch.clamp(mask_t0.sum() * 5.0, min=1.0)
        loss_t1 = (F.mse_loss(team1_sum, target_ones, reduction="none") * mask_t1).sum() / torch.clamp(mask_t1.sum() * 5.0, min=1.0)
        composition_loss = loss_t0 + loss_t1

        self._composition_loss = composition_loss

        # 2. Policy Role Collision Loss
        team_t = x_draft[:, :, 1].unsqueeze(2)
        team_s = x_draft[:, :, 1].unsqueeze(1)
        same_team = (team_t == team_s)

        past_active_picks_mask = causal_mask.unsqueeze(0) & same_team & is_pick_s.unsqueeze(1) & valid_hero_s.unsqueeze(1)
        num_picks = past_active_picks_mask.float().sum(dim=2, keepdim=True)
        inhibition_enabled = (num_picks > 0).float()

        team_role_occupancy = torch.bmm(past_active_picks_mask.float(), seq_role_probs)  # (B, 24, 5)
        role_collision = torch.bmm(team_role_occupancy, hero_role_probs.transpose(1, 2))  # (B, 24, K+1)

        p_policy = F.softmax(mlm_logits, dim=-1)
        expected_collision = (p_policy * role_collision).sum(dim=-1, keepdim=True)

        self._collision_loss = (expected_collision * inhibition_enabled).mean()
        self.inhibition_penalty = role_collision * inhibition_enabled

        logits = self.set_transformer_head(z, x_draft)

        return logits, mlm_logits

    def predict_proba(
        self,
        x_draft: torch.Tensor,
        player_pref_vectors: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.forward(x_draft, player_pref_vectors, patch_ids=patch_ids)
        return torch.sigmoid(logits)

    def get_collision_loss(self) -> torch.Tensor:
        loss = self._collision_loss.clone()
        self._collision_loss.zero_()
        return loss

    def get_composition_loss(self) -> torch.Tensor:
        loss = self._composition_loss.clone()
        self._composition_loss.zero_()
        return loss


class MatchNetwork(nn.Module):
    """Top-level Match Network."""

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
        super().__init__()
        self.d_model = d_model
        self.player_input_dim = player_input_dim

        from dota2drafter.models.player_network import PlayerComfortNetwork as _PCN

        self.player_network = _PCN(
            input_dim=player_input_dim,
            d_model=d_model,
        )

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
        player_pref_vectors = self.player_network(player_comfort)
        logits, mlm_logits = self.match_network(x_draft, player_pref_vectors, patch_ids=patch_ids)
        return logits, mlm_logits

    def predict_proba(
        self,
        x_draft: torch.Tensor,
        player_comfort: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.forward(x_draft, player_comfort, patch_ids=patch_ids)
        return torch.sigmoid(logits)

    def get_entropy_loss(self) -> torch.Tensor:
        return self.match_network.mlm_head.get_entropy_loss()

    def get_collision_loss(self) -> torch.Tensor:
        return self.match_network.get_collision_loss()

    def get_composition_loss(self) -> torch.Tensor:
        return self.match_network.get_composition_loss()

    @torch.no_grad()
    def fuse_embeddings_for_inference(self):
        self.match_network.joint_embedding.fuse_embeddings_for_inference()
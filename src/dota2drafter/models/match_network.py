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

        self.sab = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.pma_r = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.pma_d = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        self.r2d_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.d2r_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)

        self.seed_r = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.seed_d = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

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
        batch_size = x_draft.size(0)

        action_mask = x_draft[:, :, 0] == 1.0
        hero_valid = x_draft[:, :, 2] >= 0.0
        pick_mask = action_mask & hero_valid
        team_mask = x_draft[:, :, 1] == 0.0

        r_mask = pick_mask & team_mask
        d_mask = pick_mask & (~team_mask)

        max_heroes = 5

        r_cumsum = torch.cumsum(r_mask.long(), dim=-1)
        r_pick_idx = torch.where(r_mask, r_cumsum, torch.tensor(0, device=r_mask.device))
        r_valid_pick = (r_pick_idx >= 1) & (r_pick_idx <= max_heroes)
        rb_coords, rt_coords = torch.where(r_valid_pick)
        rdest_coords = r_pick_idx[rb_coords, rt_coords] - 1

        r_embeds = torch.zeros(batch_size, max_heroes, self.d_model, device=hero_embeddings.device)
        r_embeds[rb_coords, rdest_coords] = hero_embeddings[rb_coords, rt_coords]

        r_pad_mask = torch.ones(batch_size, max_heroes, dtype=torch.bool, device=hero_embeddings.device)
        r_pad_mask[rb_coords, rdest_coords] = False

        d_cumsum = torch.cumsum(d_mask.long(), dim=-1)
        d_pick_idx = torch.where(d_mask, d_cumsum, torch.tensor(0, device=d_mask.device))
        d_valid_pick = (d_pick_idx >= 1) & (d_pick_idx <= max_heroes)
        db_coords, dt_coords = torch.where(d_valid_pick)
        ddest_coords = d_pick_idx[db_coords, dt_coords] - 1

        d_embeds = torch.zeros(batch_size, max_heroes, self.d_model, device=hero_embeddings.device)
        d_embeds[db_coords, ddest_coords] = hero_embeddings[db_coords, dt_coords]

        d_pad_mask = torch.ones(batch_size, max_heroes, dtype=torch.bool, device=hero_embeddings.device)
        d_pad_mask[db_coords, ddest_coords] = False

        empty_r = r_pad_mask.all(dim=-1, keepdim=True)
        empty_d = d_pad_mask.all(dim=-1, keepdim=True)

        safe_r_pad_mask = r_pad_mask.clone()
        safe_r_pad_mask[empty_r.squeeze(-1), 0] = False

        safe_d_pad_mask = d_pad_mask.clone()
        safe_d_pad_mask[empty_d.squeeze(-1), 0] = False

        r_syn, _ = self.sab(r_embeds, r_embeds, r_embeds, key_padding_mask=safe_r_pad_mask)
        d_syn, _ = self.sab(d_embeds, d_embeds, d_embeds, key_padding_mask=safe_d_pad_mask)

        r_cross, _ = self.r2d_attn(r_syn, d_syn, d_syn, key_padding_mask=safe_d_pad_mask)
        d_cross, _ = self.d2r_attn(d_syn, r_syn, r_syn, key_padding_mask=safe_r_pad_mask)

        seed_r = self.seed_r.expand(batch_size, 1, self.d_model)
        seed_d = self.seed_d.expand(batch_size, 1, self.d_model)

        v_r, _ = self.pma_r(seed_r, r_cross, r_cross, key_padding_mask=safe_r_pad_mask)
        v_d, _ = self.pma_d(seed_d, d_cross, d_cross, key_padding_mask=safe_d_pad_mask)

        v_r = v_r.squeeze(1)
        v_d = v_d.squeeze(1)

        v_r = v_r * (~empty_r).float()
        v_d = v_d * (~empty_d).float()

        concat = torch.cat([v_r, v_d], dim=-1)
        logits = self.value_mlp(concat).squeeze(-1)

        return logits


class JointEmbedding(nn.Module):
    def __init__(self, d_model: int, num_heroes: int, h_gnn: torch.Tensor, num_patches: int = 30) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("h_gnn", h_gnn)

        self.project = nn.Linear(d_model, d_model)
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
        self, hero_indices: torch.Tensor, patch_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = hero_indices.device
        valid_heroes = (hero_indices >= 0)
        clamped_indices = hero_indices.clamp(min=0).long()

        h_gnn = self.h_gnn.to(device)
        hero_embeds = h_gnn[clamped_indices]
        hero_embeds = hero_embeds * valid_heroes.unsqueeze(-1).float()
        hero_projected = self.project(hero_embeds)

        if patch_ids is not None:
            patch_ids_clamped = torch.clamp(patch_ids.to(device), 0, self.w_patch.num_embeddings - 1)
            e_patch = self.w_patch(patch_ids_clamped)
            gamma = self.film_gamma(e_patch).unsqueeze(1)
            beta = self.film_beta(e_patch).unsqueeze(1)
            hero_projected = gamma * hero_projected + beta

        return hero_projected

    @torch.no_grad()
    def fuse_embeddings_for_inference(self):
        if isinstance(self.project, nn.Identity):
            return
        fused_h_gnn = self.project(self.h_gnn)
        del self.h_gnn
        self.register_buffer("h_gnn", fused_h_gnn)
        self.project = nn.Identity()

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

        h_gnn = self.h_gnn.to(device)
        hero_embeds = h_gnn[clamped_indices]
        hero_embeds = hero_embeds * valid_heroes.unsqueeze(-1).float()
        hero_projected = self.project(hero_embeds)

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
        E_hero = self.joint_embedding.get_pure_hero_embeddings(all_hero_indices, patch_ids)

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
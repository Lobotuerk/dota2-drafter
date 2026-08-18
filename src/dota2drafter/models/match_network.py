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

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=24)

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
        hero_embeds = self.h_gnn[hero_indices]  # (B, 24, d_model)
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
        
        # Add patch embeddings if provided
        if patch_ids is not None:
            # Clamp in case of unknown patch
            patch_ids_clamped = torch.clamp(patch_ids, 0, self.w_patch.num_embeddings - 1)
            patch_embeds = self.w_patch(patch_ids_clamped).unsqueeze(1) # (B, 1, d_model)
            z = z + patch_embeds

        return z


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

        # Output head: masked global average pooling + MLP + sigmoid
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

        # MLM head for Masked Language Modeling pre-training
        self.mlm_head = nn.Linear(d_model, num_heroes + 1)

    def forward(
        self,
        x_draft: torch.Tensor,
        player_pref_vectors: torch.Tensor,
        mlm_mode: bool = False,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the Match Network.

        Args:
            x_draft: Draft sequence tensor of shape (B, 24, 4).
            player_pref_vectors: Player preference vectors from PlayerComfortNetwork,
                                 shape (B, 10, d_model).
            mlm_mode: If True, return MLM logits for masked hero prediction
                      instead of win probability.
            patch_ids: Patch ID tensor of shape (B,).

        Returns:
            If mlm_mode is False: Win probability scalar per sample, shape (B,).
            If mlm_mode is True: MLM logits per step, shape (B, 24, num_heroes + 1).
        """
        batch_size = x_draft.size(0)

        # Compute joint embeddings
        z = self.joint_embedding(x_draft, patch_ids)  # (B, 24, d_model)

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
        # A token is padded if its step_index is 0.0, except for step 0.
        pad_mask = (x_draft[:, :, 3] == 0.0)
        if pad_mask.size(1) > 0:
            pad_mask[:, 0] = False
        pad_mask = pad_mask.to(tgt.device)

        # Run through transformer decoder
        # tgt_mask: (seq_len, seq_len) for self-attention within target
        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )  # (B, 24, d_model)

        if mlm_mode:
            # MLM mode: return per-step hero prediction logits
            mlm_logits = self.mlm_head(decoder_output)
            return mlm_logits

        # Win prediction mode: masked global average pooling + output head
        is_pick = x_draft[:, :, 0]  # (B, 24)
        valid_mask = is_pick == 1.0  # (B, 24)
        valid_mask = valid_mask.unsqueeze(-1).float()  # (B, 24, 1)

        masked_output = decoder_output * valid_mask  # (B, 24, d_model)
        sum_mask = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).squeeze(-1)  # (B, 1)
        pooled = masked_output.sum(dim=1) / sum_mask  # (B, d_model)

        logits = self.output_head(pooled).squeeze(-1)  # (B,)

        return logits

    def predict_proba(self, x_draft: torch.Tensor, player_pref_vectors: torch.Tensor) -> torch.Tensor:
        """Compute win probability using sigmoid on logits.

        Args:
            x_draft: Draft sequence tensor of shape (B, 24, 4).
            player_pref_vectors: Player preference vectors, shape (B, 10, d_model).

        Returns:
            Win probability, shape (B,).
        """
        logits = self.forward(x_draft, player_pref_vectors, mlm_mode=False)
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
        mlm_mode: bool = False,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the full Match Network.

        Args:
            x_draft: Draft sequence tensor of shape (B, 24, 4).
            player_comfort: Player comfort tensor of shape (B, 10, C).
            mlm_mode: If True, return MLM logits for masked hero prediction.
            patch_ids: Patch ID tensor of shape (B,).

        Returns:
            If mlm_mode is False: Win probability logits, shape (B,).
            If mlm_mode is True: MLM logits, shape (B, 24, num_heroes + 1).
        """
        player_pref_vectors = self.player_network(player_comfort)

        if mlm_mode:
            logits = self.match_network(x_draft, player_pref_vectors, mlm_mode=True, patch_ids=patch_ids)
        else:
            logits = self.match_network(x_draft, player_pref_vectors, mlm_mode=False, patch_ids=patch_ids)

        return logits

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
        logits = self.forward(x_draft, player_comfort, mlm_mode=False, patch_ids=patch_ids)
        return torch.sigmoid(logits)

"""Multi-modal fusion transformer."""
from __future__ import annotations

import torch
from torch import nn

from robovla.config import FusionConfig


class FusionTransformer(nn.Module):
    """Concatenates per-modality tokens, add modality type embeddings,
    and runs self-attention over unified sequence.

    Input shapes:
        vision_tokens: (B, N_v, embed_dim)
        text_tokens:   (B, N_t, embed_dim)
        proprio_tokens:(B, N_p, embed_dim)
    Output:
        (B, N_v + N_t + N_p, embed_dim) fused context tokens.
    """

    def __init__(self, config: FusionConfig, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.modality_embedding = nn.Embedding(3,embed_dim)
        self.register_buffer("modality_ids", torch.tensor([0,1,2],dtype=torch.long))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = config.num_heads,
            dim_feedforward = int(embed_dim*config.mlp_ratio),
            dropout = config.dropout,
            activation = "gelu",
            batch_first = True,
            norm_first = True,
        )

        # enable_nested_tensor=False silences a harmless PyTorch warning
        # when combined with norm_first=True; the nested tensor fast path
        # is incompatible with pre-LN layers.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

    def forward(self,vision_tokens: torch.Tensor, text_tokens: torch.Tensor, proprio_tokens: torch.Tensor) -> torch.Tensor:
        """Fuse the three token streams via concat + self.attention.

        Each modality is tagged with a learned type embedding before concatenation so encoder can distinguish
        """
        
        v_embed = self.modality_embedding(self.modality_ids[0]) # (D,)
        t_embed = self.modality_embedding(self.modality_ids[1]) # (D,)
        p_embed = self.modality_embedding(self.modality_ids[2]) # (D,)
        
        vision_tokens = vision_tokens + v_embed # broadcasts (D,) to (B,N_v,D)
        text_tokens = text_tokens + t_embed
        proprio_tokens = proprio_tokens + p_embed

        fused = torch.cat([vision_tokens,text_tokens,proprio_tokens], dim=1) # (B, N_total, D)
        return self.encoder(fused)


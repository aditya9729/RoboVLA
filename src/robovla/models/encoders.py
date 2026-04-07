"""
Author: Aditya Gudal

Encoders.
Per-modality encoders that project to a shared embed_dim.
"""
from __future__ import annotations
from typing_extensions import override

import torch
from torch import nn
import torch.nn.functional as F

import open_clip

from robovla.config import ProprioConfig, TextConfig, VisionConfig


class VisionEncoder(nn.Module):
    """CLIP visual backbone returning per-patch tokens projected to embed_dim.
    Output shape: (batch, num_tokens, embed_dim) where num_tokens = patches + CLS
    """
    def __init__(self,config: VisionConfig,embed_dim: int) -> None:
        super().__init__()

        self.config = config

        #Vision transformer + torchvision transforms - resize, center crop and normalize
        model, _, self.preprocess = open_clip.create_model_and_transforms(self.config.model_name,pretrained=self.config.pretrained)
        self.visual = model.visual
        del model # Garbage collect
        self.visual.output_tokens = True # gives tokens as well - Tuple(pooled, tokens)

        transformer_width = self.visual.transformer.width
        self.proj = nn.Linear(transformer_width, embed_dim)

        if self.config.freeze:
            for param in self.visual.parameters():
                param.requires_grad = False
            self.visual.eval()

    @override
    def train(self,mode: bool = True) -> "VisionEncoder":
        super().train(mode)

        if self.config.freeze:
            self.visual.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, tokens = self.visual(images)
        return self.proj(tokens)


class TextEncoder(nn.Module):
    """Clip Text tower returning per-token embeddings projected to embed_dim.
    
    Output shape: (batch, context_length, embed_dim)
    """
    def __init__(self,config: TextConfig, embed_dim: int) -> None:
        super().__init__()
        self.config = config

        model, _, _ = open_clip.create_model_and_transforms(                     
              config.model_name, pretrained=config.pretrained,
        )

        # Get components
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final

        # Since attn_mask is a buffer, let's register it so it moves to device
        self.register_buffer("attn_mask", model.attn_mask)

        self.tokenizer = open_clip.get_tokenizer(self.config.model_name)
        text_width = model.transformer.width

        self.proj = nn.Linear(text_width, embed_dim)

        # GC
        del model

        if config.freeze:
            for module in (self.token_embedding,self.transformer,self.ln_final):

                for param in module.parameters():
                    param.requires_grad = False

            self.positional_embedding.requires_grad = False # Since its a param
            self.transformer.eval()
    
    @override
    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)

        if self.config.freeze:
            self.transformer.eval()
        return self

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenized text to per-token embeddings.

        Args:
            token_ids: (B, context_length) int64  from self.tokenizer
        
        Returns:
            (B, context_length, embed_dim) per-token embeddings.
        """

        x = self.token_embedding(token_ids) #(B,L,width)
        x = x + self.positional_embedding[:x.shape[1]]
        x = x.permute(1,0,2) #(L, B, width)

        x = self.transformer(x, attn_mask = self.attn_mask)  # causal self-attention

        x = x.permute(1,0,2) # (B, L, width)
        x = self.ln_final(x) # (B, L, width)
        return self.proj(x) # (B, L, embed_dim)


class ProprioEncoder(nn.Module):
    """MLP that maps a flat proprio vector to (num_tokens, embed_dim).

    Output shape: (batch, num_tokens, embed_dim)
    """
    def __init__(self, config: ProprioConfig, embed_dim: int) -> None:
        super().__init__()
        self.num_tokens = config.num_tokens
        self.embed_dim = embed_dim 
        self.linear1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.linear2 = nn.Linear(config.hidden_dim,config.num_tokens * embed_dim)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """Encode proprioception to (num_tokens, embed_dim) per sample.

        Args:
            proprio: (B, input_dim) raw proprio vector.
        
        Returns:
            (B, num_tokens, embed_dim) tokens
        """
        # Input: (B, input_dim)
        x = F.elu(self.linear1(proprio)) # (B, hidden_dim)
        x = self.linear2(x)             # (B, num_tokens * embed_dim)
        return x.view(x.shape[0], self.num_tokens, self.embed_dim)  # Reshape -> (B, num_tokens, embed_dim)

       

    



        




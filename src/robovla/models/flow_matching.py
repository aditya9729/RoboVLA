"""Flow matching action head with a small DiT-style denoiser."""                 
                                                                                   
from __future__ import annotations
                                                                                
import math                                                                     

import torch                                                                     
import torch.nn.functional as F
from torch import nn

from robovla.config import FlowMatchingConfig
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) -> (B, dim) sine embedding"""
    if dim % 2 != 0:
          raise ValueError(f"sinusoidal dim must be even, got {dim}")
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device = t.device) / half)

    args = t[:, None] * freqs[None] # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim = -1) # (B, dim)


class FlowMatchingDenoiser(nn.Module):
    """Predicts velocity v(a_t,t,context) for rectified flow matching.

    Input:
        noisy_actions: [B,H,A]
        t: (B,) in U[0,1]
        context: (B,N,D) from fusion transformer
    Output:
        velocity: (B,H,A)
    """
    def __init__(self, config: FlowMatchingConfig, embed_dim: int, action_dim: int, action_horizon: int) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.action_in = nn.Linear(action_dim, embed_dim)
        self.action_pos_embed = nn.Parameter(torch.zeros(1, action_horizon, embed_dim))
        nn.init.trunc_normal_(self.action_pos_embed, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim,embed_dim)
        )

        # DiT-lite: TransformerDecoder with self-attn + cross-attn to context
        decoder_layer = nn.TransformerDecoderLayer(d_model = embed_dim, nhead = config.num_heads,
                                                    dim_feedforward = int(embed_dim * config.mlp_ratio),
                                                    dropout = 0.0,
                                                    batch_first = True, norm_first = True, activation = "gelu")
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = config.num_layers)
        self.action_out = nn.Linear(embed_dim,action_dim)
    
    def forward(self, noisy_actions: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.action_in(noisy_actions) # BHA -> BHD
        x = x +  self.action_pos_embed
        t_emb = sinusoidal_embedding(t,self.embed_dim) # BD
        t_emb = self.time_mlp(t_emb) # BD
        x = x + t_emb[:,None,:] #broadcast - BHD + B1D
        x = self.decoder(tgt=x,memory=context) #BHD
        return self.action_out(x) # BHA


class FlowMatchingActionHead(nn.Module):
    """Denoiser with training (compute_loss) and inference (sample)"""
    def __init__(self, config: FlowMatchingConfig,embed_dim: int, action_dim: int, action_horizon: int) -> None:
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.denoiser = FlowMatchingDenoiser(config,embed_dim,action_dim, action_horizon)

    def compute_loss(self, action_chunks: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Rectified flow matching loss. Sample a random time t ~ U(0,1) and noise a_1 ~ N(0, I)
        Linearly interpolate a_t = (1-t)*a_0 + t*a_1, and trains the denoiser to predict the constant velocity (a_1 - a_0)

        Returns:
            MSE loss
        """
        batch_size = action_chunks.shape[0]
        device = action_chunks.device 

        # Sample noise with same shape as chunks
        noise = torch.randn_like(action_chunks) # a_1

        # Sample a flow time per batch element
        t = torch.rand(batch_size, device=device) # (B,) in [0,1]
        t_expanded = t.view(-1,1,1) #(B,1,1)

        # Linear interpolation along straight line flow
        noisy_actions = (1.0 - t_expanded) * action_chunks + t_expanded * noise

        target_velocity = noise - action_chunks

        pred_velocity = self.denoiser(noisy_actions, t, context)
        return F.mse_loss(pred_velocity,target_velocity)

    # Sampling
    @torch.no_grad()
    def sample(self, context: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """Generate an action chunk by integrating the learned velocity field.

        Args:
            context: (B,N,D) fused context tokens.
            num_steps: Number of Euler steps. - config.num_sampling_steps.

        Returns:
            (B,H,A) sampled action chunk.
        """

        steps = num_steps if num_steps is not None else self.config.num_sampling_steps

        batch_size = context.shape[0]
        device = context.device 

        # Start with white noise at t = 1
        a = torch.randn(batch_size, self.action_horizon,self.action_dim,device=device)

        # Time grid from 1.0 down to 0.0, inclusive
        ts = torch.linspace(1.0,0.0,steps + 1, device=device)

        # Euler integration backward along the flow.
        for i in range(steps):
            t_curr = ts[i].expand(batch_size)  #(B,)
            dt = ts[i] - ts[i+1]
            v = self.denoiser(a,t_curr,context)
            a = a - dt*v # Noise to data

        return a










"""Top-level VLA policy: encoders -> fusion -> flow matching action head."""     
                                                                                   
from __future__ import annotations                                               
                                                                                
import torch                                                                   
from torch import nn

from robovla.config import VLAPolicyConfig                                       
from robovla.models.encoders import ProprioEncoder, TextEncoder, VisionEncoder
from robovla.models.fusion import FusionTransformer                              
from robovla.models.flow_matching import FlowMatchingActionHead 


class VLAPolicy(nn.Module):
    """ Vision - Language - Action policy with a flow matching action head.

    Inputs (in forward/sample):
        images: (B,3,H,W) preprocessed image tensor
        text_ids: (B,L) tokenized ids from tokenizer 
        proprio:  (B,P) raw proprioception vector
    
    Training:
        loss = policy(images, text_ods, proprio, action_chunks)

    Inference:
        actions = policy.sample(imgaes, text_ids, proprio) # (B,H,A)
    """
    def __init__(self, config: VLAPolicyConfig) -> None:
        super().__init__()
        self.config = config

        self.vision = VisionEncoder(config.vision, config.embed_dim)
        self.text = TextEncoder(config.text, config.embed_dim)
        self.proprio = ProprioEncoder(config.proprio, config.embed_dim)
        self.fusion = FusionTransformer(config.fusion, config.embed_dim)
        self.action_head = FlowMatchingActionHead(
            config = config.flow,
            embed_dim=config.embed_dim,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )

    def encode_context(self, images: torch.Tensor, text_ids: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Run all encoders + fusion. Returns (B,N,D) context tokens"""
        visual_latent = self.vision(images)
        text_latent = self.text(text_ids)
        proprio_latent= self.proprio(proprio)
        return self.fusion(visual_latent,text_latent,proprio_latent)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor, proprio: torch.Tensor,action_chunks: torch.Tensor) -> torch.Tensor:
        """Training forward: returns flow matching MSE loss"""
        context = self.encode_context(images, text_ids, proprio)
        return self.action_head.compute_loss(action_chunks,context)

    @torch.no_grad()
    def sample(self, images: torch.Tensor, text_ids: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Inference: returns sampled action chunk/rollout (B,H,A)"""
        context = self.encode_context(images, text_ids, proprio)
        return self.action_head.sample(context)

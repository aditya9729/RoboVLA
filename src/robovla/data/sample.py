
import torch
from typing import NamedTuple

class Sample(NamedTuple):
    """One synthetic sample

    Fields:
        image: Preprocessed image tensor (C,H,W)
        instruction: Raw language string(which will be padded and tokenized by collate)
        proprio: Proprioception vector (proprio_dim,)
        action_chunk: Ground-truth action chunk (action_horizon, action_dim).
    """
    image: torch.Tensor
    instruction: str
    proprio: torch.Tensor
    action_chunk: torch.Tensor

class BatchedSample(NamedTuple):
    """A collated batch of samples with instructions already tokenized."""
    images: torch.Tensor # (B,3,H,W)
    text_ids: torch.Tensor # (B,L)
    proprios: torch.Tensor # (B,P)
    action_chunks: torch.Tensor # (B,H,A)
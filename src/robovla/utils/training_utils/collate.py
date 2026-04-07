from typing import Callable
import torch
from robovla.data.sample import Sample, BatchedSample

def collate_fn(tokenizer: Callable[[list[str]],torch.Tensor]) -> Callable[[list[Sample]], BatchedSample]:
    """Builds a collate function that batches samples and tokenizes instructions"""

    def collate(batch: list[Sample]) -> BatchedSample:
        images = torch.stack([s.image for s in batch],dim = 0) # (B,3,H,W)

        instructions = [s.instruction for s in batch]   # list[str]
        text_ids = tokenizer(instructions)  #(B,L)
        proprios = torch.stack([s.proprio for s in batch], dim=0) #(B,P)
        actions = torch.stack([s.action_chunk for s in batch], dim=0) # (B,H,A)
        return BatchedSample(
            images = images,
            text_ids = text_ids,
            proprios=proprios,
            action_chunks=actions,
        )
    return collate
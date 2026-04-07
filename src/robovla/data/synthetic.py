"""Toy dataset Generation"""
from __future__ import annotations

from collections.abc import Callable

import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from robovla.config import SyntheticDataConfig
from robovla.data.sample import Sample




class SyntheticDataset(Dataset[Sample]):
    """Random tensors mimicking teleop dataset to be collected"""
    def __init__(self, config: SyntheticDataConfig,preprocess: Callable[[Image.Image],torch.Tensor]) -> None:
        self.config = config
        self.preprocess = preprocess

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> Sample:

        del index # fresh sample with some randomness
        size = self.config.image_size
        image_array = np.random.randint(0,256,(size,size,3), dtype=np.uint8)

        pil_image = Image.fromarray(image_array)
        image  = self.preprocess(pil_image)
        
        instruction = random.choice(self.config.vocabulary)

        proprio = torch.randn(self.config.proprio_dim)

        action_chunk = torch.randn(self.config.action_horizon,self.config.action_dim)

        return Sample(
            image = image,
            instruction = instruction,
            proprio=proprio, 
            action_chunk=action_chunk
        )
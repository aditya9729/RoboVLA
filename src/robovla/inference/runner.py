"""Inference runner: load a checkpoint and expose a simple act() API."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from robovla.data.sample import BatchedSample
from robovla.inference.checkpoint import load_policy_from_checkpoint


class InferenceRunner:
    """Loads a trained VLA policy and exposes single-sample and batch APIs.

    Usage:
        runner = InferenceRunner(Path("artifacts/run_step000050.pt"), device="cpu")
        actions = runner.act(image=pil_image, instruction="pick up the cup", proprio=p)
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.policy, self.policy_config = load_policy_from_checkpoint(checkpoint_path, self.device)
        # Convenience handles for callers that want to preprocess outside.
        self.preprocess = self.policy.vision.preprocess
        self.tokenizer = self.policy.text.tokenizer

    @torch.no_grad()
    def act(self,image: Image.Image,instruction: str,proprio: torch.Tensor) -> torch.Tensor:
        """Run inference on a single sample.

        Args:
            image: Raw PIL image (preprocessed via the policy's CLIP transform).
            instruction: Raw natural-language instruction.
            proprio: Proprioception vector of shape ``(P,)``.

        Returns:
            Action chunk of shape ``(action_horizon, action_dim)``.
        """
        if proprio.ndim != 1:
            raise ValueError(f"proprio must be 1-D, got shape {tuple(proprio.shape)}")

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        text_ids = self.tokenizer([instruction]).to(self.device)            # (1, L)
        proprio_batch = proprio.unsqueeze(0).to(self.device)                # (1, P)

        actions = self.policy.sample(images=image_tensor,text_ids=text_ids,proprio=proprio_batch)
        return actions[0]  # drop batch dim -> (H, A)

    @torch.no_grad()
    def act_batch(self, batch: BatchedSample) -> torch.Tensor:
        """Run inference on an already-collated batch.

        Args:
            batch: A ``BatchedSample`` whose tensors live on any device — they
                will be moved to ``self.device`` before sampling.

        Returns:
            Sampled action chunks of shape ``(B, action_horizon, action_dim)``.
        """
        return self.policy.sample(
            images=batch.images.to(self.device),
            text_ids=batch.text_ids.to(self.device),
            proprio=batch.proprios.to(self.device),
        )

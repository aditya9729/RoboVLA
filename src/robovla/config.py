"""Pydantic configuration objects for RoboVLA.

All model, data, and training configuration lives here. Configs are
hierarchical: ``VLAPolicyConfig`` composes the per-component configs so a
single object fully describes a model.

Conventions
-----------
- ``embed_dim`` is the shared token width across vision, proprio, language,
  fusion, and the action head. Keeping it shared simplifies fusion.
- ``action_horizon`` is the number of future timesteps the policy predicts
  per forward pass (action chunking, à la pi0 / Diffusion Policy).
"""

from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field


class StrictBase(BaseModel):
    """Project-wide strict Pydantic base.

    Forbids unknown fields and freezes instances so configs are hashable
    and safe to share across processes / dataloaders.
    """

    model_config = ConfigDict(extra="forbid",frozen=True,arbitrary_types_allowed=False,validate_assignment=True)


# Model Components

# For vision encoder
class VisionConfig(StrictBase):
    """Vision encoder configuration.

    We use ``open_clip_torch`` so vision and text share the same CLIP
    pretraining. The encoder is frozen by default; only its projected token
    outputs flow into the fusion transformer.
    """

    model_name: str = Field(default="ViT-B-32",description="open_clip model architecture name.")
    pretrained: str = Field(default="openai",description="open_clip pretrained tag (e.g. 'openai', 'laion2b_s34b_b79k').")
    image_size: int = Field(default=224, description="Square input image side length in pixels.")
    output_dim: int = Field(default=512, description="Native token width emitted by the backbone.")
    freeze: bool = Field(default=True, description="Freeze backbone parameters during training.")
    num_tokens: int = Field(default=50,description="Number of visual tokens (CLS + patch tokens for ViT-B/32 at 224px = 50).")


# For text encoder
class TextConfig(StrictBase):
    """Language encoder configuration.

    Reuses the CLIP text tower from ``open_clip_torch`` so vision and language
    share a pretraining distribution. 
    Tokenization uses the matching CLIP tokenizer.
    """

    model_name: str = Field(default="ViT-B-32",description="open_clip model architecture name (must match VisionConfig).")
    pretrained: str = Field(default="openai",description="open_clip pretrained tag (must match VisionConfig).")
    output_dim: int = Field(default=512, description="Native token width emitted by the tower.")
    context_length: int = Field(default=77,description="open_clip text context length (default CLIP value).")
    freeze: bool = Field(default=True, description="Freeze text tower parameters during training.")


class ProprioConfig(StrictBase):
    """Proprioceptive (joint state) encoder configuration."""

    input_dim: int = Field(description="Raw proprioception vector size (e.g., joint angles).")
    hidden_dim: int = Field(default=256, description="Hidden width of the proprio MLP.")
    num_tokens: int = Field(default=1,description="Number of proprio tokens emitted (>=1 lets us reshape into multi-token).")


class FusionConfig(StrictBase):
    """Multi-modal fusion transformer configuration."""

    num_layers: int = Field(default=4, description="Transformer encoder layers.")
    num_heads: int = Field(default=8, description="Attention heads per layer.")
    mlp_ratio: float = Field(default=4.0, description="FFN width = embed_dim * mlp_ratio.")
    dropout: float = Field(default=0.0, description="Dropout inside the fusion stack.")


class FlowMatchingConfig(StrictBase):
    """Flow matching action head configuration.

    The denoiser is a small DiT-style transformer that predicts the velocity
    field along straight-line paths between data and noise (rectified flow).
    """

    num_layers: int = Field(default=4, description="Denoiser transformer layers.")
    num_heads: int = Field(default=8, description="Denoiser attention heads per layer.")
    mlp_ratio: float = Field(default=4.0, description="Denoiser FFN expansion ratio.")
    num_sampling_steps: int = Field(default=8,description="Number of Euler integration steps used at inference.")


class VLAPolicyConfig(StrictBase):
    """Full VLA policy configuration composing every sub-component."""

    embed_dim: int = Field(default=512,description="Shared token embedding dimension across vision/text/proprio/fusion/action.")
    action_dim: int = Field(default=128,description="Per-timestep action vector dimensionality.")
    action_horizon: int = Field(default=8,description="Number of future action steps predicted per forward pass (chunking).")
    vision: VisionConfig = Field(default_factory=VisionConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    proprio: ProprioConfig = Field(default_factory=ProprioConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    flow: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)


# Data  generating data for now.
class SyntheticDataConfig(StrictBase):
    """Configuration for the synthetic toy dataset used to validate shapes."""

    num_samples: int = Field(default=256, description="Total samples in the synthetic dataset.")
    image_size: int = Field(default=224, description="Square image side length.")
    proprio_dim: int = Field(description="Proprio vector size — must match policy.proprio.input_dim.")
    action_dim: int = Field(description="Action vector size — must match policy.action_dim.")
    action_horizon: int = Field(default=8, description="Action chunk length per sample.")
    vocabulary: tuple[str, ...] = Field(
        default=(
            "pick up the red block",
            "place the cup on the shelf",
            "open the drawer",
            "push the button",
        ),description="Synthetic instruction strings sampled per example.",
    )

# For Training loop 
class TrainingConfig(StrictBase):
    """Toy training loop configuration."""

    batch_size: int = Field(default=8, description="Mini-batch size.")
    num_steps: int = Field(default=50, description="Total optimizer steps to run.")
    learning_rate: float = Field(default=3e-4, description="AdamW learning rate.")
    weight_decay: float = Field(default=1e-4, description="AdamW weight decay.")
    grad_clip: float = Field(default=1.0, description="Global gradient norm clipping value.")
    log_every: int = Field(default=5, description="Steps between log prints.")
    seed: int = Field(default=0, description="Global RNG seed.")
    device: str = Field(default="cpu", description="Torch device string ('cpu' / 'cuda' / 'mps').")

    artifacts_dir: Path = Field(default=Path("artifacts"), description="Checkpoint output directory.")                                                            
    logs_dir: Path = Field(default=Path("logs"), description="Log file directory.")  
    checkpoint_every: int = Field(default=25, description="Steps between checkpoint saves.")                                                                         
    run_name: str = Field(default="run", description="Used as filename prefix for logs and ckpts.") 
"""Checkpoint discovery and loading utilities."""
from __future__ import annotations
from pathlib import Path
import torch
from robovla.config import VLAPolicyConfig
from robovla.models.policy import VLAPolicy


def find_latest_checkpoint(artifacts_dir: Path, run_name: str | None = None) -> Path:
    """Return the most recently modified checkpoint in ``artifacts_dir``.

    Args:
        artifacts_dir: Directory containing ``.pt`` checkpoint files.
        run_name: If provided, only consider files prefixed with ``f"{run_name}_"``.

    Returns:
        Absolute path to the most recent matching checkpoint.

    Raises:
        FileNotFoundError: If ``artifacts_dir`` doesn't exist or no matching
            checkpoints are present.
    """
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"artifacts_dir does not exist: {artifacts_dir}")

    pattern = f"{run_name}_*.pt" if run_name is not None else "*.pt"
    candidates = sorted(
        artifacts_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching {pattern!r} found in {artifacts_dir}")
        
    return candidates[0]


def load_policy_from_checkpoint(ckpt_path: Path,device: torch.device) -> tuple[VLAPolicy, VLAPolicyConfig]:
    """Reconstruct a VLAPolicy and its config from a checkpoint file.

    Args:
        ckpt_path: Path to a ``.pt`` file written by ``Runner._save_checkpoint``.
        device: Target device.

    Returns:
        ``(policy_in_eval_mode, policy_config)``.
    """
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy_config = VLAPolicyConfig.model_validate(payload["policy_config"])
    policy = VLAPolicy(policy_config).to(device)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    return policy, policy_config
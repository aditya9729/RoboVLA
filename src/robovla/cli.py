"""Command-line interface for RoboVLA.

Exposes two subcommands:

    robovla train   # run a toy training loop on synthetic data
    robovla sample  # load the latest checkpoint and sample an action chunk
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
from PIL import Image

from robovla.config import (
    ProprioConfig,
    SyntheticDataConfig,
    TrainingConfig,
    VLAPolicyConfig,
)
from robovla.inference import InferenceRunner, find_latest_checkpoint
from robovla.training.runner import Runner


@click.group()
def main() -> None:
    """RoboVLA — a minimal Vision-Language-Action policy."""


@main.command()
@click.option("--run-name", default="run", show_default=True, help="Run identifier.")
@click.option("--num-steps", type=int, default=None, help="Override TrainingConfig.num_steps.")
@click.option(
    "--batch-size", type=int, default=None, help="Override TrainingConfig.batch_size.",
)
@click.option("--device", default="cpu", show_default=True, help="torch device string.")
@click.option(
    "--artifacts-dir",
    default="artifacts",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where to write checkpoints.",
)
@click.option(
    "--logs-dir",
    default="logs",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where to write log files.",
)
@click.option("--proprio-dim", type=int, default=14, show_default=True)
@click.option("--action-dim", type=int, default=7, show_default=True)
@click.option("--action-horizon", type=int, default=8, show_default=True)
def train(
    run_name: str,
    num_steps: int | None,
    batch_size: int | None,
    device: str,
    artifacts_dir: Path,
    logs_dir: Path,
    proprio_dim: int,
    action_dim: int,
    action_horizon: int,
) -> None:
    """Run a toy training loop on synthetic data."""
    policy_config = VLAPolicyConfig(
        action_dim=action_dim,
        action_horizon=action_horizon,
        proprio=ProprioConfig(input_dim=proprio_dim),
    )
    data_config = SyntheticDataConfig(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )
    train_overrides: dict[str, object] = {
        "run_name": run_name,
        "device": device,
        "artifacts_dir": artifacts_dir,
        "logs_dir": logs_dir,
    }
    if num_steps is not None:
        train_overrides["num_steps"] = num_steps
    if batch_size is not None:
        train_overrides["batch_size"] = batch_size

    train_config = TrainingConfig(**train_overrides)

    runner = Runner(policy_config, data_config, train_config)
    runner.train()


@main.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Explicit checkpoint path. If omitted, the latest in --artifacts-dir is used.",
)
@click.option(
    "--artifacts-dir",
    default="artifacts",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option("--run-name", default=None, help="Filter checkpoints by run_name prefix.")
@click.option("--device", default="cpu", show_default=True)
@click.option(
    "--instruction",
    default="pick up the red block",
    show_default=True,
    help="Instruction to condition the sampled action chunk on.",
)
def sample(
    checkpoint: Path | None,
    artifacts_dir: Path,
    run_name: str | None,
    device: str,
    instruction: str,
) -> None:
    """Load a trained checkpoint and sample an action chunk for one synthetic input."""
    ckpt_path = (
        checkpoint
        if checkpoint is not None
        else find_latest_checkpoint(artifacts_dir, run_name=run_name)
    )

    runner = InferenceRunner(ckpt_path, device=device)

    image_size = runner.policy_config.vision.image_size
    dummy_image = Image.new("RGB", (image_size, image_size), color=(127, 127, 127))
    dummy_proprio = torch.zeros(runner.policy_config.proprio.input_dim)

    actions = runner.act(
        image=dummy_image, instruction=instruction, proprio=dummy_proprio,
    )

    click.echo(f"Loaded checkpoint: {ckpt_path}")
    click.echo(f"Action chunk shape: {tuple(actions.shape)}")
    click.echo(f"Actions:\n{actions}")


if __name__ == "__main__":
    main()

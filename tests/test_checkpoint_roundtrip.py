"""Checkpoint save/load roundtrip via Runner and InferenceRunner."""

from __future__ import annotations

from pathlib import Path

import torch

from robovla.config import (
    FlowMatchingConfig,
    FusionConfig,
    ProprioConfig,
    SyntheticDataConfig,
    TextConfig,
    TrainingConfig,
    VisionConfig,
    VLAPolicyConfig,
)
from robovla.data.synthetic import SyntheticDataset
from robovla.inference import InferenceRunner, find_latest_checkpoint
from robovla.training.runner import Runner
from robovla.utils.training_utils.collate import collate_fn


def _make_configs(artifacts_dir: Path, logs_dir: Path) -> tuple[VLAPolicyConfig, SyntheticDataConfig, TrainingConfig]:
    policy_config = VLAPolicyConfig(
        embed_dim=512,
        action_dim=7,
        action_horizon=4,
        proprio=ProprioConfig(input_dim=14, hidden_dim=64, num_tokens=1),
        vision=VisionConfig(),
        text=TextConfig(),
        fusion=FusionConfig(num_layers=2, num_heads=8),
        flow=FlowMatchingConfig(num_layers=2, num_heads=8, num_sampling_steps=4),
    )
    data_config = SyntheticDataConfig(
        proprio_dim=14,
        action_dim=7,
        action_horizon=4,
        num_samples=8,
    )
    train_config = TrainingConfig(
        batch_size=2,
        num_steps=3,
        log_every=1,
        checkpoint_every=2,
        run_name="test_roundtrip",
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        device="cpu",
    )
    return policy_config, data_config, train_config


def test_runner_saves_checkpoint_findable_by_inference(tmp_path: Path) -> None:
    expected_substring = "test_roundtrip_step"
    expected_suffix = ".pt"

    artifacts_dir = tmp_path / "artifacts"
    logs_dir = tmp_path / "logs"
    policy_config, data_config, train_config = _make_configs(artifacts_dir, logs_dir)

    runner = Runner(policy_config, data_config, train_config)
    runner.train()

    found = find_latest_checkpoint(artifacts_dir, run_name="test_roundtrip")

    result_substring_present = expected_substring in found.name
    result_suffix = found.suffix

    assert result_substring_present is True
    assert result_suffix == expected_suffix


def test_inference_runner_matches_trained_policy(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    logs_dir = tmp_path / "logs"
    policy_config, data_config, train_config = _make_configs(artifacts_dir, logs_dir)

    runner = Runner(policy_config, data_config, train_config)
    runner.train()
    trained_policy = runner.policy
    trained_policy.eval()

    found = find_latest_checkpoint(artifacts_dir, run_name="test_roundtrip")
    inference = InferenceRunner(found, device="cpu")

    dataset = SyntheticDataset(data_config, preprocess=trained_policy.vision.preprocess)
    collate = collate_fn(trained_policy.text.tokenizer)
    batch = collate([dataset[i] for i in range(2)])

    with torch.no_grad():
        expected_context = trained_policy.encode_context(
            images=batch.images,
            text_ids=batch.text_ids,
            proprio=batch.proprios,
        )
        result_context = inference.policy.encode_context(
            images=batch.images,
            text_ids=batch.text_ids,
            proprio=batch.proprios,
        )

    assert torch.allclose(expected_context, result_context, atol=1e-6)

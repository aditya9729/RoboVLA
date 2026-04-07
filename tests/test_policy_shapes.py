"""Shape-correctness tests for the VLA policy forward and sample paths."""

from __future__ import annotations

import torch

from robovla.config import (
    FlowMatchingConfig,
    FusionConfig,
    ProprioConfig,
    SyntheticDataConfig,
    TextConfig,
    VisionConfig,
    VLAPolicyConfig,
)
from robovla.data.synthetic import SyntheticDataset
from robovla.models.policy import VLAPolicy
from robovla.utils.training_utils.collate import collate_fn


def _toy_policy_config() -> VLAPolicyConfig:
    """Build a small policy config that exercises every component."""
    return VLAPolicyConfig(
        embed_dim=512,
        action_dim=7,
        action_horizon=4,
        proprio=ProprioConfig(input_dim=14, hidden_dim=64, num_tokens=1),
        vision=VisionConfig(),
        text=TextConfig(),
        fusion=FusionConfig(num_layers=2, num_heads=8),
        flow=FlowMatchingConfig(num_layers=2, num_heads=8, num_sampling_steps=4),
    )


def _toy_data_config() -> SyntheticDataConfig:
    return SyntheticDataConfig(
        proprio_dim=14,
        action_dim=7,
        action_horizon=4,
        num_samples=4,
    )


def test_policy_forward_returns_scalar_loss() -> None:
    expected_loss_shape: tuple[int, ...] = ()
    expected_loss_is_finite = True

    policy_config = _toy_policy_config()
    data_config = _toy_data_config()
    policy = VLAPolicy(policy_config)
    dataset = SyntheticDataset(data_config, preprocess=policy.vision.preprocess)
    collate = collate_fn(policy.text.tokenizer)
    batch = collate([dataset[i] for i in range(2)])

    loss = policy(
        images=batch.images,
        text_ids=batch.text_ids,
        proprio=batch.proprios,
        action_chunks=batch.action_chunks,
    )

    result_loss_shape = tuple(loss.shape)
    result_loss_is_finite = bool(torch.isfinite(loss).item())

    assert result_loss_shape == expected_loss_shape
    assert result_loss_is_finite == expected_loss_is_finite


def test_policy_sample_returns_action_chunk_shape() -> None:
    batch_size = 2
    expected_actions_shape: tuple[int, int, int] = (batch_size, 4, 7)

    policy_config = _toy_policy_config()
    data_config = _toy_data_config()
    policy = VLAPolicy(policy_config)
    dataset = SyntheticDataset(data_config, preprocess=policy.vision.preprocess)
    collate = collate_fn(policy.text.tokenizer)
    batch = collate([dataset[i] for i in range(batch_size)])

    actions = policy.sample(
        images=batch.images,
        text_ids=batch.text_ids,
        proprio=batch.proprios,
    )

    result_actions_shape = tuple(actions.shape)
    assert result_actions_shape == expected_actions_shape


def test_proprio_wrong_input_dim_raises() -> None:
    """Edge case: a proprio vector of the wrong size must fail in the encoder."""
    expected_exception: type[BaseException] = RuntimeError

    policy_config = _toy_policy_config()
    policy = VLAPolicy(policy_config)
    bad_proprio = torch.randn(2, 99)            # config says input_dim=14
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_text_ids = torch.zeros(2, 77, dtype=torch.long)
    dummy_actions = torch.randn(2, 4, 7)

    raised: type[BaseException] | None = None
    try:
        policy(
            images=dummy_image,
            text_ids=dummy_text_ids,
            proprio=bad_proprio,
            action_chunks=dummy_actions,
        )
    except RuntimeError as exc:
        raised = type(exc)

    assert raised is expected_exception

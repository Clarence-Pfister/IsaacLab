# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the bounded G1 jump policy distribution."""

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.distributions import TanhGaussianDistribution


def test_tanh_gaussian_deterministic_output_is_bounded() -> None:
    """The exported policy mean must stay inside the normalized action range."""
    distribution = TanhGaussianDistribution(output_dim=3, init_std=0.5)
    latent_mean = torch.tensor(((-100.0, 0.25, 100.0),), dtype=torch.float32)

    action = distribution.deterministic_output(latent_mean)

    expected_latent_mean = distribution.LATENT_MEAN_LIMIT * torch.tanh(latent_mean / distribution.LATENT_MEAN_LIMIT)
    assert torch.all(action > -1.0)
    assert torch.all(action < 1.0)
    torch.testing.assert_close(action, torch.tanh(expected_latent_mean))


def test_tanh_gaussian_sample_and_log_probability_are_finite() -> None:
    """Squashed samples and their corrected densities must remain finite."""
    distribution = TanhGaussianDistribution(output_dim=3, init_std=1.0)
    distribution.update(torch.tensor(((12.0, 0.0, -12.0),), dtype=torch.float32))

    torch.manual_seed(0)
    action = distribution.sample()
    log_probability = distribution.log_prob(action)

    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert torch.isfinite(log_probability).all()


def test_tanh_gaussian_extreme_logits_remain_trainable() -> None:
    """Extreme network logits must not create endpoint actions or invalid densities."""
    distribution = TanhGaussianDistribution(output_dim=3, init_std=0.5, std_type="log")
    distribution.update(torch.tensor(((-1.0e6, 0.0, 1.0e6),), dtype=torch.float32))

    torch.manual_seed(0)
    action = distribution.sample()
    log_probability = distribution.log_prob(action)

    assert torch.all(torch.abs(distribution.mean) <= distribution.LATENT_MEAN_LIMIT)
    assert torch.all(action > -1.0)
    assert torch.all(action < 1.0)
    assert torch.isfinite(log_probability).all()


def test_tanh_gaussian_scalar_std_cannot_become_negative() -> None:
    """A legacy scalar standard-deviation parameter must fail safe if optimized below zero."""
    distribution = TanhGaussianDistribution(output_dim=2, init_std=0.5, std_type="scalar")
    with torch.no_grad():
        distribution.std_param.fill_(-0.1)
    distribution.update(torch.zeros((1, 2)))

    action = distribution.sample()

    assert torch.all(distribution.std > 0.0)
    assert torch.isfinite(action).all()


def test_tanh_gaussian_export_module_matches_deterministic_output() -> None:
    """JIT and ONNX export must retain the output squashing transform."""
    distribution = TanhGaussianDistribution(output_dim=3, init_std=0.5)
    latent_mean = torch.tensor(((0.5, -0.5, 2.0),), dtype=torch.float32)

    exported_output = distribution.as_deterministic_output_module()(latent_mean)

    torch.testing.assert_close(exported_output, distribution.deterministic_output(latent_mean))


def test_tanh_gaussian_export_module_is_torchscript_compatible() -> None:
    """The deterministic transform must compile through the deployment JIT path."""
    distribution = TanhGaussianDistribution(output_dim=3, init_std=0.5)
    latent_mean = torch.tensor(((0.5, -0.5, 2.0),), dtype=torch.float32)

    scripted_module = torch.jit.script(distribution.as_deterministic_output_module())

    torch.testing.assert_close(scripted_module(latent_mean), distribution.deterministic_output(latent_mean))


@pytest.mark.parametrize("std_type", ("scalar", "log"))
def test_tanh_gaussian_can_reset_exploration_std(std_type: str) -> None:
    distribution = TanhGaussianDistribution(output_dim=3, init_std=0.5, std_type=std_type)

    distribution.set_std(0.15)
    distribution.update(torch.zeros(2, 3))

    torch.testing.assert_close(distribution.std, torch.full((2, 3), 0.15))


@pytest.mark.parametrize("std_type", ("scalar", "log"))
def test_tanh_gaussian_can_use_low_residual_exploration_std(std_type: str) -> None:
    distribution = TanhGaussianDistribution(output_dim=3, init_std=0.5, std_type=std_type)

    distribution.set_std(0.02)
    distribution.update(torch.zeros(2, 3))

    torch.testing.assert_close(distribution.std, torch.full((2, 3), 0.02))


def test_tanh_gaussian_rejects_invalid_reset_std() -> None:
    distribution = TanhGaussianDistribution(output_dim=3, init_std=0.5)

    with pytest.raises(ValueError, match="standard deviation"):
        distribution.set_std(0.001)

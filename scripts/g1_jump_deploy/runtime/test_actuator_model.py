# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the deployment actuator velocity-limit model."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.g1_jump_deploy.runtime.actuator_model import saturate_torque_at_velocity_limit


def test_positive_direction_rolls_off_accelerating_torque() -> None:
    torque = np.full(5, 10.0)
    velocity = np.asarray((8.9, 9.0, 9.5, 10.0, 11.0))

    saturated = saturate_torque_at_velocity_limit(torque, velocity, np.full(5, 10.0))

    np.testing.assert_allclose(saturated, (10.0, 10.0, 5.0, 0.0, 0.0), rtol=0.0, atol=1.0e-12)


def test_negative_direction_rolls_off_accelerating_torque_symmetrically() -> None:
    torque = np.full(5, -8.0)
    velocity = np.asarray((-8.9, -9.0, -9.25, -10.0, -11.0))

    saturated = saturate_torque_at_velocity_limit(torque, velocity, np.full(5, 10.0))

    np.testing.assert_allclose(saturated, (-8.0, -8.0, -6.0, 0.0, 0.0), rtol=0.0, atol=1.0e-12)


def test_braking_torque_is_unchanged_past_both_limits() -> None:
    torque = np.asarray((-4.0, 5.0, 0.0))
    velocity = np.asarray((12.0, -15.0, 20.0))

    saturated = saturate_torque_at_velocity_limit(torque, velocity, np.full(3, 10.0))

    np.testing.assert_array_equal(saturated, torque)


@pytest.mark.parametrize(
    ("torque", "velocity", "limit", "match"),
    [
        (np.zeros(2), np.zeros(3), np.ones(2), "identical shapes"),
        (np.zeros(2), np.zeros(2), np.ones(3), "identical shapes"),
        (np.asarray((0.0, np.nan)), np.zeros(2), np.ones(2), "finite values"),
        (np.zeros(2), np.asarray((0.0, np.inf)), np.ones(2), "finite values"),
        (np.zeros(2), np.zeros(2), np.asarray((1.0, 0.0)), "positive finite"),
    ],
)
def test_rejects_invalid_array_inputs(
    torque: np.ndarray,
    velocity: np.ndarray,
    limit: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        saturate_torque_at_velocity_limit(torque, velocity, limit)


@pytest.mark.parametrize("knee_width", (0.0, -0.1, 1.1, np.inf, True))
def test_rejects_invalid_knee_width(knee_width: float) -> None:
    with pytest.raises(ValueError, match="knee_width"):
        saturate_torque_at_velocity_limit(np.zeros(1), np.zeros(1), np.ones(1), knee_width=knee_width)

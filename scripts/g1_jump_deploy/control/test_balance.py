# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the portable G1 balance controller."""

import math

import numpy as np
import pytest

from scripts.g1_jump_deploy.control.balance import BalanceController, BalanceControllerConfig, project_ankle_target

_JOINT_NAMES = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


def _quaternion_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    roll_half = 0.5 * roll
    pitch_half = 0.5 * pitch
    return np.asarray(
        (
            math.cos(roll_half) * math.cos(pitch_half),
            math.sin(roll_half) * math.cos(pitch_half),
            math.cos(roll_half) * math.sin(pitch_half),
            -math.sin(roll_half) * math.sin(pitch_half),
        )
    )


def _compute(controller: BalanceController, quaternion: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    joints = np.zeros(4)
    return controller.compute(
        joints if target is None else target,
        quaternion,
        np.zeros(3),
        joints,
        joints,
        0.002,
    )


def test_zero_attitude_error_gives_zero_offset() -> None:
    config = BalanceControllerConfig(target_roll=0.03, target_pitch=0.12, initial_pitch_integral=0.0)
    controller = BalanceController(_JOINT_NAMES, config)

    result = _compute(controller, _quaternion_from_roll_pitch(config.target_roll, config.target_pitch))

    np.testing.assert_allclose(result, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(controller.last_ankle_offset, 0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    ("requested", "expected"),
    (
        ((10.0, 0.0), (0.5236, 0.0)),
        ((0.0, 10.0), (0.0, 0.2618)),
        ((0.0, -10.0), (0.0, -0.2618)),
    ),
)
def test_projection_saturates_at_joint_limits(requested: tuple[float, float], expected: tuple[float, float]) -> None:
    np.testing.assert_allclose(project_ankle_target(*requested), expected, atol=1.0e-12)


def test_projection_keeps_targets_inside_tendon_polytope() -> None:
    constraints = (
        (-0.2618, 0.2618, 0.13708),
        (0.2618, 0.2618, 0.13708),
        (-0.14967, -0.2618, 0.22846),
        (0.14967, -0.2618, 0.22846),
    )
    for requested_pitch in np.linspace(-2.0, 2.0, 17):
        for requested_roll in np.linspace(-1.0, 1.0, 17):
            pitch, roll = project_ankle_target(requested_pitch, requested_roll)
            assert -0.87267 - 1.0e-12 <= pitch <= 0.5236 + 1.0e-12
            assert -0.2618 - 1.0e-12 <= roll <= 0.2618 + 1.0e-12
            assert all(c_roll * roll + c_pitch * pitch <= limit + 1.0e-12 for c_roll, c_pitch, limit in constraints)


def test_integral_anti_windup_does_not_run_away() -> None:
    config = BalanceControllerConfig(
        target_roll=0.0,
        target_pitch=0.0,
        roll_kp=0.0,
        pitch_kp=0.0,
        roll_kd=0.0,
        pitch_kd=0.0,
        roll_ki=1.0,
        pitch_ki=1.0,
        integral_limit=0.03,
        initial_pitch_integral=0.0,
    )
    controller = BalanceController(_JOINT_NAMES, config)
    quaternion = _quaternion_from_roll_pitch(0.4, 0.4)

    for _ in range(100_000):
        _compute(controller, quaternion)

    np.testing.assert_allclose(controller.integral_error, 0.03, atol=1.0e-15)


def test_positive_attitude_error_commands_positive_ankle_response() -> None:
    config = BalanceControllerConfig(
        target_roll=0.0,
        target_pitch=0.0,
        roll_kp=0.2,
        pitch_kp=0.2,
        roll_kd=0.0,
        pitch_kd=0.0,
        integral_enabled=False,
    )
    controller = BalanceController(_JOINT_NAMES, config)

    result = _compute(controller, _quaternion_from_roll_pitch(0.1, 0.1))

    assert result[0] > 0.0
    assert result[1] > 0.0
    assert result[2] > 0.0
    assert result[3] > 0.0


def test_target_attitude_can_be_recalibrated() -> None:
    controller = BalanceController(
        _JOINT_NAMES,
        BalanceControllerConfig(target_roll=0.0, target_pitch=0.0, integral_enabled=False),
    )
    target_roll = 0.04
    target_pitch = -0.06
    quaternion = _quaternion_from_roll_pitch(target_roll, target_pitch)

    assert np.linalg.norm(_compute(controller, quaternion)) > 0.0

    controller.set_target_attitude(target_roll, target_pitch)

    np.testing.assert_allclose(_compute(controller, quaternion), 0.0, atol=1.0e-14)
    assert controller.config.target_roll == pytest.approx(target_roll)
    assert controller.config.target_pitch == pytest.approx(target_pitch)
    with pytest.raises(ValueError, match="finite"):
        controller.set_target_attitude(float("nan"), target_pitch)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the deployment-time implicit-PD torque projection."""

import numpy as np
import pytest

from scripts.g1_jump_deploy.runtime import project_pd_position_target, project_position_target_to_lower_limit


def test_lower_limit_projection_adds_velocity_stopping_margin() -> None:
    projected = project_position_target_to_lower_limit(
        joint_pos_target=np.asarray((0.1, 0.2, 0.3)),
        joint_vel=np.asarray((-4.0, 2.0, -100.0)),
        position_lower=np.asarray((0.1, -0.5, -0.2)),
        position_upper=np.asarray((2.0, 1.0, 0.8)),
        velocity_lookahead=np.asarray((0.05, 0.05, 0.05)),
    )

    np.testing.assert_allclose(projected, (0.3, 0.2, 0.8))


def test_lower_limit_projection_rejects_invalid_bounds() -> None:
    values = np.ones(2)

    with pytest.raises(ValueError, match="strictly below"):
        project_position_target_to_lower_limit(values, values, values, values, values)


def test_projection_preserves_safe_targets_and_caps_unsafe_targets() -> None:
    position = np.asarray((0.1, -0.2))
    velocity = np.asarray((0.5, -1.0))
    requested = np.asarray((0.2, 0.5))
    stiffness = np.asarray((100.0, 50.0))
    damping = np.asarray((2.0, 4.0))
    effort_limit = np.asarray((20.0, 10.0))
    effort_limit_ratio = np.asarray((0.6, 0.6))

    projected = project_pd_position_target(
        requested,
        position,
        velocity,
        stiffness,
        damping,
        effort_limit,
        effort_limit_ratio,
    )
    torque = stiffness * (projected - position) - damping * velocity

    assert projected[0] == pytest.approx(requested[0])
    assert torque[0] == pytest.approx(9.0)
    assert torque[1] == pytest.approx(6.0)
    assert np.all(np.abs(torque) <= effort_limit_ratio * effort_limit + 1.0e-12)


def test_projection_rejects_zero_stiffness() -> None:
    values = np.ones(2)

    with pytest.raises(ValueError, match="stiffness"):
        project_pd_position_target(values, values, values, np.asarray((1.0, 0.0)), values, values, values)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for actuator velocity-limit emulation in the MuJoCo deploy loop."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from deploy_mujoco import _compute_actuator_control, _parse_args


@pytest.mark.parametrize("use_implicit_pd", (False, True))
def test_velocity_limit_emulation_reduces_only_opted_in_actuator_torque(use_implicit_pd: bool) -> None:
    target = np.asarray((1.0,))
    position = np.asarray((0.0,))
    velocity = np.asarray((11.0,))
    stiffness = np.asarray((10.0,))
    damping = np.asarray((0.0,))
    effort_limit = np.asarray((100.0,))
    velocity_limit = np.asarray((10.0,))

    unchanged = _compute_actuator_control(
        target,
        position,
        velocity,
        stiffness,
        damping,
        effort_limit,
        velocity_limit,
        use_implicit_pd=use_implicit_pd,
        emulate_velocity_limit=False,
    )
    saturated = _compute_actuator_control(
        target,
        position,
        velocity,
        stiffness,
        damping,
        effort_limit,
        velocity_limit,
        use_implicit_pd=use_implicit_pd,
        emulate_velocity_limit=True,
    )

    if use_implicit_pd:
        unchanged_torque = stiffness * (unchanged - position) - damping * velocity
        saturated_torque = stiffness * (saturated - position) - damping * velocity
    else:
        unchanged_torque = unchanged
        saturated_torque = saturated
    np.testing.assert_allclose(unchanged_torque, 10.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(saturated_torque, 0.0, rtol=0.0, atol=1.0e-12)


def test_velocity_limit_emulation_cli_defaults_off_and_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["deploy_mujoco.py"])
    assert not _parse_args().emulate_velocity_limit

    monkeypatch.setattr(sys, "argv", ["deploy_mujoco.py", "--emulate_velocity_limit"])
    assert _parse_args().emulate_velocity_limit


def test_velocity_limit_emulation_rejects_legacy_velocity_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["deploy_mujoco.py", "--emulate_velocity_limit", "--clamp_joint_velocity"],
    )

    with pytest.raises(SystemExit):
        _parse_args()

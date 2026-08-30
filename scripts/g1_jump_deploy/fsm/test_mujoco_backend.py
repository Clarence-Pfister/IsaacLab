# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for measured-state MuJoCo stand replays."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from scripts.g1_jump_deploy.fsm.mujoco_backend import MujocoRobot

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"
_MODEL = _REPO_ROOT / "data_storage" / "g1_23dof_holo_compat.xml"
_OVERLAY = _REPO_ROOT / "scripts" / "g1_jump_deploy" / "mujoco" / "model_overlay.xml"


def test_measured_state_reset_and_scaled_command_limits() -> None:
    robot = MujocoRobot(
        _MANIFEST,
        _MODEL,
        _OVERLAY,
        effort_scale=0.5,
        target_rate_limit_rad_s=0.5,
    )
    measured_position = robot.joint_positions + 0.01

    robot.reset_state(measured_position, np.asarray((2.0, 0.0, 0.0, 0.0)))

    np.testing.assert_allclose(robot.joint_positions, measured_position, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(robot.command_target, measured_position, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(robot.imu_quaternion, (1.0, 0.0, 0.0, 0.0), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(robot.command_effort_limits, 0.5 * robot.effort_limits)


def test_joint_target_rate_limit_matches_hardware_period() -> None:
    robot = MujocoRobot(
        _MANIFEST,
        _MODEL,
        _OVERLAY,
        effort_scale=0.75,
        target_rate_limit_rad_s=0.5,
    )
    measured_position = robot.joint_positions
    robot.command_joint_position_target(
        measured_position + 0.1,
        np.ones(robot.joint_count),
        np.zeros(robot.joint_count),
    )

    robot.step_physics(np.zeros(robot.joint_count))

    expected_step = 0.5 * robot.sim_dt
    np.testing.assert_allclose(
        robot.command_target,
        measured_position + expected_step,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_torque_projection_is_applied_at_the_physics_rate() -> None:
    robot = MujocoRobot(_MANIFEST, _MODEL, _OVERLAY)
    ratio = np.full(robot.joint_count, 0.6)
    robot._manifest = replace(robot._manifest, effort_limit_ratio=ratio)
    position = robot.joint_positions
    stiffness = np.full(robot.joint_count, 5_000.0)
    robot.command_joint_position_target(position + 0.1, stiffness, np.zeros(robot.joint_count))

    robot.step_physics(np.zeros(robot.joint_count))

    np.testing.assert_allclose(
        robot.applied_torque,
        ratio * robot.effort_limits,
        rtol=0.0,
        atol=1.0e-10,
    )
    assert np.all(robot.command_target < position + 0.1)


def test_gantry_support_reduces_initial_downward_acceleration() -> None:
    unsupported = MujocoRobot(_MANIFEST, _MODEL, _OVERLAY)
    supported = MujocoRobot(_MANIFEST, _MODEL, _OVERLAY, gantry_support_fraction=0.5)
    zero = np.zeros(unsupported.joint_count)
    for robot in (unsupported, supported):
        robot.command_joint_position_target(robot.joint_positions, zero, zero)
        for _ in range(10):
            robot.step_physics(zero)

    expected_force = (
        0.5 * float(np.linalg.norm(supported.model.opt.gravity)) * float(mujoco.mj_getTotalmass(supported.model))
    )
    assert supported.gantry_support_force_world[2] == pytest.approx(expected_force)
    assert supported.pelvis_linear_velocity[2] > unsupported.pelvis_linear_velocity[2]


def test_gantry_support_leaves_horizontal_translation_unrestrained() -> None:
    robot = MujocoRobot(_MANIFEST, _MODEL, _OVERLAY, gantry_support_fraction=0.5)
    robot.data.qpos[robot._root_qpos_address] += 0.1
    robot.data.qvel[robot._root_dof_address] = 1.0

    robot._apply_gantry_wrench()

    np.testing.assert_array_equal(robot.data.xfrc_applied[robot._root_body_id, :2], np.zeros(2))


def test_contactless_rehearsal_model_disables_only_ground_collision() -> None:
    robot = MujocoRobot(
        _MANIFEST,
        _MODEL,
        _OVERLAY,
        gantry_support_fraction=1.0,
        ground_contact_enabled=False,
    )

    assert not robot.ground_contact_enabled
    assert robot.model.geom_contype[robot._ground_geom_id] == 0
    assert robot.model.geom_conaffinity[robot._ground_geom_id] == 0
    assert np.any(np.delete(robot.model.geom_contype, robot._ground_geom_id) != 0)

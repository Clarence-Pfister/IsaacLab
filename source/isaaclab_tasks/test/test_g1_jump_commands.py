# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.commands import (
    JumpGoalCommand,
    _next_longitudinal_cycle_goal,
    _sample_planar_displacement,
)


def test_planar_displacement_sampler_can_anchor_every_goal_at_zero() -> None:
    dx, dy = _sample_planar_displacement(
        128,
        (-0.2, 0.2),
        (-0.15, 0.15),
        zero_goal_probability=1.0,
        boundary_goal_probability=0.0,
        device="cpu",
    )

    torch.testing.assert_close(dx, torch.zeros_like(dx))
    torch.testing.assert_close(dy, torch.zeros_like(dy))


def test_planar_displacement_sampler_can_select_only_range_boundaries() -> None:
    dx, dy = _sample_planar_displacement(
        256,
        (-0.2, 0.3),
        (-0.15, 0.1),
        zero_goal_probability=0.0,
        boundary_goal_probability=1.0,
        device="cpu",
    )

    assert torch.all((dx == -0.2) | (dx == 0.3))
    assert torch.all((dy == -0.15) | (dy == 0.1))


def test_planar_displacement_sampler_rejects_overlapping_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to at most 1"):
        _sample_planar_displacement(
            1,
            (-0.2, 0.2),
            (-0.15, 0.15),
            zero_goal_probability=0.6,
            boundary_goal_probability=0.5,
            device="cpu",
        )


def test_longitudinal_cycle_goal_balances_both_command_orders() -> None:
    previous_goal = torch.tensor((-0.1, 0.0, 0.1, -0.1, 0.0, 0.1))
    reverse_cycle = torch.tensor((False, False, False, True, True, True))

    next_goal = _next_longitudinal_cycle_goal(
        previous_goal,
        pos_x_range=(-0.1, 0.1),
        reverse_cycle=reverse_cycle,
    )

    torch.testing.assert_close(
        next_goal,
        torch.tensor((0.0, 0.1, -0.1, 0.1, -0.1, 0.0)),
    )


def test_jump_goal_metrics_wrap_yaw_error_at_pi() -> None:
    current_yaw = torch.tensor((math.radians(179.0), math.radians(10.0)))
    target_yaw = torch.tensor((math.radians(-179.0), math.radians(-20.0)))
    zeros = torch.zeros(2)
    root_quat_w = quat_from_euler_xyz(zeros, zeros, current_yaw)
    target_quat_w = quat_from_euler_xyz(zeros, zeros, target_yaw)

    command = object.__new__(JumpGoalCommand)
    command._env = SimpleNamespace(num_envs=2, device="cpu")
    command.robot = SimpleNamespace(
        is_initialized=True,
        data=SimpleNamespace(root_pos_w=torch.zeros((2, 3)), root_quat_w=root_quat_w),
    )
    command.pose_command_w = torch.cat((torch.zeros((2, 3)), target_quat_w), dim=-1)
    command.metrics = {}

    command._update_metrics()

    torch.testing.assert_close(command.metrics["position_error"], torch.zeros(2))
    torch.testing.assert_close(
        command.metrics["yaw_error"],
        torch.tensor((math.radians(2.0), math.radians(30.0))),
    )

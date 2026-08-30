# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.torque_projection import (
    project_pd_position_target,
    project_position_target_to_lower_limit,
)


def test_project_position_target_to_lower_limit_adds_velocity_stopping_margin() -> None:
    projected = project_position_target_to_lower_limit(
        joint_pos_target=torch.tensor([[0.1, 0.2, 0.3]]),
        joint_vel=torch.tensor([[-4.0, 2.0, -100.0]]),
        position_lower=torch.tensor([[0.1, -0.5, -0.2]]),
        position_upper=torch.tensor([[2.0, 1.0, 0.8]]),
        velocity_lookahead=torch.tensor([[0.05, 0.05, 0.05]]),
    )

    torch.testing.assert_close(projected, torch.tensor([[0.3, 0.2, 0.8]]))


def test_project_pd_position_target_enforces_per_joint_effort_ratio() -> None:
    joint_pos_target = torch.tensor([[1.0, -1.0]])
    joint_pos = torch.tensor([[0.2, -0.4]])
    joint_vel = torch.tensor([[0.1, -0.2]])
    stiffness = torch.tensor([[10.0, 20.0]])
    damping = torch.tensor([[2.0, 3.0]])
    effort_limit = torch.tensor([[5.0, 8.0]])
    effort_limit_ratio = torch.tensor([[1.0, 0.5]])

    projected = project_pd_position_target(
        joint_pos_target,
        joint_pos,
        joint_vel,
        stiffness,
        damping,
        effort_limit,
        effort_limit_ratio,
    )
    projected_demand = stiffness * (projected - joint_pos) - damping * joint_vel

    torch.testing.assert_close(projected, torch.tensor([[0.72, -0.63]]))
    torch.testing.assert_close(projected_demand, torch.tensor([[5.0, -4.0]]))


def test_project_pd_position_target_preserves_target_within_effort_envelope() -> None:
    joint_pos_target = torch.tensor([[0.3]])
    joint_pos = torch.tensor([[0.2]])
    joint_vel = torch.tensor([[0.1]])
    stiffness = torch.tensor([[10.0]])
    damping = torch.tensor([[2.0]])

    projected = project_pd_position_target(
        joint_pos_target,
        joint_pos,
        joint_vel,
        stiffness,
        damping,
        effort_limit=torch.tensor([[5.0]]),
        effort_limit_ratio=torch.tensor([[0.5]]),
    )

    torch.testing.assert_close(projected, joint_pos_target)

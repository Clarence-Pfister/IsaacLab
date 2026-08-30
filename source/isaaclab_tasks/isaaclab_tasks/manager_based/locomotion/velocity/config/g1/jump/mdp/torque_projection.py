# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torque-envelope projection for G1 joint-position targets."""

from __future__ import annotations

import torch


def project_position_target_to_lower_limit(
    joint_pos_target: torch.Tensor,
    joint_vel: torch.Tensor,
    position_lower: torch.Tensor,
    position_upper: torch.Tensor,
    velocity_lookahead: torch.Tensor,
) -> torch.Tensor:
    """Raise position targets early enough to brake motion toward a lower limit.

    The projected lower target is the configured position bound plus the current
    approach speed multiplied by a lookahead time. A zero lookahead preserves the
    ordinary position clip.

    Args:
        joint_pos_target: Requested joint positions [rad].
        joint_vel: Measured joint velocities [rad/s].
        position_lower: Lower command bounds [rad].
        position_upper: Upper command bounds [rad].
        velocity_lookahead: Lower-limit braking lookahead [s].

    Returns:
        Joint position targets with velocity-aware lower-limit braking [rad].
    """
    approach_speed = torch.clamp(-joint_vel, min=0.0)
    braking_target = position_lower + velocity_lookahead * approach_speed
    braking_target = torch.minimum(braking_target, position_upper)
    return torch.maximum(joint_pos_target, braking_target)


def project_pd_position_target(
    joint_pos_target: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    effort_limit: torch.Tensor,
    effort_limit_ratio: torch.Tensor,
) -> torch.Tensor:
    """Project a position target into an implicit-PD effort envelope.

    The projection preserves targets whose instantaneous PD demand is already within the
    envelope. Targets outside it are moved to the nearest position that produces the
    requested signed effort boundary. Callers must provide strictly positive stiffness.

    Args:
        joint_pos_target: Requested joint positions [rad].
        joint_pos: Measured joint positions [rad].
        joint_vel: Measured joint velocities [rad/s].
        stiffness: Joint position gains [N·m/rad].
        damping: Joint velocity gains [N·m·s/rad].
        effort_limit: Absolute actuator effort limits [N·m].
        effort_limit_ratio: Available fraction of each effort limit.

    Returns:
        Projected joint position targets [rad].
    """
    torque_demand = stiffness * (joint_pos_target - joint_pos) - damping * joint_vel
    torque_limit = effort_limit_ratio * effort_limit
    projected_torque = torch.clamp(torque_demand, min=-torque_limit, max=torque_limit)
    return joint_pos + (projected_torque + damping * joint_vel) / stiffness

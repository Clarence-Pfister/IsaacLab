# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the G1 jump task."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from ..constants import REFERENCE_DURATION_S
from .motion import (
    get_env_time,
    get_jump_phase,
    get_jump_phases,
    get_loader,
    get_phase_id,
    get_root_relative_foot_pos,
    warp_to_torch,
)


def reference_motion_complete(env, hold_duration_s: float = 0.0) -> torch.Tensor:
    """Terminate after the reference motion and an optional final-pose hold.

    Args:
        env: Environment from which to read the reference clock.
        hold_duration_s: Additional duration at the final reference pose [s].

    Returns:
        Boolean timeout mask for all environments.

    Raises:
        ValueError: If :paramref:`hold_duration_s` is negative or non-finite.
    """
    if not math.isfinite(hold_duration_s) or hold_duration_s < 0.0:
        raise ValueError(f"hold_duration_s must be finite and non-negative, got {hold_duration_s}.")
    reference_duration_s = getattr(getattr(env, "cfg", None), "reference_duration_s", REFERENCE_DURATION_S)
    return get_env_time(env) >= reference_duration_s + hold_duration_s


def ground_contact(env, threshold: float, sensor_names: Sequence[str]) -> torch.Tensor:
    """Terminates when any explicit non-foot contact sensor hits the ground."""
    terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    for sensor_name in sensor_names:
        contact_sensor = env.scene.sensors[sensor_name]
        forces = contact_sensor.data.net_forces_w_history
        forces_t = warp_to_torch(forces)
        force_norm = torch.linalg.norm(forces_t, dim=-1)
        max_force = force_norm.reshape(env.num_envs, -1).max(dim=1).values
        sensor_terminated = max_force > threshold
        terminated |= sensor_terminated

    return terminated


def foot_tracking_error(
    env,
    threshold: float,
    active_phases: Sequence[str] | None = None,
) -> torch.Tensor:
    """Terminates if root-relative foot position error exceeds the threshold."""
    loader = get_loader(env)
    robot = env.scene["robot"]
    current_time = get_env_time(env)
    current_foot_pos_rel, ref_foot_pos_rel = get_root_relative_foot_pos(env, loader, robot, current_time)

    error = torch.linalg.norm(current_foot_pos_rel - ref_foot_pos_rel, dim=-1)
    terminated = torch.any(error > threshold, dim=-1)

    if active_phases is not None:
        current_phase = get_jump_phase(env)
        active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for phase in active_phases:
            active |= current_phase == get_phase_id(phase, get_jump_phases(env))
        terminated &= active

    return terminated


def task_completion_error(
    env,
    pos_threshold: float,
    yaw_threshold: float,
    start_phase: str = "STAND",
) -> torch.Tensor:
    """Terminates if task error exceeds bounds once recovery should begin."""
    current_phase = get_jump_phase(env)

    current_xy = warp_to_torch(env.scene["robot"].data.root_pos_w)[:, :2]
    target_xy = env.command_manager.get_command("jump_goal")[:, :2]
    pos_error = torch.linalg.norm(current_xy - target_xy, dim=-1)

    current_quat = warp_to_torch(env.scene["robot"].data.root_quat_w)
    target_quat = env.command_manager.get_command("jump_goal")[:, 3:7]
    _, _, current_yaw = euler_xyz_from_quat(current_quat)
    _, _, target_yaw = euler_xyz_from_quat(target_quat)

    yaw_error = torch.abs(torch.atan2(torch.sin(current_yaw - target_yaw), torch.cos(current_yaw - target_yaw)))

    error_exceeded = (pos_error > pos_threshold) | (yaw_error > yaw_threshold)
    phase_exceeded = current_phase >= get_phase_id(start_phase, get_jump_phases(env))

    return phase_exceeded & error_exceeded

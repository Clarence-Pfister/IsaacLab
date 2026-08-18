# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the G1 jump task."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from ..constants import JUMP_PHASES, REFERENCE_MOTION_FPS
from .motion import get_env_time, get_jump_phase, get_loader, warp_to_torch


def obs_future_reference_preview(env) -> torch.Tensor:
    """Return the reference preview at offsets of 1, 4, and 7 reference frames.

    The offsets are 0.0333, 0.1333, and 0.2333 seconds at 30 FPS. The preview is
    ``[qz^r(t), qm^r(t+1), qm^r(t+4), qm^r(t+7)]`` and has 70 elements.
    """
    loader = get_loader(env)
    current_time = get_env_time(env)
    reference_dt = 1.0 / REFERENCE_MOTION_FPS

    # Define future time offsets for preview
    t_0 = current_time
    t_1 = current_time + (1 * reference_dt)
    t_4 = current_time + (4 * reference_dt)
    t_7 = current_time + (7 * reference_dt)

    # Fetch reference states at respective times
    _, _, ref_root_0, _, _, _, _ = loader.get_state(t_0)
    ref_pos_1, _, _, _, _, _, _ = loader.get_state(t_1)
    ref_pos_4, _, _, _, _, _, _ = loader.get_state(t_4)
    ref_pos_7, _, _, _, _, _, _ = loader.get_state(t_7)

    # qz^r(t) is the root z position
    qz_t = ref_root_0[:, 2:3]

    # Concatenate [qz^r(t), qm^r(t+1), qm^r(t+4), qm^r(t+7)]
    preview = torch.cat((qz_t, ref_pos_1, ref_pos_4, ref_pos_7), dim=-1)
    return preview


def obs_goal_command(env) -> torch.Tensor:
    """Return the jump goal in the frame of the robot's pose before the jump.

    The command manager's public command is expressed in the world frame, which the policy
    cannot use: it would have to infer its own global position to act on it, and the real
    robot has no such feedback. The body-frame command is the goal as the paper defines it,
    ``[cx, cy, cz]`` plus the turning direction as a quaternion, and is invariant to where
    in the world the episode happens to start. The quaternion is used rather than a yaw
    angle so the observation stays continuous when the turning range grows past +/-180 deg.
    """
    return env.command_manager.get_term("jump_goal").pose_command_b


def obs_goal_remaining(env) -> torch.Tensor:
    """Return the displacement still to cover to the goal, in the current heading frame [m].

    This is the deployable form of the robot's own position. :func:`obs_goal_command` fixes
    the goal against the pose the episode started from and never changes, so on its own it
    cannot tell the policy how much of the jump is already done; that feedback used to come
    from observing the root position in the environment frame, which no sensor on the robot
    produces. The difference between the goal and the current root position carries the same
    information, and a real robot computes it the same way, by carrying the goal forward with
    its odometry.

    Only the heading component of the root rotation is removed, so the observation does not
    swing with the pitch and roll of the body during flight. The vertical component is kept:
    height above the landing point is measurable from leg kinematics in contact and is what
    the landing has to be timed against.
    """
    robot = env.scene["robot"]
    goal_w = env.command_manager.get_term("jump_goal").pose_command_w[:, :3]
    root_pos_w = warp_to_torch(robot.data.root_pos_w)[:, :3]
    root_quat_w = warp_to_torch(robot.data.root_quat_w)
    return quat_apply_inverse(yaw_quat(root_quat_w), goal_w - root_pos_w)


def obs_jump_phase(env) -> torch.Tensor:
    """Returns the current jump phase as a one-hot policy observation."""
    phase = get_jump_phase(env)
    phase_obs = torch.zeros((env.num_envs, len(JUMP_PHASES)), device=env.device)
    phase_obs.scatter_(1, phase.unsqueeze(-1), 1.0)
    return phase_obs

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the G1 jump task."""

from __future__ import annotations

import torch

from ..constants import FOOT_CONTACT_SENSOR_NAMES, JUMP_PHASES, REFERENCE_MOTION_FPS
from .motion import (
    get_env_time,
    get_loader,
    get_phase_weight,
    get_reward,
    get_root_relative_foot_pos,
    warp_to_torch,
)


def track_joint_pos(env, gradient, phase_weights):
    loader = get_loader(env)
    robot = env.scene["robot"]
    current_time = get_env_time(env)
    ref_joint_pos, _, _, _, _, _, _ = loader.get_state(current_time)

    if loader.joint_ids is None:
        joint_ids, _ = robot.find_joints(loader.joint_names, preserve_order=True)
        loader.joint_ids = torch.tensor(joint_ids, device=env.device)

    joint_pos = warp_to_torch(robot.data.joint_pos)
    current_joint_pos = joint_pos[:, loader.joint_ids]

    r = get_reward(current_joint_pos, ref_joint_pos, gradient)
    return r * get_phase_weight(env, phase_weights)


def track_joint_vel(env, gradient, phase_weights):
    loader = get_loader(env)
    robot = env.scene["robot"]
    current_time = get_env_time(env)
    _, ref_joint_vel, _, _, _, _, _ = loader.get_state(current_time)

    if loader.joint_ids is None:
        joint_ids, _ = robot.find_joints(loader.joint_names, preserve_order=True)
        loader.joint_ids = torch.tensor(joint_ids, device=env.device)

    joint_vel = warp_to_torch(robot.data.joint_vel)
    current_joint_vel = joint_vel[:, loader.joint_ids]

    r = get_reward(current_joint_vel, ref_joint_vel, gradient)
    return r * get_phase_weight(env, phase_weights)


def track_root_pos_z(env, gradient, phase_weights):
    current_time = get_env_time(env)
    _, _, ref_root_pos, _, _, _, _ = get_loader(env).get_state(current_time)
    cz = env.command_manager.get_command("jump_goal")[:, 2:3]
    ref_root_pos_z = ref_root_pos[:, 2:3] + cz
    current_root_pos_z = warp_to_torch(env.scene["robot"].data.root_pos_w)[:, 2:3]

    r = get_reward(current_root_pos_z, ref_root_pos_z, gradient)
    return r * get_phase_weight(env, phase_weights)


def track_root_vel_z(env, gradient, phase_weights):
    current_time = get_env_time(env)
    _, _, _, ref_root_vel, _, _, _ = get_loader(env).get_state(current_time)
    ref_root_vel_z = ref_root_vel[:, 2:3]
    # Reference velocity is link-derived, so use the matching link-frame accessor.
    current_root_vel_z = warp_to_torch(env.scene["robot"].data.root_link_lin_vel_w)[:, 2:3]

    r = get_reward(current_root_vel_z, ref_root_vel_z, gradient)
    return r * get_phase_weight(env, phase_weights)


def track_root_orientation(env, gradient, phase_weights):
    current_time = get_env_time(env)
    _, _, _, _, ref_root_quat, _, _ = get_loader(env).get_state(current_time)
    current_root_quat = warp_to_torch(env.scene["robot"].data.root_quat_w)
    dot_product = torch.sum(current_root_quat * ref_root_quat, dim=-1, keepdim=True)
    quat_error = 1.0 - torch.square(dot_product)

    r = torch.exp(-gradient * quat_error).squeeze(-1)
    return r * get_phase_weight(env, phase_weights)


def track_root_angular_rate(env, gradient, phase_weights):
    current_time = get_env_time(env)
    _, _, _, _, _, ref_root_ang_vel, _ = get_loader(env).get_state(current_time)
    # Reference velocity is link-derived, so use the matching link-frame accessor.
    current_root_ang_vel = warp_to_torch(env.scene["robot"].data.root_link_ang_vel_w)

    r = get_reward(current_root_ang_vel, ref_root_ang_vel, gradient)
    return r * get_phase_weight(env, phase_weights)


def track_foot_z(env, gradient, phase_weights):
    loader = get_loader(env)
    robot = env.scene["robot"]
    current_time = get_env_time(env)
    current_foot_pos_rel, ref_foot_pos_rel = get_root_relative_foot_pos(env, loader, robot, current_time)

    r = get_reward(
        current_foot_pos_rel[..., 2],
        ref_foot_pos_rel[..., 2],
        gradient,
    )
    return r * get_phase_weight(env, phase_weights)


def track_foot_xy(env, gradient, phase_weights):
    loader = get_loader(env)
    robot = env.scene["robot"]
    current_time = get_env_time(env)
    current_foot_pos_rel, ref_foot_pos_rel = get_root_relative_foot_pos(env, loader, robot, current_time)

    r = get_reward(
        current_foot_pos_rel[..., :2].reshape(env.num_envs, -1),
        ref_foot_pos_rel[..., :2].reshape(env.num_envs, -1),
        gradient,
    )
    return r * get_phase_weight(env, phase_weights)


def target_position(env, gradient, phase_weights):
    current_xy = warp_to_torch(env.scene["robot"].data.root_pos_w)[:, :2]
    target_xy = env.command_manager.get_command("jump_goal")[:, :2]

    r = get_reward(current_xy, target_xy, gradient)
    return r * get_phase_weight(env, phase_weights)


def target_velocity(env, gradient, phase_weights):
    current_vel_xy = warp_to_torch(env.scene["robot"].data.root_lin_vel_w)[:, :2]
    active_frames = sum(
        end - start for weight, (start, end) in zip(phase_weights, JUMP_PHASES.values()) if weight != 0.0
    )
    active_duration_s = active_frames / REFERENCE_MOTION_FPS
    if active_duration_s == 0:
        return torch.zeros(env.num_envs, device=env.device)
    target_displacement_xy = env.command_manager.get_term("jump_goal").target_displacement_w
    target_vel_xy = target_displacement_xy / active_duration_s

    r = get_reward(current_vel_xy, target_vel_xy, gradient)
    return r * get_phase_weight(env, phase_weights)


def target_orientation(env, gradient, phase_weights):
    current_quat = warp_to_torch(env.scene["robot"].data.root_quat_w)
    target_quat = env.command_manager.get_command("jump_goal")[:, 3:7]
    dot_product = torch.sum(current_quat * target_quat, dim=-1, keepdim=True)
    quat_error = 1.0 - torch.square(dot_product)

    r = torch.exp(-gradient * quat_error).squeeze(-1)
    return r * get_phase_weight(env, phase_weights)


def target_angular_rate(env, gradient, phase_weights):
    current_ang_vel = warp_to_torch(env.scene["robot"].data.root_ang_vel_w)
    active_frames = sum(
        end - start for weight, (start, end) in zip(phase_weights, JUMP_PHASES.values()) if weight != 0.0
    )
    active_duration_s = active_frames / REFERENCE_MOTION_FPS
    if active_duration_s == 0:
        return torch.zeros(env.num_envs, device=env.device)
    target_yaw_displacement = env.command_manager.get_term("jump_goal").target_yaw_displacement_w
    target_ang_vel = torch.zeros(env.num_envs, 3, device=env.device)
    target_ang_vel[:, 2] = target_yaw_displacement / active_duration_s

    r = get_reward(current_ang_vel, target_ang_vel, gradient)
    return r * get_phase_weight(env, phase_weights)


def penalize_ground_impact(env, gradient, phase_weights):
    fz_total = torch.zeros(env.num_envs, 1, device=env.device)
    for sensor_name in FOOT_CONTACT_SENSOR_NAMES:
        contact_sensor = env.scene.sensors[sensor_name]
        forces = contact_sensor.data.net_forces_w
        forces_t = warp_to_torch(forces)
        fz_total += torch.sum(torch.abs(forces_t[..., 2]).reshape(env.num_envs, -1), dim=1, keepdim=True)
    r = get_reward(fz_total, torch.zeros_like(fz_total), gradient)
    return r * get_phase_weight(env, phase_weights)


def penalize_torque_consumption(env, gradient, phase_weights):
    torques = warp_to_torch(env.scene["robot"].data.applied_torque)
    r = get_reward(torques, torch.zeros_like(torques), gradient)
    return r * get_phase_weight(env, phase_weights)


def penalize_joint_vel(env, gradient, phase_weights):
    joint_vel = warp_to_torch(env.scene["robot"].data.joint_vel)
    r = get_reward(joint_vel, torch.zeros_like(joint_vel), gradient)
    return r * get_phase_weight(env, phase_weights)


def penalize_joint_acc(env, gradient, phase_weights):
    accel = warp_to_torch(env.scene["robot"].data.joint_acc)
    r = get_reward(accel, torch.zeros_like(accel), gradient)
    return r * get_phase_weight(env, phase_weights)

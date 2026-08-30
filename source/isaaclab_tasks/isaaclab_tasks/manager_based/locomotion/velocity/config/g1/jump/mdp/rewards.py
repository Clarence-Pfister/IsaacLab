# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the G1 jump task."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from ..constants import FOOT_CONTACT_SENSOR_NAMES, JOINT_ACTION_SCALES, JUMP_PHASES, REFERENCE_MOTION_FPS
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


def target_position_error(env, phase_weights, retrigger_only=False):
    """Return unsaturated planar landing-position error.

    Args:
        env: Environment from which to read root position and the jump command.
        phase_weights: Error multiplier for each ordered jump phase.
        retrigger_only: Whether to score only episodes initialized from a prior
            policy landing.

    Returns:
        Squared planar position error [m^2] multiplied by the current phase
        weight. Fresh-start episodes return zero when :paramref:`retrigger_only`
        is enabled.
    """
    current_xy = warp_to_torch(env.scene["robot"].data.root_pos_w)[:, :2]
    target_xy = env.command_manager.get_command("jump_goal")[:, :2]
    error = torch.sum(torch.square(current_xy - target_xy), dim=-1)
    if retrigger_only:
        retrigger_mask = getattr(env, "retrigger_reset_mask", None)
        if retrigger_mask is None:
            retrigger_mask = torch.zeros_like(error, dtype=torch.bool)
        error = error * retrigger_mask.to(device=error.device, dtype=error.dtype)
    return error * get_phase_weight(env, phase_weights)


def _target_velocity_xy(env, phase_weights):
    active_frames = sum(
        end - start for weight, (start, end) in zip(phase_weights, JUMP_PHASES.values()) if weight != 0.0
    )
    active_duration_s = active_frames / REFERENCE_MOTION_FPS
    if active_duration_s == 0:
        return torch.zeros((env.num_envs, 2), device=env.device)
    target_displacement_xy = env.command_manager.get_term("jump_goal").target_displacement_w
    return target_displacement_xy / active_duration_s


def target_velocity(env, gradient, phase_weights):
    current_vel_xy = warp_to_torch(env.scene["robot"].data.root_lin_vel_w)[:, :2]
    target_vel_xy = _target_velocity_xy(env, phase_weights)

    r = get_reward(current_vel_xy, target_vel_xy, gradient)
    return r * get_phase_weight(env, phase_weights)


def target_velocity_error(env, phase_weights):
    """Return an unsaturated planar velocity tracking error.

    The desired velocity spreads the commanded planar displacement across the jump phases
    whose configured weights are non-zero. Unlike :func:`target_velocity`, this term does
    not exponentially flatten large errors and is intended for use with a negative reward
    weight during early command-conditioning curricula.

    Args:
        env: Environment from which to read root velocity and the jump command.
        phase_weights: Error multiplier for each ordered jump phase.

    Returns:
        Squared planar velocity error [(m/s)^2] multiplied by the current phase weight.
    """
    current_vel_xy = warp_to_torch(env.scene["robot"].data.root_lin_vel_w)[:, :2]
    target_vel_xy = _target_velocity_xy(env, phase_weights)
    error = torch.sum(torch.square(current_vel_xy - target_vel_xy), dim=-1)
    return error * get_phase_weight(env, phase_weights)


def target_orientation(env, gradient, phase_weights):
    current_quat = warp_to_torch(env.scene["robot"].data.root_quat_w)
    target_quat = env.command_manager.get_command("jump_goal")[:, 3:7]
    dot_product = torch.sum(current_quat * target_quat, dim=-1, keepdim=True)
    quat_error = 1.0 - torch.square(dot_product)

    r = torch.exp(-gradient * quat_error).squeeze(-1)
    return r * get_phase_weight(env, phase_weights)


def target_heading(env, gradient, phase_weights):
    """Reward wrapped yaw accuracy independently of landing roll and pitch.

    Args:
        env: Environment from which to read the command and root attitude.
        gradient: Exponential squared-error gradient [rad^-2].
        phase_weights: Reward multiplier for each ordered jump phase.

    Returns:
        Dimensionless per-environment heading reward.
    """
    current_quat = warp_to_torch(env.scene["robot"].data.root_quat_w)
    target_quat = env.command_manager.get_command("jump_goal")[:, 3:7]
    _, _, current_yaw = euler_xyz_from_quat(current_quat)
    _, _, target_yaw = euler_xyz_from_quat(target_quat)
    yaw_delta = current_yaw - target_yaw
    yaw_error = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
    reward = torch.exp(-gradient * torch.square(yaw_error))
    return reward * get_phase_weight(env, phase_weights)


def target_angular_rate(env, gradient, phase_weights):
    """Reward the commanded yaw rate without constraining jump pitch and roll rates.

    Args:
        env: Environment from which to read the command and root angular velocity.
        gradient: Exponential squared-error gradient [(rad/s)^-2].
        phase_weights: Reward multiplier for each ordered jump phase.

    Returns:
        Dimensionless per-environment yaw-rate reward.
    """
    current_yaw_rate = warp_to_torch(env.scene["robot"].data.root_ang_vel_w)[:, 2:3]
    active_frames = sum(
        end - start for weight, (start, end) in zip(phase_weights, JUMP_PHASES.values()) if weight != 0.0
    )
    active_duration_s = active_frames / REFERENCE_MOTION_FPS
    if active_duration_s == 0:
        return torch.zeros(env.num_envs, device=env.device)
    target_yaw_displacement = env.command_manager.get_term("jump_goal").target_yaw_displacement_w
    target_yaw_rate = (target_yaw_displacement / active_duration_s).unsqueeze(-1)

    r = get_reward(current_yaw_rate, target_yaw_rate, gradient)
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


def joint_torque_demand_limit(env, soft_ratio, maximum_excess, phase_weights):
    """Penalize implicit-PD torque demand beyond a fraction of the effort limit.

    Unlike applied torque, the demand is evaluated before actuator clipping. A policy therefore
    cannot make a target arbitrarily more aggressive without paying an additional cost after the
    actuator reaches its limit.

    Args:
        env: Environment from which to read joint state and controller properties.
        soft_ratio: Effort fraction at which the penalty begins.
        maximum_excess: Per-joint normalized excess at which the squared cost is capped.
        phase_weights: Cost multiplier for each ordered jump phase.

    Returns:
        Dimensionless summed squared excess for each environment.

    Raises:
        ValueError: If :paramref:`soft_ratio` is outside ``(0, 1]`` or
            :paramref:`maximum_excess` is not positive.
    """
    if not 0.0 < soft_ratio <= 1.0:
        raise ValueError(f"soft_ratio must be in (0, 1], got {soft_ratio}.")
    if maximum_excess <= 0.0:
        raise ValueError(f"maximum_excess must be positive, got {maximum_excess}.")

    data = env.scene["robot"].data
    joint_pos_target = warp_to_torch(data.joint_pos_target)
    joint_pos = warp_to_torch(data.joint_pos)
    joint_vel = warp_to_torch(data.joint_vel)
    stiffness = warp_to_torch(data.joint_stiffness)
    damping = warp_to_torch(data.joint_damping)
    effort_limit = warp_to_torch(data.joint_effort_limits)

    torque_demand = stiffness * (joint_pos_target - joint_pos) - damping * joint_vel
    normalized_demand = torch.abs(torque_demand) / torch.clamp_min(effort_limit, torch.finfo(joint_pos.dtype).eps)
    excess = torch.clamp(normalized_demand - soft_ratio, min=0.0, max=maximum_excess)
    return torch.sum(torch.square(excess), dim=1) * get_phase_weight(env, phase_weights)


def joint_target_lower_limit(env, lower_limit, normalization, phase_weights, asset_cfg):
    """Penalize joint-position targets below a configured safety floor.

    Args:
        env: Environment from which to read joint position targets.
        lower_limit: Minimum requested joint position [rad].
        normalization: Joint-position shortfall represented by unit cost [rad].
        phase_weights: Cost multiplier for each ordered jump phase.
        asset_cfg: Scene entity selecting the joints to evaluate.

    Returns:
        Summed squared normalized target shortfall for each environment.

    Raises:
        ValueError: If :paramref:`lower_limit` is not finite or
            :paramref:`normalization` is not positive and finite.
    """
    if not math.isfinite(lower_limit):
        raise ValueError(f"lower_limit must be finite, got {lower_limit}.")
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError(f"normalization must be positive and finite, got {normalization}.")

    joint_targets = warp_to_torch(env.scene[asset_cfg.name].data.joint_pos_target)[:, asset_cfg.joint_ids]
    normalized_shortfall = torch.clamp((lower_limit - joint_targets) / normalization, min=0.0)
    return torch.sum(torch.square(normalized_shortfall), dim=1) * get_phase_weight(env, phase_weights)


def joint_position_limit_margin(
    env,
    margin,
    phase_weights,
    asset_cfg,
    retrigger_only=False,
    use_soft_joint_limits=True,
):
    """Penalize joint positions that enter a margin next to either soft limit.

    Args:
        env: Environment from which to read joint positions and soft limits.
        margin: Width of the penalized region inside each joint limit [rad].
        phase_weights: Cost multiplier for each ordered jump phase.
        asset_cfg: Scene entity selecting the joints to evaluate.
        retrigger_only: Whether to score only episodes carried from a prior
            policy landing.
        use_soft_joint_limits: Whether to measure margin against training soft
            limits instead of physical articulation limits.

    Returns:
        Summed joint-limit margin intrusion [rad] for each environment.

    Raises:
        ValueError: If :paramref:`margin` is not positive and finite.
    """
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError(f"margin must be positive and finite, got {margin}.")

    data = env.scene[asset_cfg.name].data
    joint_pos = warp_to_torch(data.joint_pos)[:, asset_cfg.joint_ids]
    joint_limit_data = data.soft_joint_pos_limits if use_soft_joint_limits else data.joint_pos_limits
    joint_pos_limits = warp_to_torch(joint_limit_data)[:, asset_cfg.joint_ids]
    lower_intrusion = torch.clamp(joint_pos_limits[..., 0] + margin - joint_pos, min=0.0)
    upper_intrusion = torch.clamp(joint_pos - (joint_pos_limits[..., 1] - margin), min=0.0)
    intrusion = torch.sum(lower_intrusion + upper_intrusion, dim=1)
    if retrigger_only:
        retrigger_mask = getattr(env, "retrigger_reset_mask", None)
        if retrigger_mask is None:
            retrigger_mask = torch.zeros_like(intrusion, dtype=torch.bool)
        intrusion = intrusion * retrigger_mask.to(
            device=intrusion.device,
            dtype=intrusion.dtype,
        )
    return intrusion * get_phase_weight(env, phase_weights)


def reference_joint_target_deviation(env, phase_weights):
    """Measure commanded joint-target deviation from the reference motion.

    The position targets and reference positions [rad] are normalized by each joint's
    configured action scale [rad]. This exposes policies that reproduce the reference state
    by repeatedly driving the PD controller into its effort limit instead of commanding a
    reference-like target trajectory.

    Args:
        env: Environment from which to read joint targets and reference state.
        phase_weights: Cost multiplier for each ordered jump phase.

    Returns:
        Mean squared normalized target deviation for each environment.
    """
    loader = get_loader(env)
    robot = env.scene["robot"]
    current_time = get_env_time(env)
    ref_joint_pos, _, _, _, _, _, _ = loader.get_state(current_time)

    if loader.joint_ids is None:
        joint_ids, _ = robot.find_joints(loader.joint_names, preserve_order=True)
        loader.joint_ids = torch.tensor(joint_ids, device=env.device)
    if not hasattr(loader, "joint_action_scales"):
        loader.joint_action_scales = torch.tensor(
            [JOINT_ACTION_SCALES[name] for name in loader.joint_names],
            device=env.device,
            dtype=ref_joint_pos.dtype,
        )

    joint_pos_target = warp_to_torch(robot.data.joint_pos_target)[:, loader.joint_ids]
    normalized_error = (joint_pos_target - ref_joint_pos) / loader.joint_action_scales
    return torch.mean(torch.square(normalized_error), dim=1) * get_phase_weight(env, phase_weights)


def penalize_joint_vel(env, gradient, phase_weights):
    joint_vel = warp_to_torch(env.scene["robot"].data.joint_vel)
    r = get_reward(joint_vel, torch.zeros_like(joint_vel), gradient)
    return r * get_phase_weight(env, phase_weights)


def penalize_joint_acc(env, gradient, phase_weights):
    accel = warp_to_torch(env.scene["robot"].data.joint_acc)
    r = get_reward(accel, torch.zeros_like(accel), gradient)
    return r * get_phase_weight(env, phase_weights)

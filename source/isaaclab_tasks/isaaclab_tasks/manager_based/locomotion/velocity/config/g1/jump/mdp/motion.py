# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-motion loading, interpolation and phase helpers for the G1 jump task."""

from __future__ import annotations

import os
from collections.abc import Sequence

import pandas as pd
import torch

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import axis_angle_from_quat, convert_quat, normalize, quat_conjugate, quat_mul

from ..constants import (
    CSV_MOTION_PATH,
    JOINT_NAMES,
    JUMP_PHASES,
    NUMBER_OF_JOINTS,
    REFERENCE_MOTION_FPS,
    REFERENCE_NUM_FRAMES,
)


def get_reference_initial_pose() -> tuple[
    dict[str, float], tuple[float, float, float], tuple[float, float, float, float]
]:
    """Read the joint and root pose at reference frame zero.

    Returns:
        Joint positions [rad] keyed by joint name, root position [m] in XYZ order,
        and the normalized world-from-root quaternion in XYZW order.
    """
    frame0 = pd.read_csv(CSV_MOTION_PATH, nrows=1).iloc[0]
    joint_pos = {joint_name: float(frame0[joint_name]) for joint_name in JOINT_NAMES}
    root_pos = tuple(float(frame0[name]) for name in ("root_translateX", "root_translateY", "root_translateZ"))
    root_quat_wxyz = tuple(
        float(frame0[name]) for name in ("root_quaternionW", "root_quaternionX", "root_quaternionY", "root_quaternionZ")
    )
    quaternion_norm = sum(value * value for value in root_quat_wxyz) ** 0.5
    if quaternion_norm <= 0.0:
        raise ValueError("Reference frame-zero root quaternion has zero norm.")
    root_quat_xyzw = tuple(root_quat_wxyz[index] / quaternion_norm for index in (1, 2, 3, 0))
    return joint_pos, root_pos, root_quat_xyzw


def get_reference_initial_state() -> tuple[dict[str, float], float]:
    """Read frame zero joint positions and root height for robot initialization.

    Returns:
        The joint positions [rad] keyed by joint name and the root height [m].
    """
    joint_pos, root_pos, _ = get_reference_initial_pose()
    return joint_pos, root_pos[2]


def slerp_quat(
    q0: torch.Tensor,
    q1: torch.Tensor,
    blend: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Batched SLERP for xyzw quaternions."""
    q0 = normalize(q0)
    q1 = normalize(q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.clamp(torch.abs(dot), max=1.0)

    use_lerp = dot > 0.9995
    lerp = normalize((1.0 - blend) * q0 + blend * q1)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * blend
    sin_theta = torch.sin(theta)
    s0 = torch.cos(theta) - dot * sin_theta / torch.clamp(sin_theta_0, min=eps)
    s1 = sin_theta / torch.clamp(sin_theta_0, min=eps)
    slerp = normalize(s0 * q0 + s1 * q1)
    return torch.where(use_lerp, lerp, slerp)


class MotionLoader:
    """Handles loading and interpolation of joint and root motion data from a CSV file."""

    def __init__(self, csv_path: str, device: str):
        self.device = device
        self.csv_path = os.path.abspath(csv_path)

        self.joint_names = JOINT_NAMES
        self.joint_ids = None
        self.foot_ids = None

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Reference motion CSV not found: {self.csv_path}.")

        df = pd.read_csv(self.csv_path)
        motion_dt = 1 / REFERENCE_MOTION_FPS

        # Joint Positions & Velocities
        self.ref_joint_pos = torch.tensor(df[JOINT_NAMES].values, device=device, dtype=torch.float32)
        # Central differences with one-sided boundary differences.
        self.ref_joint_vel = torch.zeros_like(self.ref_joint_pos)
        self.ref_joint_vel[1:-1] = (self.ref_joint_pos[2:] - self.ref_joint_pos[:-2]) / (2 * motion_dt)
        self.ref_joint_vel[0] = (self.ref_joint_pos[1] - self.ref_joint_pos[0]) / motion_dt
        self.ref_joint_vel[-1] = (self.ref_joint_pos[-1] - self.ref_joint_pos[-2]) / motion_dt

        # Root Translation & Linear Velocity
        root_pos_cols = ["root_translateX", "root_translateY", "root_translateZ"]
        self.ref_root_pos = torch.tensor(df[root_pos_cols].values, device=device, dtype=torch.float32)
        # Central differences with one-sided boundary differences.
        self.ref_root_vel = torch.zeros_like(self.ref_root_pos)
        self.ref_root_vel[1:-1] = (self.ref_root_pos[2:] - self.ref_root_pos[:-2]) / (2 * motion_dt)
        self.ref_root_vel[0] = (self.ref_root_pos[1] - self.ref_root_pos[0]) / motion_dt
        self.ref_root_vel[-1] = (self.ref_root_pos[-1] - self.ref_root_pos[-2]) / motion_dt

        # Root Quaternions & Angular Rate
        root_quat_cols = [
            "root_quaternionW",
            "root_quaternionX",
            "root_quaternionY",
            "root_quaternionZ",
        ]
        self.ref_root_quat = normalize(
            convert_quat(
                torch.tensor(df[root_quat_cols].values, device=device, dtype=torch.float32),
                to="xyzw",
            )
        )
        # Central quaternion differences with one-sided boundary differences.
        self.ref_root_ang_vel = torch.zeros_like(self.ref_root_pos)
        root_delta_quat = quat_mul(self.ref_root_quat[2:], quat_conjugate(self.ref_root_quat[:-2]))
        self.ref_root_ang_vel[1:-1] = axis_angle_from_quat(root_delta_quat) / (2 * motion_dt)
        root_delta_quat_start = quat_mul(self.ref_root_quat[1], quat_conjugate(self.ref_root_quat[0]))
        self.ref_root_ang_vel[0] = axis_angle_from_quat(root_delta_quat_start) / motion_dt
        root_delta_quat_end = quat_mul(self.ref_root_quat[-1], quat_conjugate(self.ref_root_quat[-2]))
        self.ref_root_ang_vel[-1] = axis_angle_from_quat(root_delta_quat_end) / motion_dt

        # Foot positions in the reference motion world frame.
        foot_pos_cols = [
            "left_foot_x",
            "left_foot_y",
            "left_foot_z",
            "right_foot_x",
            "right_foot_y",
            "right_foot_z",
        ]
        self.ref_foot_pos = torch.tensor(
            df[foot_pos_cols].values.reshape(-1, 2, 3),
            device=device,
            dtype=torch.float32,
        )

        self.length = self.ref_joint_pos.shape[0]
        self.num_joints = self.ref_joint_pos.shape[1]

        if self.length != REFERENCE_NUM_FRAMES:
            raise ValueError(f"Expected {REFERENCE_NUM_FRAMES} frames in reference motion, but found {self.length}.")
        if self.num_joints != NUMBER_OF_JOINTS:
            raise ValueError(f"Expected {NUMBER_OF_JOINTS} joints in reference motion, but found {self.num_joints}.")
        print(f"Loaded reference motion from {self.csv_path}")
        print(f"Motion length: {self.length}, Number of joints: {self.num_joints}")

    def get_state(self, current_time: torch.Tensor):
        """Returns linearly interpolated reference states."""
        frame_idx_float = current_time * REFERENCE_MOTION_FPS
        idx_low = torch.floor(frame_idx_float).to(torch.long)
        idx_high = torch.ceil(frame_idx_float).to(torch.long)

        gradient = (frame_idx_float - idx_low).unsqueeze(-1)
        # Defensively clamp negative times even though get_env_time cannot produce them.
        idx_low = torch.clamp(idx_low, min=0, max=self.length - 1)
        idx_high = torch.clamp(idx_high, min=0, max=self.length - 1)
        is_clamp = (idx_low >= self.length - 1).unsqueeze(-1)

        # Interpolate
        interp_joint_pos = (1.0 - gradient) * self.ref_joint_pos[idx_low] + gradient * self.ref_joint_pos[idx_high]
        interp_root_pos = (1.0 - gradient) * self.ref_root_pos[idx_low] + gradient * self.ref_root_pos[idx_high]
        interp_foot_pos = (1.0 - gradient.unsqueeze(-1)) * self.ref_foot_pos[idx_low] + gradient.unsqueeze(
            -1
        ) * self.ref_foot_pos[idx_high]
        interp_root_quat = slerp_quat(self.ref_root_quat[idx_low], self.ref_root_quat[idx_high], gradient)

        interp_joint_vel = (1.0 - gradient) * self.ref_joint_vel[idx_low] + gradient * self.ref_joint_vel[idx_high]
        interp_root_vel = (1.0 - gradient) * self.ref_root_vel[idx_low] + gradient * self.ref_root_vel[idx_high]
        interp_root_ang_vel = (1.0 - gradient) * self.ref_root_ang_vel[idx_low] + gradient * self.ref_root_ang_vel[
            idx_high
        ]

        # Zero velocities if animation finished
        interp_joint_vel = torch.where(is_clamp, torch.zeros_like(interp_joint_vel), interp_joint_vel)
        interp_root_vel = torch.where(is_clamp[..., :3], torch.zeros_like(interp_root_vel), interp_root_vel)
        interp_root_ang_vel = torch.where(
            is_clamp[..., :3],
            torch.zeros_like(interp_root_ang_vel),
            interp_root_ang_vel,
        )

        return (
            interp_joint_pos,
            interp_joint_vel,
            interp_root_pos,
            interp_root_vel,
            interp_root_quat,
            interp_root_ang_vel,
            interp_foot_pos,
        )


def get_loader(env) -> MotionLoader:
    """Singleton accessor for MotionLoader."""
    if not hasattr(env, "motion_loader"):
        env.motion_loader = MotionLoader(CSV_MOTION_PATH, env.device)
    return env.motion_loader


def get_reward(u: torch.Tensor, v: torch.Tensor, gradient: float) -> torch.Tensor:
    """Computes bounded reward using exponential of negative squared error."""
    return torch.exp(-gradient * torch.sum(torch.square(u - v), dim=-1))


def get_jump_phase(env) -> torch.Tensor:
    """Returns the current reference jump phase for each environment."""
    current_time = get_env_time(env)
    # Internal boundaries are phase starts; right=True preserves [start, end) semantics.
    phase_ends = torch.tensor(
        [end / REFERENCE_MOTION_FPS for _, end in JUMP_PHASES.values()][:-1],
        device=env.device,
        dtype=current_time.dtype,
    )
    return torch.bucketize(current_time, phase_ends, right=True)


def get_phase_weight(env, phase_weights: Sequence[float]) -> torch.Tensor:
    """Returns a per-environment scalar weight selected by jump phase."""
    weights = torch.tensor(phase_weights, device=env.device, dtype=torch.float32)
    return weights[get_jump_phase(env)]


def get_phase_id(phase_name: str) -> int:
    """Returns the integer phase id from the ordered JUMP_PHASES mapping."""
    try:
        return list(JUMP_PHASES).index(phase_name)
    except ValueError as exc:
        raise ValueError(f"Unknown jump phase: {phase_name}.") from exc


def get_env_time(env) -> torch.Tensor:
    if not hasattr(env, "start_times"):
        env.start_times = torch.zeros(env.num_envs, device=env.device)
    # A negative start time intentionally holds the deployment reference at frame zero
    # while a repeated-jump handoff fills observation and action history. Reference
    # lookups, phase labels, and the exported table all use frame zero during this hold.
    return torch.clamp_min(env.start_times + env.episode_length_buf * env.step_dt, 0.0)


def warp_to_torch(warp_array):
    """Utility to convert a Warp array to a PyTorch tensor with zero-copy if possible."""
    if isinstance(warp_array, torch.Tensor):
        return warp_array
    else:
        import warp

        return warp.to_torch(warp_array)


def set_contact_sensor(prim_path: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=prim_path,
        history_length=3,
    )


def get_current_foot_pos_w(env, loader: MotionLoader, robot) -> torch.Tensor:
    """Returns current left/right foot positions in world frame."""
    if loader.foot_ids is None:
        foot_ids, _ = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        loader.foot_ids = torch.tensor(foot_ids, device=env.device)

    body_pos_w = warp_to_torch(robot.data.body_pos_w)
    return body_pos_w[:, loader.foot_ids, :]


def get_root_relative_foot_pos(env, loader: MotionLoader, robot, current_time: torch.Tensor):
    """Returns current and reference left/right foot positions relative to the root."""
    _, _, ref_root_pos, _, _, _, ref_foot_pos = loader.get_state(current_time)

    current_foot_pos = get_current_foot_pos_w(env, loader, robot)
    current_root_pos = warp_to_torch(robot.data.root_pos_w).unsqueeze(1)

    current_foot_pos_rel = current_foot_pos - current_root_pos
    ref_foot_pos_rel = ref_foot_pos - ref_root_pos.unsqueeze(1)
    return current_foot_pos_rel, ref_foot_pos_rel

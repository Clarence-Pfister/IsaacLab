# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment-time projection of joint targets into an implicit-PD torque envelope."""

from __future__ import annotations

import numpy as np


def project_position_target_to_lower_limit(
    joint_pos_target: np.ndarray,
    joint_vel: np.ndarray,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
    velocity_lookahead: np.ndarray,
) -> np.ndarray:
    """Raise position targets to brake motion toward a lower command limit.

    Args:
        joint_pos_target: Requested joint positions [rad].
        joint_vel: Measured joint velocities [rad/s].
        position_lower: Lower command bounds [rad].
        position_upper: Upper command bounds [rad].
        velocity_lookahead: Lower-limit braking lookahead [s].

    Returns:
        Joint position targets with velocity-aware lower-limit braking [rad].

    Raises:
        ValueError: If inputs have inconsistent shapes, contain non-finite
            values, or define invalid bounds or lookahead times.
    """
    named_values = {
        "joint_pos_target": joint_pos_target,
        "joint_vel": joint_vel,
        "position_lower": position_lower,
        "position_upper": position_upper,
        "velocity_lookahead": velocity_lookahead,
    }
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in named_values.items()}
    shape = arrays["joint_pos_target"].shape
    if not shape:
        raise ValueError("joint_pos_target must contain at least one joint.")
    for name, value in arrays.items():
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must have shape {shape} and contain only finite values.")
    if np.any(arrays["position_lower"] >= arrays["position_upper"]):
        raise ValueError("position_lower values must be strictly below position_upper values.")
    if np.any(arrays["velocity_lookahead"] < 0.0):
        raise ValueError("velocity_lookahead values must be non-negative.")

    approach_speed = np.maximum(-arrays["joint_vel"], 0.0)
    braking_target = arrays["position_lower"] + arrays["velocity_lookahead"] * approach_speed
    braking_target = np.minimum(braking_target, arrays["position_upper"])
    return np.maximum(arrays["joint_pos_target"], braking_target)


def project_pd_position_target(
    joint_pos_target: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    effort_limit: np.ndarray,
    effort_limit_ratio: np.ndarray,
) -> np.ndarray:
    """Project a joint-position target into an instantaneous PD effort envelope.

    Args:
        joint_pos_target: Requested joint positions [rad].
        joint_pos: Measured joint positions [rad].
        joint_vel: Measured joint velocities [rad/s].
        stiffness: Joint position gains [N·m/rad].
        damping: Joint velocity gains [N·m·s/rad].
        effort_limit: Absolute actuator effort limits [N·m].
        effort_limit_ratio: Available fraction of each effort limit.

    Returns:
        Nearest joint-position targets whose instantaneous PD demand is inside
        the requested effort envelope [rad].

    Raises:
        ValueError: If inputs have inconsistent shapes, contain non-finite
            values, or define an invalid actuator envelope.
    """
    named_values = {
        "joint_pos_target": joint_pos_target,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "stiffness": stiffness,
        "damping": damping,
        "effort_limit": effort_limit,
        "effort_limit_ratio": effort_limit_ratio,
    }
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in named_values.items()}
    shape = arrays["joint_pos_target"].shape
    if not shape:
        raise ValueError("joint_pos_target must contain at least one joint.")
    for name, value in arrays.items():
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must have shape {shape} and contain only finite values.")
    if np.any(arrays["stiffness"] <= 0.0):
        raise ValueError("stiffness must be strictly positive for torque projection.")
    if np.any(arrays["damping"] < 0.0):
        raise ValueError("damping must be non-negative for torque projection.")
    if np.any(arrays["effort_limit"] <= 0.0):
        raise ValueError("effort_limit must be strictly positive for torque projection.")
    ratio = arrays["effort_limit_ratio"]
    if np.any(ratio <= 0.0) or np.any(ratio > 1.0):
        raise ValueError("effort_limit_ratio values must be in (0, 1].")

    torque_demand = arrays["stiffness"] * (arrays["joint_pos_target"] - arrays["joint_pos"])
    torque_demand -= arrays["damping"] * arrays["joint_vel"]
    torque_limit = ratio * arrays["effort_limit"]
    projected_torque = np.clip(torque_demand, -torque_limit, torque_limit)
    return arrays["joint_pos"] + (projected_torque + arrays["damping"] * arrays["joint_vel"]) / arrays["stiffness"]

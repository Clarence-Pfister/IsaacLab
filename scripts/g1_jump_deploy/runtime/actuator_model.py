# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actuator models shared by the G1 deployment simulators."""

from __future__ import annotations

import math

import numpy as np


def saturate_torque_at_velocity_limit(
    torque: np.ndarray,
    joint_velocity: np.ndarray,
    velocity_limit: np.ndarray,
    *,
    knee_width: float = 0.1,
) -> np.ndarray:
    """Roll off motion-accelerating torque near each joint velocity limit.

    Args:
        torque: Requested joint torque [N·m].
        joint_velocity: Joint velocity [rad/s].
        velocity_limit: Symmetric joint velocity limit [rad/s].
        knee_width: Fraction of each velocity limit occupied by the linear
            roll-off region.

    Returns:
        Saturated joint torque [N·m]. Braking torque is unchanged.

    Raises:
        ValueError: If the inputs have different shapes, contain non-finite
            values, have non-positive velocity limits, or ``knee_width`` is
            outside ``(0, 1]``.
    """
    torque_array = np.asarray(torque, dtype=np.float64)
    velocity_array = np.asarray(joint_velocity, dtype=np.float64)
    limit_array = np.asarray(velocity_limit, dtype=np.float64)
    if torque_array.shape != velocity_array.shape or torque_array.shape != limit_array.shape:
        raise ValueError("torque, joint_velocity, and velocity_limit must have identical shapes.")
    if not np.all(np.isfinite(torque_array)) or not np.all(np.isfinite(velocity_array)):
        raise ValueError("torque and joint_velocity must contain only finite values.")
    if not np.all(np.isfinite(limit_array)) or np.any(limit_array <= 0.0):
        raise ValueError("velocity_limit must contain only positive finite values.")
    if not isinstance(knee_width, (int, float)) or isinstance(knee_width, bool):
        raise ValueError("knee_width must be a finite number in (0, 1].")
    knee_width_value = float(knee_width)
    if not math.isfinite(knee_width_value) or not 0.0 < knee_width_value <= 1.0:
        raise ValueError("knee_width must be a finite number in (0, 1].")

    scale = np.clip(
        (limit_array - np.abs(velocity_array)) / (limit_array * knee_width_value),
        0.0,
        1.0,
    )
    accelerates_motion = torque_array * velocity_array > 0.0
    return np.where(accelerates_motion, torque_array * scale, torque_array)

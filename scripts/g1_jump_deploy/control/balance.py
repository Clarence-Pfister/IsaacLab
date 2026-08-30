# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IMU-feedback ankle strategy for standing the G1 robot.

This module intentionally depends only on NumPy and the Python standard
library so the same controller can run on the robot onboard computer.
Quaternion inputs use WXYZ order and describe the body orientation in the
world frame. Gyroscope inputs use body-frame XYZ axes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

_ANKLE_PITCH_LIMIT = (-0.87267, 0.5236)
_ANKLE_ROLL_LIMIT = (-0.2618, 0.2618)
_TENDON_CONSTRAINTS = (
    (0.2618, -0.2618, 0.13708),
    (0.2618, 0.2618, 0.13708),
    (-0.2618, -0.14967, 0.22846),
    (-0.2618, 0.14967, 0.22846),
)
_FEASIBILITY_TOLERANCE = 1.0e-12


def _ankle_constraints() -> np.ndarray:
    """Return half planes as rows of ``pitch_coef, roll_coef, limit``."""
    pitch_lower, pitch_upper = _ANKLE_PITCH_LIMIT
    roll_lower, roll_upper = _ANKLE_ROLL_LIMIT
    return np.asarray(
        _TENDON_CONSTRAINTS
        + (
            (1.0, 0.0, pitch_upper),
            (-1.0, 0.0, -pitch_lower),
            (0.0, 1.0, roll_upper),
            (0.0, -1.0, -roll_lower),
        ),
        dtype=np.float64,
    )


_ANKLE_CONSTRAINTS = _ankle_constraints()


def _is_ankle_target_feasible(point: np.ndarray, tolerance: float = _FEASIBILITY_TOLERANCE) -> bool:
    return bool(np.all(_ANKLE_CONSTRAINTS[:, :2] @ point <= _ANKLE_CONSTRAINTS[:, 2] + tolerance))


def project_ankle_target(pitch: float, roll: float) -> tuple[float, float]:
    """Project an ankle target onto the joint-and-tendon feasible polygon.

    The result is the exact Euclidean projection in pitch-roll joint space. The
    four fixed-tendon upper limits and both joint box limits are enforced.

    Args:
        pitch: Requested ankle pitch position [rad].
        roll: Requested ankle roll position [rad].

    Returns:
        Feasible ankle pitch and roll positions [rad].

    Raises:
        ValueError: If either input is non-finite or the constraints are empty.
    """
    point = np.asarray((pitch, roll), dtype=np.float64)
    if not np.all(np.isfinite(point)):
        raise ValueError("Ankle target must contain finite pitch and roll values.")
    if _is_ankle_target_feasible(point, tolerance=0.0):
        return float(point[0]), float(point[1])

    candidates: list[np.ndarray] = []
    for pitch_coef, roll_coef, limit in _ANKLE_CONSTRAINTS:
        normal = np.asarray((pitch_coef, roll_coef), dtype=np.float64)
        candidate = point - ((normal @ point - limit) / (normal @ normal)) * normal
        if _is_ankle_target_feasible(candidate):
            candidates.append(candidate)

    for first_index, first in enumerate(_ANKLE_CONSTRAINTS):
        for second in _ANKLE_CONSTRAINTS[first_index + 1 :]:
            coefficients = np.asarray((first[:2], second[:2]), dtype=np.float64)
            determinant = float(np.linalg.det(coefficients))
            if abs(determinant) <= 1.0e-15:
                continue
            candidate = np.linalg.solve(coefficients, np.asarray((first[2], second[2])))
            if _is_ankle_target_feasible(candidate):
                candidates.append(candidate)

    if not candidates:
        raise ValueError("Ankle constraints define an empty feasible polygon.")
    projected = min(candidates, key=lambda candidate: float(np.sum(np.square(candidate - point))))
    return float(projected[0]), float(projected[1])


def quaternion_to_roll_pitch(quaternion_wxyz: Sequence[float]) -> tuple[float, float]:
    """Convert a world-from-body quaternion to ZYX roll and pitch angles.

    Args:
        quaternion_wxyz: World-from-body unit quaternion in WXYZ order.

    Returns:
        Body roll and pitch attitude [rad].

    Raises:
        ValueError: If the quaternion does not contain four finite values or has
            zero norm.
    """
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("IMU quaternion must contain four finite WXYZ values.")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("IMU quaternion must have non-zero norm.")
    w, x, y, z = quaternion / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


@dataclass(frozen=True)
class BalanceControllerConfig:
    """Configuration for :class:`BalanceController`."""

    target_roll: float = math.radians(0.43)
    """Target pelvis roll attitude [rad]."""

    target_pitch: float = math.radians(7.49)
    """Target pelvis pitch attitude [rad]."""

    roll_kp: float = 3.2
    """Ankle-roll offset gain [rad/rad]."""

    pitch_kp: float = 3.2
    """Ankle-pitch offset gain [rad/rad]."""

    roll_kd: float = 0.16
    """Ankle-roll angular-rate gain [rad/(rad/s)]."""

    pitch_kd: float = 0.16
    """Ankle-pitch angular-rate gain [rad/(rad/s)]."""

    roll_ki: float = 1.0
    """Ankle-roll integral gain [rad/(rad*s)]."""

    pitch_ki: float = 1.0
    """Ankle-pitch integral gain [rad/(rad*s)]."""

    integral_enabled: bool = True
    """Whether to integrate attitude error."""

    integral_limit: float = 0.2
    """Symmetric roll/pitch attitude-error integral limit [rad*s]."""

    initial_roll_integral: float = 0.0
    """Roll integral state restored by :meth:`BalanceController.reset` [rad*s]."""

    initial_pitch_integral: float = 0.2
    """Pitch integral state restored by :meth:`BalanceController.reset` [rad*s]."""

    def __post_init__(self) -> None:
        values = (
            self.target_roll,
            self.target_pitch,
            self.roll_kp,
            self.pitch_kp,
            self.roll_kd,
            self.pitch_kd,
            self.roll_ki,
            self.pitch_ki,
            self.integral_limit,
            self.initial_roll_integral,
            self.initial_pitch_integral,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Balance controller configuration values must be finite.")
        if min(self.roll_kp, self.pitch_kp, self.roll_kd, self.pitch_kd, self.roll_ki, self.pitch_ki) < 0.0:
            raise ValueError("Balance controller gains must be non-negative.")
        if self.integral_limit < 0.0:
            raise ValueError("Integral limit must be non-negative.")
        if max(abs(self.initial_roll_integral), abs(self.initial_pitch_integral)) > self.integral_limit:
            raise ValueError("Initial integral state must not exceed the integral limit.")


class BalanceController:
    """Add IMU-feedback ankle offsets to an arbitrary stand pose.

    Positive measured roll or pitch error produces a positive target offset on
    both corresponding ankle joints. This is the empirically checked G1/MuJoCo
    sign convention: a positive target offset reduced the finite-horizon
    attitude error while a negative target offset increased it.
    """

    def __init__(self, joint_names: Sequence[str], config: BalanceControllerConfig | None = None):
        """Initialize the balance controller.

        Args:
            joint_names: Joint ordering used by all position and velocity arrays.
            config: Feedback targets, gains, and integral anti-windup settings.

        Raises:
            ValueError: If joint names are duplicated or required ankles are absent.
        """
        self.joint_names = tuple(joint_names)
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Joint names must be unique.")
        required_names = tuple(f"{side}_ankle_{axis}_joint" for side in ("left", "right") for axis in ("pitch", "roll"))
        missing = [name for name in required_names if name not in self.joint_names]
        if missing:
            raise ValueError(f"Joint names are missing required ankles: {missing}.")
        self._ankle_indices = {
            side: (
                self.joint_names.index(f"{side}_ankle_pitch_joint"),
                self.joint_names.index(f"{side}_ankle_roll_joint"),
            )
            for side in ("left", "right")
        }
        self.config = config or BalanceControllerConfig()
        self._initial_integral = np.asarray(
            (self.config.initial_roll_integral, self.config.initial_pitch_integral), dtype=np.float64
        )
        self._integral = self._initial_integral.copy() if self.config.integral_enabled else np.zeros(2)
        self.last_attitude_error = np.zeros(2, dtype=np.float64)
        self.last_ankle_offset = np.zeros(2, dtype=np.float64)

    @property
    def integral_error(self) -> np.ndarray:
        """Clamped roll-pitch attitude-error integral [rad*s]."""
        return self._integral.copy()

    def reset(self) -> None:
        """Restore configured integral state before a new stand interval."""
        if self.config.integral_enabled:
            self._integral[:] = self._initial_integral
        else:
            self._integral.fill(0.0)
        self.last_attitude_error.fill(0.0)
        self.last_ankle_offset.fill(0.0)

    def set_target_attitude(self, target_roll: float, target_pitch: float) -> None:
        """Set the roll-pitch balance target and reset controller state.

        Args:
            target_roll: Target pelvis roll attitude [rad].
            target_pitch: Target pelvis pitch attitude [rad].

        Raises:
            ValueError: If either target is non-finite.
        """
        if not math.isfinite(target_roll) or not math.isfinite(target_pitch):
            raise ValueError("Balance target attitudes must be finite.")
        self.config = replace(
            self.config,
            target_roll=float(target_roll),
            target_pitch=float(target_pitch),
            initial_roll_integral=0.0,
            initial_pitch_integral=0.0,
        )
        self._initial_integral.fill(0.0)
        self.reset()

    def compute(
        self,
        stand_target: Sequence[float],
        imu_quaternion_wxyz: Sequence[float],
        imu_angular_velocity: Sequence[float],
        joint_positions: Sequence[float],
        joint_velocities: Sequence[float],
        dt: float,
    ) -> np.ndarray:
        """Return a tendon-feasible joint target with balance offsets applied.

        Args:
            stand_target: Caller-provided joint position target [rad].
            imu_quaternion_wxyz: Measured world-from-body quaternion in WXYZ order.
            imu_angular_velocity: Measured body angular velocity XYZ [rad/s].
            joint_positions: Measured joint positions [rad].
            joint_velocities: Measured joint velocities [rad/s].
            dt: Elapsed controller time [s].

        Returns:
            Joint position target in ``joint_names`` order [rad].

        Raises:
            ValueError: If an input has the wrong shape, is non-finite, or ``dt``
                is not positive.
        """
        target = self._joint_vector(stand_target, "Stand target").copy()
        self._joint_vector(joint_positions, "Joint positions")
        self._joint_vector(joint_velocities, "Joint velocities")
        angular_velocity = np.asarray(imu_angular_velocity, dtype=np.float64)
        if angular_velocity.shape != (3,) or not np.all(np.isfinite(angular_velocity)):
            raise ValueError("IMU angular velocity must contain three finite XYZ values.")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Controller dt must be a positive finite value.")

        measured_roll, measured_pitch = quaternion_to_roll_pitch(imu_quaternion_wxyz)
        # This measured-minus-target sign is deliberate and checked by
        # positive/negative target perturbations in validate_stand.py.
        error = np.asarray(
            (measured_roll - self.config.target_roll, measured_pitch - self.config.target_pitch), dtype=np.float64
        )
        if self.config.integral_enabled:
            self._integral = np.clip(
                self._integral + error * dt, -self.config.integral_limit, self.config.integral_limit
            )
        else:
            self._integral.fill(0.0)

        offset = np.asarray(
            (
                self.config.pitch_kp * error[1]
                + self.config.pitch_kd * angular_velocity[1]
                + self.config.pitch_ki * self._integral[1],
                self.config.roll_kp * error[0]
                + self.config.roll_kd * angular_velocity[0]
                + self.config.roll_ki * self._integral[0],
            ),
            dtype=np.float64,
        )
        for pitch_index, roll_index in self._ankle_indices.values():
            target[pitch_index], target[roll_index] = project_ankle_target(
                target[pitch_index] + offset[0], target[roll_index] + offset[1]
            )

        self.last_attitude_error = error
        self.last_ankle_offset = offset
        return target

    def _joint_vector(self, values: Sequence[float], label: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        expected_shape = (len(self.joint_names),)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must contain {len(self.joint_names)} finite values.")
        return result

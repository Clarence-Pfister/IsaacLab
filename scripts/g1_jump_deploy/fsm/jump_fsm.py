# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Safety state machine for a deployed G1 jump policy.

This module deliberately depends only on Python, NumPy, and the shared
:class:`JumpGoalRuntime`. Robot and operator backends are supplied through
structural protocols, so the same controller can drive simulation or hardware.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from scripts.g1_jump_deploy.control.balance import (
    BalanceController,
    BalanceControllerConfig,
    quaternion_to_roll_pitch,
)
from scripts.g1_jump_deploy.runtime.jump_goal_runtime import JumpGoalRuntime

_MEASURED_STEADY_STATE_BALANCE_OFFSET_RAD = 0.184974632273767
_DEFAULT_BALANCE_OFFSET_LIMIT_RAD = 1.25 * _MEASURED_STEADY_STATE_BALANCE_OFFSET_RAD


class JumpControllerState(str, Enum):
    """States of the jump controller."""

    PASSIVE = "PASSIVE"
    DAMPING = "DAMPING"
    STAND = "STAND"
    GOTO_START = "GOTO_START"
    ARMED = "ARMED"
    JUMP = "JUMP"
    SETTLE = "SETTLE"
    FAULT = "FAULT"


@dataclass(frozen=True)
class JumpGoal:
    """Operator-requested landing displacement relative to the trigger pose.

    Attributes:
        dx: Forward displacement [m].
        dy: Lateral displacement [m].
        dyaw: Heading displacement [rad].
        roll: Landing roll displacement [rad].
        pitch: Landing pitch displacement [rad].
    """

    dx: float
    dy: float
    dyaw: float
    roll: float = 0.0
    pitch: float = 0.0


@dataclass(frozen=True)
class StandGainConfig:
    """Overrides used to build the stand gain set from manifest jump gains.

    The default ankle kp of 80 is four times the policy's kp of 20, which is
    too soft to hold the measured static pose. An ankle kd of 5 adds damping
    for the stiffer loop. Every non-ankle joint retains its manifest value.
    Named overrides make any further departure from the manifest explicit.

    Attributes:
        ankle_stiffness: Stand ankle position gain [N·m/rad]. Set to ``None``
            to retain manifest values.
        ankle_damping: Stand ankle velocity gain [N·m·s/rad]. Set to ``None``
            to retain manifest values.
        stiffness_overrides: Additional position gains by manifest joint name
            [N·m/rad].
        damping_overrides: Additional velocity gains by manifest joint name
            [N·m·s/rad].
    """

    ankle_stiffness: float | None = 80.0
    ankle_damping: float | None = 5.0
    stiffness_overrides: Mapping[str, float] | None = None
    damping_overrides: Mapping[str, float] | None = None


@dataclass(frozen=True)
class JumpControllerConfig:
    """Timing and safety configuration for :class:`JumpControllerFSM`.

    Attributes:
        goto_start_duration_s: Nominal quintic start-pose blend duration [s].
        goto_start_timeout_s: Start-pose convergence timeout [s].
        goto_start_load_compensation_scale: Fraction of the measured static
            target-tracking error added to the non-ankle start-pose target.
        goto_start_load_compensation_limit_rad: Maximum absolute static-load
            compensation applied to each non-ankle joint [rad].
        policy_prepare_duration_s: Duration of the goal-conditioned policy
            warm-up at frozen reference phase zero [s]. Zero preserves the
            direct stand-to-armed behavior.
        policy_prepare_retain_balance: Whether to retain the preparation-start
            independent balance correction until the transition to JUMP.
        policy_prepare_pose_tolerance_rad: Maximum measured tracking error
            from the prepared policy target when arming [rad].
        policy_stand_after_jump: Whether to keep evaluating the final STAND
            reference after the finite jump instead of handing immediately to
            the independent balance controller. Before a subsequent goal, the
            balance attitude is recalibrated from the landed pose and the
            robot is normalized back to the policy's default start pose.
        policy_stand_pose_tolerance_rad: Maximum non-ankle tracking error
            accepted for policy-native standing [rad].
        policy_stand_tilt_limit_rad: Maximum absolute body tilt accepted for
            policy-native standing [rad].
        policy_stand_retrigger_prepare_duration_s: Frozen phase-zero policy
            preparation used to move directly from policy-native standing to
            a subsequent goal [s]. Zero uses default-pose normalization.
        policy_stand_direct_retrigger: Whether a subsequent goal remains under
            the preceding episode's final STAND controller until confirmation,
            then starts a fresh policy episode through the normal IDLE handoff.
            This avoids both default-pose normalization and frozen phase-zero
            preparation.
        policy_terminal_return_steps: Number of final policy steps over which
            to blend the policy target and gains to the validated stand target
            while restoring independent balance. Zero disables the return.
        jump_blend_out_duration_s: Duration of the post-episode quintic blend
            from the policy's final reference and jump gains to the validated
            stand target and gains [s]. Zero preserves the configured SETTLE
            behavior.
        jump_target_blend_steps: Number of initial policy steps used to blend
            the held stand target into the policy target. ``None`` uses the
            complete manifest IDLE interval; zero disables target blending.
        jump_gain_blend_steps: Number of initial policy steps used to blend
            stand gains into policy gains. ``None`` uses the complete manifest
            IDLE interval; zero selects policy gains immediately.
        jump_balance_blend_steps: Number of initial policy steps used to fade
            the independent stand balance correction to zero. ``None`` uses
            the complete manifest IDLE interval; zero disables it immediately.
        armed_timeout_s: Maximum confirmation wait in :attr:`~JumpControllerState.ARMED` [s].
        settle_duration_s: Nominal quintic post-jump blend duration [s].
        settle_timeout_s: Maximum time allowed for measured stand convergence
            after a jump [s].
        settle_pose_tolerance_rad: Maximum non-ankle position error allowed
            when completing settlement [rad].
        settle_joint_velocity_tolerance_rad_s: Maximum absolute joint velocity
            allowed when completing settlement [rad/s].
        pose_tolerance_rad: Per-joint start-pose tolerance [rad] for joints
            outside the ankle balance loop.
        joint_velocity_tolerance_rad_s: Maximum absolute joint velocity
            permitted when arming [rad/s].
        foot_contact_threshold_n: Minimum load required on each foot [N].
        prearm_tilt_limit_rad: Maximum roll-pitch deviation from the balance
            target attitude when arming [rad].
        balance_offset_limit_rad: Maximum ankle balance-offset vector magnitude
            permitted when arming [rad]. The default is 25 percent above the
            measured 10-second steady-state magnitude of 0.184974632274 rad.
        goal_position_z_w: Landing-surface height in the odometry world frame [m].
        jump_abort_tilt_limit_rad: Body tilt that requests an in-jump abort [rad].
        joint_limit_abort_margin_rad: Distance a joint may travel beyond its
            manifest position limit during JUMP before requesting an abort
            [rad]. Zero preserves the exact-limit abort behavior.
        contact_safety_mode: Foot-contact safety contract. ``MEASURED`` requires
            measured bilateral support before arming and uses measured contact
            for touchdown-aware abort handling. ``GANTRY_REHEARSAL`` is only
            for a mechanically constrained, ground-clear motion rehearsal and
            makes every abort enter damping immediately. ``UNMEASURED_GROUND``
            is a no-contact validation contract that never reads foot
            contact; it latches post-takeoff aborts until the policy episode
            ends because touchdown cannot be detected.
        latched_abort_upright_settle: Whether a latched post-takeoff abort may
            enter :attr:`~JumpControllerState.SETTLE` after the complete policy
            episode. Settlement is allowed only when the complete latched abort
            reason set is exactly ``{"joint limit exceeded"}`` and the body
            remains upright. Every other abort reason enters damping.
        latched_abort_settle_tilt_limit_rad: Maximum body tilt allowed when
            :attr:`latched_abort_upright_settle` is enabled [rad].
        stiffness_slew_per_s: Maximum normal-transition kp change per second
            [N·m/(rad·s)].
        damping_slew_per_s: Maximum normal-transition kd change per second
            [N·m/rad].
        stand_entry_duration_s: Nominal quintic measured-to-reference stand
            target blend duration [s].
        stand_balance_target_entry_duration_s: Initial linear measured-to-reference
            balance-attitude target transition [s]. Zero applies the configured
            target immediately.
        stand_hold_measured_pose: Whether STAND holds the measured entry pose
            instead of blending to the manifest reference.
    """

    class ContactSafetyMode(str, Enum):
        """Available foot-contact safety contracts."""

        MEASURED = "measured"
        GANTRY_REHEARSAL = "gantry_rehearsal"
        UNMEASURED_GROUND = "unmeasured_ground"

    goto_start_duration_s: float = 2.0
    goto_start_timeout_s: float = 4.0
    goto_start_load_compensation_scale: float = 1.0
    goto_start_load_compensation_limit_rad: float = 0.1
    policy_prepare_duration_s: float = 0.0
    policy_prepare_retain_balance: bool = False
    policy_prepare_pose_tolerance_rad: float = 0.1
    policy_stand_after_jump: bool = False
    policy_stand_pose_tolerance_rad: float = 0.2
    policy_stand_tilt_limit_rad: float = math.radians(15.0)
    policy_stand_retrigger_prepare_duration_s: float = 0.0
    policy_stand_direct_retrigger: bool = False
    policy_terminal_return_steps: int = 0
    jump_blend_out_duration_s: float = 0.0
    jump_target_blend_steps: int | None = None
    jump_gain_blend_steps: int | None = None
    jump_balance_blend_steps: int | None = None
    armed_timeout_s: float = 5.0
    settle_duration_s: float = 0.5
    settle_timeout_s: float = 4.0
    settle_pose_tolerance_rad: float = 0.05
    settle_joint_velocity_tolerance_rad_s: float = 0.5
    pose_tolerance_rad: float = 0.05
    joint_velocity_tolerance_rad_s: float = 0.5
    foot_contact_threshold_n: float = 20.0
    prearm_tilt_limit_rad: float = math.radians(5.0)
    balance_offset_limit_rad: float = _DEFAULT_BALANCE_OFFSET_LIMIT_RAD
    goal_position_z_w: float = 0.0
    jump_abort_tilt_limit_rad: float = math.radians(45.0)
    joint_limit_abort_margin_rad: float = 0.0
    contact_safety_mode: ContactSafetyMode = ContactSafetyMode.MEASURED
    latched_abort_upright_settle: bool = False
    latched_abort_settle_tilt_limit_rad: float = math.radians(20.0)
    stiffness_slew_per_s: float = 300.0
    damping_slew_per_s: float = 100.0
    stand_entry_duration_s: float = 1.0
    stand_balance_target_entry_duration_s: float = 0.0
    stand_hold_measured_pose: bool = False


@runtime_checkable
class RobotInterface(Protocol):
    """Structural boundary implemented by a simulator or robot backend."""

    @property
    def joint_positions(self) -> np.ndarray:
        """Measured joint positions [rad], in manifest order."""

    @property
    def joint_velocities(self) -> np.ndarray:
        """Measured joint velocities [rad/s], in manifest order."""

    @property
    def base_angular_velocity(self) -> np.ndarray:
        """Base angular velocity in the body frame [rad/s], shape ``(3,)``."""

    @property
    def imu_quaternion(self) -> np.ndarray:
        """IMU world-from-body quaternion in WXYZ order, shape ``(4,)``."""

    @property
    def odometry_position(self) -> np.ndarray:
        """Odometry root position in world coordinates [m], shape ``(3,)``."""

    @property
    def odometry_quaternion(self) -> np.ndarray:
        """Odometry world-from-body quaternion in WXYZ order, shape ``(4,)``."""

    @property
    def foot_contact_forces(self) -> np.ndarray:
        """Left and right supporting contact forces [N], shape ``(2,)``."""

    @property
    def joint_limit_violations(self) -> np.ndarray:
        """Per-joint flags indicating a joint at or outside its limit."""

    @property
    def feedback_stale(self) -> bool:
        """Whether the state feedback missed its freshness deadline."""

    @property
    def control_deadline_missed(self) -> bool:
        """Whether the control loop missed its execution deadline."""

    def command_joint_position_target(
        self,
        target: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ) -> None:
        """Command base joint targets and PD gains in manifest order.

        The caller's fast loop adds the correction returned by
        :meth:`JumpControllerFSM.update_balance` before evaluating joint PD.

        Args:
            target: Base joint position targets [rad].
            stiffness: Position gains [N·m/rad].
            damping: Velocity gains [N·m·s/rad].
        """


@runtime_checkable
class OperatorInterface(Protocol):
    """Structural boundary for level-valued operator intents."""

    @property
    def pending_goal(self) -> JumpGoal | None:
        """Goal offered for latching while the controller is standing."""

    @property
    def request_start(self) -> bool:
        """Intent to begin moving toward the jump start pose."""

    @property
    def confirm(self) -> bool:
        """Intent to confirm a separately armed jump."""

    @property
    def abort(self) -> bool:
        """Intent to enter the safe damping behavior."""


@dataclass(frozen=True)
class _ManifestData:
    policy_dt: float
    episode_steps: int
    joint_names: tuple[str, ...]
    default_position: np.ndarray
    position_target_lower: np.ndarray
    position_target_upper: np.ndarray
    joint_position_lower: np.ndarray
    joint_position_upper: np.ndarray
    jump_stiffness: np.ndarray
    jump_damping: np.ndarray
    goal_ranges: dict[str, tuple[float, float]]
    flight_freeze_enabled: bool
    idle_end_step: int
    flight_start_step: int


def _finite_vector(value: object, length: int, name: str, *, dtype: np.dtype = np.dtype(np.float64)) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} numeric values.") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape ({length},) and contain only finite values.")
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite positive number.")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _load_manifest(path: str | Path, *, require_joint_position_limits: bool = False) -> _ManifestData:  # noqa: C901
    manifest_path = Path(path).resolve()
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read deployment manifest: {manifest_path}.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Deployment manifest root must be an object.")

    control = manifest.get("control")
    joints = manifest.get("joints")
    actuators = manifest.get("actuators")
    action = manifest.get("action")
    goal = manifest.get("goal")
    reference = manifest.get("reference")
    tables = manifest.get("tables")
    for name, value in (
        ("control", control),
        ("joints", joints),
        ("actuators", actuators),
        ("action", action),
        ("goal", goal),
        ("reference", reference),
        ("tables", tables),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"Manifest field {name} must be an object.")

    policy_dt = _positive_float(control.get("policy_dt"), "control.policy_dt")
    episode_steps_value = control.get("episode_steps")
    if isinstance(episode_steps_value, bool) or not isinstance(episode_steps_value, int) or episode_steps_value <= 0:
        raise ValueError("control.episode_steps must be a positive integer.")
    episode_steps = episode_steps_value

    joint_names_value = joints.get("names")
    if not isinstance(joint_names_value, list) or not joint_names_value:
        raise ValueError("joints.names must be a non-empty array.")
    if not all(isinstance(name, str) and name for name in joint_names_value):
        raise ValueError("joints.names must contain non-empty strings.")
    joint_names = tuple(joint_names_value)
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("joints.names must not contain duplicates.")
    joint_count = len(joint_names)
    default_position = _finite_vector(joints.get("default_pos"), joint_count, "joints.default_pos")
    position_limits_value = joints.get("position_limits") if require_joint_position_limits else None
    if position_limits_value is None and not require_joint_position_limits:
        joint_position_lower = np.full(joint_count, -math.inf, dtype=np.float64)
        joint_position_upper = np.full(joint_count, math.inf, dtype=np.float64)
    elif position_limits_value is None:
        raise ValueError("joint_limit_abort_margin_rad requires manifest joints.position_limits.")
    else:
        try:
            joint_position_limits = np.asarray(position_limits_value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"joints.position_limits must contain {joint_count} finite lower-upper pairs.") from exc
        if joint_position_limits.shape != (joint_count, 2) or not np.all(np.isfinite(joint_position_limits)):
            raise ValueError(f"joints.position_limits must have finite shape ({joint_count}, 2).")
        joint_position_lower = joint_position_limits[:, 0]
        joint_position_upper = joint_position_limits[:, 1]
        if np.any(joint_position_lower >= joint_position_upper):
            raise ValueError("joints.position_limits lower bounds must be less than upper bounds.")
    try:
        position_target_clip = np.asarray(action.get("clip"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action.clip must contain {joint_count} finite lower-upper pairs.") from exc
    if position_target_clip.shape != (joint_count, 2) or not np.all(np.isfinite(position_target_clip)):
        raise ValueError(f"action.clip must have finite shape ({joint_count}, 2).")
    position_target_lower = position_target_clip[:, 0]
    position_target_upper = position_target_clip[:, 1]
    if np.any(position_target_lower > position_target_upper):
        raise ValueError("action.clip lower bounds must not exceed upper bounds.")
    if np.any(default_position < position_target_lower) or np.any(default_position > position_target_upper):
        raise ValueError("joints.default_pos must lie within action.clip.")
    if np.any(default_position < joint_position_lower) or np.any(default_position > joint_position_upper):
        raise ValueError("joints.default_pos must lie within joints.position_limits.")
    if np.any(position_target_lower < joint_position_lower) or np.any(position_target_upper > joint_position_upper):
        raise ValueError("action.clip must lie within joints.position_limits.")
    jump_stiffness = _finite_vector(actuators.get("stiffness"), joint_count, "actuators.stiffness")
    jump_damping = _finite_vector(actuators.get("damping"), joint_count, "actuators.damping")
    if np.any(jump_stiffness < 0.0) or np.any(jump_damping < 0.0):
        raise ValueError("Manifest actuator gains must be non-negative.")

    ranges_value = goal.get("ranges")
    if not isinstance(ranges_value, dict):
        raise ValueError("Manifest field goal.ranges must be an object.")
    goal_ranges = {}
    for name in ("pos_x", "pos_y", "roll", "pitch", "yaw"):
        bounds = _finite_vector(ranges_value.get(name), 2, f"goal.ranges.{name}")
        if bounds[0] > bounds[1]:
            raise ValueError(f"goal.ranges.{name} lower bound exceeds its upper bound.")
        goal_ranges[name] = (float(bounds[0]), float(bounds[1]))
    flight_freeze = goal.get("flight_freeze")
    if not isinstance(flight_freeze, dict):
        raise ValueError("Manifest field goal.flight_freeze must be an object.")
    flight_freeze_enabled = flight_freeze.get("enabled")
    if not isinstance(flight_freeze_enabled, bool):
        raise ValueError("goal.flight_freeze.enabled must be a boolean.")

    phase_names_value = reference.get("phase_names")
    if (
        not isinstance(phase_names_value, list)
        or not all(isinstance(name, str) and name for name in phase_names_value)
        or "IDLE" not in phase_names_value
        or "FLIGHT" not in phase_names_value
    ):
        raise ValueError("reference.phase_names must contain non-empty IDLE and FLIGHT names.")
    reference_fps = _positive_float(reference.get("fps"), "reference.fps")
    phase_frame_ranges = reference.get("phase_frame_ranges")
    if not isinstance(phase_frame_ranges, list) or len(phase_frame_ranges) != len(phase_names_value):
        raise ValueError("reference.phase_frame_ranges must have one range per phase name.")
    idle_frame_range = phase_frame_ranges[phase_names_value.index("IDLE")]
    if (
        not isinstance(idle_frame_range, list)
        or len(idle_frame_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in idle_frame_range)
        or idle_frame_range[0] != 0
        or idle_frame_range[1] <= idle_frame_range[0]
    ):
        raise ValueError("The IDLE phase frame range must start at zero and have a positive integer length.")
    idle_duration_s = (idle_frame_range[1] - idle_frame_range[0]) / reference_fps
    idle_end_step = int(round(idle_duration_s / policy_dt))
    if idle_end_step <= 0 or idle_end_step >= episode_steps:
        raise ValueError("The manifest IDLE phase must span at least one policy step before the episode ends.")
    phase_filename = tables.get("jump_phase")
    if not isinstance(phase_filename, str) or not phase_filename:
        raise ValueError("tables.jump_phase must name a table beside the manifest.")
    relative_phase_path = Path(phase_filename)
    if relative_phase_path.is_absolute() or relative_phase_path.name != phase_filename:
        raise ValueError("tables.jump_phase must name a table beside the manifest.")
    phase_path = (manifest_path.parent / relative_phase_path).resolve()
    if phase_path.parent != manifest_path.parent:
        raise ValueError("tables.jump_phase must name a table beside the manifest.")
    try:
        phase_table = np.load(phase_path, allow_pickle=False)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read jump phase table: {phase_path}.") from exc
    expected_shape = (episode_steps, len(phase_names_value))
    if phase_table.shape != expected_shape or not np.all(np.isfinite(phase_table)):
        raise ValueError(f"Jump phase table must be finite with shape {expected_shape}.")
    binary = np.logical_or(np.isclose(phase_table, 0.0), np.isclose(phase_table, 1.0))
    if not np.all(binary) or not np.allclose(np.sum(phase_table, axis=1), 1.0):
        raise ValueError("Jump phase table must contain one-hot rows.")
    phase_ids = np.argmax(phase_table, axis=1)
    flight_steps = np.flatnonzero(phase_ids == phase_names_value.index("FLIGHT"))
    if flight_steps.size == 0:
        raise ValueError("Jump phase table has no FLIGHT samples.")

    return _ManifestData(
        policy_dt=policy_dt,
        episode_steps=episode_steps,
        joint_names=joint_names,
        default_position=default_position,
        position_target_lower=position_target_lower,
        position_target_upper=position_target_upper,
        joint_position_lower=joint_position_lower,
        joint_position_upper=joint_position_upper,
        jump_stiffness=jump_stiffness,
        jump_damping=jump_damping,
        goal_ranges=goal_ranges,
        flight_freeze_enabled=flight_freeze_enabled,
        idle_end_step=idle_end_step,
        flight_start_step=int(flight_steps[0]),
    )


def _validate_controller_config(config: JumpControllerConfig) -> None:
    positive_fields = (
        "stand_entry_duration_s",
        "goto_start_duration_s",
        "goto_start_timeout_s",
        "goto_start_load_compensation_limit_rad",
        "policy_prepare_pose_tolerance_rad",
        "policy_stand_pose_tolerance_rad",
        "policy_stand_tilt_limit_rad",
        "armed_timeout_s",
        "settle_duration_s",
        "settle_timeout_s",
        "settle_pose_tolerance_rad",
        "settle_joint_velocity_tolerance_rad_s",
        "pose_tolerance_rad",
        "joint_velocity_tolerance_rad_s",
        "foot_contact_threshold_n",
        "prearm_tilt_limit_rad",
        "balance_offset_limit_rad",
        "jump_abort_tilt_limit_rad",
        "latched_abort_settle_tilt_limit_rad",
        "stiffness_slew_per_s",
        "damping_slew_per_s",
    )
    for field_name in positive_fields:
        _positive_float(getattr(config, field_name), field_name)
    if (
        isinstance(config.joint_limit_abort_margin_rad, bool)
        or not isinstance(config.joint_limit_abort_margin_rad, (int, float))
        or not math.isfinite(float(config.joint_limit_abort_margin_rad))
        or config.joint_limit_abort_margin_rad < 0.0
    ):
        raise ValueError("joint_limit_abort_margin_rad must be a finite non-negative angle.")
    if (
        isinstance(config.stand_balance_target_entry_duration_s, bool)
        or not isinstance(config.stand_balance_target_entry_duration_s, (int, float))
        or not math.isfinite(float(config.stand_balance_target_entry_duration_s))
        or config.stand_balance_target_entry_duration_s < 0.0
    ):
        raise ValueError("stand_balance_target_entry_duration_s must be a finite non-negative duration.")
    if (
        isinstance(config.goal_position_z_w, bool)
        or not isinstance(config.goal_position_z_w, (int, float))
        or not math.isfinite(float(config.goal_position_z_w))
    ):
        raise ValueError("goal_position_z_w must be a finite number.")
    if (
        isinstance(config.goto_start_load_compensation_scale, bool)
        or not isinstance(config.goto_start_load_compensation_scale, (int, float))
        or not math.isfinite(float(config.goto_start_load_compensation_scale))
        or not 0.0 <= float(config.goto_start_load_compensation_scale) <= 1.0
    ):
        raise ValueError("goto_start_load_compensation_scale must be a finite number in [0, 1].")
    if (
        isinstance(config.policy_prepare_duration_s, bool)
        or not isinstance(config.policy_prepare_duration_s, (int, float))
        or not math.isfinite(float(config.policy_prepare_duration_s))
        or float(config.policy_prepare_duration_s) < 0.0
    ):
        raise ValueError("policy_prepare_duration_s must be a finite non-negative duration.")
    if not isinstance(config.policy_prepare_retain_balance, bool):
        raise ValueError("policy_prepare_retain_balance must be a boolean.")
    if (
        isinstance(config.policy_stand_retrigger_prepare_duration_s, bool)
        or not isinstance(config.policy_stand_retrigger_prepare_duration_s, (int, float))
        or not math.isfinite(float(config.policy_stand_retrigger_prepare_duration_s))
        or float(config.policy_stand_retrigger_prepare_duration_s) < 0.0
    ):
        raise ValueError("policy_stand_retrigger_prepare_duration_s must be a finite non-negative duration.")
    if not isinstance(config.stand_hold_measured_pose, bool):
        raise ValueError("stand_hold_measured_pose must be a boolean.")
    if not isinstance(config.policy_stand_after_jump, bool):
        raise ValueError("policy_stand_after_jump must be a boolean.")
    if not isinstance(config.policy_stand_direct_retrigger, bool):
        raise ValueError("policy_stand_direct_retrigger must be a boolean.")
    if config.policy_stand_direct_retrigger and not config.policy_stand_after_jump:
        raise ValueError("policy_stand_direct_retrigger requires policy_stand_after_jump.")
    if config.policy_stand_direct_retrigger and config.policy_stand_retrigger_prepare_duration_s > 0.0:
        raise ValueError(
            "policy_stand_direct_retrigger cannot be combined with policy_stand_retrigger_prepare_duration_s."
        )
    if (
        isinstance(config.policy_terminal_return_steps, bool)
        or not isinstance(config.policy_terminal_return_steps, int)
        or config.policy_terminal_return_steps < 0
    ):
        raise ValueError("policy_terminal_return_steps must be a non-negative integer.")
    if (
        isinstance(config.jump_blend_out_duration_s, bool)
        or not isinstance(config.jump_blend_out_duration_s, (int, float))
        or not math.isfinite(float(config.jump_blend_out_duration_s))
        or float(config.jump_blend_out_duration_s) < 0.0
    ):
        raise ValueError("jump_blend_out_duration_s must be a finite non-negative duration.")
    if config.policy_terminal_return_steps > 0 and config.policy_stand_after_jump:
        raise ValueError("policy_terminal_return_steps cannot be combined with policy_stand_after_jump.")
    for field_name in (
        "jump_target_blend_steps",
        "jump_gain_blend_steps",
        "jump_balance_blend_steps",
    ):
        value = getattr(config, field_name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{field_name} must be None or a non-negative integer.")
    if not isinstance(config.contact_safety_mode, JumpControllerConfig.ContactSafetyMode):
        raise ValueError("contact_safety_mode must be a JumpControllerConfig.ContactSafetyMode value.")
    if not isinstance(config.latched_abort_upright_settle, bool):
        raise ValueError("latched_abort_upright_settle must be a boolean.")
    required_goto_duration_s = max(
        config.goto_start_duration_s + config.policy_prepare_duration_s,
        config.policy_stand_retrigger_prepare_duration_s,
    )
    if config.goto_start_timeout_s < required_goto_duration_s:
        raise ValueError("goto_start_timeout_s must cover goto_start_duration_s plus policy_prepare_duration_s.")
    required_settle_duration_s = max(config.settle_duration_s, config.jump_blend_out_duration_s)
    if config.settle_timeout_s < required_settle_duration_s:
        raise ValueError("settle_timeout_s must not be shorter than the configured settle or blend-out duration.")


def _quintic(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))


def _body_tilt(quaternion_wxyz: object) -> float:
    quaternion = _finite_vector(quaternion_wxyz, 4, "imu_quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("imu_quaternion must be non-zero.")
    _, x, y, _ = quaternion / norm
    body_up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(body_up_z, -1.0, 1.0)))


class JumpControllerFSM:
    """Sequence and safeguard one deployed jump controller.

    Args:
        manifest_path: Deployment manifest and table bundle path.
        robot: Robot or simulator backend.
        operator: Source of operator intents.
        policy: NumPy policy callable mapping one observation to one raw action.
        stand_gains: Explicit stand-gain configuration.
        config: Controller timing, interlock, and slew configuration.
        balance_config: Optional balance feedback configuration.
        runtime: Optional preconstructed runtime, primarily for composition and
            testing. The manifest remains the sole source of deployment data.
    """

    _ACTIVE_STATES = {
        JumpControllerState.STAND,
        JumpControllerState.GOTO_START,
        JumpControllerState.ARMED,
        JumpControllerState.JUMP,
        JumpControllerState.SETTLE,
    }

    def __init__(
        self,
        manifest_path: str | Path,
        robot: RobotInterface,
        operator: OperatorInterface,
        policy: Callable[[np.ndarray], np.ndarray],
        *,
        stand_gains: StandGainConfig = StandGainConfig(),
        config: JumpControllerConfig = JumpControllerConfig(),
        balance_config: BalanceControllerConfig | None = None,
        runtime: JumpGoalRuntime | None = None,
    ):
        _validate_controller_config(config)
        self._manifest = _load_manifest(
            manifest_path,
            require_joint_position_limits=config.joint_limit_abort_margin_rad > 0.0,
        )
        if config.policy_terminal_return_steps > self._manifest.episode_steps:
            raise ValueError(
                "policy_terminal_return_steps exceeds the complete policy episode: "
                f"{config.policy_terminal_return_steps} > {self._manifest.episode_steps}."
            )
        for field_name in (
            "jump_target_blend_steps",
            "jump_gain_blend_steps",
            "jump_balance_blend_steps",
        ):
            value = getattr(config, field_name)
            if value is not None and value > self._manifest.idle_end_step:
                raise ValueError(
                    f"{field_name}={value} exceeds the manifest IDLE interval "
                    f"of {self._manifest.idle_end_step} policy steps."
                )
        self._robot = robot
        self._operator = operator
        self._policy = policy
        self._runtime = (
            JumpGoalRuntime(
                manifest_path,
                freeze_during_flight=self._manifest.flight_freeze_enabled,
            )
            if runtime is None
            else runtime
        )
        self._config = config
        self._jump_target_blend_steps = self._resolve_jump_blend_steps(config.jump_target_blend_steps)
        self._jump_gain_blend_steps = self._resolve_jump_blend_steps(config.jump_gain_blend_steps)
        self._jump_balance_blend_steps = self._resolve_jump_blend_steps(config.jump_balance_blend_steps)
        self._jump_stiffness = self._manifest.jump_stiffness.copy()
        self._jump_damping = self._manifest.jump_damping.copy()
        self._stand_stiffness, self._stand_damping = self._build_stand_gains(stand_gains)
        self._balance = BalanceController(self._manifest.joint_names, balance_config)
        self._balance_target_roll = self._balance.config.target_roll
        self._balance_target_pitch = self._balance.config.target_pitch
        self._zeros = np.zeros(len(self._manifest.joint_names), dtype=np.float64)

        self.state = JumpControllerState.PASSIVE
        self.transition_history = [self.state]
        self.last_report: str | None = None
        self.fault_reason: str | None = None
        self.latched_goal: JumpGoal | None = None
        self.abort_latched = False
        self.latched_abort_reason: str | None = None
        self.latched_abort_reasons: set[str] = set()
        self.joint_limit_touches: dict[str, float] = {}
        self.episode_step = 0
        self.phase_clock_history: list[int] = []

        self._state_elapsed_s = 0.0
        self._stand_balance_gate = 0.0
        self._stand_balance_target_start_roll = self._balance_target_roll
        self._stand_balance_target_start_pitch = self._balance_target_pitch
        self._stand_balance_target_entry_active = False
        self._goto_start_balance_gate = 0.0
        self._goto_balance_gate = 0.0
        self._policy_prepare_start_balance_gate = 0.0
        self._jump_start_balance_gate = 0.0
        self._jump_balance_gate = 0.0
        self._settle_balance_gate = 0.0
        self._stand_start_position = self._manifest.default_position.copy()
        self._goto_start_position = self._manifest.default_position.copy()
        self._goto_target_position = self._manifest.default_position.copy()
        self._policy_prepare_start_position = self._manifest.default_position.copy()
        self._policy_prepare_start_stiffness = self._zeros.copy()
        self._policy_prepare_start_damping = self._zeros.copy()
        self._active_policy_prepare_duration_s = config.policy_prepare_duration_s
        self._jump_start_position = self._manifest.default_position.copy()
        self._jump_start_stiffness = self._zeros.copy()
        self._jump_start_damping = self._zeros.copy()
        self._settle_start_position = self._manifest.default_position.copy()
        self._last_target = self._manifest.default_position.copy()
        self._last_stiffness = self._zeros.copy()
        self._last_damping = self._zeros.copy()
        self._previous_request_start = False
        self._previous_confirm = False
        self._airborne_seen = False
        self._damping_after_touchdown = False
        self._policy_prepare_started = False
        self._policy_prepared = False
        self._policy_prepare_start_elapsed_s = 0.0
        self._policy_stand_active = False
        self._policy_stand_retrigger = False
        self._direct_policy_stand_jump = False
        self._warnings: list[str] = []

    def drain_warnings(self) -> tuple[str, ...]:
        """Return and clear non-terminal safety warnings accumulated by the FSM."""

        warnings = tuple(self._warnings)
        self._warnings.clear()
        return warnings

    @property
    def policy_dt(self) -> float:
        """Controller period [s] from the deployment manifest."""

        return self._manifest.policy_dt

    @property
    def episode_steps(self) -> int:
        """Number of policy steps in one complete jump."""

        return self._manifest.episode_steps

    @property
    def flight_start_step(self) -> int:
        """First FLIGHT policy step, derived from the manifest phase table."""

        return self._manifest.flight_start_step

    @property
    def stand_stiffness(self) -> np.ndarray:
        """Configured stand position gains [N·m/rad], in manifest order."""

        return self._stand_stiffness.copy()

    @property
    def stand_damping(self) -> np.ndarray:
        """Configured stand velocity gains [N·m·s/rad], in manifest order."""

        return self._stand_damping.copy()

    @property
    def jump_stiffness(self) -> np.ndarray:
        """Manifest jump position gains [N·m/rad], in manifest order."""

        return self._jump_stiffness.copy()

    @property
    def jump_damping(self) -> np.ndarray:
        """Manifest jump velocity gains [N·m·s/rad], in manifest order."""

        return self._jump_damping.copy()

    @property
    def policy_prepared(self) -> bool:
        """Whether a goal-conditioned frozen-phase policy handoff is ready."""

        return self._policy_prepared

    @property
    def policy_stand_active(self) -> bool:
        """Whether STAND is held by the policy's final reference row."""

        return self._policy_stand_active

    @property
    def balance_integral_error(self) -> np.ndarray:
        """Balance-controller roll-pitch integral error [rad·s]."""

        return self._balance.integral_error

    @property
    def balance_ankle_offset(self) -> np.ndarray:
        """Most recently computed pitch-roll ankle target offset [rad]."""

        return self._balance.last_ankle_offset.copy()

    @property
    def balance_gate(self) -> float:
        """Fraction of the fast-loop balance correction currently enabled."""

        if self.state is JumpControllerState.STAND:
            return 0.0 if self._policy_stand_active else self._stand_balance_gate
        if self.state is JumpControllerState.GOTO_START:
            return self._goto_balance_gate
        if self.state is JumpControllerState.ARMED:
            if self._policy_prepared and self._config.policy_prepare_retain_balance:
                return self._goto_balance_gate
            return 0.0 if self._policy_prepared or self._policy_stand_retrigger else 1.0
        if self.state is JumpControllerState.JUMP:
            return self._jump_balance_gate
        if self.state is JumpControllerState.SETTLE:
            return self._settle_balance_gate
        return 0.0

    def update_balance(self, dt: float) -> np.ndarray:
        """Advance balance feedback and return its gated joint-target correction.

        This method is intended for the fast actuator loop. The FSM's
        :meth:`step` method updates the held base target and balance gate at the
        policy rate, while this method evaluates IMU feedback at the supplied
        fast-loop period.

        Args:
            dt: Elapsed fast-loop time [s].

        Returns:
            Gated joint-position correction in manifest order [rad]. This is
            exactly zero in PASSIVE, DAMPING, policy-prepared ARMED, and JUMP
            after its manifest IDLE handoff has completed.
        """
        fast_dt = _positive_float(dt, "dt")
        if self._stand_balance_target_entry_active and self.state is JumpControllerState.STAND:
            duration_s = self._config.stand_balance_target_entry_duration_s
            progress = min(self._state_elapsed_s / duration_s, 1.0)
            target_roll = self._stand_balance_target_start_roll + progress * (
                self._balance_target_roll - self._stand_balance_target_start_roll
            )
            target_pitch = self._stand_balance_target_start_pitch + progress * (
                self._balance_target_pitch - self._stand_balance_target_start_pitch
            )
            self._balance.update_target_attitude(target_roll, target_pitch)
            self._stand_balance_target_entry_active = progress < 1.0
        gate = self.balance_gate
        if gate == 0.0:
            return self._zeros.copy()
        base_target = self._last_target.copy()
        balanced_target = self._balance.compute(
            base_target,
            self._robot.imu_quaternion,
            self._robot.base_angular_velocity,
            self._joint_positions_or_default(),
            self._joint_velocities_or_zero(),
            fast_dt,
        )
        return gate * (balanced_target - base_target)

    def enable(self) -> None:
        """Enable standing from :attr:`~JumpControllerState.PASSIVE`."""

        if self.state is JumpControllerState.PASSIVE:
            self.last_report = "Standing enabled."
            self._transition(JumpControllerState.STAND)

    def set_balance_target_attitude(self, target_roll: float, target_pitch: float) -> None:
        """Calibrate the stand balance target while control is passive.

        Args:
            target_roll: Target pelvis roll attitude [rad].
            target_pitch: Target pelvis pitch attitude [rad].

        Raises:
            RuntimeError: If the FSM is already active.
            ValueError: If either target is non-finite.
        """
        if self.state is not JumpControllerState.PASSIVE:
            raise RuntimeError("Balance target calibration requires PASSIVE FSM state.")
        self._balance.set_target_attitude(target_roll, target_pitch)
        self._balance_target_roll = float(target_roll)
        self._balance_target_pitch = float(target_pitch)

    def reset(self) -> None:
        """Reset :attr:`~JumpControllerState.DAMPING` to passive control."""

        if self.state is JumpControllerState.DAMPING:
            self.fault_reason = None
            self.abort_latched = False
            self.latched_abort_reason = None
            self.latched_abort_reasons.clear()
            self.last_report = "Controller reset to passive."
            self._transition(JumpControllerState.PASSIVE)

    def report_fault(self, reason: str) -> None:
        """Record an external fault and immediately command damping.

        Args:
            reason: Human-readable fault reason.
        """

        self._enter_fault(reason)
        self._command_damping()

    def step(self) -> None:  # noqa: C901
        """Run one controller period and issue exactly one normal-state command."""

        request_start = bool(self._operator.request_start)
        confirm = bool(self._operator.confirm)
        abort = bool(self._operator.abort)
        request_start_rising = request_start and not self._previous_request_start
        confirm_rising = confirm and not self._previous_confirm
        state_before = self.state
        try:
            if abort and self.state in self._ACTIVE_STATES:
                if self.state is JumpControllerState.JUMP:
                    self._request_jump_abort("operator abort")
                else:
                    self.last_report = "Abort requested; damping enabled."
                    self._transition(JumpControllerState.DAMPING)

            if self.state is JumpControllerState.PASSIVE:
                self._command_passive()
            elif self.state is JumpControllerState.DAMPING:
                self._command_damping()
            elif self.state is JumpControllerState.STAND:
                self._command_stand()
                if request_start_rising:
                    self._request_start()
            elif self.state is JumpControllerState.GOTO_START:
                self._step_goto_start()
            elif self.state is JumpControllerState.ARMED:
                if confirm_rising:
                    self._confirm_jump()
                if self.state is JumpControllerState.JUMP:
                    self._check_in_jump_abort_conditions()
                    if self.state is JumpControllerState.JUMP:
                        self._step_jump()
                    else:
                        self._command_damping()
                elif self.state is JumpControllerState.ARMED:
                    self._step_armed()
                else:
                    self._command_stand()
                if self.state is JumpControllerState.ARMED and (
                    self._state_elapsed_s + self.policy_dt >= self._config.armed_timeout_s
                ):
                    self.latched_goal = None
                    if self._policy_stand_retrigger:
                        self.last_report = "Policy-stand re-arm timed out before confirmation; damping enabled."
                        self._transition(JumpControllerState.DAMPING)
                    else:
                        self.last_report = "Arming timed out before confirmation."
                        self._transition(JumpControllerState.STAND)
            elif self.state is JumpControllerState.JUMP:
                self._check_in_jump_abort_conditions()
                if self.state is JumpControllerState.JUMP:
                    self._step_jump()
                else:
                    self._command_damping()
            elif self.state is JumpControllerState.SETTLE:
                self._step_settle()
        except Exception as exc:  # A control-path exception is itself a safety fault.
            self._enter_fault(f"{type(exc).__name__}: {exc}")
            self._command_damping()
        finally:
            if self.state is state_before:
                self._state_elapsed_s += self.policy_dt
            self._previous_request_start = request_start
            self._previous_confirm = confirm

    def _build_stand_gains(self, config: StandGainConfig) -> tuple[np.ndarray, np.ndarray]:
        stiffness = self._jump_stiffness.copy()
        damping = self._jump_damping.copy()
        ankle_indices = [index for index, name in enumerate(self._manifest.joint_names) if "ankle" in name]
        if not ankle_indices:
            raise ValueError("Manifest joint names do not identify any ankle joints.")
        if config.ankle_stiffness is not None:
            ankle_stiffness = _positive_float(config.ankle_stiffness, "stand_gains.ankle_stiffness")
            stiffness[ankle_indices] = ankle_stiffness
        if config.ankle_damping is not None:
            ankle_damping = _positive_float(config.ankle_damping, "stand_gains.ankle_damping")
            damping[ankle_indices] = ankle_damping
        self._apply_gain_overrides(stiffness, config.stiffness_overrides, "stiffness")
        self._apply_gain_overrides(damping, config.damping_overrides, "damping")
        return stiffness, damping

    def _resolve_jump_blend_steps(self, configured_steps: int | None) -> int:
        """Resolve a jump handoff duration against the manifest IDLE interval."""

        return self._manifest.idle_end_step if configured_steps is None else configured_steps

    @staticmethod
    def _jump_blend(clock: int, steps: int) -> float:
        """Return a quintic handoff fraction for one policy clock sample."""

        if steps == 0:
            return 1.0
        return _quintic(min((clock + 1) / steps, 1.0))

    def _apply_gain_overrides(
        self,
        values: np.ndarray,
        overrides: Mapping[str, float] | None,
        gain_name: str,
    ) -> None:
        if overrides is None:
            return
        indices = {name: index for index, name in enumerate(self._manifest.joint_names)}
        unknown = sorted(set(overrides) - set(indices))
        if unknown:
            raise ValueError(f"Unknown stand {gain_name} override joints: {unknown}.")
        for name, value in overrides.items():
            values[indices[name]] = _positive_float(value, f"stand_gains.{gain_name}_overrides[{name!r}]")

    def _request_start(self) -> None:
        goal = self._operator.pending_goal
        if goal is None:
            self.last_report = "Goal rejected: no pending goal."
            return
        if not isinstance(goal, JumpGoal):
            self.last_report = "Goal rejected: pending_goal must be a JumpGoal."
            return
        retrigger_from_policy_stand = self._policy_stand_active
        direct_policy_stand_retrigger = retrigger_from_policy_stand and self._config.policy_stand_direct_retrigger
        try:
            self._validate_goal(goal)
            measured_balance_target = (
                quaternion_to_roll_pitch(self._robot.imu_quaternion) if retrigger_from_policy_stand else None
            )
            if not direct_policy_stand_retrigger:
                self._runtime.arm(goal.dx, goal.dy, goal.dyaw, roll=goal.roll, pitch=goal.pitch)
        except (TypeError, ValueError, RuntimeError) as exc:
            self.last_report = f"Goal rejected: {exc}"
            return
        self._policy_stand_active = False
        self._policy_stand_retrigger = retrigger_from_policy_stand
        if measured_balance_target is not None:
            self._balance.set_target_attitude(*measured_balance_target)
        self.latched_goal = goal
        self.last_report = f"Goal latched: {goal}."
        self._goto_start_position = self._last_target.copy()
        if direct_policy_stand_retrigger:
            self._goto_target_position = self._last_target.copy()
            self._goto_start_balance_gate = 0.0
            self._goto_balance_gate = 0.0
            self._policy_prepare_started = False
            self._policy_prepared = False
            self._policy_prepare_start_elapsed_s = 0.0
            self._transition(JumpControllerState.GOTO_START)
            return
        if retrigger_from_policy_stand and self._config.policy_stand_retrigger_prepare_duration_s > 0.0:
            self._goto_target_position = self._last_target.copy()
            self._goto_start_balance_gate = 0.0
            self._goto_balance_gate = 0.0
            self._policy_prepare_started = False
            self._policy_prepared = False
            self._policy_prepare_start_elapsed_s = 0.0
            self._transition(JumpControllerState.GOTO_START)
            return
        measured_position = self._joint_positions_or_default()
        load_compensation = self._config.goto_start_load_compensation_scale * (self._last_target - measured_position)
        ankle_mask = np.asarray(["ankle" in name for name in self._manifest.joint_names], dtype=np.bool_)
        load_compensation[ankle_mask] = 0.0
        compensation_limit = self._config.goto_start_load_compensation_limit_rad
        load_compensation = np.clip(load_compensation, -compensation_limit, compensation_limit)
        self._goto_target_position = np.clip(
            self._manifest.default_position + load_compensation,
            self._manifest.position_target_lower,
            self._manifest.position_target_upper,
        )
        self._goto_start_balance_gate = self._stand_balance_gate
        self._goto_balance_gate = self._goto_start_balance_gate
        self._policy_prepare_started = False
        self._policy_prepared = False
        self._policy_prepare_start_elapsed_s = 0.0
        self._transition(JumpControllerState.GOTO_START)

    def _validate_goal(self, goal: JumpGoal) -> None:
        for name, value in (
            ("pos_x", goal.dx),
            ("pos_y", goal.dy),
            ("roll", goal.roll),
            ("pitch", goal.pitch),
            ("yaw", goal.dyaw),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Goal {name} must be a finite number.")
            lower, upper = self._manifest.goal_ranges[name]
            if not lower <= float(value) <= upper:
                raise ValueError(f"Goal {name}={value} is outside manifest range [{lower}, {upper}].")

    def _step_goto_start(self) -> None:
        if self._policy_stand_retrigger and self._config.policy_stand_direct_retrigger:
            self._step_direct_policy_stand_retrigger()
        elif self._policy_stand_retrigger and self._config.policy_stand_retrigger_prepare_duration_s > 0.0:
            if self._policy_prepare_started:
                self._step_policy_preparation()
            else:
                self._begin_policy_preparation(self._config.policy_stand_retrigger_prepare_duration_s)
        elif self._policy_prepare_started:
            self._step_policy_preparation()
        else:
            self._step_static_goto_start()

        if self.state is not JumpControllerState.GOTO_START:
            return
        if self._state_elapsed_s + self.policy_dt >= self._config.goto_start_timeout_s:
            if self.last_report is None or not (
                self.last_report.startswith("Pre-arm refused:")
                or self.last_report.startswith("Policy preparation not ready:")
                or self.last_report.startswith("Policy-stand re-arm not ready:")
            ):
                self.last_report = "Pre-arm refused: start pose or policy preparation did not converge before timeout."
            self.latched_goal = None
            if self._policy_stand_retrigger:
                self.last_report += "; damping enabled."
                self._transition(JumpControllerState.DAMPING)
            else:
                self._transition(JumpControllerState.STAND)

    def _step_direct_policy_stand_retrigger(self) -> None:
        """Hold the preceding policy's final STAND row while arming a new goal."""
        target = self._infer_policy_target(
            advance=False,
            reference_step=self._runtime.stand_reference_step,
        )
        self._goto_target_position = target.copy()
        self._goto_balance_gate = 0.0
        self._command(target, self._jump_stiffness, self._jump_damping, slew=False)
        self.last_report = f"Direct policy-stand goal armed: {self.latched_goal}."
        self._transition(JumpControllerState.ARMED)

    def _step_static_goto_start(self) -> None:
        """Blend to the static start pose before policy preparation."""
        progress = min((self._state_elapsed_s + self.policy_dt) / self._config.goto_start_duration_s, 1.0)
        blend = _quintic(progress)
        target = self._goto_start_position + blend * (self._goto_target_position - self._goto_start_position)
        self._goto_balance_gate = self._goto_start_balance_gate + blend * (1.0 - self._goto_start_balance_gate)
        self._command(target, self._stand_stiffness, self._stand_damping, slew=False)

        if progress >= 1.0:
            failures = self._prearm_failures()
            if not failures:
                if self._config.policy_prepare_duration_s > 0.0:
                    self._begin_policy_preparation()
                else:
                    self._policy_stand_retrigger = False
                    self.last_report = f"Armed goal: {self.latched_goal}."
                    self._transition(JumpControllerState.ARMED)
                return
            self.last_report = "Pre-arm refused: " + "; ".join(failures)

    def _begin_policy_preparation(self, duration_s: float | None = None) -> None:
        """Initialize a frozen phase-zero runtime for the latched goal."""
        self._runtime.trigger(
            self._robot.odometry_position,
            self._robot.odometry_quaternion,
            self._robot.joint_positions,
            goal_pos_z_w=self._config.goal_position_z_w,
            retrigger=self._policy_stand_retrigger,
        )
        self._policy_prepare_started = True
        self._policy_prepared = False
        self._active_policy_prepare_duration_s = (
            self._config.policy_prepare_duration_s if duration_s is None else duration_s
        )
        self._policy_prepare_start_elapsed_s = self._state_elapsed_s + self.policy_dt
        self._policy_prepare_start_position = self._last_target.copy()
        self._policy_prepare_start_stiffness = self._last_stiffness.copy()
        self._policy_prepare_start_damping = self._last_damping.copy()
        self._policy_prepare_start_balance_gate = self._goto_balance_gate
        self.last_report = f"Preparing policy at frozen phase zero for goal: {self.latched_goal}."

    def _step_policy_preparation(self) -> None:
        """Warm policy history under the goal without advancing its clock."""
        elapsed_s = self._state_elapsed_s + self.policy_dt - self._policy_prepare_start_elapsed_s
        progress = min(elapsed_s / self._active_policy_prepare_duration_s, 1.0)
        blend = _quintic(progress)
        policy_target = self._infer_policy_target(advance=False)
        target = self._policy_prepare_start_position + blend * (policy_target - self._policy_prepare_start_position)
        stiffness = self._policy_prepare_start_stiffness + blend * (
            self._jump_stiffness - self._policy_prepare_start_stiffness
        )
        damping = self._policy_prepare_start_damping + blend * (self._jump_damping - self._policy_prepare_start_damping)
        if self._config.policy_prepare_retain_balance:
            self._goto_balance_gate = self._policy_prepare_start_balance_gate
        else:
            self._goto_balance_gate = self._policy_prepare_start_balance_gate * (1.0 - blend)
        self._command(target, stiffness, damping, slew=False)

        if progress < 1.0:
            return
        failures = self._prearm_failures(prepared_target=target)
        if failures:
            self.last_report = "Policy preparation not ready: " + "; ".join(failures)
            return
        self._goto_target_position = target.copy()
        self._policy_prepared = True
        self.last_report = f"Goal-conditioned policy prepared and armed: {self.latched_goal}."
        self._transition(JumpControllerState.ARMED)

    def _step_armed(self) -> None:
        """Hold either the static or goal-conditioned prepared start state."""
        if self._policy_stand_retrigger and self._config.policy_stand_direct_retrigger:
            target = self._infer_policy_target(
                advance=False,
                reference_step=self._runtime.stand_reference_step,
            )
            self._goto_target_position = target.copy()
            self._command(target, self._jump_stiffness, self._jump_damping, slew=False)
            return
        if self._policy_prepared and self._policy_stand_retrigger:
            self._command(
                self._goto_target_position,
                self._jump_stiffness,
                self._jump_damping,
                slew=False,
            )
            return
        if not self._policy_prepared:
            self._command(
                self._goto_target_position,
                self._stand_stiffness,
                self._stand_damping,
                slew=False,
            )
            return
        target = self._infer_policy_target(advance=False)
        self._command(target, self._jump_stiffness, self._jump_damping, slew=False)

    def _prearm_failures(self, *, prepared_target: np.ndarray | None = None) -> list[str]:
        failures = []
        joint_count = len(self._manifest.joint_names)
        velocities = None
        try:
            positions = _finite_vector(self._robot.joint_positions, joint_count, "joint_positions")
            velocities = _finite_vector(self._robot.joint_velocities, joint_count, "joint_velocities")
        except ValueError as exc:
            failures.append(f"joint_reporting: {exc}")
            positions = None
        if prepared_target is None:
            pose_indices = np.asarray(
                [index for index, name in enumerate(self._manifest.joint_names) if "ankle" not in name],
                dtype=np.int32,
            )
            if positions is not None and np.any(
                np.abs(positions[pose_indices] - self._manifest.default_position[pose_indices])
                > self._config.pose_tolerance_rad
            ):
                failures.append("pose: a joint is outside the start-pose tolerance")
        else:
            target = _finite_vector(prepared_target, joint_count, "prepared_target")
            if positions is not None and np.any(
                np.abs(positions - target) > self._config.policy_prepare_pose_tolerance_rad
            ):
                failures.append("pose: a joint is outside the prepared-policy tracking tolerance")
        if velocities is not None and np.any(np.abs(velocities) > self._config.joint_velocity_tolerance_rad_s):
            failures.append("joint_velocity: a joint is moving too quickly to arm")

        try:
            limit_flags = np.asarray(self._robot.joint_limit_violations, dtype=np.bool_)
            if limit_flags.shape != (joint_count,):
                raise ValueError(f"expected shape ({joint_count},), got {limit_flags.shape}")
            if np.any(limit_flags):
                failures.append("joint_limits: at least one joint is at a limit")
        except (TypeError, ValueError) as exc:
            failures.append(f"joint_reporting: invalid joint-limit flags ({exc})")

        if self._config.contact_safety_mode is JumpControllerConfig.ContactSafetyMode.MEASURED:
            try:
                contacts = _finite_vector(self._robot.foot_contact_forces, 2, "foot_contact_forces")
                if np.any(contacts <= self._config.foot_contact_threshold_n):
                    failures.append("feet_loaded: both feet must exceed the contact-force threshold")
            except ValueError as exc:
                failures.append(f"feet_loaded: {exc}")
        if prepared_target is None:
            attitude_error_magnitude = float(np.linalg.norm(self._balance.last_attitude_error))
        else:
            try:
                measured_roll, measured_pitch = quaternion_to_roll_pitch(self._robot.imu_quaternion)
                roll_error = math.remainder(measured_roll - self._balance.config.target_roll, 2.0 * math.pi)
                pitch_error = math.remainder(measured_pitch - self._balance.config.target_pitch, 2.0 * math.pi)
                attitude_error_magnitude = math.hypot(roll_error, pitch_error)
            except ValueError as exc:
                failures.append(f"body_tilt: {exc}")
                attitude_error_magnitude = math.inf
        if attitude_error_magnitude >= self._config.prearm_tilt_limit_rad:
            failures.append("body_tilt: IMU attitude deviation exceeds the target-relative pre-arm limit")
        if prepared_target is None:
            balance_offset_magnitude = float(np.linalg.norm(self._balance.last_ankle_offset))
            if balance_offset_magnitude > self._config.balance_offset_limit_rad:
                failures.append(
                    "balance_offset: ankle offset magnitude "
                    f"{balance_offset_magnitude:.12f} rad exceeds the "
                    f"{self._config.balance_offset_limit_rad:.12f} rad limit"
                )
        if self.latched_goal is None:
            failures.append("goal: no goal is latched")
        else:
            try:
                self._validate_goal(self.latched_goal)
            except ValueError as exc:
                failures.append(f"goal: {exc}")
        return failures

    def _confirm_jump(self) -> None:
        direct_policy_stand_retrigger = self._policy_stand_retrigger and self._config.policy_stand_direct_retrigger
        failures = self._prearm_failures(
            prepared_target=(self._last_target if self._policy_prepared or direct_policy_stand_retrigger else None)
        )
        if failures:
            self.last_report = "Confirmation refused: " + "; ".join(failures)
            self.latched_goal = None
            self._transition(JumpControllerState.STAND)
            return
        if direct_policy_stand_retrigger:
            if self.latched_goal is None:
                raise RuntimeError("Direct policy-stand retrigger has no latched goal.")
            self._runtime.arm(
                self.latched_goal.dx,
                self.latched_goal.dy,
                self.latched_goal.dyaw,
                roll=self.latched_goal.roll,
                pitch=self.latched_goal.pitch,
            )
            self._runtime.trigger(
                self._robot.odometry_position,
                self._robot.odometry_quaternion,
                self._robot.joint_positions,
                goal_pos_z_w=self._config.goal_position_z_w,
                retrigger=True,
            )
        elif self._policy_prepared:
            self._runtime.reanchor_goal(
                self._robot.odometry_position,
                self._robot.odometry_quaternion,
                goal_pos_z_w=self._config.goal_position_z_w,
            )
        else:
            self._runtime.trigger(
                self._robot.odometry_position,
                self._robot.odometry_quaternion,
                self._robot.joint_positions,
                goal_pos_z_w=self._config.goal_position_z_w,
            )
        self.episode_step = 0
        self.phase_clock_history.clear()
        self.abort_latched = False
        self.latched_abort_reason = None
        self.latched_abort_reasons.clear()
        self.joint_limit_touches.clear()
        self._warnings.clear()
        self._airborne_seen = False
        self._damping_after_touchdown = False
        self._jump_start_position = self._last_target.copy()
        self._jump_start_stiffness = self._last_stiffness.copy()
        self._jump_start_damping = self._last_damping.copy()
        self._jump_start_balance_gate = (
            0.0 if self._policy_prepared or direct_policy_stand_retrigger else self.balance_gate
        )
        self._jump_balance_gate = self._jump_start_balance_gate
        self._direct_policy_stand_jump = direct_policy_stand_retrigger
        self._policy_stand_retrigger = False
        self.last_report = f"Jump started for goal: {self.latched_goal}."
        self._transition(JumpControllerState.JUMP)

    def _check_in_jump_abort_conditions(self) -> None:
        reasons = []
        try:
            if _body_tilt(self._robot.imu_quaternion) > self._config.jump_abort_tilt_limit_rad:
                reasons.append("body tilt limit exceeded")
        except ValueError as exc:
            reasons.append(f"invalid IMU feedback: {exc}")
        try:
            flags = np.asarray(self._robot.joint_limit_violations, dtype=np.bool_)
            if flags.shape != (len(self._manifest.joint_names),):
                reasons.append("joint-limit feedback has the wrong shape")
            elif self._config.joint_limit_abort_margin_rad == 0.0:
                if np.any(flags):
                    reasons.append("joint limit exceeded")
            else:
                positions = _finite_vector(
                    self._robot.joint_positions,
                    len(self._manifest.joint_names),
                    "joint_positions",
                )
                depths = np.maximum(
                    self._manifest.joint_position_lower - positions,
                    positions - self._manifest.joint_position_upper,
                )
                depths = np.maximum(depths, 0.0)
                margin = self._config.joint_limit_abort_margin_rad
                beyond_margin = depths > margin
                if np.any(beyond_margin):
                    reasons.append("joint limit exceeded")
                touched_within_margin = (flags | (depths > 0.0)) & ~beyond_margin
                for index in np.flatnonzero(touched_within_margin):
                    name = self._manifest.joint_names[int(index)]
                    depth = float(depths[index])
                    previous_depth = self.joint_limit_touches.get(name)
                    self.joint_limit_touches[name] = max(depth, 0.0 if previous_depth is None else previous_depth)
                    if previous_depth is None:
                        self._warnings.append(
                            f"WARNING: joint limit touched within abort margin: {name} depth={depth:.6f} rad "
                            f"(margin={margin:.6f} rad); continuing."
                        )
        except (TypeError, ValueError):
            reasons.append("joint-limit feedback is invalid")
        if bool(self._robot.control_deadline_missed):
            reasons.append("control deadline missed")
        if bool(self._robot.feedback_stale):
            reasons.append("state feedback stale")
        if reasons:
            self._request_jump_abort(reasons)
        self._update_touchdown_abort()

    def _update_touchdown_abort(self) -> None:
        if self._config.contact_safety_mode is not JumpControllerConfig.ContactSafetyMode.MEASURED:
            return
        try:
            contacts = _finite_vector(self._robot.foot_contact_forces, 2, "foot_contact_forces")
        except ValueError:
            return
        if self.episode_step >= self.flight_start_step and np.all(contacts <= self._config.foot_contact_threshold_n):
            self._airborne_seen = True
        elif self.abort_latched and self._airborne_seen and np.all(contacts > self._config.foot_contact_threshold_n):
            self._damping_after_touchdown = True
            self.last_report = "Latched post-takeoff abort reached touchdown; damping while finishing the episode."

    def _request_jump_abort(self, reasons: str | list[str]) -> None:
        reason_values = [reasons] if isinstance(reasons, str) else list(dict.fromkeys(reasons))
        reason_set = set(reason_values)
        reason = ", ".join(reason_values)
        if self._config.contact_safety_mode is JumpControllerConfig.ContactSafetyMode.GANTRY_REHEARSAL:
            self.last_report = f"Gantry rehearsal aborted: {reason}; damping enabled."
            self._transition(JumpControllerState.DAMPING)
            return
        if self.episode_step < self.flight_start_step:
            self.last_report = f"Jump aborted before takeoff: {reason}."
            self._transition(JumpControllerState.DAMPING)
            return
        self.abort_latched = True
        self.latched_abort_reasons.update(reason_set)
        if self.latched_abort_reason is None:
            self.latched_abort_reason = reason
        self.last_report = f"Post-takeoff abort latched until episode end: {reason}."

    def _step_jump(self) -> None:
        clock = self.episode_step
        target = self._infer_policy_target(advance=True)
        if self._damping_after_touchdown:
            self._command_damping()
        else:
            if self._policy_prepared or self._direct_policy_stand_jump:
                stiffness = self._jump_stiffness
                damping = self._jump_damping
                self._jump_balance_gate = 0.0
            else:
                target_blend = self._jump_blend(clock, self._jump_target_blend_steps)
                gain_blend = self._jump_blend(clock, self._jump_gain_blend_steps)
                balance_blend = self._jump_blend(clock, self._jump_balance_blend_steps)
                target = self._jump_start_position + target_blend * (target - self._jump_start_position)
                stiffness = self._jump_start_stiffness + gain_blend * (
                    self._jump_stiffness - self._jump_start_stiffness
                )
                damping = self._jump_start_damping + gain_blend * (self._jump_damping - self._jump_start_damping)
                self._jump_balance_gate = self._jump_start_balance_gate * (1.0 - balance_blend)

            return_steps = self._config.policy_terminal_return_steps
            return_start_step = self.episode_steps - return_steps
            if return_steps > 0 and clock >= return_start_step:
                progress = (clock - return_start_step + 1) / return_steps
                return_blend = _quintic(progress)
                target = target + return_blend * (self._manifest.default_position - target)
                stiffness = stiffness + return_blend * (self._stand_stiffness - stiffness)
                damping = damping + return_blend * (self._stand_damping - damping)
                self._jump_balance_gate += return_blend * (1.0 - self._jump_balance_gate)
            self._command(target, stiffness, damping, slew=False)
        self.phase_clock_history.append(clock)
        self.episode_step += 1
        self._settle_start_position = np.asarray(target, dtype=np.float64).copy()

        if self._runtime.done:
            if self.episode_step != self.episode_steps:
                raise RuntimeError(
                    f"Runtime ended at controller step {self.episode_step}, expected {self.episode_steps}."
                )
            if self.abort_latched:
                reason = self.latched_abort_reason or "unspecified abort"
                joint_limit_only = self.latched_abort_reasons == {"joint limit exceeded"}
                if self._config.latched_abort_upright_settle and joint_limit_only:
                    try:
                        tilt = _body_tilt(self._robot.imu_quaternion)
                    except ValueError as exc:
                        self.last_report = (
                            f"Latched post-takeoff abort ({reason}) could not verify upright settlement: {exc}; "
                            "damping enabled."
                        )
                        self._transition(JumpControllerState.DAMPING)
                        self._command_damping()
                    else:
                        if tilt <= self._config.latched_abort_settle_tilt_limit_rad:
                            self.last_report = (
                                f"Latched post-takeoff abort ({reason}) completed upright at "
                                f"{math.degrees(tilt):.1f} deg; settling."
                            )
                            self._transition(JumpControllerState.SETTLE)
                        else:
                            self.last_report = (
                                f"Latched post-takeoff abort ({reason}) completed at "
                                f"{math.degrees(tilt):.1f} deg, above the "
                                f"{math.degrees(self._config.latched_abort_settle_tilt_limit_rad):.1f} deg "
                                "upright-settle limit; damping enabled."
                            )
                            self._transition(JumpControllerState.DAMPING)
                            self._command_damping()
                else:
                    eligibility = (
                        "upright settlement is disabled"
                        if not self._config.latched_abort_upright_settle
                        else f"abort reason set {sorted(self.latched_abort_reasons)!r} is terminal"
                    )
                    self.last_report = f"Latched post-takeoff abort ({reason}) applied after the complete episode."
                    if self._config.latched_abort_upright_settle:
                        self.last_report = f"Latched post-takeoff abort ({reason}); {eligibility}; damping enabled."
                    self._transition(JumpControllerState.DAMPING)
                    if not self._damping_after_touchdown:
                        self._command_damping()
            else:
                self.last_report = "Jump episode complete; settling."
                self._transition(JumpControllerState.SETTLE)
        elif self.episode_step >= self.episode_steps:
            raise RuntimeError("Runtime did not report done at manifest episode_steps.")

    def _infer_policy_target(
        self,
        *,
        advance: bool,
        reference_step: int | None = None,
    ) -> np.ndarray:
        """Evaluate the policy once and transform its action to a joint target."""
        observation = self._runtime.step(
            self._robot.joint_positions,
            self._robot.joint_velocities,
            self._robot.base_angular_velocity,
            self._robot.imu_quaternion,
            self._robot.odometry_position,
            self._robot.odometry_quaternion,
            advance=advance,
            reference_step=reference_step,
        )
        raw_action = self._policy(observation)
        return self._runtime.transform_action(raw_action)

    def _step_settle(self) -> None:
        if self._config.jump_blend_out_duration_s > 0.0:
            self._step_jump_blend_out()
            return
        if self._config.policy_stand_after_jump:
            self._step_policy_stand_settle()
            return
        progress = min((self._state_elapsed_s + self.policy_dt) / self._config.settle_duration_s, 1.0)
        blend = _quintic(progress)
        target = self._settle_start_position + blend * (self._manifest.default_position - self._settle_start_position)
        stiffness = self._jump_stiffness + blend * (self._stand_stiffness - self._jump_stiffness)
        damping = self._jump_damping + blend * (self._stand_damping - self._jump_damping)
        self._settle_balance_gate = 1.0 if self._config.policy_terminal_return_steps > 0 else blend
        self._command(target, stiffness, damping, slew=True)
        gains_ready = np.allclose(self._last_stiffness, self._stand_stiffness, rtol=0.0, atol=1.0e-12) and np.allclose(
            self._last_damping, self._stand_damping, rtol=0.0, atol=1.0e-12
        )
        if progress >= 1.0 and gains_ready:
            convergence_failures = self._settle_convergence_failures()
            if not convergence_failures:
                self.latched_goal = None
                self.last_report = "Measured joints settled to stand."
                self._transition(JumpControllerState.STAND)
            elif self._state_elapsed_s + self.policy_dt >= self._config.settle_timeout_s:
                self.latched_goal = None
                self.last_report = "Settle timed out: " + "; ".join(convergence_failures) + "; damping enabled."
                self._transition(JumpControllerState.DAMPING)
            else:
                self.last_report = "Waiting for measured stand convergence: " + "; ".join(convergence_failures)

    def _step_jump_blend_out(self) -> None:
        """Blend the held final policy row and jump gains into STAND."""
        duration_s = self._config.jump_blend_out_duration_s
        progress = min((self._state_elapsed_s + self.policy_dt) / duration_s, 1.0)
        blend = _quintic(progress)
        target = self._settle_start_position + blend * (self._manifest.default_position - self._settle_start_position)
        stiffness = self._jump_stiffness + blend * (self._stand_stiffness - self._jump_stiffness)
        damping = self._jump_damping + blend * (self._stand_damping - self._jump_damping)
        self._settle_balance_gate = blend
        self._command(target, stiffness, damping, slew=False)
        gains_ready = np.allclose(self._last_stiffness, self._stand_stiffness, rtol=0.0, atol=1.0e-12) and np.allclose(
            self._last_damping, self._stand_damping, rtol=0.0, atol=1.0e-12
        )
        if progress >= 1.0 and gains_ready:
            convergence_failures = self._settle_convergence_failures()
            if not convergence_failures:
                self.latched_goal = None
                self._policy_stand_active = False
                self.last_report = "Jump blend-out reached measured stand."
                self._transition(JumpControllerState.STAND)
            elif self._state_elapsed_s + self.policy_dt >= self._config.settle_timeout_s:
                self.latched_goal = None
                self.last_report = "Jump blend-out timed out: " + "; ".join(convergence_failures) + "; damping enabled."
                self._transition(JumpControllerState.DAMPING)
            else:
                self.last_report = "Waiting for measured stand convergence after blend-out: " + "; ".join(
                    convergence_failures
                )

    def _step_policy_stand_settle(self) -> None:
        """Continue closed-loop inference on the policy's final STAND row."""
        target = self._infer_policy_target(
            advance=False,
            reference_step=self._runtime.stand_reference_step,
        )
        self._settle_balance_gate = 0.0
        self._command(target, self._jump_stiffness, self._jump_damping, slew=False)
        if self._state_elapsed_s + self.policy_dt < self._config.settle_duration_s:
            return

        failures = self._policy_stand_convergence_failures(target)
        if not failures:
            self.latched_goal = None
            self._policy_stand_active = True
            self.last_report = "Policy-native stand settled."
            self._transition(JumpControllerState.STAND)
        elif self._state_elapsed_s + self.policy_dt >= self._config.settle_timeout_s:
            self.latched_goal = None
            self.last_report = "Policy-native stand timed out: " + "; ".join(failures) + "; damping enabled."
            self._transition(JumpControllerState.DAMPING)
        else:
            self.last_report = "Waiting for policy-native stand convergence: " + "; ".join(failures)

    def _policy_stand_convergence_failures(self, target: np.ndarray) -> list[str]:
        """Return measured-state reasons that policy-native stand is not ready."""
        joint_count = len(self._manifest.joint_names)
        try:
            positions = _finite_vector(self._robot.joint_positions, joint_count, "joint_positions")
            velocities = _finite_vector(self._robot.joint_velocities, joint_count, "joint_velocities")
        except ValueError as exc:
            return [f"joint reporting is invalid ({exc})"]

        target_array = _finite_vector(target, joint_count, "policy stand target")
        pose_indices = np.asarray(
            [index for index, name in enumerate(self._manifest.joint_names) if "ankle" not in name],
            dtype=np.int32,
        )
        errors = np.abs(positions[pose_indices] - target_array[pose_indices])
        maximum_error_offset = int(np.argmax(errors))
        maximum_error_index = int(pose_indices[maximum_error_offset])
        maximum_speed_index = int(np.argmax(np.abs(velocities)))
        failures = []
        if errors[maximum_error_offset] > self._config.policy_stand_pose_tolerance_rad:
            failures.append(
                f"{self._manifest.joint_names[maximum_error_index]} tracking error "
                f"{errors[maximum_error_offset]:.3f} rad exceeds "
                f"{self._config.policy_stand_pose_tolerance_rad:.3f} rad"
            )
        if abs(velocities[maximum_speed_index]) > self._config.settle_joint_velocity_tolerance_rad_s:
            failures.append(
                f"{self._manifest.joint_names[maximum_speed_index]} speed "
                f"{abs(velocities[maximum_speed_index]):.3f} rad/s exceeds "
                f"{self._config.settle_joint_velocity_tolerance_rad_s:.3f} rad/s"
            )
        if self._config.joint_limit_abort_margin_rad > 0.0:
            projected_positions = positions + self.policy_dt * velocities
            projected_limit_crossing = (projected_positions <= self._manifest.joint_position_lower) | (
                projected_positions >= self._manifest.joint_position_upper
            )
            if np.any(projected_limit_crossing):
                index = int(np.flatnonzero(projected_limit_crossing)[0])
                failures.append(
                    f"{self._manifest.joint_names[index]} velocity projects across its manifest limit "
                    f"within {self.policy_dt:.3f} s"
                )
        try:
            limit_flags = np.asarray(self._robot.joint_limit_violations, dtype=np.bool_)
            if limit_flags.shape != (joint_count,):
                raise ValueError(f"expected shape ({joint_count},), got {limit_flags.shape}")
            if np.any(limit_flags):
                failures.append("at least one joint is at a limit")
        except (TypeError, ValueError) as exc:
            failures.append(f"joint-limit reporting is invalid ({exc})")
        try:
            if _body_tilt(self._robot.imu_quaternion) > self._config.policy_stand_tilt_limit_rad:
                failures.append("body tilt exceeds the policy-stand limit")
        except ValueError as exc:
            failures.append(f"IMU reporting is invalid ({exc})")
        return failures

    def _settle_convergence_failures(self) -> list[str]:
        """Return measured-state reasons that settlement cannot complete."""
        joint_count = len(self._manifest.joint_names)
        try:
            positions = _finite_vector(self._robot.joint_positions, joint_count, "joint_positions")
            velocities = _finite_vector(self._robot.joint_velocities, joint_count, "joint_velocities")
        except ValueError as exc:
            return [f"joint reporting is invalid ({exc})"]

        pose_indices = np.asarray(
            [index for index, name in enumerate(self._manifest.joint_names) if "ankle" not in name],
            dtype=np.int32,
        )
        pose_errors = np.abs(positions[pose_indices] - self._manifest.default_position[pose_indices])
        worst_pose_offset = int(np.argmax(pose_errors))
        worst_pose_index = int(pose_indices[worst_pose_offset])
        maximum_pose_error = float(pose_errors[worst_pose_offset])
        maximum_speed_index = int(np.argmax(np.abs(velocities)))
        maximum_speed = float(abs(velocities[maximum_speed_index]))

        failures = []
        if maximum_pose_error > self._config.settle_pose_tolerance_rad:
            failures.append(
                f"{self._manifest.joint_names[worst_pose_index]} pose error {maximum_pose_error:.3f} rad exceeds "
                f"{self._config.settle_pose_tolerance_rad:.3f} rad"
            )
        if maximum_speed > self._config.settle_joint_velocity_tolerance_rad_s:
            failures.append(
                f"{self._manifest.joint_names[maximum_speed_index]} speed {maximum_speed:.3f} rad/s exceeds "
                f"{self._config.settle_joint_velocity_tolerance_rad_s:.3f} rad/s"
            )
        return failures

    def _command_stand(self) -> None:
        if self._policy_stand_active:
            target = self._infer_policy_target(
                advance=False,
                reference_step=self._runtime.stand_reference_step,
            )
            self._stand_balance_gate = 0.0
            self._command(target, self._jump_stiffness, self._jump_damping, slew=False)
            return
        progress = min(self._state_elapsed_s / self._config.stand_entry_duration_s, 1.0)
        blend = _quintic(progress)
        if self._config.stand_hold_measured_pose:
            target = self._stand_start_position
        else:
            target = self._stand_start_position + blend * (self._manifest.default_position - self._stand_start_position)
        # Neither the reference nor a latched measured pose is open-loop
        # stable, so balance remains active throughout standing.
        self._stand_balance_gate = 1.0
        self._command(
            target,
            self._stand_stiffness,
            self._stand_damping,
            slew=False,
        )

    def _command_passive(self) -> None:
        self._command(self._joint_positions_or_default(), self._zeros, self._zeros, slew=False)

    def _command_damping(self) -> None:
        self._command(self._joint_positions_or_default(), self._zeros, self._jump_damping, slew=False)

    def _command(
        self,
        target: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
        *,
        slew: bool,
    ) -> None:
        joint_count = len(self._manifest.joint_names)
        target_array = _finite_vector(target, joint_count, "command target")
        stiffness_array = _finite_vector(stiffness, joint_count, "command stiffness")
        damping_array = _finite_vector(damping, joint_count, "command damping")
        if np.any(stiffness_array < 0.0) or np.any(damping_array < 0.0):
            raise ValueError("Command gains must be non-negative.")
        if slew:
            stiffness_limit = self._config.stiffness_slew_per_s * self.policy_dt
            damping_limit = self._config.damping_slew_per_s * self.policy_dt
            stiffness_array = self._last_stiffness + np.clip(
                stiffness_array - self._last_stiffness,
                -stiffness_limit,
                stiffness_limit,
            )
            damping_array = self._last_damping + np.clip(
                damping_array - self._last_damping,
                -damping_limit,
                damping_limit,
            )
        self._robot.command_joint_position_target(
            target_array.copy(),
            stiffness_array.copy(),
            damping_array.copy(),
        )
        self._last_target = target_array.copy()
        self._last_stiffness = stiffness_array.copy()
        self._last_damping = damping_array.copy()

    def _joint_positions_or_default(self) -> np.ndarray:
        try:
            return _finite_vector(
                self._robot.joint_positions,
                len(self._manifest.joint_names),
                "joint_positions",
            )
        except ValueError:
            return self._manifest.default_position.copy()

    def _joint_velocities_or_zero(self) -> np.ndarray:
        try:
            return _finite_vector(
                self._robot.joint_velocities,
                len(self._manifest.joint_names),
                "joint_velocities",
            )
        except ValueError:
            return self._zeros.copy()

    def _enter_fault(self, reason: str) -> None:
        self.fault_reason = reason
        self.last_report = f"Fault: {reason}"
        self._transition(JumpControllerState.FAULT)
        self._transition(JumpControllerState.DAMPING)

    def _transition(self, state: JumpControllerState) -> None:
        if self.state is state:
            return
        previous_state = self.state
        if previous_state is JumpControllerState.JUMP and self._config.policy_terminal_return_steps == 0:
            self._balance.reset()
        self.state = state
        if state is JumpControllerState.SETTLE:
            self._settle_balance_gate = 1.0 if self._config.policy_terminal_return_steps > 0 else 0.0
        if state in (JumpControllerState.PASSIVE, JumpControllerState.DAMPING) or (
            state is JumpControllerState.STAND and not self._policy_stand_active
        ):
            self._runtime.cancel()
            self._policy_prepare_started = False
            self._policy_prepared = False
            self._policy_prepare_start_elapsed_s = 0.0
            self._policy_stand_active = False
            self._policy_stand_retrigger = False
            self._direct_policy_stand_jump = False
        if state is JumpControllerState.STAND:
            self._policy_stand_retrigger = False
            self._balance.reset()
            self._stand_start_position = self._joint_positions_or_default()
            self._stand_balance_gate = 0.0
            if (
                previous_state is JumpControllerState.PASSIVE
                and self._config.stand_balance_target_entry_duration_s > 0.0
            ):
                (
                    self._stand_balance_target_start_roll,
                    self._stand_balance_target_start_pitch,
                ) = quaternion_to_roll_pitch(self._robot.imu_quaternion)
                self._balance.update_target_attitude(
                    self._stand_balance_target_start_roll,
                    self._stand_balance_target_start_pitch,
                )
                self._stand_balance_target_entry_active = True
        self._state_elapsed_s = 0.0
        self.transition_history.append(state)

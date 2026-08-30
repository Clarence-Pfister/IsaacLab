# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run read-only policy preflight, stand control, or a contactless gantry rehearsal.

No mode in this module authorizes a ground jump without measured foot contact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    # Unitree SDK2 currently uses Python 3.10, where pip vendors the same parser.
    from pip._vendor import tomli as tomllib

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.g1_jump_deploy.control.balance import (  # noqa: E402
    BalanceControllerConfig,
    quaternion_to_roll_pitch,
)
from scripts.g1_jump_deploy.fsm import (  # noqa: E402
    JumpControllerConfig,
    JumpControllerFSM,
    JumpControllerState,
    JumpGoal,
    StandGainConfig,
)
from scripts.g1_jump_deploy.runtime import (  # noqa: E402
    JumpGoalRuntime,
    OnnxPolicy,
    project_pd_position_target,
    project_position_target_to_lower_limit,
)

_DEFAULT_MANIFEST = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"
_DEFAULT_VALIDATION_RECORD = _SCRIPT_DIR / "validated_bundle.toml"
_DEFAULT_AUDIO_CUE = _SCRIPT_DIR / "assets" / "jump_mode.wav"
_LOW_STATE_TOPIC = "rt/lowstate"
_USER_COMMAND_TOPIC = "rt/user_lowcmd"
_AUDIO_SAMPLE_RATE = 16_000
_AUDIO_CHANNEL_COUNT = 1
_AUDIO_SAMPLE_WIDTH_BYTES = 2
_AUDIO_CHUNK_DURATION_S = 1.0
_FAST_DT = 0.002
_FEEDBACK_TIMEOUT_S = 0.02
_MAX_CONTROL_GAP_S = 0.02
_MAX_TILT_RAD = math.radians(20.0)
_MAX_BASE_ANGULAR_SPEED_RAD_S = 6.0
_NATIVE_WALKRUN_HANDOFF_MAX_TILT_RAD = math.radians(10.0)
_NATIVE_WALKRUN_MONITOR_MAX_TILT_RAD = math.radians(15.0)
_MAX_JOINT_SPEED_RAD_S = 4.0
_SHADOW_MAX_JOINT_SPEED_RAD_S = 0.5
_NATIVE_WALKRUN_HANDOFF_MAX_SPEED_RAD_S = 0.5
_NATIVE_WALKRUN_MONITOR_MAX_SPEED_RAD_S = 2.5
_MAX_MOTOR_TEMPERATURE_C = 80
_TARGET_RATE_LIMIT_RAD_S = 1.2
_TAKEOVER_DAMPING = 1.5
_STAND_ENTRY_LEG_ERROR_LIMIT_RAD = 0.35
_GANTRY_STAND_ENTRY_LEG_ERROR_LIMIT_RAD = 0.65
_GANTRY_STANDUP_DURATION_S = 4.0
_NATIVE_WALKRUN_RETURN_MIN_DURATION_S = 8.0
_RESTORE_RETRY_COUNT = 5
_PASSIVE_FSM_ID = 1
_NATIVE_WALKRUN_FSM_ID = 801
_NATIVE_STAND_FSM_IDS = frozenset((500, 801))
_PASSIVE_BRIDGE_TIMEOUT_S = 2.0
_NATIVE_WALKRUN_RESTORE_TIMEOUT_S = 2.0
_NATIVE_WALKRUN_MONITOR_DURATION_S = 2.0
_NATIVE_HANDOFF_BLEND_DURATION_S = 4.0
_NATIVE_HANDOFF_SETTLE_DURATION_S = 1.0
_NATIVE_HANDOFF_MAX_POSITION_ERROR_RAD = 0.15
_GANTRY_REHEARSAL_MIN_DURATION_S = 15.0
_GANTRY_REHEARSAL_MAX_EFFORT_SCALE = 0.1
_EXPECTED_SHADOW_STEPS = 152
_EXPECTED_SHADOW_GOALS_X = (-0.1, 0.0, 0.1)
_REMOTE_A_MASK = 0x01
_REMOTE_B_MASK = 0x02
_REMOTE_Y_MASK = 0x08
_REMOTE_L1_MASK = 0x02
_REMOTE_R1_MASK = 0x01
_ACTIVATION_HOLD_S = 2.0
_ACTIVATION_TIMEOUT_S = 60.0
_ACTIVATION_RELEASE_TIMEOUT_S = 5.0
_REHEARSAL_NEUTRAL_HOLD_S = 0.5
_REHEARSAL_STABILIZATION_S = 4.5
_REHEARSAL_ARMED_TIMEOUT_S = 15.0
_OBSERVATION_DIM = 326
_POLICY_BENCHMARK_SAMPLES = 100


class SafetyFault(RuntimeError):
    """Raised when a hardware safety interlock refuses or stops control."""


@dataclass(frozen=True)
class HardwareManifest:
    """Manifest fields required by the G1 hardware boundary.

    Attributes:
        joint_names: Policy joint names in command order.
        sdk_slots: Unitree SDK2 motor slots for each policy joint.
        default_position: Manifest stand joint positions [rad].
        joint_position_lower: Lower physical joint limits [rad].
        joint_position_upper: Upper physical joint limits [rad].
        target_position_lower: Lower joint-target limits [rad].
        target_position_upper: Upper joint-target limits [rad].
        effort_limit: Manifest motor effort limits [N·m].
        velocity_limit: Manifest motor velocity limits [rad/s].
        stiffness: Manifest position gains [N·m/rad].
        damping: Manifest damping gains [N·m·s/rad].
        initial_root_height: Reference pelvis height above the floor [m].
        policy_dt: FSM period [s].
        effort_limit_ratio: Optional manifest PD torque-envelope ratios.
        lower_limit_velocity_lookahead: Optional lower-limit braking lookahead [s].
    """

    joint_names: tuple[str, ...]
    sdk_slots: tuple[int, ...]
    default_position: np.ndarray
    joint_position_lower: np.ndarray
    joint_position_upper: np.ndarray
    target_position_lower: np.ndarray
    target_position_upper: np.ndarray
    effort_limit: np.ndarray
    velocity_limit: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    initial_root_height: float
    policy_dt: float
    effort_limit_ratio: np.ndarray | None = None
    lower_limit_velocity_lookahead: np.ndarray | None = None

    @property
    def joint_count(self) -> int:
        """Number of actuated joints."""
        return len(self.joint_names)


@dataclass(frozen=True)
class FeedbackSnapshot:
    """Validated G1 feedback in manifest order."""

    received_at: float
    tick: int
    mode_pr: int
    mode_machine: int
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torque_estimates: np.ndarray
    imu_quaternion: np.ndarray
    imu_gyroscope: np.ndarray
    wireless_remote: bytes
    maximum_temperature_c: int


@dataclass(frozen=True)
class ShadowPolicyReport:
    """Read-only diagnostics from policy step 0 on live G1 feedback.

    Attributes:
        policy_sha256: SHA-256 digest of the evaluated ONNX file.
        inference_median_ms: Median warm inference latency [ms].
        inference_p99_ms: 99th-percentile warm inference latency [ms].
        inference_maximum_ms: Maximum warm inference latency [ms].
        raw_action_minimum: Minimum raw action value.
        raw_action_maximum: Maximum raw action value.
        maximum_target_delta_rad: Maximum absolute projected target change
            from the measured pose [rad].
        maximum_torque_fraction: Maximum projected PD torque as a fraction of
            the manifest motor effort limit.
        maximum_torque_joint: Joint producing :attr:`maximum_torque_fraction`.
    """

    policy_sha256: str
    inference_median_ms: float
    inference_p99_ms: float
    inference_maximum_ms: float
    raw_action_minimum: float
    raw_action_maximum: float
    maximum_target_delta_rad: float
    maximum_torque_fraction: float
    maximum_torque_joint: str


@dataclass(frozen=True)
class ShadowEpisodeReport:
    """Read-only diagnostics from one complete policy timeline.

    Attributes:
        policy_sha256: SHA-256 digest of the evaluated ONNX file.
        steps: Number of evaluated policy steps.
        elapsed_s: Wall-clock shadow duration [s].
        inference_median_ms: Median warm inference latency [ms].
        inference_p99_ms: 99th-percentile warm inference latency [ms].
        inference_maximum_ms: Maximum warm inference latency [ms].
        raw_action_minimum: Minimum raw action value.
        raw_action_maximum: Maximum raw action value.
        maximum_target_delta_rad: Maximum absolute projected target change
            from the measured pose [rad].
        maximum_torque_fraction: Maximum projected PD torque as a fraction of
            the manifest motor effort limit.
        maximum_torque_joint: Joint producing :attr:`maximum_torque_fraction`.
        maximum_unprojected_torque_fraction: Maximum requested PD torque as a
            fraction of the manifest motor effort limit before projection.
        torque_projection_steps: Number of policy steps on which torque
            projection modified at least one joint target.
        maximum_body_tilt_deg: Maximum measured body tilt [deg].
        maximum_joint_speed_rad_s: Maximum measured joint speed [rad/s].
        maximum_measured_torque_fraction: Maximum measured motor torque as a
            fraction of the manifest motor effort limit.
        maximum_measured_torque_joint: Joint producing
            :attr:`maximum_measured_torque_fraction`.
        log_path: Newly created NPZ diagnostic log.
    """

    policy_sha256: str
    steps: int
    elapsed_s: float
    inference_median_ms: float
    inference_p99_ms: float
    inference_maximum_ms: float
    raw_action_minimum: float
    raw_action_maximum: float
    maximum_target_delta_rad: float
    maximum_torque_fraction: float
    maximum_torque_joint: str
    maximum_unprojected_torque_fraction: float
    torque_projection_steps: int
    maximum_body_tilt_deg: float
    maximum_joint_speed_rad_s: float
    maximum_measured_torque_fraction: float
    maximum_measured_torque_joint: str
    log_path: Path


@dataclass(frozen=True)
class _PublishedCommand:
    """One successfully published command and its source feedback."""

    published_at: float
    feedback: FeedbackSnapshot
    target: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray


class _RehearsalRecorder:
    """Collect and atomically create one contactless-rehearsal audit."""

    def __init__(
        self,
        log_path: Path,
        manifest_path: Path,
        policy_path: Path,
        admission_path: Path,
        manifest: HardwareManifest,
        goal: JumpGoal,
        effort_scale: float,
    ):
        self.log_path = log_path.resolve()
        if self.log_path.exists():
            raise ValueError(f"Rehearsal log already exists and will not be overwritten: {self.log_path}")
        self._manifest_path = manifest_path.resolve()
        self._policy_path = policy_path.resolve()
        self._admission_path = admission_path.resolve()
        self._manifest = manifest
        self._goal = goal
        self._effort_scale = effort_scale
        self._created_unix_time_s = time.time()
        self._published_at: list[float] = []
        self._feedback_received_at: list[float] = []
        self._ticks: list[int] = []
        self._mode_pr: list[int] = []
        self._mode_machine: list[int] = []
        self._temperatures: list[int] = []
        self._joint_positions: list[np.ndarray] = []
        self._joint_velocities: list[np.ndarray] = []
        self._joint_torque_estimates: list[np.ndarray] = []
        self._imu_quaternions: list[np.ndarray] = []
        self._imu_gyroscopes: list[np.ndarray] = []
        self._wireless_remotes: list[np.ndarray] = []
        self._fsm_states: list[str] = []
        self._episode_steps: list[int] = []
        self._command_targets: list[np.ndarray] = []
        self._command_stiffness: list[np.ndarray] = []
        self._command_damping: list[np.ndarray] = []
        self._balance_offsets: list[np.ndarray] = []

    @property
    def sample_count(self) -> int:
        """Number of successful command publications in the audit."""
        return len(self._published_at)

    def record(self, robot: _G1Robot, fsm: JumpControllerFSM, balance_offset: np.ndarray) -> None:
        """Record the last command only after DDS accepted it."""
        command = robot.last_published_command
        if command is None:
            raise RuntimeError("cannot audit before a command has been published")
        if self._published_at and command.published_at <= self._published_at[-1]:
            raise RuntimeError("cannot audit the same or an out-of-order command twice")
        feedback = command.feedback
        state = fsm.state.value if isinstance(fsm.state, JumpControllerState) else str(fsm.state)
        self._published_at.append(command.published_at)
        self._feedback_received_at.append(feedback.received_at)
        self._ticks.append(feedback.tick)
        self._mode_pr.append(feedback.mode_pr)
        self._mode_machine.append(feedback.mode_machine)
        self._temperatures.append(feedback.maximum_temperature_c)
        self._joint_positions.append(feedback.joint_positions.copy())
        self._joint_velocities.append(feedback.joint_velocities.copy())
        self._joint_torque_estimates.append(feedback.joint_torque_estimates.copy())
        self._imu_quaternions.append(feedback.imu_quaternion.copy())
        self._imu_gyroscopes.append(feedback.imu_gyroscope.copy())
        self._wireless_remotes.append(np.frombuffer(feedback.wireless_remote, dtype=np.uint8).copy())
        self._fsm_states.append(state)
        self._episode_steps.append(int(getattr(fsm, "episode_step", 0)))
        self._command_targets.append(command.target.copy())
        self._command_stiffness.append(command.stiffness.copy())
        self._command_damping.append(command.damping.copy())
        self._balance_offsets.append(
            _finite_vector(balance_offset, self._manifest.joint_count, "audit balance offset").copy()
        )

    def write(self, success: bool, reason: str, state_buffer: _StateBuffer) -> None:
        """Create the immutable NPZ audit, including failed attempts."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        count = self.sample_count
        joint_count = self._manifest.joint_count

        def joint_matrix(samples: list[np.ndarray]) -> np.ndarray:
            return np.asarray(samples, dtype=np.float64).reshape(count, joint_count)

        metadata = {
            "schema_version": "1.0",
            "mode": "contactless_gantry_policy_rehearsal",
            "control_result": "pass" if success else "stopped",
            "reason": reason,
            "ground_jump_authorized": False,
            "contact_sensor_available": False,
            "feet_required_ground_clear": True,
            "created_unix_time_s": self._created_unix_time_s,
            "written_unix_time_s": time.time(),
            "manifest_path": str(self._manifest_path),
            "manifest_sha256": _sha256(self._manifest_path),
            "policy_path": str(self._policy_path),
            "policy_sha256": _sha256(self._policy_path),
            "shadow_admission_path": str(self._admission_path),
            "shadow_admission_sha256": _sha256(self._admission_path),
            "joint_names": list(self._manifest.joint_names),
            "goal": {
                "pos_x": self._goal.dx,
                "pos_y": self._goal.dy,
                "yaw": self._goal.dyaw,
                "roll": self._goal.roll,
                "pitch": self._goal.pitch,
            },
            "effort_scale": self._effort_scale,
            "target_rate_limit_rad_s": _TARGET_RATE_LIMIT_RAD_S,
            "policy_dt_s": self._manifest.policy_dt,
            "command_dt_s": _FAST_DT,
            "command_samples": count,
            "feedback_counters": {
                "valid_packets": int(state_buffer.valid_packets),
                "crc_errors": int(state_buffer.crc_errors),
                "invalid_packets": int(state_buffer.invalid_packets),
            },
        }
        published_at = np.asarray(self._published_at, dtype=np.float64)
        feedback_received_at = np.asarray(self._feedback_received_at, dtype=np.float64)
        command_target = joint_matrix(self._command_targets)
        command_stiffness = joint_matrix(self._command_stiffness)
        command_damping = joint_matrix(self._command_damping)
        joint_position = joint_matrix(self._joint_positions)
        joint_velocity = joint_matrix(self._joint_velocities)
        estimated_command_torque = command_stiffness * (command_target - joint_position) - (
            command_damping * joint_velocity
        )
        try:
            with self.log_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    published_at=published_at,
                    feedback_received_at=feedback_received_at,
                    feedback_age_ms=(published_at - feedback_received_at) * 1000.0,
                    tick=np.asarray(self._ticks, dtype=np.int64),
                    mode_pr=np.asarray(self._mode_pr, dtype=np.int32),
                    mode_machine=np.asarray(self._mode_machine, dtype=np.int32),
                    maximum_temperature_c=np.asarray(self._temperatures, dtype=np.int16),
                    joint_position=joint_position,
                    joint_velocity=joint_velocity,
                    joint_torque_estimate=joint_matrix(self._joint_torque_estimates),
                    imu_quaternion_wxyz=np.asarray(self._imu_quaternions, dtype=np.float64).reshape(count, 4),
                    imu_gyroscope=np.asarray(self._imu_gyroscopes, dtype=np.float64).reshape(count, 3),
                    wireless_remote=np.asarray(self._wireless_remotes, dtype=np.uint8).reshape(count, 40),
                    fsm_state=np.asarray(self._fsm_states, dtype="<U16"),
                    episode_step=np.asarray(self._episode_steps, dtype=np.int32),
                    command_target=command_target,
                    command_stiffness=command_stiffness,
                    command_damping=command_damping,
                    estimated_command_torque=estimated_command_torque,
                    balance_offset=joint_matrix(self._balance_offsets),
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
        except OSError as exc:
            raise ValueError(f"Cannot create rehearsal log {self.log_path}: {exc}") from exc


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def _load_hardware_manifest(path: Path) -> HardwareManifest:
    try:
        with path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except OSError as exc:
        raise ValueError(f"Cannot read manifest {path}: {exc}") from exc

    if not isinstance(manifest, dict) or manifest.get("schema_version") not in ("1.5", "1.6", "1.7"):
        raise ValueError("The G1 hardware boundary requires deployment manifest schema 1.5, 1.6, or 1.7")

    joints = manifest.get("joints")
    action = manifest.get("action")
    actuators = manifest.get("actuators")
    control = manifest.get("control")
    reference = manifest.get("reference")
    if not all(isinstance(section, dict) for section in (joints, action, actuators, control, reference)):
        raise ValueError("Manifest must contain joints, action, actuators, control, and reference objects")

    names_value = joints.get("names")
    slots_value = joints.get("unitree_sdk2_slots")
    if not isinstance(names_value, list) or not all(isinstance(name, str) and name for name in names_value):
        raise ValueError("joints.names must contain non-empty strings")
    joint_names = tuple(names_value)
    if len(joint_names) != 23 or len(set(joint_names)) != len(joint_names):
        raise ValueError("This runner requires 23 unique G1 joint names")
    if (
        not isinstance(slots_value, list)
        or len(slots_value) != len(joint_names)
        or any(isinstance(slot, bool) or not isinstance(slot, int) for slot in slots_value)
    ):
        raise ValueError("joints.unitree_sdk2_slots must contain 23 integer slots")
    sdk_slots = tuple(slots_value)
    if len(set(sdk_slots)) != len(sdk_slots) or any(slot < 0 or slot >= 35 for slot in sdk_slots):
        raise ValueError("Unitree SDK2 slots must be unique integers in [0, 35)")

    try:
        joint_position_limits = np.asarray(joints.get("position_limits"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("joints.position_limits must contain 23 finite increasing ranges") from exc
    if (
        joint_position_limits.shape != (len(joint_names), 2)
        or not np.all(np.isfinite(joint_position_limits))
        or np.any(joint_position_limits[:, 0] >= joint_position_limits[:, 1])
    ):
        raise ValueError("joints.position_limits must contain 23 finite increasing ranges")

    clip_value = action.get("clip")
    clip = np.asarray(clip_value, dtype=np.float64)
    if clip.shape != (len(joint_names), 2) or not np.all(np.isfinite(clip)) or np.any(clip[:, 0] >= clip[:, 1]):
        raise ValueError("action.clip must contain one finite increasing range per joint")
    default_position = _finite_vector(joints.get("default_pos"), len(joint_names), "joints.default_pos")
    if np.any(default_position < clip[:, 0]) or np.any(default_position > clip[:, 1]):
        raise ValueError("joints.default_pos must remain inside action.clip")
    if np.any(default_position < joint_position_limits[:, 0]) or np.any(default_position > joint_position_limits[:, 1]):
        raise ValueError("joints.default_pos must remain inside joints.position_limits")
    if np.any(clip[:, 0] < joint_position_limits[:, 0]) or np.any(clip[:, 1] > joint_position_limits[:, 1]):
        raise ValueError("action.clip must remain inside joints.position_limits")
    effort_limit = _finite_vector(actuators.get("effort_limit"), len(joint_names), "actuators.effort_limit")
    velocity_limit = _finite_vector(actuators.get("velocity_limit"), len(joint_names), "actuators.velocity_limit")
    stiffness = _finite_vector(actuators.get("stiffness"), len(joint_names), "actuators.stiffness")
    damping = _finite_vector(actuators.get("damping"), len(joint_names), "actuators.damping")
    if (
        np.any(effort_limit <= 0.0)
        or np.any(velocity_limit <= 0.0)
        or np.any(stiffness <= 0.0)
        or np.any(damping < 0.0)
    ):
        raise ValueError("Actuator effort/velocity limits and stiffness must be positive; damping must be non-negative")
    torque_projection = action.get("torque_projection")
    if torque_projection is None:
        raise ValueError("Schema 1.5--1.7 hardware deployment requires action.torque_projection")
    if isinstance(torque_projection, dict) and torque_projection.get("type") == "instantaneous_pd":
        projection_period_s = torque_projection.get("period_s")
        if (
            isinstance(projection_period_s, bool)
            or not isinstance(projection_period_s, (int, float))
            or not math.isclose(float(projection_period_s), _FAST_DT, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise ValueError("Torque projection must run at control.sim_dt")
        effort_limit_ratio = _finite_vector(
            torque_projection.get("effort_limit_ratio"),
            len(joint_names),
            "action.torque_projection.effort_limit_ratio",
        )
        if np.any(effort_limit_ratio <= 0.0) or np.any(effort_limit_ratio > 1.0):
            raise ValueError("Torque-projection effort-limit ratios must be in (0, 1]")
        expected_projection_formula = (
            "q_target = q + (clip(kp*(q_requested-q)-kd*dq, -ratio*effort_limit, ratio*effort_limit)+kd*dq)/kp"
        )
        if torque_projection.get("formula") != expected_projection_formula:
            raise ValueError("Torque-projection formula is unsupported")
    else:
        raise ValueError("action.torque_projection.type must be instantaneous_pd")
    lower_limit_brake = action.get("lower_limit_brake")
    if lower_limit_brake is None:
        raise ValueError("Schema 1.5--1.7 hardware deployment requires action.lower_limit_brake")
    if isinstance(lower_limit_brake, dict) and lower_limit_brake.get("type") == "velocity_lookahead":
        brake_period_s = lower_limit_brake.get("period_s")
        if (
            isinstance(brake_period_s, bool)
            or not isinstance(brake_period_s, (int, float))
            or not math.isclose(float(brake_period_s), _FAST_DT, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise ValueError("Lower-limit braking must run at control.sim_dt")
        brake_lower = _finite_vector(
            lower_limit_brake.get("position_lower"),
            len(joint_names),
            "action.lower_limit_brake.position_lower",
        )
        brake_upper = _finite_vector(
            lower_limit_brake.get("position_upper"),
            len(joint_names),
            "action.lower_limit_brake.position_upper",
        )
        lower_limit_velocity_lookahead = _finite_vector(
            lower_limit_brake.get("velocity_lookahead_s"),
            len(joint_names),
            "action.lower_limit_brake.velocity_lookahead_s",
        )
        if not np.array_equal(brake_lower, clip[:, 0]) or not np.array_equal(brake_upper, clip[:, 1]):
            raise ValueError("Lower-limit brake position bounds must exactly equal action.clip")
        if np.any(lower_limit_velocity_lookahead < 0.0) or not np.any(lower_limit_velocity_lookahead > 0.0):
            raise ValueError("Lower-limit brake lookahead must be non-negative and active")
        expected_brake_formula = "q_requested = max(q_filtered, min(q_upper, q_lower + t_lookahead*max(-dq, 0)))"
        if lower_limit_brake.get("formula") != expected_brake_formula:
            raise ValueError("Lower-limit brake formula is unsupported")
    else:
        raise ValueError("action.lower_limit_brake.type must be velocity_lookahead")
    root_frame0 = reference.get("root_frame0")
    if not isinstance(root_frame0, dict):
        raise ValueError("reference.root_frame0 must be an object")
    initial_root_position = _finite_vector(root_frame0.get("pos"), 3, "reference.root_frame0.pos")
    initial_root_height = float(initial_root_position[2])
    if initial_root_height <= 0.0:
        raise ValueError("reference.root_frame0.pos[2] must be positive")
    policy_dt = control.get("policy_dt")
    if (
        isinstance(policy_dt, bool)
        or not isinstance(policy_dt, (int, float))
        or not math.isclose(float(policy_dt), 0.02, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError("The G1 hardware FSM requires control.policy_dt=0.02 s")
    sim_dt = control.get("sim_dt")
    if (
        isinstance(sim_dt, bool)
        or not isinstance(sim_dt, (int, float))
        or not math.isclose(float(sim_dt), _FAST_DT, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError("The G1 hardware FSM requires control.sim_dt=0.002 s")

    return HardwareManifest(
        joint_names=joint_names,
        sdk_slots=sdk_slots,
        default_position=default_position,
        joint_position_lower=joint_position_limits[:, 0].copy(),
        joint_position_upper=joint_position_limits[:, 1].copy(),
        target_position_lower=clip[:, 0].copy(),
        target_position_upper=clip[:, 1].copy(),
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        stiffness=stiffness,
        damping=damping,
        initial_root_height=initial_root_height,
        policy_dt=float(policy_dt),
        effort_limit_ratio=effort_limit_ratio,
        lower_limit_velocity_lookahead=lower_limit_velocity_lookahead,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Cannot read artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _verify_validated_bundle(
    manifest_path: Path,
    validation_record_path: Path,
    policy_path: Path | None,
) -> None:
    """Verify hardware artifacts against the accepted sim2sim record.

    Args:
        manifest_path: Deployment manifest path.
        validation_record_path: Accepted artifact-digest record.
        policy_path: ONNX policy path, or ``None`` when policy inference is not
            requested.

    Raises:
        ValueError: If the record is malformed, an artifact is missing, or an
            artifact digest differs from the accepted bundle.
    """
    try:
        with validation_record_path.open("rb") as stream:
            record = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Cannot read validation record {validation_record_path}: {exc}") from exc
    if not isinstance(record, dict) or record.get("schema_version") != "1.0":
        raise ValueError("Hardware validation record must use schema 1.0")
    raw_artifacts = record.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise ValueError("Hardware validation record must contain an artifacts table")

    expected_digests: dict[str, str] = {}
    for filename, raw_artifact in raw_artifacts.items():
        if not isinstance(filename, str) or Path(filename).name != filename or not isinstance(raw_artifact, dict):
            raise ValueError("Validation record artifact names must be plain filenames")
        digest = raw_artifact.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Validation record artifact {filename!r} has an invalid SHA-256 digest")
        expected_digests[filename] = digest

    def verify(path: Path, record_filename: str) -> None:
        expected = expected_digests.get(record_filename)
        if expected is None:
            raise ValueError(f"Validation record has no digest for {record_filename}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Hardware artifact {path} does not match the accepted {record_filename}: "
                f"expected {expected}, got {actual}"
            )

    verify(manifest_path, "deploy_manifest.json")
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read deployment manifest {manifest_path}: {exc}") from exc
    tables = manifest.get("tables") if isinstance(manifest, dict) else None
    if not isinstance(tables, dict):
        raise ValueError("Deployment manifest must contain a tables object")
    for table_field in ("reference_preview", "jump_phase"):
        filename = tables.get(table_field)
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Manifest tables.{table_field} must be a plain filename")
        verify(manifest_path.with_name(filename), filename)
    if policy_path is not None:
        verify(policy_path, "policy.onnx")


def _verify_shadow_admission(
    admission_path: Path,
    manifest_path: Path,
    policy_path: Path,
    manifest: HardwareManifest,
) -> None:
    """Verify the exact read-only hardware-shadow evidence and its logs."""
    try:
        with admission_path.open(encoding="utf-8") as stream:
            admission = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read shadow admission {admission_path}: {exc}") from exc
    if not isinstance(admission, dict) or admission.get("schema_version") != "1.0":
        raise ValueError("Shadow admission must use schema 1.0")
    if admission.get("read_only_shadow_admission") is not True:
        raise ValueError("Shadow admission does not attest to successful read-only replay")
    if admission.get("authorizes_motor_control") is not False:
        raise ValueError("Shadow admission must explicitly deny motor-control authorization")
    if admission.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Shadow admission manifest digest differs from the accepted manifest")
    if admission.get("policy_sha256") != _sha256(policy_path):
        raise ValueError("Shadow admission policy digest differs from the accepted policy")
    raw_logs = admission.get("logs")
    if not isinstance(raw_logs, list) or len(raw_logs) != len(_EXPECTED_SHADOW_GOALS_X):
        raise ValueError("Shadow admission must contain exactly three logs")

    def metric(entry: dict[str, Any], name: str) -> float:
        value = entry.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Shadow admission log metric {name!r} must be finite")
        return float(value)

    goals: list[float] = []
    resolved_paths: set[Path] = set()
    for raw_log in raw_logs:
        if not isinstance(raw_log, dict):
            raise ValueError("Shadow admission logs must be objects")
        raw_path = raw_log.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Shadow admission log path must be a non-empty string")
        resolved_path = Path(raw_path).resolve()
        if resolved_path in resolved_paths:
            raise ValueError("Shadow admission log paths must be distinct")
        resolved_paths.add(resolved_path)
        digest = raw_log.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Shadow admission contains an invalid log SHA-256 digest")
        if _sha256(resolved_path) != digest:
            raise ValueError(f"Shadow log {resolved_path} differs from its admitted digest")
        steps = raw_log.get("steps")
        unique_ticks = raw_log.get("unique_feedback_ticks")
        if steps != _EXPECTED_SHADOW_STEPS or unique_ticks != _EXPECTED_SHADOW_STEPS:
            raise ValueError("Each admitted shadow must contain 152 distinct policy samples")
        goals.append(metric(raw_log, "goal_pos_x_m"))
        if metric(raw_log, "inference_maximum_ms") > manifest.policy_dt * 1000.0:
            raise ValueError("Admitted shadow inference exceeded the policy deadline")
        if metric(raw_log, "feedback_age_maximum_ms") > _FEEDBACK_TIMEOUT_S * 1000.0:
            raise ValueError("Admitted shadow feedback exceeded the freshness deadline")
        if metric(raw_log, "body_tilt_maximum_deg") > math.degrees(_MAX_TILT_RAD):
            raise ValueError("Admitted shadow body tilt exceeded the hardware limit")
        if metric(raw_log, "joint_speed_maximum_rad_s") > _SHADOW_MAX_JOINT_SPEED_RAD_S:
            raise ValueError("Admitted shadow was not stationary")
        if metric(raw_log, "projected_torque_maximum_fraction") > 0.600001:
            raise ValueError("Admitted shadow exceeded the validated torque envelope")
    if not np.allclose(
        np.sort(np.asarray(goals, dtype=np.float64)),
        np.asarray(_EXPECTED_SHADOW_GOALS_X, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Shadow admission must cover goals -0.1, 0.0, and 0.1 m")


def _project_shadow_target(
    runtime: JumpGoalRuntime,
    raw_action: np.ndarray,
    manifest: HardwareManifest,
    snapshot: FeedbackSnapshot,
    effort_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the complete deployment command guard without publishing.

    Args:
        runtime: Armed jump-policy observation/action runtime.
        raw_action: Raw policy action in manifest order.
        manifest: Validated hardware manifest.
        snapshot: Live G1 feedback sample.
        effort_scale: Fraction of manifest effort available to the guard.

    Returns:
        Requested target [rad], projected target [rad], unprojected PD torque
        [N·m], projected PD torque [N·m], and effective effort ratios.

    Raises:
        SafetyFault: If the projected command violates its torque envelope.
        ValueError: If :paramref:`effort_scale` is invalid.
    """
    if not math.isfinite(effort_scale) or not 0.0 < effort_scale <= 1.0:
        raise ValueError("effort_scale must be finite and in (0, 1]")
    requested_target = runtime.transform_action(raw_action)
    requested_target = np.clip(
        requested_target,
        manifest.target_position_lower,
        manifest.target_position_upper,
    )
    if manifest.lower_limit_velocity_lookahead is not None:
        requested_target = project_position_target_to_lower_limit(
            requested_target,
            snapshot.joint_velocities,
            manifest.target_position_lower,
            manifest.target_position_upper,
            manifest.lower_limit_velocity_lookahead,
        )
    effort_ratio = np.full(manifest.joint_count, effort_scale, dtype=np.float64)
    if manifest.effort_limit_ratio is not None:
        effort_ratio = np.minimum(effort_ratio, manifest.effort_limit_ratio)
    unprojected_torque = manifest.stiffness * (requested_target - snapshot.joint_positions) - (
        manifest.damping * snapshot.joint_velocities
    )
    projected_target = project_pd_position_target(
        requested_target,
        snapshot.joint_positions,
        snapshot.joint_velocities,
        manifest.stiffness,
        manifest.damping,
        manifest.effort_limit,
        effort_ratio,
    )
    projected_target = np.clip(
        projected_target,
        manifest.target_position_lower,
        manifest.target_position_upper,
    )
    projected_torque = manifest.stiffness * (projected_target - snapshot.joint_positions) - (
        manifest.damping * snapshot.joint_velocities
    )
    torque_fraction = np.abs(projected_torque) / manifest.effort_limit
    if np.any(torque_fraction > effort_ratio + 1.0e-9):
        raise SafetyFault("Read-only policy target exceeded the projected torque envelope")
    return requested_target, projected_target, unprojected_torque, projected_torque, effort_ratio


def _evaluate_shadow_policy(
    manifest_path: Path,
    policy_path: Path,
    manifest: HardwareManifest,
    snapshot: FeedbackSnapshot,
    *,
    goal_pos_x: float,
    goal_pos_y: float,
    goal_yaw: float,
    goal_roll: float,
    goal_pitch: float,
    effort_scale: float,
) -> ShadowPolicyReport:
    """Evaluate policy step 0 from live feedback without publishing a command.

    Args:
        manifest_path: Deployment manifest path.
        policy_path: ONNX policy path.
        manifest: Validated hardware manifest.
        snapshot: Live G1 feedback sample.
        goal_pos_x: Forward landing displacement [m].
        goal_pos_y: Lateral landing displacement [m].
        goal_yaw: Heading displacement [rad].
        goal_roll: Roll displacement [rad].
        goal_pitch: Pitch displacement [rad].
        effort_scale: Fraction of manifest effort available to the guard.

    Returns:
        Read-only inference and command-envelope diagnostics.

    Raises:
        SafetyFault: If warm inference misses the policy deadline or the
            projected command violates its torque envelope.
        ValueError: If an input or deployment artifact violates its contract.
    """
    runtime = JumpGoalRuntime(manifest_path, freeze_during_flight=True)
    runtime.arm(goal_pos_x, goal_pos_y, goal_yaw, roll=goal_roll, pitch=goal_pitch)
    root_position = np.asarray((0.0, 0.0, manifest.initial_root_height), dtype=np.float64)
    runtime.trigger(
        root_position,
        snapshot.imu_quaternion,
        snapshot.joint_positions,
        goal_pos_z_w=0.0,
    )
    observation = runtime.step(
        snapshot.joint_positions,
        snapshot.joint_velocities,
        snapshot.imu_gyroscope,
        snapshot.imu_quaternion,
        root_position,
        snapshot.imu_quaternion,
    )

    policy = OnnxPolicy(policy_path, _OBSERVATION_DIM, manifest.joint_count)
    policy.warm_up()
    latencies_ms = np.empty(_POLICY_BENCHMARK_SAMPLES, dtype=np.float64)
    action = np.zeros(manifest.joint_count, dtype=np.float64)
    for index in range(_POLICY_BENCHMARK_SAMPLES):
        started_at = time.perf_counter()
        action = policy(observation)
        latencies_ms[index] = (time.perf_counter() - started_at) * 1000.0
    inference_p99_ms = float(np.percentile(latencies_ms, 99.0))
    inference_maximum_ms = float(np.max(latencies_ms))
    if inference_p99_ms > 0.5 * manifest.policy_dt * 1000.0 or inference_maximum_ms > manifest.policy_dt * 1000.0:
        raise SafetyFault(
            f"ONNX inference timing is unsafe: p99={inference_p99_ms:.3f} ms, "
            f"maximum={inference_maximum_ms:.3f} ms, policy period={manifest.policy_dt * 1000.0:.3f} ms"
        )

    _, target, _, estimated_torque, _ = _project_shadow_target(
        runtime,
        action,
        manifest,
        snapshot,
        effort_scale,
    )
    torque_fraction = np.abs(estimated_torque) / manifest.effort_limit
    maximum_torque_index = int(np.argmax(torque_fraction))
    return ShadowPolicyReport(
        policy_sha256=_sha256(policy_path),
        inference_median_ms=float(np.median(latencies_ms)),
        inference_p99_ms=inference_p99_ms,
        inference_maximum_ms=inference_maximum_ms,
        raw_action_minimum=float(np.min(action)),
        raw_action_maximum=float(np.max(action)),
        maximum_target_delta_rad=float(np.max(np.abs(target - snapshot.joint_positions))),
        maximum_torque_fraction=float(torque_fraction[maximum_torque_index]),
        maximum_torque_joint=manifest.joint_names[maximum_torque_index],
    )


def _run_shadow_policy_episode(  # noqa: C901
    manifest_path: Path,
    policy_path: Path,
    manifest: HardwareManifest,
    state_buffer: _StateBuffer,
    log_path: Path,
    *,
    goal_pos_x: float,
    goal_pos_y: float,
    goal_yaw: float,
    goal_roll: float,
    goal_pitch: float,
    effort_scale: float,
) -> ShadowEpisodeReport:
    """Run one complete policy timeline against live feedback without publishing.

    The robot must remain stationary under its existing native controller. The
    resulting targets are counterfactual diagnostics: they exercise every
    observation, inference, action-transform, knee-brake, and torque-projection
    step, but they are never sent to a command topic.

    Args:
        manifest_path: Deployment manifest path.
        policy_path: ONNX policy path.
        manifest: Validated hardware manifest.
        state_buffer: Live, validated G1 feedback buffer.
        log_path: New NPZ file to create. Existing files are never overwritten.
        goal_pos_x: Forward landing displacement [m].
        goal_pos_y: Lateral landing displacement [m].
        goal_yaw: Heading displacement [rad].
        goal_roll: Roll displacement [rad].
        goal_pitch: Pitch displacement [rad].
        effort_scale: Fraction of manifest effort available to the guard.

    Returns:
        Full-episode timing and command-envelope diagnostics.

    Raises:
        SafetyFault: If feedback, timing, or command-envelope checks fail.
        ValueError: If an input or deployment artifact violates its contract.
    """
    resolved_log_path = log_path.resolve()
    if resolved_log_path.exists():
        raise ValueError(f"Shadow log already exists and will not be overwritten: {resolved_log_path}")
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = state_buffer.snapshot()
    if snapshot is None:
        raise SafetyFault("no valid G1 feedback before policy shadow")
    if np.max(np.abs(snapshot.joint_velocities)) > _SHADOW_MAX_JOINT_SPEED_RAD_S:
        raise SafetyFault(f"policy shadow requires joint speeds below {_SHADOW_MAX_JOINT_SPEED_RAD_S:.2f} rad/s")

    runtime = JumpGoalRuntime(manifest_path, freeze_during_flight=True)
    runtime.arm(goal_pos_x, goal_pos_y, goal_yaw, roll=goal_roll, pitch=goal_pitch)
    root_position = np.asarray((0.0, 0.0, manifest.initial_root_height), dtype=np.float64)
    policy = OnnxPolicy(policy_path, _OBSERVATION_DIM, manifest.joint_count)
    policy.warm_up()
    initial_mode_pr = snapshot.mode_pr
    initial_mode_machine = snapshot.mode_machine
    episode_triggered = False

    sample_times_s: list[float] = []
    feedback_ages_ms: list[float] = []
    ticks: list[int] = []
    mode_pr_values: list[int] = []
    mode_machines: list[int] = []
    temperatures_c: list[int] = []
    joint_positions: list[np.ndarray] = []
    joint_velocities: list[np.ndarray] = []
    joint_torque_estimates: list[np.ndarray] = []
    imu_quaternions: list[np.ndarray] = []
    imu_gyroscopes: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    delayed_actions: list[np.ndarray] = []
    requested_targets: list[np.ndarray] = []
    projected_targets: list[np.ndarray] = []
    unprojected_torques: list[np.ndarray] = []
    projected_torques: list[np.ndarray] = []
    effort_ratios: list[np.ndarray] = []
    inference_latencies_ms: list[float] = []
    body_tilts_deg: list[float] = []
    maximum_joint_speeds: list[float] = []
    torque_projection_steps = 0

    started_at = time.monotonic()
    next_policy = started_at
    while not runtime.done:
        _sleep_until(next_policy)
        snapshot = state_buffer.snapshot()
        if snapshot is None:
            raise SafetyFault("G1 feedback disappeared during policy shadow")
        step_started_at = time.monotonic()
        feedback_fault = _feedback_fault(snapshot, manifest, step_started_at)
        if feedback_fault is not None:
            raise SafetyFault(f"policy shadow stopped: {feedback_fault}")
        maximum_joint_speed = float(np.max(np.abs(snapshot.joint_velocities)))
        if maximum_joint_speed > _SHADOW_MAX_JOINT_SPEED_RAD_S:
            raise SafetyFault(
                f"policy shadow stopped: joint speed {maximum_joint_speed:.3f} rad/s exceeded "
                f"{_SHADOW_MAX_JOINT_SPEED_RAD_S:.2f} rad/s"
            )
        if snapshot.mode_machine != initial_mode_machine:
            raise SafetyFault(
                f"policy shadow stopped: mode_machine changed from {initial_mode_machine} to {snapshot.mode_machine}"
            )
        if snapshot.mode_pr != initial_mode_pr:
            raise SafetyFault(f"policy shadow stopped: mode_pr changed from {initial_mode_pr} to {snapshot.mode_pr}")
        if _remote_b_pressed(snapshot.wireless_remote):
            raise SafetyFault("policy shadow cancelled by B")
        if state_buffer.crc_errors or state_buffer.invalid_packets:
            raise SafetyFault(
                "policy shadow feedback integrity error "
                f"(CRC={state_buffer.crc_errors}, invalid={state_buffer.invalid_packets})"
            )

        if not episode_triggered:
            runtime.trigger(
                root_position,
                snapshot.imu_quaternion,
                snapshot.joint_positions,
                goal_pos_z_w=0.0,
            )
            episode_triggered = True
        observation = runtime.step(
            snapshot.joint_positions,
            snapshot.joint_velocities,
            snapshot.imu_gyroscope,
            snapshot.imu_quaternion,
            root_position,
            snapshot.imu_quaternion,
        )
        inference_started_at = time.perf_counter()
        raw_action = policy(observation)
        inference_latency_ms = (time.perf_counter() - inference_started_at) * 1000.0
        if inference_latency_ms > manifest.policy_dt * 1000.0:
            raise SafetyFault(
                f"policy shadow inference missed its deadline: {inference_latency_ms:.3f} ms "
                f"> {manifest.policy_dt * 1000.0:.3f} ms"
            )
        requested_target, projected_target, unprojected_torque, projected_torque, effort_ratio = _project_shadow_target(
            runtime, raw_action, manifest, snapshot, effort_scale
        )
        if np.any(np.abs(unprojected_torque) > effort_ratio * manifest.effort_limit + 1.0e-9):
            torque_projection_steps += 1

        sample_times_s.append(step_started_at - started_at)
        feedback_ages_ms.append((step_started_at - snapshot.received_at) * 1000.0)
        ticks.append(snapshot.tick)
        mode_pr_values.append(snapshot.mode_pr)
        mode_machines.append(snapshot.mode_machine)
        temperatures_c.append(snapshot.maximum_temperature_c)
        joint_positions.append(snapshot.joint_positions)
        joint_velocities.append(snapshot.joint_velocities)
        joint_torque_estimates.append(snapshot.joint_torque_estimates)
        imu_quaternions.append(snapshot.imu_quaternion)
        imu_gyroscopes.append(snapshot.imu_gyroscope)
        observations.append(observation)
        raw_actions.append(raw_action)
        delayed_actions.append(runtime.delayed_action)
        requested_targets.append(requested_target)
        projected_targets.append(projected_target)
        unprojected_torques.append(unprojected_torque)
        projected_torques.append(projected_torque)
        effort_ratios.append(effort_ratio)
        inference_latencies_ms.append(inference_latency_ms)
        body_tilts_deg.append(math.degrees(_body_tilt(snapshot.imu_quaternion)))
        maximum_joint_speeds.append(maximum_joint_speed)

        next_policy += manifest.policy_dt
        schedule_lag_s = time.monotonic() - next_policy
        if not runtime.done and schedule_lag_s > manifest.policy_dt:
            raise SafetyFault(f"policy shadow schedule fell behind by {schedule_lag_s * 1000.0:.3f} ms")

    elapsed_s = time.monotonic() - started_at
    inference_array = np.asarray(inference_latencies_ms, dtype=np.float64)
    inference_p99_ms = float(np.percentile(inference_array, 99.0))
    inference_maximum_ms = float(np.max(inference_array))
    if inference_p99_ms > 0.5 * manifest.policy_dt * 1000.0:
        raise SafetyFault(
            f"policy shadow inference p99 is unsafe: {inference_p99_ms:.3f} ms > "
            f"{0.5 * manifest.policy_dt * 1000.0:.3f} ms"
        )

    raw_action_array = np.asarray(raw_actions, dtype=np.float64)
    joint_position_array = np.asarray(joint_positions, dtype=np.float64)
    measured_torque_array = np.asarray(joint_torque_estimates, dtype=np.float64)
    projected_target_array = np.asarray(projected_targets, dtype=np.float64)
    unprojected_torque_array = np.asarray(unprojected_torques, dtype=np.float64)
    projected_torque_array = np.asarray(projected_torques, dtype=np.float64)
    unprojected_fraction = np.abs(unprojected_torque_array) / manifest.effort_limit[None, :]
    projected_fraction = np.abs(projected_torque_array) / manifest.effort_limit[None, :]
    maximum_torque_flat_index = int(np.argmax(projected_fraction))
    _, maximum_torque_joint_index = np.unravel_index(maximum_torque_flat_index, projected_fraction.shape)
    measured_torque_fraction = np.abs(measured_torque_array) / manifest.effort_limit[None, :]
    maximum_measured_torque_flat_index = int(np.argmax(measured_torque_fraction))
    _, maximum_measured_torque_joint_index = np.unravel_index(
        maximum_measured_torque_flat_index,
        measured_torque_fraction.shape,
    )
    policy_sha256 = _sha256(policy_path)
    metadata = {
        "schema_version": "1.0",
        "read_only": True,
        "command_publisher_created": False,
        "feedback_mode": "stationary_live_lowstate_counterfactual_targets",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha256,
        "joint_names": list(manifest.joint_names),
        "goal": {
            "pos_x": goal_pos_x,
            "pos_y": goal_pos_y,
            "yaw": goal_yaw,
            "roll": goal_roll,
            "pitch": goal_pitch,
        },
        "effort_scale": effort_scale,
        "policy_dt_s": manifest.policy_dt,
        "feedback_counters": {
            "valid_packets": int(state_buffer.valid_packets),
            "crc_errors": int(state_buffer.crc_errors),
            "invalid_packets": int(state_buffer.invalid_packets),
        },
    }
    try:
        with resolved_log_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                time=np.asarray(sample_times_s, dtype=np.float64),
                feedback_age_ms=np.asarray(feedback_ages_ms, dtype=np.float64),
                tick=np.asarray(ticks, dtype=np.int64),
                mode_pr=np.asarray(mode_pr_values, dtype=np.int32),
                mode_machine=np.asarray(mode_machines, dtype=np.int32),
                maximum_temperature_c=np.asarray(temperatures_c, dtype=np.int16),
                joint_position=joint_position_array,
                joint_velocity=np.asarray(joint_velocities, dtype=np.float64),
                joint_torque_estimate=measured_torque_array,
                imu_quaternion_wxyz=np.asarray(imu_quaternions, dtype=np.float64),
                imu_gyroscope=np.asarray(imu_gyroscopes, dtype=np.float64),
                observation=np.asarray(observations, dtype=np.float32),
                raw_action=raw_action_array,
                delayed_action=np.asarray(delayed_actions, dtype=np.float64),
                requested_target=np.asarray(requested_targets, dtype=np.float64),
                projected_target=projected_target_array,
                unprojected_torque=unprojected_torque_array,
                projected_torque=projected_torque_array,
                effort_ratio=np.asarray(effort_ratios, dtype=np.float64),
                inference_latency_ms=inference_array,
                body_tilt_deg=np.asarray(body_tilts_deg, dtype=np.float64),
                maximum_joint_speed_rad_s=np.asarray(maximum_joint_speeds, dtype=np.float64),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
    except OSError as exc:
        raise ValueError(f"Cannot create shadow log {resolved_log_path}: {exc}") from exc

    return ShadowEpisodeReport(
        policy_sha256=policy_sha256,
        steps=len(raw_actions),
        elapsed_s=elapsed_s,
        inference_median_ms=float(np.median(inference_array)),
        inference_p99_ms=inference_p99_ms,
        inference_maximum_ms=inference_maximum_ms,
        raw_action_minimum=float(np.min(raw_action_array)),
        raw_action_maximum=float(np.max(raw_action_array)),
        maximum_target_delta_rad=float(np.max(np.abs(projected_target_array - joint_position_array))),
        maximum_torque_fraction=float(np.max(projected_fraction)),
        maximum_torque_joint=manifest.joint_names[maximum_torque_joint_index],
        maximum_unprojected_torque_fraction=float(np.max(unprojected_fraction)),
        torque_projection_steps=torque_projection_steps,
        maximum_body_tilt_deg=float(np.max(body_tilts_deg)),
        maximum_joint_speed_rad_s=float(np.max(maximum_joint_speeds)),
        maximum_measured_torque_fraction=float(np.max(measured_torque_fraction)),
        maximum_measured_torque_joint=manifest.joint_names[maximum_measured_torque_joint_index],
        log_path=resolved_log_path,
    )


def _read_audio_cue(path: Path) -> tuple[bytes, float]:
    """Read and validate a G1-compatible PCM WAV cue.

    Args:
        path: WAV file to load.

    Returns:
        Raw PCM bytes and cue duration [s].
    """
    try:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("audio cue must be uncompressed PCM")
            if wav_file.getnchannels() != _AUDIO_CHANNEL_COUNT:
                raise ValueError("audio cue must be mono")
            if wav_file.getsampwidth() != _AUDIO_SAMPLE_WIDTH_BYTES:
                raise ValueError("audio cue must use 16-bit samples")
            if wav_file.getframerate() != _AUDIO_SAMPLE_RATE:
                raise ValueError("audio cue sample rate must be 16000 Hz")
            frame_count = wav_file.getnframes()
            pcm_data = wav_file.readframes(frame_count)
    except (FileNotFoundError, wave.Error) as exc:
        raise ValueError(f"cannot read audio cue {path}: {exc}") from exc
    if not pcm_data:
        raise ValueError("audio cue contains no samples")
    return pcm_data, frame_count / _AUDIO_SAMPLE_RATE


def _play_audio_cue(audio_client: Any, pcm_data: bytes, duration_s: float) -> None:
    """Play one PCM cue before motor-control ownership changes."""
    if not pcm_data or not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("audio cue data and duration must be valid")
    app_name = "g1_jump_deploy_mode"
    stream_id = str(time.time_ns())
    bytes_per_second = _AUDIO_SAMPLE_RATE * _AUDIO_CHANNEL_COUNT * _AUDIO_SAMPLE_WIDTH_BYTES
    chunk_size = int(bytes_per_second * _AUDIO_CHUNK_DURATION_S)
    for chunk_index, offset in enumerate(range(0, len(pcm_data), chunk_size)):
        chunk = pcm_data[offset : offset + chunk_size]
        result = audio_client.PlayStream(app_name, stream_id, chunk)
        if not isinstance(result, tuple) or len(result) != 2:
            raise SafetyFault(f"audio stream chunk {chunk_index} returned an invalid response")
        return_code, _ = result
        if return_code != 0:
            raise SafetyFault(f"audio stream chunk {chunk_index} returned code {return_code}")
        time.sleep(len(chunk) / bytes_per_second)
    time.sleep(0.25)
    stop_code = audio_client.PlayStop(app_name)
    if stop_code != 0:
        raise SafetyFault(f"audio PlayStop returned code {stop_code}")
    print(f"PASS: played Jump mode cue ({duration_s:.2f} s).")


def _remote_face_button_pressed(
    wireless_remote: bytes | bytearray | list[int],
    mask: int,
) -> bool:
    """Return whether one face-button bit is active in a Unitree remote sample."""
    return len(wireless_remote) == 40 and bool(int(wireless_remote[3]) & mask)


def _remote_a_pressed(wireless_remote: bytes | bytearray | list[int]) -> bool:
    """Return whether the Unitree handheld remote's A button is pressed."""
    return _remote_face_button_pressed(wireless_remote, _REMOTE_A_MASK)


def _remote_b_pressed(wireless_remote: bytes | bytearray | list[int]) -> bool:
    """Return whether the Unitree handheld remote's B button is pressed."""
    return _remote_face_button_pressed(wireless_remote, _REMOTE_B_MASK)


def _remote_y_pressed(wireless_remote: bytes | bytearray | list[int]) -> bool:
    """Return whether the Unitree handheld remote's Y button is pressed."""
    return _remote_face_button_pressed(wireless_remote, _REMOTE_Y_MASK)


def _remote_rehearsal_buttons_released(wireless_remote: bytes | bytearray | list[int]) -> bool:
    """Return whether A, Y, and B are all released."""
    return not any(
        (
            _remote_a_pressed(wireless_remote),
            _remote_b_pressed(wireless_remote),
            _remote_y_pressed(wireless_remote),
        )
    )


def _remote_activation_pressed(wireless_remote: bytes | bytearray | list[int]) -> bool:
    """Return whether both L1 and R1 are pressed."""
    if len(wireless_remote) != 40:
        return False
    shoulder_buttons = int(wireless_remote[2])
    return shoulder_buttons & (_REMOTE_L1_MASK | _REMOTE_R1_MASK) == (_REMOTE_L1_MASK | _REMOTE_R1_MASK)


def _body_tilt(quaternion_wxyz: np.ndarray) -> float:
    quaternion = _finite_vector(quaternion_wxyz, 4, "IMU quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("IMU quaternion must be non-zero")
    _, x, y, _ = quaternion / norm
    return math.acos(float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def _feedback_fault(snapshot: FeedbackSnapshot, manifest: HardwareManifest, now: float) -> str | None:
    """Return a feedback interlock reason, or ``None`` when feedback is safe."""
    age_s = now - snapshot.received_at
    if not math.isfinite(age_s) or age_s < 0.0 or age_s > _FEEDBACK_TIMEOUT_S:
        return f"feedback stale for {age_s * 1000.0:.1f} ms"
    try:
        body_tilt = _body_tilt(snapshot.imu_quaternion)
    except ValueError as exc:
        return f"invalid IMU feedback: {exc}"
    if body_tilt > _MAX_TILT_RAD:
        return "body tilt exceeded 20 degrees"
    maximum_angular_speed = float(np.max(np.abs(snapshot.imu_gyroscope)))
    if maximum_angular_speed > _MAX_BASE_ANGULAR_SPEED_RAD_S:
        return (
            f"base angular speed reached {maximum_angular_speed:.2f} rad/s and exceeded "
            f"{_MAX_BASE_ANGULAR_SPEED_RAD_S:.2f} rad/s"
        )
    maximum_speed_index = int(np.argmax(np.abs(snapshot.joint_velocities)))
    maximum_speed = float(abs(snapshot.joint_velocities[maximum_speed_index]))
    if maximum_speed > _MAX_JOINT_SPEED_RAD_S:
        return (
            f"joint speed for {manifest.joint_names[maximum_speed_index]} reached "
            f"{maximum_speed:.2f} rad/s and exceeded "
            f"{_MAX_JOINT_SPEED_RAD_S:.2f} rad/s"
        )
    if np.any(snapshot.joint_positions < manifest.joint_position_lower) or np.any(
        snapshot.joint_positions > manifest.joint_position_upper
    ):
        return "a measured joint position exceeded the manifest physical limits"
    if snapshot.maximum_temperature_c > _MAX_MOTOR_TEMPERATURE_C:
        return f"motor temperature {snapshot.maximum_temperature_c} C exceeded {_MAX_MOTOR_TEMPERATURE_C} C"
    return None


def _stand_entry_pose_fault(
    snapshot: FeedbackSnapshot,
    manifest: HardwareManifest,
    limit_rad: float = _STAND_ENTRY_LEG_ERROR_LIMIT_RAD,
) -> str | None:
    """Return why the measured leg posture is too far from stand entry."""
    if not math.isfinite(limit_rad) or limit_rad <= 0.0:
        raise ValueError("stand-entry leg error limit must be positive and finite")
    leg_indices = np.asarray(
        [
            index
            for index, name in enumerate(manifest.joint_names)
            if any(part in name for part in ("_hip_", "_knee_", "_ankle_"))
        ],
        dtype=np.int32,
    )
    if leg_indices.size != 12:
        return f"manifest identifies {leg_indices.size} leg joints instead of 12"
    error = np.abs(snapshot.joint_positions[leg_indices] - manifest.default_position[leg_indices])
    worst_leg_offset = int(np.argmax(error))
    if error[worst_leg_offset] <= limit_rad:
        return None
    joint_index = int(leg_indices[worst_leg_offset])
    return (
        f"stand-entry {manifest.joint_names[joint_index]} differs from the manifest pose by "
        f"{error[worst_leg_offset]:.3f} rad (limit {limit_rad:.2f} rad)"
    )


def _native_walkrun_handoff_fault(
    snapshot: FeedbackSnapshot,
    manifest: HardwareManifest,
    now: float,
) -> str | None:
    """Return why a successful custom stand cannot hand back to native WALKRUN."""
    feedback_fault = _feedback_fault(snapshot, manifest, now)
    if feedback_fault is not None:
        return feedback_fault
    if _remote_b_pressed(snapshot.wireless_remote):
        return "B is pressed"
    body_tilt = _body_tilt(snapshot.imu_quaternion)
    if body_tilt > _NATIVE_WALKRUN_HANDOFF_MAX_TILT_RAD:
        return f"body tilt {math.degrees(body_tilt):.2f} deg exceeds 10.00 deg"
    maximum_speed = float(np.max(np.abs(snapshot.joint_velocities)))
    if maximum_speed > _NATIVE_WALKRUN_HANDOFF_MAX_SPEED_RAD_S:
        return f"joint speed {maximum_speed:.2f} rad/s exceeds {_NATIVE_WALKRUN_HANDOFF_MAX_SPEED_RAD_S:.2f} rad/s"
    return _stand_entry_pose_fault(snapshot, manifest)


class _StateBuffer:
    def __init__(self, manifest: HardwareManifest, crc: Any):
        self._manifest = manifest
        self._crc = crc
        self._lock = threading.Lock()
        self._snapshot: FeedbackSnapshot | None = None
        self.valid_packets = 0
        self.crc_errors = 0
        self.invalid_packets = 0

    def update(self, state: Any) -> None:
        """Validate and store one SDK ``LowState`` sample."""
        if self._crc.Crc(state) != state.crc:
            with self._lock:
                self.crc_errors += 1
            return
        try:
            if len(state.motor_state) != 35:
                raise ValueError("motor_state must contain 35 entries")
            positions = np.asarray([state.motor_state[slot].q for slot in self._manifest.sdk_slots], dtype=np.float64)
            velocities = np.asarray([state.motor_state[slot].dq for slot in self._manifest.sdk_slots], dtype=np.float64)
            torque_estimates = np.asarray(
                [state.motor_state[slot].tau_est for slot in self._manifest.sdk_slots],
                dtype=np.float64,
            )
            quaternion = _finite_vector(state.imu_state.quaternion, 4, "IMU quaternion")
            gyroscope = _finite_vector(state.imu_state.gyroscope, 3, "IMU gyroscope")
            _finite_vector(positions, self._manifest.joint_count, "joint positions")
            _finite_vector(velocities, self._manifest.joint_count, "joint velocities")
            _finite_vector(torque_estimates, self._manifest.joint_count, "joint torque estimates")
            temperatures = [
                int(temperature)
                for slot in self._manifest.sdk_slots
                for temperature in state.motor_state[slot].temperature
            ]
            remote = bytes(state.wireless_remote)
            if len(remote) != 40:
                raise ValueError("wireless_remote must contain 40 bytes")
            snapshot = FeedbackSnapshot(
                received_at=time.monotonic(),
                tick=int(state.tick),
                mode_pr=int(state.mode_pr),
                mode_machine=int(state.mode_machine),
                joint_positions=positions,
                joint_velocities=velocities,
                joint_torque_estimates=torque_estimates,
                imu_quaternion=quaternion,
                imu_gyroscope=gyroscope,
                wireless_remote=remote,
                maximum_temperature_c=max(temperatures),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            with self._lock:
                self.invalid_packets += 1
            return
        with self._lock:
            self._snapshot = snapshot
            self.valid_packets += 1

    def snapshot(self) -> FeedbackSnapshot | None:
        """Return a copy of the newest valid feedback sample."""
        with self._lock:
            if self._snapshot is None:
                return None
            snapshot = self._snapshot
            return FeedbackSnapshot(
                received_at=snapshot.received_at,
                tick=snapshot.tick,
                mode_pr=snapshot.mode_pr,
                mode_machine=snapshot.mode_machine,
                joint_positions=snapshot.joint_positions.copy(),
                joint_velocities=snapshot.joint_velocities.copy(),
                joint_torque_estimates=snapshot.joint_torque_estimates.copy(),
                imu_quaternion=snapshot.imu_quaternion.copy(),
                imu_gyroscope=snapshot.imu_gyroscope.copy(),
                wireless_remote=snapshot.wireless_remote,
                maximum_temperature_c=snapshot.maximum_temperature_c,
            )


class _StandOperator:
    def __init__(self):
        self.pending_goal = None
        self.request_start = False
        self.confirm = False
        self.abort = False

    def update(self, wireless_remote: bytes) -> None:
        """Update the stand-only abort intent from one remote sample."""
        self.abort = _remote_b_pressed(wireless_remote)


class _GantryRehearsalOperator(_StandOperator):
    def __init__(self, goal: JumpGoal):
        super().__init__()
        self.pending_goal = goal

    def update(self, wireless_remote: bytes) -> None:
        """Map A/Y/B to rehearse, confirm, and abort intents."""
        super().update(wireless_remote)
        self.request_start = _remote_a_pressed(wireless_remote)
        self.confirm = _remote_y_pressed(wireless_remote)


class _InactivePolicy:
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        raise RuntimeError("The stand-only G1 runner must not invoke a policy")


class _G1Robot:
    def __init__(
        self,
        manifest: HardwareManifest,
        state_buffer: _StateBuffer,
        publisher: Any,
        low_command: Any,
        crc: Any,
    ):
        self._manifest = manifest
        self._state_buffer = state_buffer
        self._publisher = publisher
        self._low_command = low_command
        self._crc = crc
        self._base_target = np.zeros(manifest.joint_count, dtype=np.float64)
        self._stiffness = np.zeros(manifest.joint_count, dtype=np.float64)
        self._damping = np.zeros(manifest.joint_count, dtype=np.float64)
        self._last_published_target: np.ndarray | None = None
        self._last_published_command: _PublishedCommand | None = None
        self._last_publish_time: float | None = None
        self._control_deadline_missed = False
        self.maximum_estimated_torque = np.zeros(manifest.joint_count, dtype=np.float64)
        self.peak_position_torque = np.zeros(manifest.joint_count, dtype=np.float64)
        self.peak_damping_torque = np.zeros(manifest.joint_count, dtype=np.float64)
        self.peak_position_error = np.zeros(manifest.joint_count, dtype=np.float64)
        self.peak_joint_velocity = np.zeros(manifest.joint_count, dtype=np.float64)

    def _snapshot(self) -> FeedbackSnapshot:
        snapshot = self._state_buffer.snapshot()
        if snapshot is None:
            raise SafetyFault("no valid G1 feedback")
        return snapshot

    @property
    def joint_positions(self) -> np.ndarray:
        return self._snapshot().joint_positions

    @property
    def joint_velocities(self) -> np.ndarray:
        return self._snapshot().joint_velocities

    @property
    def base_angular_velocity(self) -> np.ndarray:
        return self._snapshot().imu_gyroscope

    @property
    def imu_quaternion(self) -> np.ndarray:
        return self._snapshot().imu_quaternion

    @property
    def odometry_position(self) -> np.ndarray:
        # The latched deployment policy only consumes this at trigger time. G1
        # LowState has no world position, so use the exported reference height
        # instead of incorrectly telling the policy its pelvis is on the floor.
        return np.asarray((0.0, 0.0, self._manifest.initial_root_height), dtype=np.float64)

    @property
    def odometry_quaternion(self) -> np.ndarray:
        return self.imu_quaternion

    @property
    def foot_contact_forces(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    @property
    def joint_limit_violations(self) -> np.ndarray:
        positions = self.joint_positions
        return np.logical_or(
            positions <= self._manifest.joint_position_lower,
            positions >= self._manifest.joint_position_upper,
        )

    @property
    def feedback_stale(self) -> bool:
        return time.monotonic() - self._snapshot().received_at > _FEEDBACK_TIMEOUT_S

    @property
    def control_deadline_missed(self) -> bool:
        return self._control_deadline_missed

    @property
    def command_base_target(self) -> np.ndarray:
        """Held base joint-position target [rad], in manifest order."""
        return self._base_target.copy()

    @property
    def command_stiffness(self) -> np.ndarray:
        """Held position gains [N·m/rad], in manifest order."""
        return self._stiffness.copy()

    @property
    def command_damping(self) -> np.ndarray:
        """Held velocity gains [N·m·s/rad], in manifest order."""
        return self._damping.copy()

    @property
    def last_published_command(self) -> _PublishedCommand | None:
        """Most recent command accepted by DDS, with copied source feedback."""
        command = self._last_published_command
        if command is None:
            return None
        feedback = command.feedback
        return _PublishedCommand(
            published_at=command.published_at,
            feedback=FeedbackSnapshot(
                received_at=feedback.received_at,
                tick=feedback.tick,
                mode_pr=feedback.mode_pr,
                mode_machine=feedback.mode_machine,
                joint_positions=feedback.joint_positions.copy(),
                joint_velocities=feedback.joint_velocities.copy(),
                joint_torque_estimates=feedback.joint_torque_estimates.copy(),
                imu_quaternion=feedback.imu_quaternion.copy(),
                imu_gyroscope=feedback.imu_gyroscope.copy(),
                wireless_remote=feedback.wireless_remote,
                maximum_temperature_c=feedback.maximum_temperature_c,
            ),
            target=command.target.copy(),
            stiffness=command.stiffness.copy(),
            damping=command.damping.copy(),
        )

    def command_joint_position_target(
        self,
        target: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ) -> None:
        """Hold a finite position/gain command for the 500 Hz publisher."""
        count = self._manifest.joint_count
        self._base_target = _finite_vector(target, count, "command target").copy()
        self._stiffness = _finite_vector(stiffness, count, "command stiffness").copy()
        self._damping = _finite_vector(damping, count, "command damping").copy()
        if np.any(self._stiffness < 0.0) or np.any(self._damping < 0.0):
            raise ValueError("Command gains must be non-negative")

    def set_damping(self) -> None:
        """Prepare a low-gain damping command at the measured posture."""
        self._base_target = self.joint_positions
        self._stiffness.fill(0.0)
        self._damping.fill(_TAKEOVER_DAMPING)

    def publish(
        self,
        balance_offset: np.ndarray,
        effort_scale: float,
        *,
        target_rate_limit_rad_s: float | None = _TARGET_RATE_LIMIT_RAD_S,
    ) -> None:
        """Publish one guarded, torque-limited user command.

        Args:
            balance_offset: Fast-loop balance target correction [rad].
            effort_scale: Fraction of manifest effort limits available to the
                command guard.
            target_rate_limit_rad_s: Optional joint-target slew limit [rad/s].
                ``None`` retains the policy target dynamics while the torque
                projection remains active.
        """
        snapshot = self._snapshot()
        now = time.monotonic()
        feedback_fault = _feedback_fault(snapshot, self._manifest, now)
        if feedback_fault is not None:
            raise SafetyFault(feedback_fault)
        if self._last_publish_time is not None:
            gap_s = now - self._last_publish_time
            self._control_deadline_missed = gap_s > 2.0 * _FAST_DT
            if gap_s > _MAX_CONTROL_GAP_S:
                raise SafetyFault(f"control publish gap reached {gap_s * 1000.0:.1f} ms")

        desired = self._base_target + _finite_vector(balance_offset, self._manifest.joint_count, "balance offset")
        desired = np.clip(desired, self._manifest.target_position_lower, self._manifest.target_position_upper)
        if target_rate_limit_rad_s is None:
            rate_limited = desired.copy()
        else:
            if (
                isinstance(target_rate_limit_rad_s, bool)
                or not isinstance(target_rate_limit_rad_s, (int, float))
                or not math.isfinite(float(target_rate_limit_rad_s))
                or target_rate_limit_rad_s <= 0.0
            ):
                raise ValueError("target_rate_limit_rad_s must be a positive finite number or None")
            previous = snapshot.joint_positions if self._last_published_target is None else self._last_published_target
            maximum_step = float(target_rate_limit_rad_s) * _FAST_DT
            rate_limited = previous + np.clip(desired - previous, -maximum_step, maximum_step)
        if self._manifest.lower_limit_velocity_lookahead is not None:
            rate_limited = project_position_target_to_lower_limit(
                rate_limited,
                snapshot.joint_velocities,
                self._manifest.target_position_lower,
                self._manifest.target_position_upper,
                self._manifest.lower_limit_velocity_lookahead,
            )

        position_error = rate_limited - snapshot.joint_positions
        position_torque = self._stiffness * position_error
        damping_torque = -self._damping * snapshot.joint_velocities
        estimated_torque = position_torque + damping_torque
        effort_ratio = np.full(self._manifest.joint_count, effort_scale)
        if self._manifest.effort_limit_ratio is not None:
            effort_ratio = np.minimum(effort_ratio, self._manifest.effort_limit_ratio)
            active_stiffness = self._stiffness > np.finfo(np.float64).eps
            if np.any(active_stiffness):
                projected_target = project_pd_position_target(
                    rate_limited[active_stiffness],
                    snapshot.joint_positions[active_stiffness],
                    snapshot.joint_velocities[active_stiffness],
                    self._stiffness[active_stiffness],
                    self._damping[active_stiffness],
                    self._manifest.effort_limit[active_stiffness],
                    effort_ratio[active_stiffness],
                )
                rate_limited[active_stiffness] = projected_target
                position_error = rate_limited - snapshot.joint_positions
                position_torque = self._stiffness * position_error
                estimated_torque = position_torque + damping_torque
        effort_limit = effort_ratio * self._manifest.effort_limit
        clipped_torque = np.clip(estimated_torque, -effort_limit, effort_limit)
        new_peaks = np.abs(estimated_torque) > self.maximum_estimated_torque
        self.maximum_estimated_torque[new_peaks] = np.abs(estimated_torque[new_peaks])
        self.peak_position_torque[new_peaks] = position_torque[new_peaks]
        self.peak_damping_torque[new_peaks] = damping_torque[new_peaks]
        self.peak_position_error[new_peaks] = position_error[new_peaks]
        self.peak_joint_velocity[new_peaks] = snapshot.joint_velocities[new_peaks]
        if np.any(np.abs(estimated_torque) > effort_limit + 1.0e-9):
            joint_index = int(np.argmax(np.abs(estimated_torque) / effort_limit))
            raise SafetyFault(
                f"estimated torque for {self._manifest.joint_names[joint_index]} reached "
                f"{estimated_torque[joint_index]:.2f} N m (limit {effort_limit[joint_index]:.2f} N m; "
                f"position={position_torque[joint_index]:+.2f} N m, "
                f"damping={damping_torque[joint_index]:+.2f} N m, "
                f"error={position_error[joint_index]:+.4f} rad, "
                f"velocity={snapshot.joint_velocities[joint_index]:+.3f} rad/s)"
            )

        command_target = rate_limited.copy()
        active_stiffness = self._stiffness > np.finfo(np.float64).eps
        command_target[active_stiffness] = (
            snapshot.joint_positions[active_stiffness]
            + (
                clipped_torque[active_stiffness]
                + self._damping[active_stiffness] * snapshot.joint_velocities[active_stiffness]
            )
            / self._stiffness[active_stiffness]
        )
        command_target = np.clip(
            command_target,
            self._manifest.target_position_lower,
            self._manifest.target_position_upper,
        )
        bounded_position_error = command_target - snapshot.joint_positions
        bounded_position_torque = self._stiffness * bounded_position_error
        bounded_estimated_torque = bounded_position_torque + damping_torque
        if np.any(np.abs(bounded_estimated_torque) > effort_limit + 1.0e-9):
            joint_index = int(np.argmax(np.abs(bounded_estimated_torque) / effort_limit))
            raise SafetyFault(
                f"target bounds cannot satisfy the torque envelope for "
                f"{self._manifest.joint_names[joint_index]}: "
                f"{bounded_estimated_torque[joint_index]:.2f} N m exceeds "
                f"{effort_limit[joint_index]:.2f} N m"
            )
        self._write(snapshot, command_target, self._stiffness, self._damping)
        self._last_published_target = command_target
        self._last_publish_time = now

    def publish_takeover_damping(self) -> None:
        """Publish one measured-position damping command before ownership changes."""
        snapshot = self._snapshot()
        damping = np.full(self._manifest.joint_count, _TAKEOVER_DAMPING, dtype=np.float64)
        self._write(snapshot, snapshot.joint_positions, np.zeros(self._manifest.joint_count), damping)
        self._last_published_target = snapshot.joint_positions.copy()
        self._last_publish_time = time.monotonic()

    def _write(
        self,
        snapshot: FeedbackSnapshot,
        target: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ) -> None:
        self._low_command.mode_pr = 0
        self._low_command.mode_machine = snapshot.mode_machine
        for policy_index, slot in enumerate(self._manifest.sdk_slots):
            motor = self._low_command.motor_cmd[slot]
            motor.mode = 1
            motor.q = float(target[policy_index])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(stiffness[policy_index])
            motor.kd = float(damping[policy_index])
        self._low_command.crc = self._crc.Crc(self._low_command)
        if not self._publisher.Write(self._low_command):
            raise SafetyFault("DDS rejected the user low-command sample")
        self._last_published_command = _PublishedCommand(
            published_at=time.monotonic(),
            feedback=FeedbackSnapshot(
                received_at=snapshot.received_at,
                tick=snapshot.tick,
                mode_pr=snapshot.mode_pr,
                mode_machine=snapshot.mode_machine,
                joint_positions=snapshot.joint_positions.copy(),
                joint_velocities=snapshot.joint_velocities.copy(),
                joint_torque_estimates=snapshot.joint_torque_estimates.copy(),
                imu_quaternion=snapshot.imu_quaternion.copy(),
                imu_gyroscope=snapshot.imu_gyroscope.copy(),
                wireless_remote=snapshot.wireless_remote,
                maximum_temperature_c=snapshot.maximum_temperature_c,
            ),
            target=target.copy(),
            stiffness=stiffness.copy(),
            damping=damping.copy(),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight G1 jump inference read-only, run guarded stand control, or run one "
            "contactless gantry policy rehearsal."
        )
    )
    parser.add_argument("network_interface", help="Ethernet interface connected to G1, for example enp131s0.")
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST, help="23-DOF deployment manifest.")
    parser.add_argument(
        "--validation_record",
        type=Path,
        default=_DEFAULT_VALIDATION_RECORD,
        help="Accepted artifact SHA-256 record required by the hardware boundary.",
    )
    parser.add_argument("--policy", type=Path, default=None, help="Defaults to policy.onnx beside the manifest.")
    policy_mode = parser.add_mutually_exclusive_group()
    policy_mode.add_argument(
        "--check_policy",
        action="store_true",
        help="Run policy step 0 on live feedback without creating any command or locomotion client.",
    )
    policy_mode.add_argument(
        "--shadow_policy",
        action="store_true",
        help="Run the complete policy timeline against stationary live feedback without publishing commands.",
    )
    policy_mode.add_argument(
        "--gantry_policy_rehearsal",
        action="store_true",
        help=(
            "Actuate one zero-goal policy motion with the pelvis mechanically constrained and feet "
            "ground-clear; this is not a jump or ground-test mode."
        ),
    )
    parser.add_argument(
        "--shadow_log",
        type=Path,
        default=None,
        help="New NPZ diagnostic log required by --shadow_policy; existing files are refused.",
    )
    parser.add_argument(
        "--shadow_admission",
        type=Path,
        default=None,
        help="Exact read-only shadow-admission JSON required by --gantry_policy_rehearsal.",
    )
    parser.add_argument(
        "--rehearsal_log",
        type=Path,
        default=None,
        help="New NPZ control audit required by --gantry_policy_rehearsal; existing files are refused.",
    )
    parser.add_argument(
        "--acknowledge_contactless_rehearsal",
        action="store_true",
        help=(
            "Acknowledge that the robot has no foot-contact signal and that the rehearsal requires "
            "full mechanical support with no possible ground contact."
        ),
    )
    parser.add_argument("--goal_pos_x", type=float, default=0.0, help="Read-only policy goal x displacement [m].")
    parser.add_argument("--goal_pos_y", type=float, default=0.0, help="Read-only policy goal y displacement [m].")
    parser.add_argument("--goal_yaw", type=float, default=0.0, help="Read-only policy goal yaw displacement [rad].")
    parser.add_argument("--goal_roll", type=float, default=0.0, help="Read-only policy goal roll displacement [rad].")
    parser.add_argument("--goal_pitch", type=float, default=0.0, help="Read-only policy goal pitch displacement [rad].")
    parser.add_argument(
        "--audio_cue",
        type=Path,
        default=_DEFAULT_AUDIO_CUE,
        help="16-bit, 16 kHz, mono PCM WAV played after remote activation.",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Maximum user-control duration in seconds.")
    parser.add_argument(
        "--effort_scale",
        type=float,
        default=0.7,
        help="Fraction of manifest effort limits allowed by the command guard.",
    )
    parser.add_argument(
        "--enable_control",
        action="store_true",
        help="Permit the guarded handover after B-abort verification and an L1+R1 hold.",
    )
    parser.add_argument(
        "--entry_mode",
        choices=("passive", "native_stand", "gantry_standup", "native_walkrun_gantry"),
        default="passive",
        help=("Required native mode before handover; gantry modes require a load-bearing upright harness."),
    )
    parser.add_argument(
        "--exit_mode",
        choices=("passive", "native_walkrun"),
        default="passive",
        help="Native control mode requested after a successful duration; faults and B always return to PASSIVE.",
    )
    parser.add_argument(
        "--query_fsm",
        action="store_true",
        help="Read and print the native locomotion FSM ID without changing it.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 1.0 <= args.duration <= 30.0:
        parser.error("--duration must be finite and in [1, 30] seconds")
    if not math.isfinite(args.effort_scale) or not 0.1 <= args.effort_scale <= 1.0:
        parser.error("--effort_scale must be finite and in [0.1, 1.0]")
    if (args.check_policy or args.shadow_policy) and (args.enable_control or args.query_fsm):
        parser.error("Read-only policy modes cannot be combined with control or locomotion RPCs")
    if args.shadow_policy and args.shadow_log is None:
        parser.error("--shadow_policy requires --shadow_log")
    if not args.shadow_policy and args.shadow_log is not None:
        parser.error("--shadow_log requires --shadow_policy")
    if args.gantry_policy_rehearsal:
        if not args.enable_control:
            parser.error("--gantry_policy_rehearsal requires --enable_control")
        if args.query_fsm:
            parser.error("--gantry_policy_rehearsal cannot be combined with --query_fsm")
        if args.shadow_admission is None:
            parser.error("--gantry_policy_rehearsal requires --shadow_admission")
        if args.rehearsal_log is None:
            parser.error("--gantry_policy_rehearsal requires --rehearsal_log")
        if not args.acknowledge_contactless_rehearsal:
            parser.error("--gantry_policy_rehearsal requires --acknowledge_contactless_rehearsal")
        if args.entry_mode != "gantry_standup":
            parser.error("--gantry_policy_rehearsal requires --entry_mode gantry_standup")
        if args.exit_mode != "passive":
            parser.error("--gantry_policy_rehearsal requires --exit_mode passive")
        if args.duration < _GANTRY_REHEARSAL_MIN_DURATION_S:
            parser.error(f"--gantry_policy_rehearsal requires --duration >= {_GANTRY_REHEARSAL_MIN_DURATION_S:.1f}")
        if args.effort_scale > _GANTRY_REHEARSAL_MAX_EFFORT_SCALE:
            parser.error(f"--gantry_policy_rehearsal limits --effort_scale to {_GANTRY_REHEARSAL_MAX_EFFORT_SCALE:.1f}")
        if any(
            abs(getattr(args, name)) > 1.0e-12
            for name in ("goal_pos_x", "goal_pos_y", "goal_yaw", "goal_roll", "goal_pitch")
        ):
            parser.error("--gantry_policy_rehearsal currently requires every goal component to be zero")
    else:
        if args.shadow_admission is not None:
            parser.error("--shadow_admission requires --gantry_policy_rehearsal")
        if args.rehearsal_log is not None:
            parser.error("--rehearsal_log requires --gantry_policy_rehearsal")
        if args.acknowledge_contactless_rehearsal:
            parser.error("--acknowledge_contactless_rehearsal requires --gantry_policy_rehearsal")
    for name in ("goal_pos_x", "goal_pos_y", "goal_yaw", "goal_roll", "goal_pitch"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite")
    gantry_entry = args.entry_mode in ("gantry_standup", "native_walkrun_gantry")
    if args.enable_control and gantry_entry and args.duration < _GANTRY_STANDUP_DURATION_S:
        parser.error(f"--entry_mode {args.entry_mode} requires --duration >= {_GANTRY_STANDUP_DURATION_S:.1f}")
    if args.enable_control and gantry_entry and args.effort_scale > 0.5:
        parser.error(f"--entry_mode {args.entry_mode} limits --effort_scale to 0.5")
    if args.enable_control and args.exit_mode == "native_walkrun":
        if args.entry_mode != "native_walkrun_gantry":
            parser.error("--exit_mode native_walkrun requires --entry_mode native_walkrun_gantry")
        if args.duration < _NATIVE_WALKRUN_RETURN_MIN_DURATION_S:
            parser.error(
                f"--exit_mode native_walkrun requires --duration >= {_NATIVE_WALKRUN_RETURN_MIN_DURATION_S:.1f}"
            )
    return args


def _wait_for_feedback(state_buffer: _StateBuffer, timeout_s: float) -> FeedbackSnapshot:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = state_buffer.snapshot()
        if snapshot is not None:
            return snapshot
        time.sleep(0.01)
    raise SafetyFault(f"no valid G1 feedback arrived within {timeout_s:.1f} seconds")


def _verify_remote_abort(state_buffer: _StateBuffer, timeout_s: float = 15.0) -> None:
    print("REMOTE TEST: release all buttons, then press and release B once.")
    deadline = time.monotonic() + timeout_s
    saw_release = False
    saw_press = False
    while time.monotonic() < deadline:
        snapshot = state_buffer.snapshot()
        if snapshot is None:
            time.sleep(0.01)
            continue
        pressed = _remote_b_pressed(snapshot.wireless_remote)
        if not pressed:
            saw_release = True
            if saw_press:
                print("PASS: B-button abort signal was received and released.")
                return
        elif saw_release:
            saw_press = True
        time.sleep(0.01)
    raise SafetyFault("B-button abort was not pressed and released within 15 seconds")


def _wait_for_remote_activation(
    state_buffer: _StateBuffer,
    manifest: HardwareManifest,
    *,
    hold_s: float = _ACTIVATION_HOLD_S,
    timeout_s: float = _ACTIVATION_TIMEOUT_S,
) -> None:
    """Wait for a deliberate L1+R1 hold and subsequent release."""
    print(f"ACTIVATION: release all buttons, then hold L1 + R1 together for {hold_s:.1f} seconds.")
    deadline = time.monotonic() + timeout_s
    hold_started_at: float | None = None
    saw_release = False
    while time.monotonic() < deadline:
        snapshot = state_buffer.snapshot()
        if snapshot is None:
            time.sleep(0.01)
            continue
        now = time.monotonic()
        feedback_fault = _feedback_fault(snapshot, manifest, now)
        if feedback_fault is not None:
            raise SafetyFault(f"activation refused: {feedback_fault}")
        if _remote_b_pressed(snapshot.wireless_remote):
            raise SafetyFault("activation cancelled by B")
        pressed = _remote_activation_pressed(snapshot.wireless_remote)
        if not pressed:
            saw_release = True
            hold_started_at = None
        elif saw_release:
            if hold_started_at is None:
                hold_started_at = now
            elif now - hold_started_at >= hold_s:
                break
        time.sleep(0.01)
    else:
        raise SafetyFault(f"L1+R1 was not held for {hold_s:.1f} seconds within {timeout_s:.0f} seconds")

    print("ACTIVATION ACCEPTED: release L1 + R1.")
    release_deadline = time.monotonic() + _ACTIVATION_RELEASE_TIMEOUT_S
    while time.monotonic() < release_deadline:
        snapshot = state_buffer.snapshot()
        if snapshot is None:
            time.sleep(0.01)
            continue
        if _remote_b_pressed(snapshot.wireless_remote):
            raise SafetyFault("activation cancelled by B")
        if not _remote_activation_pressed(snapshot.wireless_remote):
            print("PASS: L1+R1 activation chord was released.")
            return
        time.sleep(0.01)
    raise SafetyFault("L1+R1 was not released within 5 seconds")


def _verify_rehearsal_buttons_released(
    state_buffer: _StateBuffer,
    manifest: HardwareManifest,
    *,
    timeout_s: float = 10.0,
) -> None:
    """Require a stable neutral A/Y/B state before user-control handover."""
    print(f"REHEARSAL REMOTE: release A, Y, and B for {_REHEARSAL_NEUTRAL_HOLD_S:.1f} seconds.")
    deadline = time.monotonic() + timeout_s
    neutral_started_at: float | None = None
    while time.monotonic() < deadline:
        snapshot = state_buffer.snapshot()
        if snapshot is None:
            time.sleep(0.01)
            continue
        now = time.monotonic()
        feedback_fault = _feedback_fault(snapshot, manifest, now)
        if feedback_fault is not None:
            raise SafetyFault(f"rehearsal remote check refused: {feedback_fault}")
        if _remote_rehearsal_buttons_released(snapshot.wireless_remote):
            if neutral_started_at is None:
                neutral_started_at = now
            elif now - neutral_started_at >= _REHEARSAL_NEUTRAL_HOLD_S:
                print("PASS: A, Y, and B are released.")
                return
        else:
            neutral_started_at = None
        time.sleep(0.01)
    raise SafetyFault("A, Y, and B were not continuously released before rehearsal handover")


def _bridge_native_stand_to_passive(
    loco_client: Any,
    robot: _G1Robot,
    state_buffer: _StateBuffer,
    manifest: HardwareManifest,
    native_fsm_id: int,
    *,
    timeout_s: float = _PASSIVE_BRIDGE_TIMEOUT_S,
) -> None:
    """Bridge native standing to PASSIVE with the user command preloaded."""
    if native_fsm_id not in _NATIVE_STAND_FSM_IDS:
        raise ValueError(f"unsupported native stand FSM ID {native_fsm_id}")
    fsm_code, fsm_id = loco_client.GetFsmId()
    if fsm_code != 0 or fsm_id != native_fsm_id:
        raise SafetyFault(f"native-stand bridge expected FSM ID {native_fsm_id}, got code={fsm_code}, ID={fsm_id}")
    snapshot = state_buffer.snapshot()
    if snapshot is None:
        raise SafetyFault("G1 feedback disappeared before native-stand bridge")
    feedback_fault = _feedback_fault(snapshot, manifest, time.monotonic())
    if feedback_fault is not None:
        raise SafetyFault(f"native-stand bridge refused: {feedback_fault}")
    if _remote_b_pressed(snapshot.wireless_remote):
        raise SafetyFault("native-stand bridge cancelled by B")

    robot.publish_takeover_damping()
    velocity_code = loco_client.SetVelocity(0.0, 0.0, 0.0, 1.0)
    if velocity_code != 0:
        raise SafetyFault(f"zero native velocity request returned code {velocity_code}")
    passive_code = loco_client.SetFsmId(_PASSIVE_FSM_ID)
    if passive_code != 0:
        raise SafetyFault(f"native PASSIVE request returned code {passive_code}")

    deadline = time.monotonic() + timeout_s
    cancel_requested = False
    while time.monotonic() < deadline:
        snapshot = state_buffer.snapshot()
        if snapshot is not None:
            feedback_fault = _feedback_fault(snapshot, manifest, time.monotonic())
            if feedback_fault is not None:
                raise SafetyFault(f"native-stand bridge stopped in PASSIVE: {feedback_fault}")
            if _remote_b_pressed(snapshot.wireless_remote):
                cancel_requested = True
        fsm_code, fsm_id = loco_client.GetFsmId()
        if fsm_code == 0 and fsm_id == _PASSIVE_FSM_ID:
            if cancel_requested:
                raise SafetyFault("native-stand bridge reached PASSIVE but was cancelled by B")
            print("PASS: native stand bridged to PASSIVE; requesting user control immediately.")
            return
        time.sleep(0.005)
    raise SafetyFault(f"native controller did not report PASSIVE within {timeout_s:.1f} seconds")


def _sleep_until(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0.0:
        time.sleep(remaining)


def _restore_internal_control(
    robot: _G1Robot,
    loco_client: Any,
    internal_mode: Any,
    *,
    expected_fsm_id: int | None = None,
    request_fsm_id: int | None = None,
    timeout_s: float = _NATIVE_WALKRUN_RESTORE_TIMEOUT_S,
) -> tuple[bool, int | None, int | None]:
    """Restore a native controller and optionally verify its reported FSM ID."""
    return_code: int | None = None
    reported_fsm_id: int | None = None
    for _ in range(_RESTORE_RETRY_COUNT):
        try:
            robot.set_damping()
            robot.publish(np.zeros(robot._manifest.joint_count), effort_scale=1.0)
        except (SafetyFault, ValueError):
            pass
        try:
            return_code = loco_client.SwitchToInternalCtrl(internal_mode)
        except Exception:
            return_code = None
        if return_code == 0:
            if request_fsm_id is not None:
                try:
                    set_fsm_code = loco_client.SetFsmId(request_fsm_id)
                except Exception:
                    set_fsm_code = None
                if set_fsm_code != 0:
                    return_code = set_fsm_code
                    time.sleep(0.1)
                    continue
            if expected_fsm_id is None:
                return True, return_code, reported_fsm_id
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    fsm_code, reported_fsm_id = loco_client.GetFsmId()
                except Exception:
                    fsm_code, reported_fsm_id = -1, None
                if fsm_code == 0 and reported_fsm_id == expected_fsm_id:
                    return True, return_code, reported_fsm_id
                time.sleep(0.02)
        time.sleep(0.1)
    return False, return_code, reported_fsm_id


def _monitor_native_walkrun_return(
    state_buffer: _StateBuffer,
    loco_client: Any,
    manifest: HardwareManifest,
    *,
    duration_s: float = _NATIVE_WALKRUN_MONITOR_DURATION_S,
) -> str | None:
    """Monitor the native WALKRUN controller immediately after handback."""
    started_at = time.monotonic()
    next_fsm_check = started_at
    while time.monotonic() - started_at < duration_s:
        snapshot = state_buffer.snapshot()
        if snapshot is None:
            return "G1 feedback disappeared"
        now = time.monotonic()
        feedback_fault = _feedback_fault(snapshot, manifest, now)
        if feedback_fault is not None:
            return feedback_fault
        if _remote_b_pressed(snapshot.wireless_remote):
            return "operator pressed B during native WALKRUN handback"
        body_tilt = _body_tilt(snapshot.imu_quaternion)
        if body_tilt > _NATIVE_WALKRUN_MONITOR_MAX_TILT_RAD:
            return f"body tilt {math.degrees(body_tilt):.2f} deg exceeded 15.00 deg"
        maximum_speed = float(np.max(np.abs(snapshot.joint_velocities)))
        if maximum_speed > _NATIVE_WALKRUN_MONITOR_MAX_SPEED_RAD_S:
            return f"joint speed {maximum_speed:.2f} rad/s exceeded {_NATIVE_WALKRUN_MONITOR_MAX_SPEED_RAD_S:.2f} rad/s"
        if now >= next_fsm_check:
            try:
                fsm_code, fsm_id = loco_client.GetFsmId()
            except Exception:
                fsm_code, fsm_id = -1, None
            if fsm_code != 0 or fsm_id != _NATIVE_WALKRUN_FSM_ID:
                return f"native controller reported code={fsm_code}, FSM ID={fsm_id}"
            next_fsm_check = now + 0.1
        time.sleep(0.01)
    return None


def _blend_to_native_handoff(
    robot: _G1Robot,
    fsm: JumpControllerFSM,
    state_buffer: _StateBuffer,
    target_position: np.ndarray,
    effort_scale: float,
) -> None:
    """Blend the active stand command back to a captured native pose."""
    target = _finite_vector(target_position, robot._manifest.joint_count, "native handoff target")
    if np.any(target < robot._manifest.target_position_lower) or np.any(target > robot._manifest.target_position_upper):
        raise SafetyFault("native handoff target exceeds the manifest joint limits")
    start_target = robot.command_base_target
    stiffness = robot.command_stiffness
    damping = robot.command_damping
    if not np.any(stiffness > np.finfo(np.float64).eps):
        raise SafetyFault("native handoff cannot start without active stand stiffness")

    print(
        f"NATIVE HANDOFF: blend to the captured FSM {_NATIVE_WALKRUN_FSM_ID} pose over "
        f"{_NATIVE_HANDOFF_BLEND_DURATION_S:.1f} s, then settle for "
        f"{_NATIVE_HANDOFF_SETTLE_DURATION_S:.1f} s."
    )
    started_at = time.monotonic()
    next_fast = started_at
    total_duration_s = _NATIVE_HANDOFF_BLEND_DURATION_S + _NATIVE_HANDOFF_SETTLE_DURATION_S
    while True:
        snapshot = state_buffer.snapshot()
        if snapshot is None:
            raise SafetyFault("G1 feedback disappeared during native handoff")
        now = time.monotonic()
        if _remote_b_pressed(snapshot.wireless_remote):
            raise SafetyFault("operator pressed B during native handoff")
        feedback_fault = _feedback_fault(snapshot, robot._manifest, now)
        if feedback_fault is not None:
            raise SafetyFault(f"native handoff stopped: {feedback_fault}")
        if state_buffer.crc_errors or state_buffer.invalid_packets:
            raise SafetyFault(
                "native handoff feedback integrity error "
                f"(CRC={state_buffer.crc_errors}, invalid={state_buffer.invalid_packets})"
            )
        elapsed_s = now - started_at
        if elapsed_s >= total_duration_s:
            break
        progress = min(elapsed_s / _NATIVE_HANDOFF_BLEND_DURATION_S, 1.0)
        blend = progress * progress * progress * (10.0 + progress * (-15.0 + 6.0 * progress))
        command_target = start_target + blend * (target - start_target)
        robot.command_joint_position_target(command_target, stiffness, damping)
        balance_offset = fsm.update_balance(_FAST_DT)
        robot.publish(balance_offset, effort_scale)
        next_fast += _FAST_DT
        if now - next_fast > _MAX_CONTROL_GAP_S:
            raise SafetyFault("500 Hz native-handoff command schedule fell behind by more than 20 ms")
        _sleep_until(next_fast)

    snapshot = state_buffer.snapshot()
    if snapshot is None:
        raise SafetyFault("G1 feedback disappeared after native handoff")
    position_error = np.abs(snapshot.joint_positions - target)
    worst_index = int(np.argmax(position_error))
    if position_error[worst_index] > _NATIVE_HANDOFF_MAX_POSITION_ERROR_RAD:
        raise SafetyFault(
            f"native handoff {robot._manifest.joint_names[worst_index]} position error is "
            f"{position_error[worst_index]:.3f} rad (limit {_NATIVE_HANDOFF_MAX_POSITION_ERROR_RAD:.2f} rad)"
        )
    print(
        f"PASS: captured native pose reached; maximum joint error={position_error[worst_index]:.3f} rad "
        f"({robot._manifest.joint_names[worst_index]})."
    )


def _run_control(  # noqa: C901
    robot: _G1Robot,
    operator: _StandOperator,
    fsm: JumpControllerFSM,
    state_buffer: _StateBuffer,
    loco_client: Any,
    internal_passive_mode: Any,
    duration_s: float,
    effort_scale: float,
    *,
    success_internal_mode: Any | None = None,
    success_internal_fsm_id: int | None = None,
    success_handoff_position: np.ndarray | None = None,
    recalibrate_balance_after_user_switch: bool = False,
    gantry_policy_rehearsal: bool = False,
    rehearsal_recorder: _RehearsalRecorder | None = None,
) -> tuple[bool, str]:
    if gantry_policy_rehearsal != (rehearsal_recorder is not None):
        raise ValueError("gantry_policy_rehearsal and rehearsal_recorder must be enabled together")
    snapshot = state_buffer.snapshot()
    if snapshot is None:
        raise SafetyFault("G1 feedback disappeared before handover")
    feedback_fault = _feedback_fault(snapshot, robot._manifest, time.monotonic())
    if feedback_fault is not None:
        raise SafetyFault(feedback_fault)
    if _remote_b_pressed(snapshot.wireless_remote):
        raise SafetyFault("B must be released before user-control handover")
    fsm_code, fsm_id = loco_client.GetFsmId()
    if fsm_code != 0 or fsm_id != _PASSIVE_FSM_ID:
        raise SafetyFault(f"native control left PASSIVE before handover (code={fsm_code}, FSM ID={fsm_id})")

    robot.publish_takeover_damping()
    if rehearsal_recorder is not None:
        rehearsal_recorder.record(robot, fsm, np.zeros(robot._manifest.joint_count))
    switch_attempted = False
    control_started_at: float | None = None
    success = True
    reason = "requested stand duration completed"
    try:
        switch_attempted = True
        switch_code = loco_client.SwitchToUserCtrl()
        if switch_code != 0:
            raise SafetyFault(f"SwitchToUserCtrl returned code {switch_code}")
        if recalibrate_balance_after_user_switch:
            calibration_snapshot = state_buffer.snapshot()
            if calibration_snapshot is None:
                raise SafetyFault("G1 feedback disappeared during takeover balance calibration")
            feedback_fault = _feedback_fault(calibration_snapshot, robot._manifest, time.monotonic())
            if feedback_fault is not None:
                raise SafetyFault(f"takeover balance calibration refused: {feedback_fault}")
            target_roll, target_pitch = quaternion_to_roll_pitch(calibration_snapshot.imu_quaternion)
            fsm.set_balance_target_attitude(target_roll, target_pitch)
            print(
                f"TAKEOVER BALANCE CALIBRATION: roll={math.degrees(target_roll):+.2f} deg, "
                f"pitch={math.degrees(target_pitch):+.2f} deg."
            )

        update_operator = getattr(operator, "update", None)
        if callable(update_operator):
            update_operator(snapshot.wireless_remote)
        else:
            operator.abort = _remote_b_pressed(snapshot.wireless_remote)
        if (
            operator.abort
            or bool(getattr(operator, "request_start", False))
            or bool(getattr(operator, "confirm", False))
        ):
            raise SafetyFault("A, Y, and B must be released when user control becomes active")
        print("USER CONTROL ACTIVE: B aborts; Ctrl+C also returns to native PASSIVE.")
        if gantry_policy_rehearsal:
            print(
                f"GANTRY REHEARSAL: do not press A or Y during the {_REHEARSAL_STABILIZATION_S:.1f} s "
                "stand stabilization; wait for REHEARSAL READY."
            )
        fsm.enable()
        fsm.step()
        control_started_at = time.monotonic()
        next_fast = control_started_at
        next_policy = control_started_at + fsm.policy_dt
        rehearsal_started = False
        rehearsal_ready = False
        while True:
            snapshot = state_buffer.snapshot()
            if snapshot is None:
                raise SafetyFault("G1 feedback disappeared")
            now = time.monotonic()
            if callable(update_operator):
                update_operator(snapshot.wireless_remote)
            else:
                operator.abort = _remote_b_pressed(snapshot.wireless_remote)
            if operator.abort:
                fsm.step()
                success = False
                reason = "operator pressed B"
                break
            if gantry_policy_rehearsal:
                if not rehearsal_ready and now - control_started_at >= _REHEARSAL_STABILIZATION_S:
                    rehearsal_ready = True
                    print("REHEARSAL READY: tap and release A; wait for ARMED before tapping Y.")
                if not rehearsal_ready and (operator.request_start or operator.confirm):
                    raise SafetyFault("A or Y was pressed before REHEARSAL READY")
                if operator.confirm and fsm.state is not JumpControllerState.ARMED and not rehearsal_started:
                    raise SafetyFault("Y was pressed before the FSM reported ARMED")
            feedback_fault = _feedback_fault(snapshot, robot._manifest, now)
            if feedback_fault is not None:
                raise SafetyFault(feedback_fault)
            if state_buffer.crc_errors or state_buffer.invalid_packets:
                raise SafetyFault(
                    f"feedback integrity error (CRC={state_buffer.crc_errors}, invalid={state_buffer.invalid_packets})"
                )
            if now - control_started_at >= duration_s:
                if gantry_policy_rehearsal:
                    raise SafetyFault(f"gantry rehearsal timed out in FSM state {fsm.state.value}")
                break
            if now >= next_policy:
                state_before = fsm.state
                policy_step_started_at = time.monotonic()
                fsm.step()
                policy_step_elapsed_s = time.monotonic() - policy_step_started_at
                if policy_step_elapsed_s > fsm.policy_dt:
                    raise SafetyFault(
                        f"50 Hz FSM step took {policy_step_elapsed_s * 1000.0:.3f} ms and missed its deadline"
                    )
                next_policy += fsm.policy_dt
                if time.monotonic() - next_policy > fsm.policy_dt:
                    raise SafetyFault("50 Hz FSM deadline was missed")
                if fsm.state is not state_before:
                    print(f"FSM: {state_before.value} -> {fsm.state.value}: {fsm.last_report}")
                    if gantry_policy_rehearsal and fsm.state is JumpControllerState.ARMED:
                        print(f"CONFIRM NOW: tap and release Y within {_REHEARSAL_ARMED_TIMEOUT_S:.0f} seconds.")
                if not gantry_policy_rehearsal and fsm.state is not JumpControllerState.STAND:
                    raise SafetyFault(f"unexpected FSM state {fsm.state.value}")
                if gantry_policy_rehearsal:
                    if fsm.state is JumpControllerState.JUMP:
                        rehearsal_started = True
                    if fsm.state in (JumpControllerState.DAMPING, JumpControllerState.FAULT):
                        success = False
                        reason = fsm.last_report or f"gantry rehearsal entered {fsm.state.value}"
                        break
                    if state_before is JumpControllerState.ARMED and fsm.state is JumpControllerState.STAND:
                        success = False
                        reason = fsm.last_report or "gantry rehearsal confirmation window expired"
                        break
                    if rehearsal_started and fsm.state is JumpControllerState.STAND:
                        reason = "one gantry policy rehearsal completed and settled"
                        break
            balance_offset = fsm.update_balance(_FAST_DT)
            robot.publish(balance_offset, effort_scale)
            if rehearsal_recorder is not None:
                rehearsal_recorder.record(robot, fsm, balance_offset)
            next_fast += _FAST_DT
            if now - next_fast > _MAX_CONTROL_GAP_S:
                raise SafetyFault("500 Hz command schedule fell behind by more than 20 ms")
            _sleep_until(next_fast)
        if success and success_internal_mode is not None:
            if success_handoff_position is None:
                raise SafetyFault("native WALKRUN handback pose was not captured")
            _blend_to_native_handoff(robot, fsm, state_buffer, success_handoff_position, effort_scale)
    except KeyboardInterrupt:
        success = False
        reason = "operator pressed Ctrl+C"
    except SafetyFault as exc:
        success = False
        elapsed = "" if control_started_at is None else f" after {time.monotonic() - control_started_at:.3f} s"
        reason = f"{exc}{elapsed}"
        fsm.report_fault(reason)
    except Exception as exc:
        success = False
        reason = f"unexpected {type(exc).__name__}: {exc}"
    finally:
        restored = not switch_attempted
        return_code: int | None = None
        reported_fsm_id: int | None = None
        restore_success_mode = switch_attempted and success and success_internal_mode is not None
        if restore_success_mode:
            final_snapshot = state_buffer.snapshot()
            handoff_fault = (
                "G1 feedback disappeared"
                if final_snapshot is None
                else _native_walkrun_handoff_fault(final_snapshot, robot._manifest, time.monotonic())
            )
            if handoff_fault is not None:
                success = False
                reason += f"; native WALKRUN handback refused: {handoff_fault}"
                restore_success_mode = False
        if restore_success_mode:
            try:
                velocity_code = loco_client.SetVelocity(0.0, 0.0, 0.0, 1.0)
            except Exception:
                velocity_code = None
            if velocity_code != 0:
                success = False
                reason += f"; native zero-velocity request failed (code={velocity_code})"
                restore_success_mode = False
        if switch_attempted:
            restore_mode = success_internal_mode if restore_success_mode else internal_passive_mode
            expected_fsm_id = success_internal_fsm_id if restore_success_mode else _PASSIVE_FSM_ID
            restored, return_code, reported_fsm_id = _restore_internal_control(
                robot,
                loco_client,
                restore_mode,
                expected_fsm_id=expected_fsm_id,
                request_fsm_id=None if restore_success_mode else _PASSIVE_FSM_ID,
            )
        if not restored and restore_success_mode:
            success = False
            reason += f"; native WALKRUN handback was not verified (last code={return_code}, FSM ID={reported_fsm_id})"
            restored, return_code, _ = _restore_internal_control(
                robot,
                loco_client,
                internal_passive_mode,
                expected_fsm_id=_PASSIVE_FSM_ID,
                request_fsm_id=_PASSIVE_FSM_ID,
            )
            restore_success_mode = False
        if restored and restore_success_mode:
            monitor_fault = _monitor_native_walkrun_return(state_buffer, loco_client, robot._manifest)
            if monitor_fault is not None:
                success = False
                reason += f"; native WALKRUN handback stopped: {monitor_fault}"
                restored, return_code, _ = _restore_internal_control(
                    robot,
                    loco_client,
                    internal_passive_mode,
                    expected_fsm_id=_PASSIVE_FSM_ID,
                    request_fsm_id=_PASSIVE_FSM_ID,
                )
                restore_success_mode = False
        if not restored:
            success = False
            reason += f"; CRITICAL: could not restore native PASSIVE (last code={return_code})"
        elif restore_success_mode:
            print(
                f"Native WALKRUN control restored (FSM ID {reported_fsm_id}) and monitored for "
                f"{_NATIVE_WALKRUN_MONITOR_DURATION_S:.1f} s."
            )
        else:
            print("Native PASSIVE/damping control restored.")
    return success, reason


def main() -> int:  # noqa: C901
    """Run read-only checks, stand control, or the guarded gantry rehearsal."""
    args = _parse_args()
    available_interfaces = {name for _, name in socket.if_nameindex()}
    if args.network_interface not in available_interfaces:
        raise ValueError(
            f"Network interface {args.network_interface!r} does not exist; available: {sorted(available_interfaces)}"
        )
    manifest_path = args.manifest.resolve()
    manifest = _load_hardware_manifest(manifest_path)
    policy_requested = args.check_policy or args.shadow_policy or args.gantry_policy_rehearsal
    policy_path = (
        (args.policy.resolve() if args.policy is not None else manifest_path.with_name("policy.onnx"))
        if policy_requested
        else None
    )
    _verify_validated_bundle(manifest_path, args.validation_record.resolve(), policy_path)
    rehearsal_policy = None
    rehearsal_recorder = None
    rehearsal_goal = JumpGoal(
        args.goal_pos_x,
        args.goal_pos_y,
        args.goal_yaw,
        roll=args.goal_roll,
        pitch=args.goal_pitch,
    )
    if args.gantry_policy_rehearsal:
        _verify_shadow_admission(args.shadow_admission.resolve(), manifest_path, policy_path, manifest)
        rehearsal_log_path = args.rehearsal_log.resolve()
        rehearsal_recorder = _RehearsalRecorder(
            rehearsal_log_path,
            manifest_path,
            policy_path,
            args.shadow_admission,
            manifest,
            rehearsal_goal,
            args.effort_scale,
        )
        rehearsal_policy = OnnxPolicy(policy_path, _OBSERVATION_DIM, manifest.joint_count)
        rehearsal_policy.warm_up()

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as exc:
        raise RuntimeError("Read-only preflight requires the Unitree SDK2 Python environment") from exc

    if args.gantry_policy_rehearsal:
        print("G1 CONTACTLESS GANTRY POLICY REHEARSAL: ground jumping remains disabled.")
    else:
        print("G1 DEPLOYMENT PREFLIGHT: motor control remains stand-only.")
    print(
        "Default, --check_policy, and --shadow_policy modes are read-only; "
        "--enable_control is required to create a publisher."
    )
    ChannelFactoryInitialize(0, args.network_interface)
    crc = CRC()
    state_buffer = _StateBuffer(manifest, crc)
    subscriber = ChannelSubscriber(_LOW_STATE_TOPIC, LowState_)
    subscriber.Init(state_buffer.update, 1)
    publisher = None
    try:
        first_snapshot = _wait_for_feedback(state_buffer, 5.0)
        preflight_start = time.monotonic()
        while time.monotonic() - preflight_start < 2.0:
            time.sleep(0.01)
        snapshot = _wait_for_feedback(state_buffer, 1.0)
        feedback_fault = _feedback_fault(snapshot, manifest, time.monotonic())
        if feedback_fault is not None:
            raise SafetyFault(feedback_fault)
        if state_buffer.crc_errors or state_buffer.invalid_packets:
            raise SafetyFault(
                f"preflight feedback errors (CRC={state_buffer.crc_errors}, invalid={state_buffer.invalid_packets})"
            )
        if np.max(np.abs(snapshot.joint_velocities)) > 0.5:
            raise SafetyFault("robot must be motionless below 0.5 rad/s before user-control handover")
        print(
            f"PASS: feedback tick={snapshot.tick}, packets={state_buffer.valid_packets}, "
            f"tilt={math.degrees(_body_tilt(snapshot.imu_quaternion)):.2f} deg, "
            f"max_speed={np.max(np.abs(snapshot.joint_velocities)):.3f} rad/s, "
            f"max_temperature={snapshot.maximum_temperature_c} C, mode_machine={snapshot.mode_machine}."
        )
        if snapshot.mode_machine != first_snapshot.mode_machine:
            raise SafetyFault("mode_machine changed during preflight")
        if args.check_policy:
            report = _evaluate_shadow_policy(
                manifest_path,
                policy_path,
                manifest,
                snapshot,
                goal_pos_x=args.goal_pos_x,
                goal_pos_y=args.goal_pos_y,
                goal_yaw=args.goal_yaw,
                goal_roll=args.goal_roll,
                goal_pitch=args.goal_pitch,
                effort_scale=args.effort_scale,
            )
            print(
                f"READ-ONLY POLICY PASS: sha256={report.policy_sha256}, "
                f"latency median/p99/max={report.inference_median_ms:.3f}/"
                f"{report.inference_p99_ms:.3f}/{report.inference_maximum_ms:.3f} ms."
            )
            print(
                f"Step-0 raw action=[{report.raw_action_minimum:+.4f}, {report.raw_action_maximum:+.4f}], "
                f"max projected target delta={report.maximum_target_delta_rad:.4f} rad, "
                f"max torque={100.0 * report.maximum_torque_fraction:.1f}% "
                f"({report.maximum_torque_joint})."
            )
            print("READ-ONLY PASS: no command publisher or locomotion client was created.")
            return 0
        if args.shadow_policy:
            report = _run_shadow_policy_episode(
                manifest_path,
                policy_path,
                manifest,
                state_buffer,
                args.shadow_log,
                goal_pos_x=args.goal_pos_x,
                goal_pos_y=args.goal_pos_y,
                goal_yaw=args.goal_yaw,
                goal_roll=args.goal_roll,
                goal_pitch=args.goal_pitch,
                effort_scale=args.effort_scale,
            )
            print(
                f"READ-ONLY SHADOW PASS: {report.steps} policy steps in {report.elapsed_s:.3f} s, "
                f"sha256={report.policy_sha256}."
            )
            print(
                f"Inference median/p99/max={report.inference_median_ms:.3f}/"
                f"{report.inference_p99_ms:.3f}/{report.inference_maximum_ms:.3f} ms; "
                f"raw action=[{report.raw_action_minimum:+.4f}, {report.raw_action_maximum:+.4f}]."
            )
            print(
                f"Max projected target delta={report.maximum_target_delta_rad:.4f} rad, "
                f"projected torque={100.0 * report.maximum_torque_fraction:.1f}% "
                f"({report.maximum_torque_joint}), unprojected peak="
                f"{100.0 * report.maximum_unprojected_torque_fraction:.1f}%, "
                f"projection active on {report.torque_projection_steps}/{report.steps} steps."
            )
            print(
                f"Measured max tilt={report.maximum_body_tilt_deg:.2f} deg, "
                f"max joint speed={report.maximum_joint_speed_rad_s:.3f} rad/s, "
                f"max torque estimate={100.0 * report.maximum_measured_torque_fraction:.1f}% "
                f"({report.maximum_measured_torque_joint}); log={report.log_path}."
            )
            print("READ-ONLY PASS: no command publisher or locomotion client was created.")
            return 0
        if not args.enable_control and not args.query_fsm:
            print("READ-ONLY PASS: no command publisher or locomotion client was created.")
            return 0

        try:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        except ImportError as exc:
            raise RuntimeError("Locomotion FSM queries require the matching Unitree G1 SDK") from exc
        loco_client = LocoClient()
        loco_client.SetTimeout(5.0)
        loco_client.Init()
        fsm_code, fsm_id = loco_client.GetFsmId()
        if fsm_code != 0:
            raise SafetyFault(f"GetFsmId returned code {fsm_code}")
        if args.query_fsm:
            print(f"READ-ONLY FSM: native locomotion FSM ID={fsm_id}.")
        if not args.enable_control:
            print("READ-ONLY PASS: no command publisher or state-changing motor-control RPC was created.")
            return 0

        try:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            from unitree_sdk2py.g1.loco.g1_loco_api import InternalFsmMode
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        except ImportError as exc:
            raise RuntimeError("Stand control requires the matching Unitree G1 command SDK") from exc

        if args.entry_mode in ("passive", "gantry_standup") and fsm_id != _PASSIVE_FSM_ID:
            raise SafetyFault(f"native FSM ID is {fsm_id}, expected PASSIVE ID {_PASSIVE_FSM_ID}")
        if args.entry_mode == "native_walkrun_gantry" and fsm_id != _NATIVE_WALKRUN_FSM_ID:
            raise SafetyFault(f"native FSM ID is {fsm_id}, expected stationary WALKRUN ID {_NATIVE_WALKRUN_FSM_ID}")
        if args.entry_mode == "native_stand" and fsm_id not in _NATIVE_STAND_FSM_IDS:
            raise SafetyFault(
                f"native FSM ID is {fsm_id}, expected one of the verified native stand IDs "
                f"{sorted(_NATIVE_STAND_FSM_IDS)}"
            )
        if args.entry_mode == "native_stand":
            print(
                f"ENTRY MODE: native_stand (FSM ID {fsm_id}); G1 must already be motionless in balanced native stand."
            )
        elif args.entry_mode == "gantry_standup":
            print(
                "ENTRY MODE: gantry_standup; the harness must carry most of the torso weight and "
                "mechanically prevent pelvis roll/pitch. A loose fall-arrest cable is not sufficient."
            )
        elif args.entry_mode == "native_walkrun_gantry":
            print(
                f"ENTRY MODE: native_walkrun_gantry (FSM ID {fsm_id}); native velocity must be zero, "
                "and the harness must carry about 80% of body weight (roughly 270 N / 27 kgf) while "
                "preventing pelvis fall or rotation."
            )
        if args.exit_mode == "native_walkrun":
            print(
                "EXIT MODE: native_walkrun is requested only after successful completion; "
                "B or any fault returns to PASSIVE/damping."
            )

        try:
            pcm_data, cue_duration_s = _read_audio_cue(args.audio_cue.resolve())
        except ValueError as exc:
            raise SafetyFault(f"audio cue refused: {exc}") from exc
        gantry_entry = args.entry_mode in ("gantry_standup", "native_walkrun_gantry")
        leg_error_limit = _GANTRY_STAND_ENTRY_LEG_ERROR_LIMIT_RAD if gantry_entry else _STAND_ENTRY_LEG_ERROR_LIMIT_RAD
        entry_snapshot = state_buffer.snapshot()
        if entry_snapshot is None:
            raise SafetyFault("G1 feedback disappeared before stand-entry admission")
        pose_fault = _stand_entry_pose_fault(entry_snapshot, manifest, leg_error_limit)
        if pose_fault is not None:
            raise SafetyFault(pose_fault)
        _verify_remote_abort(state_buffer)
        if args.gantry_policy_rehearsal:
            print(
                "FINAL CHECK: load-bearing gantry rigidly constraining the pelvis, feet unable to touch "
                "the floor or any structure throughout full leg motion, area clear, hardware emergency "
                "stop staffed, and a second person holding the remote."
            )
        elif gantry_entry:
            print(
                "FINAL CHECK: gantry carrying most of the torso weight, pelvis unable to fall or rotate, "
                "feet near the floor, area clear, and another person ready."
            )
        else:
            print(
                "FINAL CHECK: gantry supporting G1, feet near the floor, area clear, "
                "and another person ready if possible."
            )
        _wait_for_remote_activation(state_buffer, manifest)
        if args.gantry_policy_rehearsal:
            _verify_rehearsal_buttons_released(state_buffer, manifest)

        try:
            audio_client = AudioClient()
            audio_client.SetTimeout(5.0)
            audio_client.Init()
            _play_audio_cue(audio_client, pcm_data, cue_duration_s)
        except SafetyFault:
            raise
        except Exception as exc:
            raise SafetyFault(f"audio cue failed before handover: {type(exc).__name__}: {exc}") from exc

        publisher = ChannelPublisher(_USER_COMMAND_TOPIC, LowCmd_)
        publisher.Init()
        low_command = unitree_hg_msg_dds__LowCmd_()
        robot = _G1Robot(manifest, state_buffer, publisher, low_command, crc)
        operator = _GantryRehearsalOperator(rehearsal_goal) if args.gantry_policy_rehearsal else _StandOperator()
        calibration_snapshot = state_buffer.snapshot()
        if calibration_snapshot is None:
            raise SafetyFault("G1 feedback disappeared before stand calibration")
        pose_fault = _stand_entry_pose_fault(calibration_snapshot, manifest, leg_error_limit)
        if pose_fault is not None:
            raise SafetyFault(pose_fault)
        target_roll, target_pitch = quaternion_to_roll_pitch(calibration_snapshot.imu_quaternion)
        native_handoff_position = (
            calibration_snapshot.joint_positions.copy() if args.entry_mode == "native_walkrun_gantry" else None
        )
        balance_config = BalanceControllerConfig(
            target_roll=target_roll,
            target_pitch=target_pitch,
            integral_enabled=False,
            initial_roll_integral=0.0,
            initial_pitch_integral=0.0,
        )
        stand_entry_duration_s = _GANTRY_STANDUP_DURATION_S if gantry_entry else 1.0
        if args.entry_mode == "native_walkrun_gantry":
            print(
                f"NATIVE POSE CAPTURE: roll={math.degrees(target_roll):+.2f} deg, "
                f"pitch={math.degrees(target_pitch):+.2f} deg; balance target will be recalibrated "
                "after user-control ownership is active."
            )
        else:
            print(
                f"STAND CALIBRATION: fixed target roll={math.degrees(target_roll):+.2f} deg, "
                f"pitch={math.degrees(target_pitch):+.2f} deg; blend measured joints to stand over "
                f"{stand_entry_duration_s:.1f} s."
            )
        fsm = JumpControllerFSM(
            manifest_path,
            robot,
            operator,
            rehearsal_policy if args.gantry_policy_rehearsal else _InactivePolicy(),
            stand_gains=StandGainConfig(ankle_stiffness=80.0, ankle_damping=7.0),
            config=JumpControllerConfig(
                stand_entry_duration_s=stand_entry_duration_s,
                stand_hold_measured_pose=False,
                armed_timeout_s=(
                    _REHEARSAL_ARMED_TIMEOUT_S
                    if args.gantry_policy_rehearsal
                    else JumpControllerConfig().armed_timeout_s
                ),
                contact_safety_mode=(
                    JumpControllerConfig.ContactSafetyMode.GANTRY_REHEARSAL
                    if args.gantry_policy_rehearsal
                    else JumpControllerConfig.ContactSafetyMode.MEASURED
                ),
            ),
            balance_config=balance_config,
        )
        if args.entry_mode in ("native_stand", "native_walkrun_gantry"):
            _bridge_native_stand_to_passive(loco_client, robot, state_buffer, manifest, fsm_id)
        success, reason = _run_control(
            robot,
            operator,
            fsm,
            state_buffer,
            loco_client,
            InternalFsmMode.PASSIVE,
            args.duration,
            args.effort_scale,
            success_internal_mode=(InternalFsmMode.WALKRUN if args.exit_mode == "native_walkrun" else None),
            success_internal_fsm_id=(_NATIVE_WALKRUN_FSM_ID if args.exit_mode == "native_walkrun" else None),
            success_handoff_position=native_handoff_position,
            recalibrate_balance_after_user_switch=args.entry_mode == "native_walkrun_gantry",
            gantry_policy_rehearsal=args.gantry_policy_rehearsal,
            rehearsal_recorder=rehearsal_recorder,
        )
        if rehearsal_recorder is not None:
            try:
                rehearsal_recorder.write(success, reason, state_buffer)
                print(
                    f"REHEARSAL AUDIT: {rehearsal_recorder.sample_count} accepted commands written to "
                    f"{rehearsal_recorder.log_path}."
                )
            except ValueError as exc:
                success = False
                reason += f"; rehearsal audit failed: {exc}"
        peak_index = int(np.argmax(robot.maximum_estimated_torque))
        print(
            f"Peak estimated command torque: {robot.maximum_estimated_torque[peak_index]:.2f} N m "
            f"({manifest.joint_names[peak_index]})."
        )
        print(
            f"Peak components: position={robot.peak_position_torque[peak_index]:+.2f} N m, "
            f"damping={robot.peak_damping_torque[peak_index]:+.2f} N m, "
            f"error={robot.peak_position_error[peak_index]:+.4f} rad, "
            f"velocity={robot.peak_joint_velocity[peak_index]:+.3f} rad/s."
        )
        print(f"{'PASS' if success else 'STOP'}: {reason}.")
        return 0 if success else 2
    except SafetyFault as exc:
        if rehearsal_recorder is not None and not rehearsal_recorder.log_path.exists():
            try:
                rehearsal_recorder.write(False, str(exc), state_buffer)
                print(f"REHEARSAL AUDIT: refusal written to {rehearsal_recorder.log_path}.")
            except ValueError as audit_exc:
                print(f"REHEARSAL AUDIT FAILED: {audit_exc}")
        print(f"REFUSED/STOPPED: {exc}")
        return 2
    finally:
        if publisher is not None:
            publisher.Close()
        subscriber.Close()


if __name__ == "__main__":
    raise SystemExit(main())

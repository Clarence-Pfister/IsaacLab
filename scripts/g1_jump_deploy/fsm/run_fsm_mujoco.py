# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run reproducible jump-FSM scenarios through the MuJoCo robot backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.g1_jump_deploy.control.balance import BalanceControllerConfig, quaternion_to_roll_pitch  # noqa: E402
from scripts.g1_jump_deploy.fsm.jump_fsm import (  # noqa: E402
    JumpControllerConfig,
    JumpControllerFSM,
    JumpControllerState,
    JumpGoal,
    OperatorInterface,
    RobotInterface,
    StandGainConfig,
)
from scripts.g1_jump_deploy.fsm.mujoco_backend import (  # noqa: E402
    MujocoRobot,
    OperatorTimelineEntry,
    ScriptedOperator,
)
from scripts.g1_jump_deploy.runtime import OnnxPolicy  # noqa: E402

_DEFAULT_MANIFEST = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"
_DEFAULT_MODEL = _REPO_ROOT / "data_storage" / "g1_23dof_holo_compat.xml"
_DEFAULT_OVERLAY = _SCRIPT_DIR.parent / "mujoco" / "model_overlay.xml"
_SCENARIOS = ("stand", "nominal", "repeat", "abort_early", "abort_late", "reject")
_START_TIME_S = 0.5
_CONFIRM_TIME_S = 2.7
_EARLY_ABORT_TIME_S = 2.9
_CONTACTLESS_REHEARSAL_ARMED_TIMEOUT_S = 15.0
_CONTACTLESS_REHEARSAL_TARGET_RATE_LIMIT_RAD_S = 1.2
_UNMEASURED_GROUND_CONFIRM_TIME_S = 2.8
_UNMEASURED_GROUND_POLICY_PREPARE_DURATION_S = 0.0
_UNMEASURED_GROUND_STAND_STIFFNESS = 200.0
_UNMEASURED_GROUND_STAND_DAMPING = 5.0
_REPEAT_STAND_DWELL_S = 0.5
_REPEAT_CONFIRM_DWELL_S = 0.35
_CONTACT_THRESHOLD_N = 20.0
_BALANCE_CONFIG = BalanceControllerConfig()
# The replay uses a 75% hardware effort scale. A 90% replay threshold therefore
# stays below 67.5% of the manifest motor limit while allowing contact transients.
_HARDWARE_MARGIN_MAX_COMMAND_EFFORT_RATIO = 0.9
_HARDWARE_MARGIN_MAX_TILT_RAD = math.radians(15.0)
_HARDWARE_MARGIN_MAX_JOINT_SPEED_RAD_S = 2.5
_HARDWARE_MARGIN_MIN_PELVIS_HEIGHT_M = 0.65
_HARDWARE_MARGIN_MAX_TARGET_ERROR_RAD = 0.2
_COMMAND_TRACKING_MAX_PLANAR_ERROR_M = 0.08
_COMMAND_TRACKING_MIN_DIRECTED_PROGRESS_M = 0.02
_PREJUMP_HOLD_MAX_TILT_RAD = math.radians(30.0)
_PREJUMP_HOLD_MIN_PELVIS_HEIGHT_M = 0.5
_PREJUMP_HOLD_STATES = frozenset(
    (JumpControllerState.STAND.value, JumpControllerState.GOTO_START.value, JumpControllerState.ARMED.value)
)


@dataclass(frozen=True)
class InitialState:
    """Measured G1 state used to initialize a MuJoCo robustness replay."""

    label: str
    joint_positions: np.ndarray
    root_quaternion_wxyz: np.ndarray


def _load_initial_state(path: Path, expected_joint_names: tuple[str, ...]) -> InitialState:
    try:
        with path.resolve().open(encoding="utf-8") as stream:
            raw = json.load(stream)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read initial-state capture: {path.resolve()}.") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Initial-state capture must be a schema-version-1 object.")
    label = raw.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("Initial-state capture label must be a non-empty string.")
    joint_names = raw.get("joint_names")
    if not isinstance(joint_names, list) or tuple(joint_names) != expected_joint_names:
        raise ValueError("Initial-state joint_names must exactly match manifest order.")
    try:
        joint_positions = np.asarray(raw.get("joint_positions_rad"), dtype=np.float64)
        quaternion = np.asarray(raw.get("root_quaternion_wxyz"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Initial-state pose fields must contain numeric values.") from exc
    if joint_positions.shape != (len(expected_joint_names),) or not np.all(np.isfinite(joint_positions)):
        raise ValueError(f"Initial-state joint_positions_rad must contain {len(expected_joint_names)} finite values.")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("Initial-state root_quaternion_wxyz must contain four finite values.")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= np.finfo(np.float64).eps:
        raise ValueError("Initial-state root_quaternion_wxyz must be non-zero.")
    return InitialState(label, joint_positions, quaternion / quaternion_norm)


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right
    return np.asarray(
        (
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ),
        dtype=np.float64,
    )


def _apply_attitude_offset(quaternion_wxyz: np.ndarray, roll_rad: float, pitch_rad: float) -> np.ndarray:
    roll_half = 0.5 * roll_rad
    pitch_half = 0.5 * pitch_rad
    roll_quaternion = np.asarray((math.cos(roll_half), math.sin(roll_half), 0.0, 0.0), dtype=np.float64)
    pitch_quaternion = np.asarray((math.cos(pitch_half), 0.0, math.sin(pitch_half), 0.0), dtype=np.float64)
    offset = _quaternion_multiply_wxyz(pitch_quaternion, roll_quaternion)
    result = _quaternion_multiply_wxyz(offset, quaternion_wxyz)
    return result / np.linalg.norm(result)


class InactivePolicy:
    """Retain zero diagnostics while forbidding inference in stand-only mode."""

    def __init__(self, observation_dim: int, action_dim: int):
        self.last_observation = np.zeros(observation_dim, dtype=np.float32)
        self.last_action = np.zeros(action_dim, dtype=np.float64)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        raise RuntimeError("The stand scenario must not invoke the jump policy.")


@dataclass
class StateInterval:
    state: JumpControllerState
    entry_time_s: float
    exit_time_s: float | None
    report: str | None


class StateTimeline:
    """Recover entry and exit times, including zero-duration transient states."""

    def __init__(self, fsm: JumpControllerFSM):
        self._history_index = 1
        self.intervals = [StateInterval(fsm.transition_history[0], 0.0, None, fsm.last_report)]

    def observe(self, fsm: JumpControllerFSM, time_s: float) -> None:
        while self._history_index < len(fsm.transition_history):
            self.intervals[-1].exit_time_s = time_s
            state = fsm.transition_history[self._history_index]
            self.intervals.append(StateInterval(state, time_s, None, fsm.last_report))
            self._history_index += 1

    def finish(self, time_s: float) -> None:
        if self.intervals[-1].exit_time_s is None:
            self.intervals[-1].exit_time_s = time_s


@dataclass(frozen=True)
class ControlTick:
    time_s: float
    state_before: JumpControllerState
    state_after: JumpControllerState
    episode_step_before: int
    episode_step_after: int
    request_start: bool
    confirm: bool
    abort: bool
    duration_s: float


class FsmLogger:
    """Collect one initial and one post-physics row per 500 Hz step."""

    def __init__(
        self,
        output_path: Path,
        metadata: dict[str, Any],
        balance_config: BalanceControllerConfig = _BALANCE_CONFIG,
    ):
        self.output_path = output_path.resolve()
        self.metadata = metadata
        self._balance_config = balance_config
        self.values: dict[str, list[np.ndarray | float | int | bool | str]] = {
            "time": [],
            "fsm_state": [],
            "fsm_episode_step": [],
            "fsm_abort_latched": [],
            "fsm_policy_prepared": [],
            "fsm_policy_stand_active": [],
            "fsm_last_report": [],
            "qpos": [],
            "qvel": [],
            "q_target": [],
            "stiffness": [],
            "damping": [],
            "applied_tau": [],
            "pelvis_pose": [],
            "pelvis_velocity": [],
            "tilt": [],
            "balance_attitude_target": [],
            "target_relative_tilt_error": [],
            "foot_contact_forces": [],
            "foot_contact_force_vectors": [],
            "joint_limit_violations": [],
            "feedback_stale": [],
            "control_deadline_missed": [],
            "control_duration": [],
            "operator_request_start": [],
            "operator_confirm": [],
            "operator_abort": [],
            "observation": [],
            "raw_action": [],
        }

    def append(
        self,
        robot: MujocoRobot,
        operator: ScriptedOperator,
        fsm: JumpControllerFSM,
        policy: OnnxPolicy,
    ) -> None:
        quaternion = robot.imu_quaternion
        position = robot.odometry_position
        row: dict[str, np.ndarray | float | int | bool | str] = {
            "time": float(robot.data.time),
            "fsm_state": fsm.state.value,
            "fsm_episode_step": fsm.episode_step,
            "fsm_abort_latched": fsm.abort_latched,
            "fsm_policy_prepared": fsm.policy_prepared,
            "fsm_policy_stand_active": fsm.policy_stand_active,
            "fsm_last_report": fsm.last_report or "",
            "qpos": robot.joint_positions,
            "qvel": robot.joint_velocities,
            "q_target": robot.command_target,
            "stiffness": robot.command_stiffness,
            "damping": robot.command_damping,
            "applied_tau": robot.applied_torque,
            "pelvis_pose": np.concatenate((position, quaternion)),
            "pelvis_velocity": np.concatenate((robot.pelvis_linear_velocity, robot.base_angular_velocity)),
            "tilt": _body_tilt(quaternion),
            "balance_attitude_target": np.asarray(
                (self._balance_config.target_roll, self._balance_config.target_pitch), dtype=np.float64
            ),
            "target_relative_tilt_error": _target_relative_tilt_error(quaternion, self._balance_config),
            "foot_contact_forces": robot.foot_contact_forces,
            "foot_contact_force_vectors": robot.foot_contact_force_vectors,
            "joint_limit_violations": robot.joint_limit_violations,
            "feedback_stale": robot.feedback_stale,
            "control_deadline_missed": robot.control_deadline_missed,
            "control_duration": robot.last_control_duration_s,
            "operator_request_start": operator.request_start,
            "operator_confirm": operator.confirm,
            "operator_abort": operator.abort,
            "observation": policy.last_observation,
            "raw_action": policy.last_action,
        }
        for name, value in row.items():
            self.values[name].append(value.copy() if isinstance(value, np.ndarray) else value)

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(values) for name, values in self.values.items()}

    def save(self) -> None:
        if not self.values["time"]:
            raise RuntimeError("Cannot save an empty FSM MuJoCo trajectory.")
        arrays = self.arrays()
        arrays["metadata_json"] = np.asarray(json.dumps(self.metadata, sort_keys=True))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_path, **arrays)


def _body_tilt(quaternion_wxyz: np.ndarray) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("Pelvis quaternion must contain four finite values with non-zero norm.")
    _, x, y, _ = quaternion / norm
    return math.acos(float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def _target_relative_tilt_error(
    quaternion_wxyz: np.ndarray,
    balance_config: BalanceControllerConfig = _BALANCE_CONFIG,
) -> float:
    roll, pitch = quaternion_to_roll_pitch(quaternion_wxyz)
    return math.hypot(roll - balance_config.target_roll, pitch - balance_config.target_pitch)


def _midpoint_goal(ranges: dict[str, tuple[float, float]], args: argparse.Namespace) -> JumpGoal:
    requested = {
        "pos_x": args.goal_pos_x,
        "pos_y": args.goal_pos_y,
        "roll": args.goal_roll,
        "pitch": args.goal_pitch,
        "yaw": args.goal_yaw,
    }
    values = {
        name: (bounds[0] + bounds[1]) * 0.5 if requested[name] is None else requested[name]
        for name, bounds in ranges.items()
    }
    return JumpGoal(
        dx=values["pos_x"],
        dy=values["pos_y"],
        dyaw=values["yaw"],
        roll=values["roll"],
        pitch=values["pitch"],
    )


def _repeat_goals(
    primary_goal: JumpGoal,
    ranges: dict[str, tuple[float, float]],
    pos_x_values: list[float] | tuple[float, ...] | None,
) -> tuple[JumpGoal, ...]:
    """Build distinct longitudinal goals for one uninterrupted repeat run."""
    lower, upper = ranges["pos_x"]
    values = [lower, 0.5 * (lower + upper), upper] if pos_x_values is None else list(pos_x_values)
    if len(values) < 2:
        raise ValueError("A repeat scenario requires at least two longitudinal goals.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Repeat longitudinal goals must be finite.")
    if len(set(values)) != len(values):
        raise ValueError("Repeat longitudinal goals must be distinct.")
    outside = [value for value in values if not lower <= value <= upper]
    if outside:
        raise ValueError(f"Repeat longitudinal goals {outside} are outside manifest range [{lower}, {upper}].")
    return tuple(
        JumpGoal(value, primary_goal.dy, primary_goal.dyaw, primary_goal.roll, primary_goal.pitch) for value in values
    )


def _scenario_timeline(
    scenario: str,
    goal: JumpGoal,
    goal_ranges: dict[str, tuple[float, float]],
    policy_dt: float,
    flight_start_step: int,
    start_time_s: float = _START_TIME_S,
    confirm_time_s: float = _CONFIRM_TIME_S,
    repeat_goals: tuple[JumpGoal, ...] = (),
    episode_steps: int | None = None,
    settle_timeout_s: float = JumpControllerConfig().settle_timeout_s,
    repeat_prepare_duration_s: float = 0.0,
) -> tuple[OperatorTimelineEntry, ...]:
    if scenario == "stand":
        return ()
    if scenario == "reject":
        lower, upper = goal_ranges["pos_x"]
        rejected_dx = upper + max(0.01, 0.01 * (upper - lower))
        rejected_goal = JumpGoal(rejected_dx, goal.dy, goal.dyaw, goal.roll, goal.pitch)
        return (
            OperatorTimelineEntry(
                start_time_s,
                goal=rejected_goal,
                request_start=True,
                label="request out-of-envelope pos_x goal",
            ),
        )

    if scenario == "repeat":
        if len(repeat_goals) < 2:
            raise ValueError("The repeat scenario requires at least two goals.")
        if episode_steps is None or episode_steps <= 0:
            raise ValueError("The repeat scenario requires a positive episode step count.")
        cycle_period_s = (
            confirm_time_s - start_time_s + episode_steps * policy_dt + settle_timeout_s + _REPEAT_STAND_DWELL_S
        )
        entries = []
        for index, repeat_goal in enumerate(repeat_goals):
            cycle_offset_s = index * cycle_period_s
            repeat_start_time_s = start_time_s + cycle_offset_s
            repeat_confirm_time_s = confirm_time_s + cycle_offset_s
            if index > 0 and repeat_prepare_duration_s > 0.0:
                repeat_confirm_time_s = repeat_start_time_s + repeat_prepare_duration_s + _REPEAT_CONFIRM_DWELL_S
            entries.extend(
                (
                    OperatorTimelineEntry(
                        repeat_start_time_s,
                        goal=repeat_goal,
                        request_start=True,
                        label=f"request repeat jump {index + 1} with goal",
                    ),
                    OperatorTimelineEntry(
                        repeat_confirm_time_s,
                        confirm=True,
                        label=f"confirm repeat jump {index + 1}",
                    ),
                )
            )
        return tuple(entries)

    entries = [
        OperatorTimelineEntry(start_time_s, goal=goal, request_start=True, label="request start with goal"),
        OperatorTimelineEntry(confirm_time_s, confirm=True, label="confirm armed goal"),
    ]
    if scenario == "abort_early":
        early_abort_time_s = confirm_time_s + (_EARLY_ABORT_TIME_S - _CONFIRM_TIME_S)
        entries.append(OperatorTimelineEntry(early_abort_time_s, abort=True, label="abort before takeoff"))
    elif scenario == "abort_late":
        # Policy step 0 runs on the confirmation tick, so this pulse is sampled
        # with episode_step == flight_start_step.
        abort_time_s = confirm_time_s + flight_start_step * policy_dt
        entries.append(OperatorTimelineEntry(abort_time_s, abort=True, label="abort after takeoff"))
    return tuple(entries)


def _timeline_json(timeline: tuple[OperatorTimelineEntry, ...]) -> list[dict[str, Any]]:
    result = []
    for entry in timeline:
        result.append(
            {
                "time_s": entry.time_s,
                "duration_s": entry.duration_s,
                "goal": None
                if entry.goal is None
                else {
                    "dx": entry.goal.dx,
                    "dy": entry.goal.dy,
                    "dyaw": entry.goal.dyaw,
                    "roll": entry.goal.roll,
                    "pitch": entry.goal.pitch,
                },
                "request_start": entry.request_start,
                "confirm": entry.confirm,
                "abort": entry.abort,
                "label": entry.label,
            }
        )
    return result


def _terminal_state_reached(
    scenario: str,
    fsm: JumpControllerFSM,
    time_s: float,
    start_time_s: float = _START_TIME_S,
    expected_jump_count: int = 1,
) -> bool:
    history = fsm.transition_history
    if fsm.state is JumpControllerState.DAMPING:
        return True
    if scenario == "reject":
        return time_s >= start_time_s + 0.5
    if (
        scenario in ("nominal", "repeat")
        and history.count(JumpControllerState.SETTLE) >= expected_jump_count
        and fsm.state is JumpControllerState.STAND
    ):
        return True
    if (
        scenario in ("nominal", "repeat")
        and JumpControllerState.GOTO_START in history
        and fsm.state is JumpControllerState.STAND
        and fsm.last_report is not None
        and ("refused" in fsm.last_report.lower() or "timed out" in fsm.last_report.lower())
    ):
        return True
    return False


def _longest_airborne_window(times: np.ndarray, airborne: np.ndarray, sim_dt: float) -> tuple[float, float] | None:
    indices = np.flatnonzero(airborne)
    if indices.size == 0:
        return None
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, split_points)
    longest = max(runs, key=len)
    return float(times[longest[0]]), float(times[longest[-1]] + sim_dt)


def _print_timeline(timeline: StateTimeline) -> None:
    print("State timeline:")
    for index, interval in enumerate(timeline.intervals):
        exit_time = interval.exit_time_s if interval.exit_time_s is not None else interval.entry_time_s
        report = "" if interval.report is None else f"  report={interval.report}"
        print(
            f"  {index:02d} {interval.state.value:<10} "
            f"entry={interval.entry_time_s:7.3f} s exit={exit_time:7.3f} s "
            f"duration={exit_time - interval.entry_time_s:6.3f} s{report}"
        )


def _print_state_summaries(
    timeline: StateTimeline,
    arrays: dict[str, np.ndarray],
    robot: MujocoRobot,
) -> None:
    print("Per-state summaries:")
    times = arrays["time"]
    for index, interval in enumerate(timeline.intervals):
        exit_time = interval.exit_time_s if interval.exit_time_s is not None else float(times[-1])
        mask = (
            (arrays["fsm_state"] == interval.state.value)
            & (times >= interval.entry_time_s - 1.0e-12)
            & (times <= exit_time + 1.0e-12)
        )
        if not np.any(mask):
            print(f"  {index:02d} {interval.state.value:<10} no 500 Hz samples (transient state)")
            continue
        pelvis_z = arrays["pelvis_pose"][mask, 2]
        absolute_tilt_deg = np.degrees(arrays["tilt"][mask])
        target_relative_tilt_error_deg = np.degrees(arrays["target_relative_tilt_error"][mask])
        torque = np.abs(arrays["applied_tau"][mask])
        torque_ratio = torque / robot.command_effort_limits[np.newaxis, :]
        torque_sample, torque_joint = np.unravel_index(int(np.argmax(torque_ratio)), torque_ratio.shape)
        support = arrays["foot_contact_forces"][mask]
        loaded_fraction = np.mean(support > _CONTACT_THRESHOLD_N, axis=0)
        line = (
            f"  {index:02d} {interval.state.value:<10} "
            f"pelvis_z={np.min(pelvis_z):.3f}..{np.max(pelvis_z):.3f} m "
            f"(end {pelvis_z[-1]:.3f}), "
            f"absolute_tilt=end {absolute_tilt_deg[-1]:.3f}/peak {np.max(absolute_tilt_deg):.3f} deg, "
            f"target_relative_tilt_error=end {target_relative_tilt_error_deg[-1]:.3f}/"
            f"peak {np.max(target_relative_tilt_error_deg):.3f} deg, "
            f"torque_peak={torque[torque_sample, torque_joint]:.1f}/"
            f"{robot.command_effort_limits[torque_joint]:.1f} N·m "
            f"({100.0 * torque_ratio[torque_sample, torque_joint]:.1f}%, {robot.joint_names[torque_joint]}), "
            f"foot_peak=[{np.max(support[:, 0]):.1f}, {np.max(support[:, 1]):.1f}] N, "
            f"loaded=[{100.0 * loaded_fraction[0]:.0f}%, {100.0 * loaded_fraction[1]:.0f}%]"
        )
        print(line)
        if interval.state is JumpControllerState.JUMP:
            jump_times = times[mask]
            apex_index = int(np.argmax(pelvis_z))
            airborne = np.all(support <= _CONTACT_THRESHOLD_N, axis=1)
            window = _longest_airborne_window(jump_times, airborne, robot.sim_dt)
            if window is None:
                airborne_text = "none"
            else:
                airborne_text = f"{window[0]:.3f}..{window[1]:.3f} s ({window[1] - window[0]:.3f} s)"
            print(
                f"       JUMP apex={pelvis_z[apex_index]:.3f} m at t={jump_times[apex_index]:.3f} s; "
                f"airborne_window={airborne_text}"
            )


def _hardware_margin_result(arrays: dict[str, np.ndarray], robot: MujocoRobot) -> tuple[bool, str]:
    torque_ratio = np.abs(arrays["applied_tau"]) / robot.command_effort_limits[np.newaxis, :]
    maximum_torque_ratio = float(np.max(torque_ratio))
    maximum_tilt = float(np.max(arrays["tilt"]))
    maximum_joint_speed = float(np.max(np.abs(arrays["qvel"])))
    minimum_pelvis_height = float(np.min(arrays["pelvis_pose"][:, 2]))
    final_target_error = float(np.max(np.abs(arrays["q_target"][-1] - arrays["qpos"][-1])))
    joint_limit_violated = bool(np.any(arrays["joint_limit_violations"]))
    checks = (
        (
            maximum_torque_ratio <= _HARDWARE_MARGIN_MAX_COMMAND_EFFORT_RATIO,
            f"command effort {100.0 * maximum_torque_ratio:.1f}% "
            f"(limit {100.0 * _HARDWARE_MARGIN_MAX_COMMAND_EFFORT_RATIO:.0f}%)",
        ),
        (
            maximum_tilt <= _HARDWARE_MARGIN_MAX_TILT_RAD,
            f"tilt {math.degrees(maximum_tilt):.1f} deg (limit {math.degrees(_HARDWARE_MARGIN_MAX_TILT_RAD):.0f} deg)",
        ),
        (
            maximum_joint_speed <= _HARDWARE_MARGIN_MAX_JOINT_SPEED_RAD_S,
            f"joint speed {maximum_joint_speed:.2f} rad/s (limit {_HARDWARE_MARGIN_MAX_JOINT_SPEED_RAD_S:.1f} rad/s)",
        ),
        (
            minimum_pelvis_height >= _HARDWARE_MARGIN_MIN_PELVIS_HEIGHT_M,
            f"pelvis height {minimum_pelvis_height:.3f} m (minimum {_HARDWARE_MARGIN_MIN_PELVIS_HEIGHT_M:.2f} m)",
        ),
        (
            final_target_error <= _HARDWARE_MARGIN_MAX_TARGET_ERROR_RAD,
            f"final target error {final_target_error:.3f} rad (limit {_HARDWARE_MARGIN_MAX_TARGET_ERROR_RAD:.1f} rad)",
        ),
        (
            not joint_limit_violated,
            "joint-limit contact absent" if not joint_limit_violated else "joint-limit contact present",
        ),
    )
    failed = [description for passed, description in checks if not passed]
    summary = "; ".join(description for _, description in checks)
    return not failed, summary


def _unmeasured_ground_contact_result(
    arrays: dict[str, np.ndarray],
    flight_start_step: int,
) -> tuple[bool, str]:
    """Audit hidden MuJoCo contact truth for an unmeasured-ground run."""
    forces = np.asarray(arrays["foot_contact_forces"], dtype=np.float64)
    states = np.asarray(arrays["fsm_state"])
    episode_steps = np.asarray(arrays["fsm_episode_step"], dtype=np.int64)
    all_jump_indices = np.flatnonzero(states == JumpControllerState.JUMP.value)
    if all_jump_indices.size == 0:
        return False, "the FSM never entered JUMP"
    split_points = np.flatnonzero(np.diff(all_jump_indices) > 1) + 1
    jump_episodes = np.split(all_jump_indices, split_points)
    airborne_windows = []
    for episode_number, jump_indices in enumerate(jump_episodes, start=1):
        first_jump_index = int(jump_indices[0])
        if first_jump_index == 0 or not np.all(forces[first_jump_index - 1] > _CONTACT_THRESHOLD_N):
            return False, f"episode {episode_number}: bilateral support was absent immediately before JUMP"

        jump_flight_indices = jump_indices[episode_steps[jump_indices] >= flight_start_step]
        airborne_indices = jump_flight_indices[np.all(forces[jump_flight_indices] <= _CONTACT_THRESHOLD_N, axis=1)]
        if airborne_indices.size == 0:
            return False, f"episode {episode_number}: MuJoCo ground truth never showed an airborne interval"
        first_airborne_index = int(airborne_indices[0])
        touchdown_indices = jump_indices[
            (jump_indices > first_airborne_index) & np.all(forces[jump_indices] > _CONTACT_THRESHOLD_N, axis=1)
        ]
        if touchdown_indices.size == 0:
            return False, f"episode {episode_number}: bilateral touchdown was absent before JUMP ended"
        first_touchdown_index = int(touchdown_indices[0])
        airborne_windows.append(
            f"{arrays['time'][first_airborne_index]:.3f}..{arrays['time'][first_touchdown_index]:.3f} s"
        )

    episode_count = len(jump_episodes)
    return (
        True,
        "hidden ground truth showed bilateral support, flight, and bilateral touchdown in "
        f"{episode_count}/{episode_count} episodes (airborne {', '.join(airborne_windows)})",
    )


def _command_tracking_result(
    arrays: dict[str, np.ndarray],
    goals: tuple[JumpGoal, ...],
) -> tuple[bool, str]:
    """Audit body-relative displacement for every commanded jump episode.

    Args:
        arrays: Logged FSM state and pelvis pose arrays. Pelvis quaternions use
            WXYZ order.
        goals: Ordered body-relative jump goals.

    Returns:
        Whether every episode met the position and directed-progress limits,
        followed by a concise per-episode summary.
    """
    states = np.asarray(arrays["fsm_state"])
    pelvis_poses = np.asarray(arrays["pelvis_pose"], dtype=np.float64)
    if pelvis_poses.shape != (len(states), 7):
        return False, f"pelvis_pose must have shape ({len(states)}, 7), got {pelvis_poses.shape}"
    jump_indices = np.flatnonzero(states == JumpControllerState.JUMP.value)
    if jump_indices.size == 0:
        return False, "the FSM never entered JUMP"
    split_points = np.flatnonzero(np.diff(jump_indices) > 1) + 1
    episodes = np.split(jump_indices, split_points)
    if len(episodes) != len(goals):
        return False, f"observed {len(episodes)} jump episodes for {len(goals)} goals"

    failures = []
    summaries = []
    for episode_number, (episode, goal) in enumerate(zip(episodes, goals), start=1):
        start_index = max(int(episode[0]) - 1, 0)
        end_index = min(int(episode[-1]) + 1, len(pelvis_poses) - 1)
        start_pose = pelvis_poses[start_index]
        end_pose = pelvis_poses[end_index]
        if not np.all(np.isfinite((start_pose, end_pose))):
            return False, f"episode {episode_number}: pelvis pose contains non-finite values"
        quat_norm = float(np.linalg.norm(start_pose[3:]))
        if quat_norm <= np.finfo(np.float64).eps:
            return False, f"episode {episode_number}: start quaternion has zero norm"
        w, x, y, z = start_pose[3:] / quat_norm
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        world_dx, world_dy = end_pose[:2] - start_pose[:2]
        dx = math.cos(yaw) * world_dx + math.sin(yaw) * world_dy
        dy = -math.sin(yaw) * world_dx + math.cos(yaw) * world_dy
        planar_error = math.hypot(dx - goal.dx, dy - goal.dy)
        goal_norm = math.hypot(goal.dx, goal.dy)
        directed_progress = 0.0
        if goal_norm > 0.0:
            directed_progress = (dx * goal.dx + dy * goal.dy) / goal_norm
        episode_failed = planar_error > _COMMAND_TRACKING_MAX_PLANAR_ERROR_M
        if goal_norm >= 0.05:
            episode_failed |= directed_progress < _COMMAND_TRACKING_MIN_DIRECTED_PROGRESS_M
        if episode_failed:
            failures.append(episode_number)
        summaries.append(
            f"{episode_number}: goal=({goal.dx:+.3f},{goal.dy:+.3f}) m, "
            f"actual=({dx:+.3f},{dy:+.3f}) m, error={planar_error:.3f} m"
        )

    summary = "; ".join(summaries)
    if failures:
        failed_text = ", ".join(str(value) for value in failures)
        return False, f"episode {failed_text} failed command tracking; {summary}"
    return True, f"tracked {len(goals)}/{len(goals)} commanded displacements; {summary}"


def _scenario_result(
    scenario: str,
    fsm: JumpControllerFSM,
    control_ticks: list[ControlTick],
    final_time_s: float,
    requested_duration_s: float,
    expected_jump_count: int = 1,
) -> tuple[bool, str]:
    history = fsm.transition_history
    abort_ticks = [tick for tick in control_ticks if tick.abort]
    if scenario == "stand":
        expected = [JumpControllerState.PASSIVE, JumpControllerState.STAND]
        duration_complete = final_time_s >= requested_duration_s - 1.0e-9
        passed = history == expected and fsm.state is JumpControllerState.STAND and duration_complete
        if not duration_complete:
            return False, f"stand viewing ended after {final_time_s:.3f} of {requested_duration_s:.3f} s"
        return passed, "held the stand controller for the requested duration" if passed else (
            f"ended in {fsm.state.value}: {fsm.last_report}"
        )
    if scenario == "nominal":
        expected = [
            JumpControllerState.PASSIVE,
            JumpControllerState.STAND,
            JumpControllerState.GOTO_START,
            JumpControllerState.ARMED,
            JumpControllerState.JUMP,
            JumpControllerState.SETTLE,
            JumpControllerState.STAND,
        ]
        passed = history == expected
        return passed, "completed the full nominal path" if passed else f"ended in {fsm.state.value}: {fsm.last_report}"
    if scenario == "repeat":
        expected = [JumpControllerState.PASSIVE, JumpControllerState.STAND]
        cycle = [
            JumpControllerState.GOTO_START,
            JumpControllerState.ARMED,
            JumpControllerState.JUMP,
            JumpControllerState.SETTLE,
            JumpControllerState.STAND,
        ]
        expected.extend(cycle * expected_jump_count)
        passed = history == expected
        count_name = "three" if expected_jump_count == 3 else str(expected_jump_count)
        jump_count = history.count(JumpControllerState.JUMP)
        return passed, (
            f"completed all {count_name} commanded jump cycles"
            if passed
            else f"ended in {fsm.state.value} after {jump_count} jump cycles: {fsm.last_report}"
        )
    if scenario == "reject":
        report = fsm.last_report or ""
        passed = JumpControllerState.JUMP not in history and "Goal rejected:" in report and "pos_x" in report
        return passed, report or "no rejection reason was reported"
    if not abort_ticks:
        return False, f"abort was never sampled; ended in {fsm.state.value}: {fsm.last_report}"
    tick = abort_ticks[0]
    if scenario == "abort_early":
        passed = (
            tick.state_before is JumpControllerState.JUMP
            and tick.episode_step_before < fsm.flight_start_step
            and tick.state_after is JumpControllerState.DAMPING
        )
        return (
            passed,
            f"abort sampled at step {tick.episode_step_before} (flight starts at {fsm.flight_start_step}); "
            f"same-tick state={tick.state_after.value}",
        )
    passed = (
        tick.state_before is JumpControllerState.JUMP
        and tick.episode_step_before >= fsm.flight_start_step
        and tick.state_after is JumpControllerState.JUMP
        and fsm.state is JumpControllerState.DAMPING
        and fsm.episode_step == fsm.episode_steps
        and fsm.phase_clock_history == list(range(fsm.episode_steps))
    )
    return (
        passed,
        f"abort sampled at step {tick.episode_step_before} (flight starts at {fsm.flight_start_step}); "
        f"same-tick state={tick.state_after.value}, final_step={fsm.episode_step}, final_state={fsm.state.value}",
    )


def _prejump_hold_upright_result(arrays: dict[str, np.ndarray]) -> tuple[bool, str]:
    """Check simulated ground truth throughout STAND, GOTO_START, and ARMED."""
    states = np.asarray(arrays["fsm_state"])
    times = np.asarray(arrays["time"], dtype=np.float64)
    tilts = np.asarray(arrays["tilt"], dtype=np.float64)
    pelvis_pose = np.asarray(arrays["pelvis_pose"], dtype=np.float64)
    if states.ndim != 1 or times.shape != states.shape or tilts.shape != states.shape:
        return False, "hold trajectory arrays have inconsistent shapes"
    if pelvis_pose.shape != (states.size, 7):
        return False, "pelvis_pose must have shape (samples, 7)"
    hold_mask = np.isin(states, tuple(_PREJUMP_HOLD_STATES))
    if not np.any(hold_mask):
        return False, "trajectory has no STAND, GOTO_START, or ARMED samples"
    hold_indices = np.flatnonzero(hold_mask)
    hold_tilts = tilts[hold_indices]
    hold_heights = pelvis_pose[hold_indices, 2]
    invalid = ~np.isfinite(hold_tilts) | ~np.isfinite(hold_heights)
    excessive_tilt = hold_tilts > _PREJUMP_HOLD_MAX_TILT_RAD
    low_pelvis = hold_heights < _PREJUMP_HOLD_MIN_PELVIS_HEIGHT_M
    failed = invalid | excessive_tilt | low_pelvis
    if np.any(failed):
        sample_index = int(hold_indices[int(np.flatnonzero(failed)[0])])
        return False, (
            f"{states[sample_index]} violated upright hold at t={times[sample_index]:.3f} s: "
            f"tilt={math.degrees(tilts[sample_index]):.2f} deg "
            f"(limit {math.degrees(_PREJUMP_HOLD_MAX_TILT_RAD):.2f}), "
            f"pelvis_z={pelvis_pose[sample_index, 2]:.3f} m "
            f"(minimum {_PREJUMP_HOLD_MIN_PELVIS_HEIGHT_M:.3f})"
        )
    return True, (
        f"peak tilt={math.degrees(float(np.max(hold_tilts))):.2f} deg, "
        f"minimum pelvis_z={float(np.min(hold_heights)):.3f} m"
    )


def _validate_contactless_gantry_rehearsal_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Enforce the exact simulated hardware-rehearsal envelope."""
    rehearsal_escalated = args.rehearsal_effort_scale_override is not None or args.rehearsal_unlimited_slew
    if not args.contactless_gantry_rehearsal:
        if rehearsal_escalated or args.acknowledge_rehearsal_escalation:
            parser.error("rehearsal escalation options require --contactless_gantry_rehearsal.")
        return
    if rehearsal_escalated and not args.acknowledge_rehearsal_escalation:
        parser.error("rehearsal escalation requires --acknowledge_rehearsal_escalation.")
    if args.acknowledge_rehearsal_escalation and not rehearsal_escalated:
        parser.error("--acknowledge_rehearsal_escalation requires a rehearsal escalation option.")
    if args.rehearsal_effort_scale_override is not None and (
        not math.isfinite(args.rehearsal_effort_scale_override) or not 0.1 < args.rehearsal_effort_scale_override <= 0.6
    ):
        parser.error("--rehearsal_effort_scale_override must be finite and in (0.1, 0.6].")
    if args.scenario != "nominal":
        parser.error("--contactless_gantry_rehearsal requires --scenario nominal.")
    if not math.isclose(args.gantry_support_fraction, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        parser.error("--contactless_gantry_rehearsal requires --gantry_support_fraction 1.0.")
    if args.rehearsal_effort_scale_override is None and not math.isclose(
        args.effort_scale, 0.1, rel_tol=0.0, abs_tol=1.0e-12
    ):
        parser.error("--contactless_gantry_rehearsal requires --effort_scale 0.1.")
    if not args.rehearsal_unlimited_slew and (
        args.target_rate_limit_rad_s is None
        or not math.isclose(args.target_rate_limit_rad_s, 1.2, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        parser.error("--contactless_gantry_rehearsal requires --target_rate_limit_rad_s 1.2.")
    if args.max_duration < 15.0:
        parser.error("--contactless_gantry_rehearsal requires --max_duration >= 15.")
    if args.start_time_s < 4.5:
        parser.error("--contactless_gantry_rehearsal requires --start_time_s >= 4.5.")
    if args.confirm_time_s - args.start_time_s < 2.5:
        parser.error("--contactless_gantry_rehearsal requires confirmation at least 2.5 seconds after start.")
    if not math.isclose(args.stand_entry_duration_s, 4.0, rel_tol=0.0, abs_tol=1.0e-12):
        parser.error("--contactless_gantry_rehearsal requires --stand_entry_duration_s 4.0.")
    if not math.isclose(args.stand_ankle_stiffness, 80.0, rel_tol=0.0, abs_tol=1.0e-12):
        parser.error("--contactless_gantry_rehearsal requires --stand_ankle_stiffness 80.0.")
    if not math.isclose(args.stand_ankle_damping, 7.0, rel_tol=0.0, abs_tol=1.0e-12):
        parser.error("--contactless_gantry_rehearsal requires --stand_ankle_damping 7.0.")
    if args.policy_terminal_return_steps != 0:
        parser.error("--contactless_gantry_rehearsal requires --policy_terminal_return_steps 0.")
    if (
        not args.balance_disable_integral
        or not math.isclose(
            args.balance_initial_roll_integral,
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            args.balance_initial_pitch_integral,
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        parser.error("--contactless_gantry_rehearsal requires disabled balance integral with zero initial errors.")
    goal_values = (args.goal_pos_x, args.goal_pos_y, args.goal_roll, args.goal_pitch, args.goal_yaw)
    if any(value is None or not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12) for value in goal_values):
        parser.error("--contactless_gantry_rehearsal requires all five explicit goals to be zero.")


def _rehearsal_target_rate_limit(
    state: JumpControllerState,
    *,
    unlimited_slew: bool,
) -> float | None:
    """Return the contactless-rehearsal slew limit for one FSM state."""

    if unlimited_slew and state in (JumpControllerState.JUMP, JumpControllerState.SETTLE):
        return None
    return _CONTACTLESS_REHEARSAL_TARGET_RATE_LIMIT_RAD_S


def _validate_unmeasured_ground_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Restrict the blind-contact runner to a ground jump-path validation."""
    if not args.unmeasured_ground_validation:
        return
    if args.scenario not in ("nominal", "repeat"):
        parser.error("--unmeasured_ground_validation requires --scenario nominal or repeat.")
    if not math.isclose(args.gantry_support_fraction, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        parser.error("--unmeasured_ground_validation requires --gantry_support_fraction 0.0.")
    preparation_interval_s = max(
        args.goto_start_duration_s + args.policy_prepare_duration_s,
        args.policy_stand_retrigger_prepare_duration_s,
    )
    earliest_confirmation_s = args.start_time_s + preparation_interval_s + 0.25
    if args.confirm_time_s < earliest_confirmation_s:
        parser.error(
            "--unmeasured_ground_validation requires --confirm_time_s at least 0.25 s after "
            "the nominal static and policy preparation intervals."
        )


def _parse_args() -> argparse.Namespace:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--policy", type=Path, default=None, help="Defaults to policy.onnx beside the manifest.")
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--overlay", type=Path, default=_DEFAULT_OVERLAY)
    parser.add_argument("--scenario", choices=_SCENARIOS, default="nominal")
    parser.add_argument("--log", type=Path, default=None, help="Defaults to fsm_mujoco_<scenario>.npz.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_duration", type=float, default=8.0, help="Maximum simulated duration [s].")
    parser.add_argument(
        "--emulate_velocity_limit",
        action="store_true",
        help="Emulate manifest actuator velocity limits with torque-speed saturation.",
    )
    contact_contract = parser.add_mutually_exclusive_group()
    contact_contract.add_argument(
        "--contactless_gantry_rehearsal",
        action="store_true",
        help="Mirror the hardware contactless-rehearsal contract with ground collision disabled.",
    )
    contact_contract.add_argument(
        "--unmeasured_ground_validation",
        action="store_true",
        help=(
            "Keep MuJoCo ground collision and contact truth for auditing while preventing the FSM "
            "from reading foot contact; simulation only."
        ),
    )
    parser.add_argument(
        "--latched_abort_upright_settle",
        action="store_true",
        help=(
            "After a joint-limit-only latched post-takeoff abort, settle instead of damping when the "
            "completed episode remains within the configured upright tilt limit."
        ),
    )
    parser.add_argument(
        "--joint_limit_abort_margin_rad",
        type=float,
        default=JumpControllerConfig().joint_limit_abort_margin_rad,
        help="In-JUMP travel allowed beyond each manifest joint limit before aborting [rad].",
    )
    parser.add_argument(
        "--rehearsal_effort_scale_override",
        type=float,
        default=None,
        help="Escalated contactless-rehearsal effort scale in (0.1, 0.6].",
    )
    parser.add_argument(
        "--rehearsal_unlimited_slew",
        action="store_true",
        help="Disable the target slew limit for an escalated contactless rehearsal.",
    )
    parser.add_argument(
        "--acknowledge_rehearsal_escalation",
        action="store_true",
        help="Acknowledge the higher-speed/higher-effort contactless-rehearsal envelope.",
    )
    parser.add_argument(
        "--start_time_s",
        type=float,
        default=_START_TIME_S,
        help="Simulation time of the request-start pulse [s].",
    )
    parser.add_argument(
        "--confirm_time_s",
        type=float,
        default=None,
        help="Simulation time of the separate confirmation pulse [s].",
    )
    parser.add_argument(
        "--goto_start_duration_s",
        type=float,
        default=JumpControllerConfig().goto_start_duration_s,
        help="Duration of the quintic transition to the validated jump start pose [s].",
    )
    parser.add_argument(
        "--policy_prepare_duration_s",
        type=float,
        default=None,
        help=("Goal-conditioned policy warm-up duration at frozen phase zero [s]. Defaults to 0 s."),
    )
    parser.add_argument(
        "--policy_prepare_pose_tolerance_rad",
        type=float,
        default=JumpControllerConfig().policy_prepare_pose_tolerance_rad,
        help="Maximum measured tracking error accepted after policy preparation [rad].",
    )
    parser.add_argument(
        "--policy_stand_after_jump",
        action="store_true",
        help="Continue closed-loop inference on the final STAND reference after the jump.",
    )
    parser.add_argument(
        "--policy_stand_retrigger_prepare_duration_s",
        type=float,
        default=JumpControllerConfig().policy_stand_retrigger_prepare_duration_s,
        help="Frozen phase-zero preparation from policy-native stand to each subsequent goal [s].",
    )
    parser.add_argument(
        "--policy_stand_direct_retrigger",
        action="store_true",
        help=(
            "Hold the preceding policy's final STAND controller until confirmation, then start "
            "the next goal through the normal IDLE handoff."
        ),
    )
    parser.add_argument(
        "--policy_terminal_return_steps",
        type=int,
        default=JumpControllerConfig().policy_terminal_return_steps,
        help="Final policy steps used to blend into the validated stand target; zero disables it.",
    )
    parser.add_argument(
        "--jump_target_blend_steps",
        type=int,
        default=None,
        help="Initial policy steps that blend stand to policy targets; zero disables the blend.",
    )
    parser.add_argument(
        "--jump_gain_blend_steps",
        type=int,
        default=None,
        help="Initial policy steps that blend stand to policy gains; zero selects policy gains immediately.",
    )
    parser.add_argument(
        "--jump_balance_blend_steps",
        type=int,
        default=None,
        help="Initial policy steps that fade stand balance; zero disables stand balance immediately.",
    )
    parser.add_argument(
        "--initial_state",
        type=Path,
        default=None,
        help="Measured-pose JSON capture used to initialize a stand robustness replay.",
    )
    parser.add_argument("--initial_roll_offset_deg", type=float, default=0.0, help="Initial pelvis roll offset [deg].")
    parser.add_argument(
        "--initial_pitch_offset_deg", type=float, default=0.0, help="Initial pelvis pitch offset [deg]."
    )
    parser.add_argument(
        "--effort_scale",
        type=float,
        default=1.0,
        help="Fraction of manifest torque available to the simulated command [0-1].",
    )
    parser.add_argument(
        "--target_rate_limit_rad_s",
        type=float,
        default=None,
        help="Optional joint-target slew limit [rad/s].",
    )
    parser.add_argument(
        "--gantry_support_fraction",
        type=float,
        default=0.0,
        help="Fraction of robot weight supported upward at the pelvis [0-1].",
    )
    parser.add_argument(
        "--stand_hold_measured_pose",
        action="store_true",
        help="Hold the measured entry pose instead of blending to the manifest reference.",
    )
    parser.add_argument("--stand_ankle_stiffness", type=float, default=80.0, help="Stand ankle kp [N·m/rad].")
    parser.add_argument("--stand_ankle_damping", type=float, default=5.0, help="Stand ankle kd [N·m·s/rad].")
    parser.add_argument("--balance_target_roll_deg", type=float, default=None, help="Balance target roll [deg].")
    parser.add_argument("--balance_target_pitch_deg", type=float, default=None, help="Balance target pitch [deg].")
    parser.add_argument(
        "--balance_target_from_initial_state",
        action="store_true",
        help="Use the initial-state pelvis roll and pitch as the fixed balance target.",
    )
    parser.add_argument(
        "--balance_initial_roll_integral",
        type=float,
        default=0.0,
        help="Initial roll attitude-error integral [rad·s].",
    )
    parser.add_argument(
        "--balance_initial_pitch_integral",
        type=float,
        default=0.2,
        help="Initial pitch attitude-error integral [rad·s].",
    )
    parser.add_argument(
        "--balance_disable_integral",
        action="store_true",
        help="Disable roll-pitch integral feedback for the stand replay.",
    )
    parser.add_argument("--balance_kp", type=float, default=3.2, help="Shared roll-pitch balance kp [rad/rad].")
    parser.add_argument("--balance_kd", type=float, default=0.16, help="Shared roll-pitch balance kd [rad/(rad/s)].")
    parser.add_argument(
        "--stand_entry_duration_s",
        type=float,
        default=1.0,
        help="Measured-to-reference stand transition duration [s].",
    )
    parser.add_argument(
        "--settle_duration_s",
        type=float,
        default=JumpControllerConfig().settle_duration_s,
        help="Post-policy quintic blend to the stand target [s].",
    )
    parser.add_argument(
        "--settle_timeout_s",
        type=float,
        default=JumpControllerConfig().settle_timeout_s,
        help="Maximum measured stand-convergence interval after a jump [s].",
    )
    parser.add_argument(
        "--stand_return_state",
        type=Path,
        default=None,
        help="Measured-pose JSON capture used as a late stand handoff target.",
    )
    parser.add_argument(
        "--stand_return_start_s",
        type=float,
        default=8.0,
        help="Simulation time at which to start blending toward --stand_return_state [s].",
    )
    parser.add_argument(
        "--stand_return_duration_s",
        type=float,
        default=4.0,
        help="Quintic stand-return blend duration [s].",
    )
    parser.add_argument(
        "--require_hardware_margin",
        action="store_true",
        help="Require conservative stand margins before reporting PASS.",
    )
    parser.add_argument("--goal_pos_x", type=float, default=None)
    parser.add_argument("--goal_pos_y", type=float, default=None)
    parser.add_argument("--goal_roll", type=float, default=None)
    parser.add_argument("--goal_pitch", type=float, default=None)
    parser.add_argument("--goal_yaw", type=float, default=None)
    parser.add_argument(
        "--repeat_goal_pos_x",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Distinct longitudinal goals for --scenario repeat [m]. "
            "Defaults to the lower bound, midpoint, and upper bound of the manifest envelope."
        ),
    )
    args = parser.parse_args()
    if args.latched_abort_upright_settle and not args.unmeasured_ground_validation:
        parser.error("--latched_abort_upright_settle requires --unmeasured_ground_validation.")
    if not math.isfinite(args.joint_limit_abort_margin_rad) or args.joint_limit_abort_margin_rad < 0.0:
        parser.error("--joint_limit_abort_margin_rad must be a finite non-negative angle.")
    if args.policy_prepare_duration_s is None:
        args.policy_prepare_duration_s = (
            _UNMEASURED_GROUND_POLICY_PREPARE_DURATION_S if args.unmeasured_ground_validation else 0.0
        )
    if args.confirm_time_s is None:
        args.confirm_time_s = (
            _UNMEASURED_GROUND_CONFIRM_TIME_S if args.unmeasured_ground_validation else _CONFIRM_TIME_S
        )
    if not math.isfinite(args.max_duration) or args.max_duration <= 0.0:
        parser.error("--max_duration must be a positive finite duration.")
    if not math.isfinite(args.start_time_s) or args.start_time_s < 0.0:
        parser.error("--start_time_s must be a finite non-negative time.")
    if not math.isfinite(args.confirm_time_s) or args.confirm_time_s <= args.start_time_s:
        parser.error("--confirm_time_s must be finite and later than --start_time_s.")
    if not math.isfinite(args.goto_start_duration_s) or args.goto_start_duration_s <= 0.0:
        parser.error("--goto_start_duration_s must be a positive finite duration.")
    if not math.isfinite(args.policy_prepare_duration_s) or args.policy_prepare_duration_s < 0.0:
        parser.error("--policy_prepare_duration_s must be a finite non-negative duration.")
    if (
        not math.isfinite(args.policy_stand_retrigger_prepare_duration_s)
        or args.policy_stand_retrigger_prepare_duration_s < 0.0
    ):
        parser.error("--policy_stand_retrigger_prepare_duration_s must be a finite non-negative duration.")
    if args.policy_stand_retrigger_prepare_duration_s > 0.0 and not args.policy_stand_after_jump:
        parser.error("--policy_stand_retrigger_prepare_duration_s requires --policy_stand_after_jump.")
    if args.policy_stand_direct_retrigger and not args.policy_stand_after_jump:
        parser.error("--policy_stand_direct_retrigger requires --policy_stand_after_jump.")
    if args.policy_stand_direct_retrigger and args.policy_stand_retrigger_prepare_duration_s > 0.0:
        parser.error(
            "--policy_stand_direct_retrigger cannot be combined with --policy_stand_retrigger_prepare_duration_s."
        )
    if args.policy_terminal_return_steps < 0:
        parser.error("--policy_terminal_return_steps must be a non-negative integer.")
    if args.policy_terminal_return_steps > 0 and args.policy_stand_after_jump:
        parser.error("--policy_terminal_return_steps cannot be combined with --policy_stand_after_jump.")
    if not math.isfinite(args.policy_prepare_pose_tolerance_rad) or args.policy_prepare_pose_tolerance_rad <= 0.0:
        parser.error("--policy_prepare_pose_tolerance_rad must be a positive finite angle.")
    for name in (
        "jump_target_blend_steps",
        "jump_gain_blend_steps",
        "jump_balance_blend_steps",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name} must be a non-negative integer.")
    if not math.isfinite(args.stand_entry_duration_s) or args.stand_entry_duration_s <= 0.0:
        parser.error("--stand_entry_duration_s must be a positive finite duration.")
    if not math.isfinite(args.settle_duration_s) or args.settle_duration_s <= 0.0:
        parser.error("--settle_duration_s must be a positive finite duration.")
    if not math.isfinite(args.settle_timeout_s) or args.settle_timeout_s < args.settle_duration_s:
        parser.error("--settle_timeout_s must be finite and not shorter than --settle_duration_s.")
    if not math.isfinite(args.stand_return_start_s) or args.stand_return_start_s < 0.0:
        parser.error("--stand_return_start_s must be a finite non-negative time.")
    if not math.isfinite(args.stand_return_duration_s) or args.stand_return_duration_s <= 0.0:
        parser.error("--stand_return_duration_s must be a positive finite duration.")
    if not math.isfinite(args.effort_scale) or not 0.0 < args.effort_scale <= 1.0:
        parser.error("--effort_scale must be finite and in (0, 1].")
    if args.target_rate_limit_rad_s is not None and (
        not math.isfinite(args.target_rate_limit_rad_s) or args.target_rate_limit_rad_s <= 0.0
    ):
        parser.error("--target_rate_limit_rad_s must be a positive finite velocity.")
    if not math.isfinite(args.gantry_support_fraction) or not 0.0 <= args.gantry_support_fraction <= 1.0:
        parser.error("--gantry_support_fraction must be finite and in [0, 1].")
    for name in ("stand_ankle_stiffness", "stand_ankle_damping"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and non-negative.")
    for name in ("balance_kp", "balance_kd"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and non-negative.")
    for name in (
        "initial_roll_offset_deg",
        "initial_pitch_offset_deg",
        "balance_target_roll_deg",
        "balance_target_pitch_deg",
        "balance_initial_roll_integral",
        "balance_initial_pitch_integral",
    ):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            parser.error(f"--{name} must be finite.")
    if args.initial_state is not None and args.scenario != "stand":
        parser.error("--initial_state is currently restricted to --scenario stand.")
    if args.stand_return_state is not None and args.scenario != "stand":
        parser.error("--stand_return_state is restricted to --scenario stand.")
    if args.stand_return_state is not None and args.max_duration < (
        args.stand_return_start_s + args.stand_return_duration_s
    ):
        parser.error("--max_duration must include the complete stand-return blend.")
    if args.initial_state is None and (
        not math.isclose(args.initial_roll_offset_deg, 0.0) or not math.isclose(args.initial_pitch_offset_deg, 0.0)
    ):
        parser.error("Initial attitude offsets require --initial_state.")
    if args.balance_target_from_initial_state and args.initial_state is None:
        parser.error("--balance_target_from_initial_state requires --initial_state.")
    if args.balance_target_from_initial_state and (
        args.balance_target_roll_deg is not None or args.balance_target_pitch_deg is not None
    ):
        parser.error("--balance_target_from_initial_state conflicts with explicit balance targets.")
    if args.require_hardware_margin and (args.scenario != "stand" or args.initial_state is None):
        parser.error("--require_hardware_margin requires --scenario stand and --initial_state.")
    if args.repeat_goal_pos_x is not None:
        if args.scenario != "repeat":
            parser.error("--repeat_goal_pos_x requires --scenario repeat.")
        if len(args.repeat_goal_pos_x) < 2:
            parser.error("--repeat_goal_pos_x requires at least two values.")
        if any(not math.isfinite(value) for value in args.repeat_goal_pos_x):
            parser.error("--repeat_goal_pos_x values must be finite.")
        if len(set(args.repeat_goal_pos_x)) != len(args.repeat_goal_pos_x):
            parser.error("--repeat_goal_pos_x values must be distinct.")
    _validate_contactless_gantry_rehearsal_args(parser, args)
    if args.rehearsal_effort_scale_override is not None:
        args.effort_scale = args.rehearsal_effort_scale_override
    if args.rehearsal_unlimited_slew:
        args.target_rate_limit_rad_s = None
    _validate_unmeasured_ground_args(parser, args)
    return args


def run(args: argparse.Namespace) -> None:  # noqa: C901
    """Run one selected FSM scenario and write its full-rate log."""
    manifest_path = args.manifest.resolve()
    policy_path = args.policy.resolve() if args.policy is not None else manifest_path.parent / "policy.onnx"
    log_path = args.log if args.log is not None else Path(f"fsm_mujoco_{args.scenario}.npz")
    robot = MujocoRobot(
        manifest_path,
        args.model,
        args.overlay,
        effort_scale=args.effort_scale,
        target_rate_limit_rad_s=args.target_rate_limit_rad_s,
        emulate_velocity_limit=args.emulate_velocity_limit,
        gantry_support_fraction=args.gantry_support_fraction,
        ground_contact_enabled=not args.contactless_gantry_rehearsal,
    )
    initial_state = None
    if args.initial_state is not None:
        initial_state = _load_initial_state(args.initial_state, robot.joint_names)
        initial_quaternion = _apply_attitude_offset(
            initial_state.root_quaternion_wxyz,
            math.radians(args.initial_roll_offset_deg),
            math.radians(args.initial_pitch_offset_deg),
        )
        robot.reset_state(initial_state.joint_positions, initial_quaternion)
    else:
        initial_quaternion = None
    stand_return_state = (
        None if args.stand_return_state is None else _load_initial_state(args.stand_return_state, robot.joint_names)
    )
    robot.print_permutations()
    goal = _midpoint_goal(robot.goal_ranges, args)
    repeat_goals = _repeat_goals(goal, robot.goal_ranges, args.repeat_goal_pos_x) if args.scenario == "repeat" else ()
    expected_jump_count = len(repeat_goals) if repeat_goals else 1

    measured_target = (
        quaternion_to_roll_pitch(robot.imu_quaternion)
        if args.contactless_gantry_rehearsal
        else (None if initial_quaternion is None else quaternion_to_roll_pitch(initial_quaternion))
    )
    balance_config = BalanceControllerConfig(
        target_roll=(
            measured_target[0]
            if args.balance_target_from_initial_state or args.contactless_gantry_rehearsal
            else (
                _BALANCE_CONFIG.target_roll
                if args.balance_target_roll_deg is None
                else math.radians(args.balance_target_roll_deg)
            )
        ),
        target_pitch=(
            measured_target[1]
            if args.balance_target_from_initial_state or args.contactless_gantry_rehearsal
            else (
                _BALANCE_CONFIG.target_pitch
                if args.balance_target_pitch_deg is None
                else math.radians(args.balance_target_pitch_deg)
            )
        ),
        initial_roll_integral=args.balance_initial_roll_integral,
        initial_pitch_integral=args.balance_initial_pitch_integral,
        integral_enabled=not args.balance_disable_integral,
        roll_kp=args.balance_kp,
        pitch_kp=args.balance_kp,
        roll_kd=args.balance_kd,
        pitch_kd=args.balance_kd,
    )
    stand_stiffness_overrides = None
    stand_damping_overrides = None
    if args.unmeasured_ground_validation:
        prepared_leg_joints = tuple(
            f"{side}_{joint}_joint" for side in ("left", "right") for joint in ("hip_pitch", "knee")
        )
        stand_stiffness_overrides = {name: _UNMEASURED_GROUND_STAND_STIFFNESS for name in prepared_leg_joints}
        stand_damping_overrides = {name: _UNMEASURED_GROUND_STAND_DAMPING for name in prepared_leg_joints}
    stand_gains = StandGainConfig(
        ankle_stiffness=args.stand_ankle_stiffness,
        ankle_damping=args.stand_ankle_damping,
        stiffness_overrides=stand_stiffness_overrides,
        damping_overrides=stand_damping_overrides,
    )
    goto_start_timeout_s = (
        max(
            args.goto_start_duration_s + args.policy_prepare_duration_s,
            args.policy_stand_retrigger_prepare_duration_s,
        )
        + 2.0
    )
    controller_config = JumpControllerConfig(
        stand_entry_duration_s=args.stand_entry_duration_s,
        stand_hold_measured_pose=args.stand_hold_measured_pose,
        goto_start_duration_s=args.goto_start_duration_s,
        goto_start_timeout_s=goto_start_timeout_s,
        policy_prepare_duration_s=args.policy_prepare_duration_s,
        policy_prepare_pose_tolerance_rad=args.policy_prepare_pose_tolerance_rad,
        policy_stand_after_jump=args.policy_stand_after_jump,
        policy_stand_retrigger_prepare_duration_s=args.policy_stand_retrigger_prepare_duration_s,
        policy_stand_direct_retrigger=args.policy_stand_direct_retrigger,
        policy_terminal_return_steps=args.policy_terminal_return_steps,
        jump_target_blend_steps=args.jump_target_blend_steps,
        jump_gain_blend_steps=args.jump_gain_blend_steps,
        jump_balance_blend_steps=args.jump_balance_blend_steps,
        settle_duration_s=args.settle_duration_s,
        settle_timeout_s=args.settle_timeout_s,
        armed_timeout_s=(
            _CONTACTLESS_REHEARSAL_ARMED_TIMEOUT_S
            if args.contactless_gantry_rehearsal
            else JumpControllerConfig().armed_timeout_s
        ),
        contact_safety_mode=(
            JumpControllerConfig.ContactSafetyMode.GANTRY_REHEARSAL
            if args.contactless_gantry_rehearsal
            else (
                JumpControllerConfig.ContactSafetyMode.UNMEASURED_GROUND
                if args.unmeasured_ground_validation
                else JumpControllerConfig.ContactSafetyMode.MEASURED
            )
        ),
        latched_abort_upright_settle=args.latched_abort_upright_settle,
        joint_limit_abort_margin_rad=args.joint_limit_abort_margin_rad,
    )

    if args.scenario == "stand":
        policy = InactivePolicy(robot.observation_dim, robot.joint_count)
    else:
        policy = OnnxPolicy(policy_path, robot.observation_dim, robot.joint_count)
        policy.warm_up()
    operator_timeline = _scenario_timeline(
        args.scenario,
        goal,
        robot.goal_ranges,
        robot.policy_dt,
        robot.flight_start_step,
        args.start_time_s,
        args.confirm_time_s,
        repeat_goals=repeat_goals,
        episode_steps=robot.episode_steps,
        settle_timeout_s=args.settle_timeout_s,
        repeat_prepare_duration_s=args.policy_stand_retrigger_prepare_duration_s,
    )
    operator = ScriptedOperator(operator_timeline)
    fsm = JumpControllerFSM(
        manifest_path,
        robot,
        operator,
        policy,
        stand_gains=stand_gains,
        config=controller_config,
        balance_config=balance_config,
    )
    if fsm.flight_start_step != robot.flight_start_step:
        raise ValueError("FSM and MuJoCo backend disagree on the manifest FLIGHT start step.")
    if not isinstance(robot, RobotInterface):
        raise TypeError("MujocoRobot does not satisfy RobotInterface.")
    if not isinstance(operator, OperatorInterface):
        raise TypeError("ScriptedOperator does not satisfy OperatorInterface.")

    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata = {
        "scenario": args.scenario,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "policy": None if args.scenario == "stand" else str(policy_path),
        "model": str(Path(args.model).resolve()),
        "overlay": str(Path(args.overlay).resolve()),
        "simulator": "mujoco",
        "sim_dt": robot.sim_dt,
        "policy_dt": robot.policy_dt,
        "decimation": robot.decimation,
        "initial_state": None if args.initial_state is None else str(args.initial_state.resolve()),
        "initial_state_label": None if initial_state is None else initial_state.label,
        "initial_roll_offset_deg": args.initial_roll_offset_deg,
        "initial_pitch_offset_deg": args.initial_pitch_offset_deg,
        "initial_root_height_offset_m": robot.initial_root_height_offset,
        "joint_names": robot.joint_names,
        "effort_limits": robot.effort_limits.tolist(),
        "command_effort_limits": robot.command_effort_limits.tolist(),
        "effort_scale": args.effort_scale,
        "emulate_velocity_limit": args.emulate_velocity_limit,
        "target_rate_limit_rad_s": (
            {
                "stand": _CONTACTLESS_REHEARSAL_TARGET_RATE_LIMIT_RAD_S,
                "armed": _CONTACTLESS_REHEARSAL_TARGET_RATE_LIMIT_RAD_S,
                "jump": None,
                "settle": None,
            }
            if args.rehearsal_unlimited_slew
            else args.target_rate_limit_rad_s
        ),
        "gantry_support_fraction": args.gantry_support_fraction,
        "gantry_support_force_world_n": robot.gantry_support_force_world.tolist(),
        "contactless_gantry_rehearsal": args.contactless_gantry_rehearsal,
        "unmeasured_ground_validation": args.unmeasured_ground_validation,
        "latched_abort_upright_settle": args.latched_abort_upright_settle,
        "joint_limit_abort_margin_rad": args.joint_limit_abort_margin_rad,
        "rehearsal_effort_scale_override": args.rehearsal_effort_scale_override,
        "rehearsal_unlimited_slew": args.rehearsal_unlimited_slew,
        "ground_contact_enabled": robot.ground_contact_enabled,
        "contact_safety_mode": controller_config.contact_safety_mode.value,
        "start_time_s": args.start_time_s,
        "confirm_time_s": args.confirm_time_s,
        "goto_start_duration_s": args.goto_start_duration_s,
        "policy_prepare_duration_s": args.policy_prepare_duration_s,
        "policy_prepare_pose_tolerance_rad": args.policy_prepare_pose_tolerance_rad,
        "policy_stand_after_jump": args.policy_stand_after_jump,
        "policy_stand_retrigger_prepare_duration_s": args.policy_stand_retrigger_prepare_duration_s,
        "policy_stand_direct_retrigger": args.policy_stand_direct_retrigger,
        "policy_terminal_return_steps": args.policy_terminal_return_steps,
        "jump_target_blend_steps": args.jump_target_blend_steps,
        "jump_gain_blend_steps": args.jump_gain_blend_steps,
        "jump_balance_blend_steps": args.jump_balance_blend_steps,
        "policy_prepare_stand_stiffness_overrides_nm_per_rad": stand_stiffness_overrides,
        "policy_prepare_stand_damping_overrides_nm_s_per_rad": stand_damping_overrides,
        "stand_hold_measured_pose": args.stand_hold_measured_pose,
        "stand_entry_duration_s": args.stand_entry_duration_s,
        "settle_duration_s": args.settle_duration_s,
        "settle_timeout_s": args.settle_timeout_s,
        "armed_timeout_s": controller_config.armed_timeout_s,
        "stand_return_state": (None if args.stand_return_state is None else str(args.stand_return_state.resolve())),
        "stand_return_state_label": None if stand_return_state is None else stand_return_state.label,
        "stand_return_start_s": args.stand_return_start_s,
        "stand_return_duration_s": args.stand_return_duration_s,
        "stand_ankle_stiffness_nm_per_rad": args.stand_ankle_stiffness,
        "stand_ankle_damping_nm_s_per_rad": args.stand_ankle_damping,
        "balance_target_roll_deg": math.degrees(balance_config.target_roll),
        "balance_target_pitch_deg": math.degrees(balance_config.target_pitch),
        "balance_target_from_initial_state": args.balance_target_from_initial_state,
        "balance_target_from_simulated_feedback": args.contactless_gantry_rehearsal,
        "balance_initial_roll_integral_rad_s": balance_config.initial_roll_integral,
        "balance_initial_pitch_integral_rad_s": balance_config.initial_pitch_integral,
        "balance_integral_enabled": balance_config.integral_enabled,
        "balance_kp": args.balance_kp,
        "balance_kd": args.balance_kd,
        "require_hardware_margin": args.require_hardware_margin,
        "policy_from_mujoco": robot.policy_from_mujoco.tolist(),
        "mujoco_from_policy": robot.mujoco_from_policy.tolist(),
        "policy_from_actuator": robot.policy_from_actuator.tolist(),
        "actuator_from_policy": robot.actuator_from_policy.tolist(),
        "operator_timeline": _timeline_json(operator_timeline),
        "expected_jump_count": expected_jump_count,
        "contact_threshold_n": _CONTACT_THRESHOLD_N,
        "pelvis_pose_convention": "position_world_xyz[m], quaternion_world_from_body_wxyz",
        "pelvis_velocity_convention": "linear_world_xyz[m/s], angular_body_xyz[rad/s]",
        "tilt_convention": "absolute body-up deviation from world vertical [rad]",
        "target_relative_tilt_error_convention": "roll-pitch norm relative to balance target [rad]",
        "sample_convention": "initial pre-control row followed by post-physics 500 Hz rows",
    }
    logger = FsmLogger(log_path, metadata, balance_config)
    state_timeline = StateTimeline(fsm)
    control_ticks: list[ControlTick] = []

    print(f"Scenario: {args.scenario}")
    if initial_state is not None:
        print(f"Initial state: {initial_state.label} ({args.initial_state.resolve()})")
    if stand_return_state is not None:
        print(
            f"Stand return: {stand_return_state.label} at t={args.stand_return_start_s:.3f} s "
            f"over {args.stand_return_duration_s:.3f} s"
        )
    target_rate_description = (
        f"{_CONTACTLESS_REHEARSAL_TARGET_RATE_LIMIT_RAD_S} rad/s outside JUMP/SETTLE, unlimited in JUMP/SETTLE"
        if args.rehearsal_unlimited_slew
        else f"{args.target_rate_limit_rad_s} rad/s"
    )
    print(f"Command limits: effort_scale={args.effort_scale:.2f}, target_rate_limit={target_rate_description}")
    if args.gantry_support_fraction > 0.0:
        print(
            f"Gantry support: {100.0 * args.gantry_support_fraction:.0f}% body weight, "
            f"force={robot.gantry_support_force_world.tolist()} N"
        )
    if args.contactless_gantry_rehearsal:
        print("Contactless rehearsal: ground collision disabled; measured foot contact is unavailable.")
    if args.unmeasured_ground_validation:
        print("Unmeasured-ground validation: ground collision is enabled; FSM foot contact is unavailable.")
        if args.latched_abort_upright_settle:
            print("Joint-limit-only latched post-takeoff aborts settle when the completed episode remains upright.")
        if args.joint_limit_abort_margin_rad > 0.0:
            print(
                f"Joint-limit abort margin: {args.joint_limit_abort_margin_rad:.6f} rad; "
                "shallower stop touches are warnings."
            )
    if args.policy_prepare_duration_s > 0.0:
        print(
            "Goal-conditioned preparation: "
            f"phase zero frozen for {args.policy_prepare_duration_s:.3f} s, "
            f"tracking tolerance={args.policy_prepare_pose_tolerance_rad:.3f} rad"
        )
    if args.policy_stand_retrigger_prepare_duration_s > 0.0:
        print(
            "Repeat preparation: policy-native stand to frozen phase zero over "
            f"{args.policy_stand_retrigger_prepare_duration_s:.3f} s"
        )
    if args.policy_stand_direct_retrigger:
        print("Repeat handoff: hold policy-native STAND until confirmation, then use the IDLE blend.")
    if args.policy_terminal_return_steps > 0:
        print(f"Terminal stand return: final {args.policy_terminal_return_steps} policy steps blend to validated stand")
    print(
        "Jump handoff steps: "
        f"target={args.jump_target_blend_steps}, gains={args.jump_gain_blend_steps}, "
        f"balance={args.jump_balance_blend_steps} (None=manifest IDLE)"
    )
    print("Operator timeline:")
    for entry in operator_timeline:
        goal_text = "" if entry.goal is None else f" goal={entry.goal}"
        print(f"  t={entry.time_s:.3f} s  {entry.label}{goal_text}")

    operator.update(0.0)
    logger.append(robot, operator, fsm, policy)
    fsm.enable()
    state_timeline.observe(fsm, 0.0)
    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(robot.model, robot.data)

    maximum_physics_steps = int(math.ceil(args.max_duration / robot.sim_dt))
    terminal_time_s: float | None = None
    stand_return_start_target: np.ndarray | None = None
    stand_return_stiffness: np.ndarray | None = None
    stand_return_damping: np.ndarray | None = None
    stop_reason = f"maximum duration {args.max_duration:.3f} s reached"
    try:
        for physics_step in range(maximum_physics_steps):
            step_start = time.monotonic()
            if physics_step % robot.decimation == 0:
                control_time_s = float(robot.data.time)
                operator.update(control_time_s)
                state_before = fsm.state
                episode_step_before = fsm.episode_step
                request_start = operator.request_start
                confirm = operator.confirm
                abort = operator.abort
                control_start = time.monotonic()
                if stand_return_state is None or control_time_s < args.stand_return_start_s:
                    fsm.step()
                    for warning in fsm.drain_warnings():
                        print(warning)
                control_duration_s = time.monotonic() - control_start
                robot.record_control_duration(control_duration_s)
                state_timeline.observe(fsm, control_time_s)
                control_ticks.append(
                    ControlTick(
                        time_s=control_time_s,
                        state_before=state_before,
                        state_after=fsm.state,
                        episode_step_before=episode_step_before,
                        episode_step_after=fsm.episode_step,
                        request_start=request_start,
                        confirm=confirm,
                        abort=abort,
                        duration_s=control_duration_s,
                    )
                )
                if _terminal_state_reached(
                    args.scenario,
                    fsm,
                    control_time_s,
                    args.start_time_s,
                    expected_jump_count,
                ):
                    if terminal_time_s is None:
                        terminal_time_s = control_time_s
                    elif control_time_s - terminal_time_s >= 0.25:
                        stop_reason = f"terminal state observed for {control_time_s - terminal_time_s:.3f} s"
                        break
                else:
                    terminal_time_s = None

            if stand_return_state is not None and float(robot.data.time) >= args.stand_return_start_s:
                if stand_return_start_target is None:
                    stand_return_start_target = robot.command_base_target
                    stand_return_stiffness = robot.command_stiffness
                    stand_return_damping = robot.command_damping
                progress = min(
                    (float(robot.data.time) - args.stand_return_start_s) / args.stand_return_duration_s,
                    1.0,
                )
                blend = progress * progress * progress * (10.0 + progress * (-15.0 + 6.0 * progress))
                stand_return_target = stand_return_start_target + blend * (
                    stand_return_state.joint_positions - stand_return_start_target
                )
                robot.command_joint_position_target(
                    stand_return_target,
                    stand_return_stiffness,
                    stand_return_damping,
                )

            if args.rehearsal_unlimited_slew:
                target_rate_limit = _rehearsal_target_rate_limit(
                    fsm.state,
                    unlimited_slew=True,
                )
                robot.set_target_rate_limit(target_rate_limit)
            balance_offset = fsm.update_balance(robot.sim_dt)
            robot.step_physics(balance_offset)
            logger.append(robot, operator, fsm, policy)
            if viewer is not None:
                if not viewer.is_running():
                    stop_reason = "viewer closed"
                    break
                viewer.sync()
                remaining_s = robot.sim_dt - (time.monotonic() - step_start)
                if remaining_s > 0.0:
                    time.sleep(remaining_s)
    except KeyboardInterrupt:
        stop_reason = "interrupted by operator"
    finally:
        if viewer is not None:
            viewer.close()

    final_time_s = float(robot.data.time)
    state_timeline.finish(final_time_s)
    metadata["stop_reason"] = stop_reason
    metadata["final_time_s"] = final_time_s
    metadata["maximum_control_duration_s"] = robot.maximum_control_duration_s
    metadata["control_deadline_miss_count"] = robot.control_deadline_miss_count
    metadata["state_timeline"] = [
        {
            "state": interval.state.value,
            "entry_time_s": interval.entry_time_s,
            "exit_time_s": interval.exit_time_s,
            "report": interval.report,
        }
        for interval in state_timeline.intervals
    ]
    logger.save()

    print(f"Stop reason: {stop_reason}")
    _print_timeline(state_timeline)
    _print_state_summaries(state_timeline, logger.arrays(), robot)
    abort_ticks = [tick for tick in control_ticks if tick.abort]
    for tick in abort_ticks:
        print(
            f"Abort proof: sampled t={tick.time_s:.3f} s, state={tick.state_before.value}, "
            f"episode_step={tick.episode_step_before}, flight_start_step={fsm.flight_start_step}, "
            f"state_after={tick.state_after.value}"
        )
    passed, result = _scenario_result(
        args.scenario,
        fsm,
        control_ticks,
        final_time_s,
        args.max_duration,
        expected_jump_count,
    )
    upright_passed, upright_result = _prejump_hold_upright_result(logger.arrays())
    print(f"Upright hold audit: {'PASS' if upright_passed else 'FAIL'} — {upright_result}")
    passed = passed and upright_passed
    if not upright_passed:
        result = f"upright hold failed: {upright_result}"
    if args.unmeasured_ground_validation:
        contact_passed, contact_result = _unmeasured_ground_contact_result(logger.arrays(), fsm.flight_start_step)
        print(f"Hidden-contact audit: {'PASS' if contact_passed else 'FAIL'} — {contact_result}")
        passed = passed and contact_passed
        if not contact_passed:
            result = f"FSM path result passed but hidden-contact audit failed: {contact_result}"
    if args.scenario in ("nominal", "repeat") and not args.contactless_gantry_rehearsal:
        command_goals = repeat_goals if args.scenario == "repeat" else (goal,)
        command_passed, command_result = _command_tracking_result(logger.arrays(), command_goals)
        print(f"Command tracking: {'PASS' if command_passed else 'FAIL'} — {command_result}")
        passed = passed and command_passed
        if not command_passed:
            result = f"FSM path result passed but command tracking failed: {command_result}"
    if args.require_hardware_margin:
        margin_passed, margin_result = _hardware_margin_result(logger.arrays(), robot)
        print(f"Hardware margin: {'PASS' if margin_passed else 'FAIL'} — {margin_result}")
        passed = passed and margin_passed
        if not margin_passed:
            result = f"stand duration completed but hardware margin failed: {margin_result}"
    if stand_return_state is not None:
        return_error = np.abs(robot.joint_positions - stand_return_state.joint_positions)
        return_worst_index = int(np.argmax(return_error))
        print(
            f"Stand-return final error: {return_error[return_worst_index]:.3f} rad "
            f"({robot.joint_names[return_worst_index]}), "
            f"max_speed={np.max(np.abs(robot.joint_velocities)):.3f} rad/s"
        )
    print(f"Scenario result: {'PASS' if passed else 'INCOMPLETE'} — {result}")
    print(
        f"Control timing: max={1.0e3 * robot.maximum_control_duration_s:.3f} ms, "
        f"deadline_misses={robot.control_deadline_miss_count}"
    )
    print(f"Wrote {len(logger.values['time'])} samples to {logger.output_path}")


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()

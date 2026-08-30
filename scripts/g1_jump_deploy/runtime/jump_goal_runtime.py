# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared observation and action runtime for deployed G1 jump policies.

The runtime accepts sensor and odometry quaternions in WXYZ order. The goal
quaternion stored in the policy observation follows the manifest's XYZW order.
Keeping the two conventions explicit is important because Unitree's IMU uses
WXYZ while the exported command observation uses XYZW.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_JOINT_COUNT = 23
_OBSERVATION_DIM = 326
_PHASE_NAMES = ("IDLE", "CROUCH", "TAKEOFF", "FLIGHT", "LAND", "STAND")
_EXPECTED_TERM_SHAPES = {
    "joint_pos": (_JOINT_COUNT, 4),
    "joint_vel": (_JOINT_COUNT, 4),
    "goal_remaining": (3, 4),
    "base_ang_vel": (3, 4),
    "projected_gravity": (3, 4),
    "last_action": (_JOINT_COUNT, 1),
    "goal_command": (7, 1),
    "reference_preview": (70, 1),
    "jump_phase": (6, 1),
}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {path} must be an object.")
    return value


def _sequence(value: Any, path: str, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Manifest field {path} must be an array.")
    if length is not None and len(value) != length:
        raise ValueError(f"Manifest field {path} must contain {length} values, got {len(value)}.")
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Manifest field {path} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Manifest field {path} must be a non-negative integer.")
    return value


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Manifest field {path} must be a finite number.")
    return float(value)


def _float_array(value: Any, path: str, length: int, *, dtype: np.dtype[Any]) -> np.ndarray:
    values = _sequence(value, path, length)
    try:
        result = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest field {path} must contain only numbers.") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Manifest field {path} must contain {length} finite numbers.")
    return result


def _observation_scale(value: Any, path: str, length: int, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Expand one observation-term scale to its per-component vector."""
    if value is None:
        return np.ones(length, dtype=dtype)
    if isinstance(value, bool):
        raise ValueError(f"Manifest field {path} must be a finite number or an array of {length} numbers.")
    if isinstance(value, (int, float)):
        scale = float(value)
        if not math.isfinite(scale):
            raise ValueError(f"Manifest field {path} must be finite.")
        return np.full(length, scale, dtype=dtype)
    return _float_array(value, path, length, dtype=dtype)


def _float_pairs(value: Any, path: str, length: int, *, dtype: np.dtype[Any]) -> np.ndarray:
    values = _sequence(value, path, length)
    try:
        result = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest field {path} must contain only numeric pairs.") from exc
    if result.shape != (length, 2) or not np.all(np.isfinite(result)):
        raise ValueError(f"Manifest field {path} must contain {length} finite [low, high] pairs.")
    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError(f"Manifest field {path} must satisfy low < high for every pair.")
    return result


def _vector(value: Any, length: int, name: str, *, dtype: np.dtype[Any] = np.dtype(np.float64)) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} numeric values.") from exc
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _quaternion_wxyz(value: Any, name: str) -> np.ndarray:
    """Validate and normalize a quaternion supplied in WXYZ order."""
    quaternion = _vector(value, 4, f"{name} (WXYZ)")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must be a non-zero quaternion in WXYZ order.")
    return quaternion / norm


def _quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by a normalized WXYZ quaternion."""
    scalar = quaternion[0]
    axis = quaternion[1:]
    twice_cross = 2.0 * np.cross(axis, vector)
    return vector + scalar * twice_cross + np.cross(axis, twice_cross)


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose two normalized WXYZ quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _yaw_rotate_inverse_wxyz(quaternion: np.ndarray, vector_w: np.ndarray) -> np.ndarray:
    """Express a world vector in the yaw-only frame of a WXYZ quaternion."""
    w, x, y, z = quaternion
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        (
            cosine * vector_w[0] + sine * vector_w[1],
            -sine * vector_w[0] + cosine * vector_w[1],
            vector_w[2],
        ),
        dtype=np.float64,
    )


class JumpGoalRuntime:
    """Build jump-policy observations and joint targets using only NumPy.

    All runtime pose, odometry, and IMU quaternions are accepted in WXYZ
    order. The goal command quaternion is emitted in XYZW order, as required
    by ``goal.quat_order`` in deployment manifest schemas 1.2 through 1.7. Depending on
    ``goal.orientation_mode``, it is either trigger-relative or the remaining
    rotation from the current IMU attitude to the target.

    Args:
        manifest_path: Path to ``deploy_manifest.json``. Its two table files
            must be siblings named by the manifest.
        freeze_during_flight: Whether to hold goal odometry during FLIGHT for
            policies whose manifest uses ``flight_frozen`` goal feedback. This
            has no effect on a ``latched`` policy, which never uses live
            odometry.

    Raises:
        ValueError: If the manifest or either table violates its schema.
    """

    def __init__(self, manifest_path: str | Path, *, freeze_during_flight: bool = True):  # noqa: C901
        if not isinstance(freeze_during_flight, bool):
            raise TypeError("freeze_during_flight must be a boolean.")

        self._manifest_path = Path(manifest_path).resolve()
        try:
            with self._manifest_path.open(encoding="utf-8") as stream:
                raw = json.load(stream)
        except OSError as exc:
            raise FileNotFoundError(f"Cannot read deployment manifest: {self._manifest_path}.") from exc
        manifest = _mapping(raw, "<root>")
        schema_version = manifest.get("schema_version")
        if schema_version not in ("1.2", "1.3", "1.4", "1.5", "1.6", "1.7"):
            raise ValueError(
                f"Expected deploy manifest schema 1.2 through 1.7, got {schema_version!r}; "
                "re-export the deployment bundle."
            )
        self._schema_version = schema_version
        self._policy_dtype = np.dtype(np.float64)

        control = _mapping(manifest.get("control"), "control")
        self._episode_steps = _positive_int(control.get("episode_steps"), "control.episode_steps")

        joints = _mapping(manifest.get("joints"), "joints")
        joint_names = _sequence(joints.get("names"), "joints.names", _JOINT_COUNT)
        if not all(isinstance(name, str) and name for name in joint_names):
            raise ValueError("Manifest field joints.names must contain non-empty strings.")
        if len(set(joint_names)) != _JOINT_COUNT:
            raise ValueError("Manifest field joints.names must contain 23 unique names.")
        self._default_pos = _float_array(
            joints.get("default_pos"), "joints.default_pos", _JOINT_COUNT, dtype=self._policy_dtype
        )
        self._default_vel = _float_array(
            joints.get("default_vel"), "joints.default_vel", _JOINT_COUNT, dtype=self._policy_dtype
        )
        if schema_version in ("1.5", "1.6", "1.7"):
            position_limits = _float_pairs(
                joints.get("position_limits"),
                "joints.position_limits",
                _JOINT_COUNT,
                dtype=self._policy_dtype,
            )
            if np.any(self._default_pos < position_limits[:, 0]) or np.any(self._default_pos > position_limits[:, 1]):
                raise ValueError("Manifest joints.default_pos must lie within joints.position_limits.")

        observation = _mapping(manifest.get("observation"), "observation")
        self._observation_dim = _positive_int(observation.get("total_dim"), "observation.total_dim")
        if self._observation_dim != _OBSERVATION_DIM:
            raise ValueError(f"Schema {schema_version} requires observation.total_dim={_OBSERVATION_DIM}.")
        if observation.get("history_order") != "oldest_first":
            raise ValueError("Only observation.history_order='oldest_first' is supported.")
        if observation.get("history_layout") != "history_major":
            raise ValueError("Only observation.history_layout='history_major' is supported.")

        self._terms: dict[str, dict[str, Any]] = {}
        occupied = np.zeros(self._observation_dim, dtype=np.bool_)
        for index, term_value in enumerate(_sequence(observation.get("terms"), "observation.terms")):
            term = _mapping(term_value, f"observation.terms[{index}]")
            name = term.get("name")
            if not isinstance(name, str) or not name or name in self._terms:
                raise ValueError(f"observation.terms[{index}].name must be a unique non-empty string.")
            offset = _nonnegative_int(term.get("offset"), f"observation.terms[{index}].offset")
            step_dim = _positive_int(term.get("step_dim"), f"observation.terms[{index}].step_dim")
            history = _positive_int(term.get("history"), f"observation.terms[{index}].history")
            total = _positive_int(term.get("total"), f"observation.terms[{index}].total")
            if total != step_dim * history:
                raise ValueError(f"Observation term {name!r} has inconsistent dimensions.")
            end = offset + total
            if end > self._observation_dim or np.any(occupied[offset:end]):
                raise ValueError(f"Observation term {name!r} overlaps another term or exceeds total_dim.")
            occupied[offset:end] = True
            self._terms[name] = {
                "offset": offset,
                "step_dim": step_dim,
                "history": history,
                "total": total,
                "scale": _observation_scale(
                    term.get("scale"),
                    f"observation.terms[{index}].scale",
                    step_dim,
                    dtype=np.dtype(np.float32),
                ),
            }
        if set(self._terms) != set(_EXPECTED_TERM_SHAPES) or not np.all(occupied):
            raise ValueError("Manifest observation terms must be exactly the fixed schema and cover total_dim.")
        for name, expected_shape in _EXPECTED_TERM_SHAPES.items():
            actual_shape = (self._terms[name]["step_dim"], self._terms[name]["history"])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Observation term {name!r} must have step_dim/history {expected_shape[0]}/{expected_shape[1]}."
                )

        action = _mapping(manifest.get("action"), "action")
        if _positive_int(action.get("dim"), "action.dim") != _JOINT_COUNT:
            raise ValueError(f"Schema {schema_version} requires action.dim={_JOINT_COUNT}.")
        self._action_scale = _float_array(action.get("scale"), "action.scale", _JOINT_COUNT, dtype=self._policy_dtype)
        self._action_offset = _float_array(
            action.get("offset"), "action.offset", _JOINT_COUNT, dtype=self._policy_dtype
        )
        self._filter_alpha = _float_array(
            action.get("filter_alpha"), "action.filter_alpha", _JOINT_COUNT, dtype=self._policy_dtype
        )
        if not np.array_equal(self._action_offset, self._default_pos):
            raise ValueError("Manifest action.offset must exactly equal joints.default_pos.")
        if np.any(self._filter_alpha <= 0.0) or np.any(self._filter_alpha > 1.0):
            raise ValueError("Manifest action.filter_alpha values must be in (0, 1].")
        delay = _mapping(action.get("delay_steps"), "action.delay_steps")
        self._delay_min = _nonnegative_int(delay.get("min"), "action.delay_steps.min")
        self._delay_max = _nonnegative_int(delay.get("max"), "action.delay_steps.max")
        if self._delay_min > self._delay_max:
            raise ValueError("action.delay_steps.min must not exceed action.delay_steps.max.")
        # A null clip is legitimate; see deploy_mujoco for why this policy needs one.
        _raw_clip = action.get("clip")
        self._action_clip = (
            None
            if _raw_clip is None
            else _float_pairs(_raw_clip, "action.clip", _JOINT_COUNT, dtype=self._policy_dtype)
        )
        expected_formula = "q_target = alpha*clip(offset + scale*a_delayed) + (1-alpha)*q_target_prev"
        if action.get("formula") != expected_formula:
            raise ValueError(f"Manifest action.formula does not match the schema {schema_version} transform.")

        reference = _mapping(manifest.get("reference"), "reference")
        phase_names = tuple(_sequence(reference.get("phase_names"), "reference.phase_names"))
        if phase_names != _PHASE_NAMES:
            raise ValueError(f"reference.phase_names must equal {_PHASE_NAMES}.")
        self._flight_phase = phase_names.index("FLIGHT")

        goal = _mapping(manifest.get("goal"), "goal")
        if goal.get("quat_order") != "xyzw":
            raise ValueError(f"Schema {schema_version} requires goal.quat_order='xyzw'.")
        ranges = _mapping(goal.get("ranges"), "goal.ranges")
        self._goal_ranges: dict[str, tuple[float, float]] = {}
        for name in ("pos_x", "pos_y", "roll", "pitch", "yaw"):
            bounds = _float_array(ranges.get(name), f"goal.ranges.{name}", 2, dtype=np.dtype(np.float64))
            if bounds[0] > bounds[1]:
                raise ValueError(f"goal.ranges.{name} lower bound exceeds its upper bound.")
            self._goal_ranges[name] = (float(bounds[0]), float(bounds[1]))
        for name in ("roll", "pitch"):
            lower, upper = self._goal_ranges[name]
            if not lower <= 0.0 <= upper:
                raise ValueError(f"goal.ranges.{name} must include the arm() default of zero.")
        flight_freeze = _mapping(goal.get("flight_freeze"), "goal.flight_freeze")
        manifest_freeze_enabled = flight_freeze.get("enabled")
        if not isinstance(manifest_freeze_enabled, bool):
            raise ValueError("goal.flight_freeze.enabled must be a boolean.")
        remaining_mode = goal.get("remaining_mode")
        if remaining_mode is None:
            # Schema 1.2 bundles exported before remaining_mode used this flag
            # as the complete observation contract.
            remaining_mode = "flight_frozen" if manifest_freeze_enabled else "live"
        if remaining_mode not in ("live", "flight_frozen", "latched"):
            raise ValueError("goal.remaining_mode must be one of 'live', 'flight_frozen', or 'latched'.")
        if remaining_mode == "flight_frozen" and not manifest_freeze_enabled:
            raise ValueError("goal.remaining_mode='flight_frozen' requires flight_freeze.enabled=true.")
        if freeze_during_flight and remaining_mode == "live":
            raise ValueError("freeze_during_flight cannot be enabled when the manifest disables flight freezing.")
        self._goal_remaining_mode = remaining_mode
        orientation_mode = goal.get("orientation_mode", "trigger_relative")
        if orientation_mode not in ("trigger_relative", "remaining"):
            raise ValueError("goal.orientation_mode must be one of 'trigger_relative' or 'remaining'.")
        self._goal_orientation_mode = orientation_mode
        if schema_version in ("1.6", "1.7"):
            retrigger_indicator = _mapping(goal.get("retrigger_indicator"), "goal.retrigger_indicator")
            expected_mode = "goal_command_z" if schema_version == "1.6" else "goal_command_z_affine_pos_x"
            if retrigger_indicator.get("mode") != expected_mode:
                raise ValueError(
                    f"goal.retrigger_indicator.mode must be {expected_mode!r} for schema {schema_version}."
                )
            fresh_value = _finite_float(
                retrigger_indicator.get("fresh_value"),
                "goal.retrigger_indicator.fresh_value",
            )
            retrigger_value = _finite_float(
                retrigger_indicator.get("retrigger_value"),
                "goal.retrigger_indicator.retrigger_value",
            )
            if fresh_value != 0.0:
                raise ValueError("goal.retrigger_indicator.fresh_value must be zero.")
            if retrigger_value == fresh_value or abs(retrigger_value) > 1.0:
                raise ValueError(
                    "goal.retrigger_indicator.retrigger_value must differ from fresh_value and lie in [-1, 1]."
                )
            goal_pos_x_scale = (
                _finite_float(
                    retrigger_indicator.get("goal_pos_x_scale"),
                    "goal.retrigger_indicator.goal_pos_x_scale",
                )
                if schema_version == "1.7"
                else 0.0
            )
            if schema_version == "1.7":
                if goal_pos_x_scale == 0.0:
                    raise ValueError("goal.retrigger_indicator.goal_pos_x_scale must be non-zero.")
                pos_x_bounds = self._goal_ranges["pos_x"]
                maximum_indicator = max(abs(retrigger_value + goal_pos_x_scale * bound) for bound in pos_x_bounds)
                if maximum_indicator > 1.0:
                    raise ValueError("The affine retrigger indicator must remain in [-1, 1] over goal.ranges.pos_x.")
        else:
            fresh_value = 0.0
            retrigger_value = 0.0
            goal_pos_x_scale = 0.0
        self._retrigger_fresh_value = fresh_value
        self._retrigger_value = retrigger_value
        self._retrigger_goal_pos_x_scale = goal_pos_x_scale
        self.freeze_during_flight = freeze_during_flight

        tables = _mapping(manifest.get("tables"), "tables")
        self._reference_preview = self._load_table(
            tables.get("reference_preview"),
            self._terms["reference_preview"]["step_dim"],
            "tables.reference_preview",
        )
        self._jump_phase = self._load_table(
            tables.get("jump_phase"), self._terms["jump_phase"]["step_dim"], "tables.jump_phase"
        )
        phase_is_binary = np.logical_or(np.isclose(self._jump_phase, 0.0), np.isclose(self._jump_phase, 1.0))
        if not np.all(phase_is_binary) or not np.allclose(np.sum(self._jump_phase, axis=1), 1.0):
            raise ValueError("Jump phase table must contain one-hot rows.")
        stand_steps = np.flatnonzero(np.argmax(self._jump_phase, axis=1) == phase_names.index("STAND"))
        if stand_steps.size == 0:
            raise ValueError("Jump phase table must contain at least one STAND row.")
        self._stand_reference_step = int(stand_steps[-1])

        self._rng = np.random.default_rng()
        self._pose_command_b: np.ndarray | None = None
        self._goal_pos_w: np.ndarray | None = None
        self._goal_quat_wxyz: np.ndarray | None = None
        self._previous_target: np.ndarray | None = None
        self._last_preflight_goal_remaining: np.ndarray | None = None
        self._last_action = np.zeros(_JOINT_COUNT, dtype=self._policy_dtype)
        self._action_buffer: list[np.ndarray] = []
        self._histories: dict[str, np.ndarray] = {}
        self._delay_steps = self._delay_min
        self._step_counter = 0
        self._previous_phase = -1
        self._triggered = False
        self._active_retrigger_value = self._retrigger_fresh_value
        self._active_retrigger = False

    def _load_table(self, filename: Any, width: int, field: str) -> np.ndarray:
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Manifest field {field} must be a non-empty filename.")
        relative_path = Path(filename)
        if relative_path.is_absolute() or relative_path.name != filename:
            raise ValueError(f"Manifest field {field} must name a table beside the manifest.")
        table_path = (self._manifest_path.parent / relative_path).resolve()
        if table_path.parent != self._manifest_path.parent:
            raise ValueError(f"Manifest field {field} must name a table beside the manifest.")
        try:
            table = np.load(table_path, allow_pickle=False)
        except OSError as exc:
            raise FileNotFoundError(f"Cannot read deployment table: {table_path}.") from exc
        expected_shape = (self._episode_steps, width)
        if table.shape != expected_shape:
            raise ValueError(f"{field} table must have shape {expected_shape}, got {table.shape}.")
        result = np.asarray(table, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{field} table contains non-finite values.")
        return result

    @property
    def done(self) -> bool:
        """Whether all manifest policy steps have been produced."""
        return self._triggered and self._step_counter >= self._episode_steps

    @property
    def stand_reference_step(self) -> int:
        """Last manifest policy step belonging to the STAND phase."""

        return self._stand_reference_step

    @property
    def delayed_action(self) -> np.ndarray:
        """Most recent delayed raw action in manifest joint order."""
        if not self._triggered:
            raise RuntimeError("trigger() must be called before reading delayed_action.")
        return self._last_action.copy()

    @property
    def goal_position_w(self) -> np.ndarray:
        """Latched landing goal position in world coordinates [m]."""
        if not self._triggered or self._goal_pos_w is None:
            raise RuntimeError("trigger() must be called before reading goal_position_w.")
        return self._goal_pos_w.copy()

    def arm(self, dx: float, dy: float, dyaw: float, *, roll: float = 0.0, pitch: float = 0.0) -> None:
        """Latch an operator goal relative to the trigger body pose.

        Args:
            dx: Forward landing displacement [m].
            dy: Lateral landing displacement [m].
            dyaw: Landing heading displacement [rad].
            roll: Landing roll displacement [rad].
            pitch: Landing pitch displacement [rad].

        Raises:
            RuntimeError: If an episode is currently active.
            ValueError: If a value is non-finite or outside its manifest range.
        """
        if self._triggered and not self.done:
            raise RuntimeError("Cannot arm a new goal during an active episode.")
        values = {"pos_x": dx, "pos_y": dy, "roll": roll, "pitch": pitch, "yaw": dyaw}
        resolved: dict[str, float] = {}
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Goal {name} must be a finite number.")
            result = float(value)
            lower, upper = self._goal_ranges[name]
            if not lower <= result <= upper:
                raise ValueError(f"Goal {name}={result} is outside manifest range [{lower}, {upper}].")
            resolved[name] = result

        cr, sr = math.cos(0.5 * resolved["roll"]), math.sin(0.5 * resolved["roll"])
        cp, sp = math.cos(0.5 * resolved["pitch"]), math.sin(0.5 * resolved["pitch"])
        cy, sy = math.cos(0.5 * resolved["yaw"]), math.sin(0.5 * resolved["yaw"])
        quaternion_xyzw = np.asarray(
            (
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ),
            dtype=self._policy_dtype,
        )
        if quaternion_xyzw[3] < 0.0:
            quaternion_xyzw *= -1.0
        self._pose_command_b = np.concatenate(
            (
                np.asarray((resolved["pos_x"], resolved["pos_y"], 0.0), dtype=self._policy_dtype),
                quaternion_xyzw,
            )
        )
        self._triggered = False

    def trigger(
        self,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
        joint_pos: np.ndarray,
        *,
        action_delay_steps: int | None = None,
        goal_pos_z_w: float | None = None,
        retrigger: bool = False,
    ) -> None:
        """Start an episode from measured robot state.

        Args:
            root_pos_w: Trigger root position in world coordinates [m], shape
                ``(3,)``.
            root_quat_w: Trigger world-from-body quaternion in WXYZ order,
                shape ``(4,)``.
            joint_pos: Measured joint positions [rad] in manifest joint order,
                shape ``(23,)``.
            action_delay_steps: Fixed raw-action delay [policy steps]. If
                omitted, sample from the manifest range.
            goal_pos_z_w: Optional landing height in world coordinates [m]. If
                omitted, retain the trigger root height.
            retrigger: Whether this episode begins from the preceding
                policy-native landing. Schema 1.6 and 1.7 policies expose this state in
                the goal observation; older policies ignore it.

        Raises:
            RuntimeError: If no operator goal has been armed.
            ValueError: If an input has the wrong shape or invalid values.
        """
        if self._pose_command_b is None:
            raise RuntimeError("arm() must be called before trigger().")
        if not isinstance(retrigger, bool):
            raise ValueError("retrigger must be a boolean.")
        root_position = _vector(root_pos_w, 3, "root_pos_w")
        root_quaternion = _quaternion_wxyz(root_quat_w, "root_quat_w")
        measured_joint_pos = _vector(joint_pos, _JOINT_COUNT, "joint_pos", dtype=self._policy_dtype)
        self._anchor_goal(root_position, root_quaternion, goal_pos_z_w)
        self._previous_target = measured_joint_pos.copy()
        self._last_action.fill(0.0)
        self._action_buffer.clear()
        self._histories.clear()
        if action_delay_steps is not None:
            if isinstance(action_delay_steps, bool) or not isinstance(action_delay_steps, int):
                raise ValueError("action_delay_steps must be an integer.")
            if not self._delay_min <= action_delay_steps <= self._delay_max:
                raise ValueError(
                    f"action_delay_steps={action_delay_steps} is outside manifest range "
                    f"[{self._delay_min}, {self._delay_max}]."
                )
            self._delay_steps = action_delay_steps
        elif self._delay_min == self._delay_max:
            self._delay_steps = self._delay_min
        else:
            self._delay_steps = int(self._rng.integers(self._delay_min, self._delay_max + 1))
        self._step_counter = 0
        self._previous_phase = -1
        self._triggered = True
        self._active_retrigger_value = self._retrigger_value if retrigger else self._retrigger_fresh_value
        self._active_retrigger = retrigger

    def reanchor_goal(
        self,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
        *,
        goal_pos_z_w: float | None = None,
    ) -> None:
        """Reanchor a prepared goal without resetting policy history.

        This is used after a frozen phase-zero preparation interval so the
        commanded displacement remains relative to the separately confirmed
        jump pose. Observation history, delayed actions, filtered targets, and
        the episode clock are preserved.

        Args:
            root_pos_w: Confirmed root position in world coordinates [m],
                shape ``(3,)``.
            root_quat_w: Confirmed world-from-body quaternion in WXYZ order,
                shape ``(4,)``.
            goal_pos_z_w: Optional landing height in world coordinates [m]. If
                omitted, retain the confirmed root height.

        Raises:
            RuntimeError: If :meth:`trigger` has not initialized preparation.
            ValueError: If an input has the wrong shape or invalid values.
        """
        if not self._triggered:
            raise RuntimeError("trigger() must be called before reanchor_goal().")
        root_position = _vector(root_pos_w, 3, "root_pos_w")
        root_quaternion = _quaternion_wxyz(root_quat_w, "root_quat_w")
        self._anchor_goal(root_position, root_quaternion, goal_pos_z_w)

    def cancel(self) -> None:
        """Cancel an armed or prepared episode and clear its mutable state."""
        self._pose_command_b = None
        self._goal_pos_w = None
        self._goal_quat_wxyz = None
        self._previous_target = None
        self._last_preflight_goal_remaining = None
        self._last_action.fill(0.0)
        self._action_buffer.clear()
        self._histories.clear()
        self._step_counter = 0
        self._previous_phase = -1
        self._triggered = False
        self._active_retrigger_value = self._retrigger_fresh_value
        self._active_retrigger = False

    def _anchor_goal(
        self,
        root_position: np.ndarray,
        root_quaternion: np.ndarray,
        goal_pos_z_w: float | None,
    ) -> None:
        """Resolve the armed body-relative command at one root pose."""
        if self._pose_command_b is None:
            raise RuntimeError("arm() must be called before anchoring a goal.")
        relative_position = self._pose_command_b[:3].astype(np.float64)
        self._goal_pos_w = root_position + _quat_rotate_wxyz(root_quaternion, relative_position)
        if goal_pos_z_w is not None:
            if (
                isinstance(goal_pos_z_w, bool)
                or not isinstance(goal_pos_z_w, (int, float))
                or not math.isfinite(float(goal_pos_z_w))
            ):
                raise ValueError("goal_pos_z_w must be a finite number.")
            self._goal_pos_w[2] = float(goal_pos_z_w)
        command_quaternion_xyzw = self._pose_command_b[3:].astype(np.float64)
        command_quaternion_wxyz = command_quaternion_xyzw[[3, 0, 1, 2]]
        self._goal_quat_wxyz = _quat_multiply_wxyz(root_quaternion, command_quaternion_wxyz)
        initial_remaining = _yaw_rotate_inverse_wxyz(root_quaternion, self._goal_pos_w - root_position)
        self._last_preflight_goal_remaining = initial_remaining.astype(np.float32)

    def step(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        imu_quat: np.ndarray,
        odom_pos_w: np.ndarray,
        odom_quat_w: np.ndarray,
        *,
        advance: bool = True,
        reference_step: int | None = None,
    ) -> np.ndarray:
        """Build the next policy observation and optionally advance the episode.

        Args:
            joint_pos: Measured joint positions [rad] in manifest joint order,
                shape ``(23,)``.
            joint_vel: Measured joint velocities [rad/s] in manifest joint
                order, shape ``(23,)``.
            base_ang_vel: Base angular velocity in the body frame [rad/s],
                shape ``(3,)``.
            imu_quat: IMU world-from-body quaternion in WXYZ order, shape
                ``(4,)``. This is orientation, not accelerometer data.
            odom_pos_w: Odometry root position in world coordinates [m], shape
                ``(3,)``.
            odom_quat_w: Odometry world-from-body quaternion in WXYZ order,
                shape ``(4,)``.
            advance: Whether to advance the reference and episode clock. Set
                this to ``False`` only while preparing the policy at phase zero.
            reference_step: Optional manifest row used for reference preview
                and jump phase while the episode clock is frozen. This permits
                policy-native standing at :attr:`stand_reference_step`,
                including after the finite episode has completed.

        Returns:
            The 326-element policy observation as ``float32``.

        Raises:
            RuntimeError: If the episode has not been triggered or is done.
            ValueError: If an input has the wrong shape or invalid values, if
                ``advance`` is not a boolean, or if ``reference_step`` is invalid.
        """
        if not isinstance(advance, bool):
            raise ValueError("advance must be a boolean.")
        if reference_step is not None:
            if isinstance(reference_step, bool) or not isinstance(reference_step, int):
                raise ValueError("reference_step must be an integer or None.")
            if not 0 <= reference_step < self._episode_steps:
                raise ValueError(f"reference_step must be in [0, {self._episode_steps - 1}].")
            if advance:
                raise ValueError("reference_step requires advance=False.")
        if not self._triggered:
            raise RuntimeError("trigger() must be called before step().")
        if self.done and reference_step is None:
            raise RuntimeError("The episode is done; trigger() before requesting another observation.")
        sample_step = self._step_counter if reference_step is None else reference_step
        measured_joint_pos = _vector(joint_pos, _JOINT_COUNT, "joint_pos", dtype=self._policy_dtype)
        measured_joint_vel = _vector(joint_vel, _JOINT_COUNT, "joint_vel", dtype=self._policy_dtype)
        angular_velocity = _vector(base_ang_vel, 3, "base_ang_vel", dtype=self._policy_dtype)
        imu_quaternion = _quaternion_wxyz(imu_quat, "imu_quat")
        odom_position = _vector(odom_pos_w, 3, "odom_pos_w")
        odom_quaternion = _quaternion_wxyz(odom_quat_w, "odom_quat_w")
        if self._goal_pos_w is None or self._goal_quat_wxyz is None or self._pose_command_b is None:
            raise RuntimeError("trigger() did not initialize the latched goal.")

        live_goal_remaining = _yaw_rotate_inverse_wxyz(odom_quaternion, self._goal_pos_w - odom_position).astype(
            np.float32
        )
        phase = int(np.argmax(self._jump_phase[sample_step]))
        if self._goal_remaining_mode == "latched":
            if self._last_preflight_goal_remaining is None:
                raise RuntimeError("Latched goal feedback has no trigger-time value.")
            goal_remaining = self._last_preflight_goal_remaining.copy()
        elif self._goal_remaining_mode == "flight_frozen" and self.freeze_during_flight and phase == self._flight_phase:
            if self._last_preflight_goal_remaining is None:
                raise RuntimeError("Flight goal freeze has no pre-flight value.")
            goal_remaining = self._last_preflight_goal_remaining.copy()
        else:
            goal_remaining = live_goal_remaining
            self._last_preflight_goal_remaining = live_goal_remaining.copy()

        world_gravity = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
        imu_inverse = imu_quaternion.copy()
        imu_inverse[1:] *= -1.0
        projected_gravity = _quat_rotate_wxyz(imu_inverse, world_gravity).astype(np.float32)

        goal_command = self._pose_command_b.copy()
        if self._goal_orientation_mode == "remaining":
            remaining_quaternion_wxyz = _quat_multiply_wxyz(imu_inverse, self._goal_quat_wxyz)
            remaining_quaternion_wxyz /= np.linalg.norm(remaining_quaternion_wxyz)
            if remaining_quaternion_wxyz[0] < 0.0:
                remaining_quaternion_wxyz *= -1.0
            goal_command[3:] = remaining_quaternion_wxyz[[1, 2, 3, 0]]
        goal_command[2] = self._active_retrigger_value
        if self._active_retrigger:
            goal_command[2] += self._retrigger_goal_pos_x_scale * goal_command[0]

        samples = {
            "joint_pos": measured_joint_pos - self._default_pos,
            "joint_vel": measured_joint_vel - self._default_vel,
            "goal_remaining": goal_remaining,
            "base_ang_vel": angular_velocity,
            "projected_gravity": projected_gravity,
            "last_action": self._last_action,
            "goal_command": goal_command,
            "reference_preview": self._reference_preview[sample_step],
            "jump_phase": self._jump_phase[sample_step],
        }
        observation = self._assemble_observation(samples)
        self._previous_phase = phase
        if advance:
            self._step_counter += 1
        return observation

    def _assemble_observation(self, samples: dict[str, np.ndarray]) -> np.ndarray:
        observation = np.empty(self._observation_dim, dtype=np.float32)
        for name, term in self._terms.items():
            sample = np.asarray(samples[name], dtype=np.float32)
            step_dim = term["step_dim"]
            if sample.shape != (step_dim,) or not np.all(np.isfinite(sample)):
                raise ValueError(f"Observation sample {name!r} must have shape ({step_dim},) and be finite.")
            sample = sample * term["scale"]
            history = term["history"]
            if history == 1:
                flat = sample
            elif name not in self._histories:
                # Isaac's CircularBuffer fills every slot from its first push.
                self._histories[name] = np.repeat(sample[None, :], history, axis=0)
                flat = self._histories[name].reshape(-1)
            else:
                values = self._histories[name]
                values[:-1] = values[1:].copy()
                values[-1] = sample
                flat = values.reshape(-1)
            offset = term["offset"]
            observation[offset : offset + term["total"]] = flat
        return observation

    def transform_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Transform a policy action into filtered joint position targets.

        Delay is applied before the manifest affine transform. The transformed
        position is clipped before the per-joint low-pass filter. The first raw
        action fills unavailable delay history, matching Isaac's
        :class:`DelayBuffer` first-push behavior.

        Args:
            raw_action: Raw policy action in manifest joint order, shape
                ``(23,)``.

        Returns:
            Filtered joint position targets [rad] in manifest joint order.

        Raises:
            RuntimeError: If the episode has not been triggered.
            ValueError: If the action has the wrong shape or invalid values.
        """
        if not self._triggered or self._previous_target is None:
            raise RuntimeError("trigger() must be called before transform_action().")
        action = _vector(raw_action, _JOINT_COUNT, "raw_action", dtype=self._policy_dtype)
        self._action_buffer.append(action.copy())
        if len(self._action_buffer) > self._delay_max + 1:
            self._action_buffer.pop(0)
        if len(self._action_buffer) > self._delay_steps:
            delayed_action = self._action_buffer[-self._delay_steps - 1]
        else:
            delayed_action = self._action_buffer[0]
        self._last_action = delayed_action.copy()

        raw_target = self._action_offset + self._action_scale * delayed_action
        if self._action_clip is None:
            clipped_target = raw_target
        else:
            clipped_target = np.clip(raw_target, self._action_clip[:, 0], self._action_clip[:, 1])
        target = self._filter_alpha * clipped_target + (1.0 - self._filter_alpha) * self._previous_target
        self._previous_target = target.astype(self._policy_dtype, copy=True)
        return self._previous_target.copy()

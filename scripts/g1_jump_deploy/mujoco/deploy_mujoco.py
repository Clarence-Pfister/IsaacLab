# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an exported G1 jump policy in MuJoCo at the deployment control rates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from model_overlay import apply_initial_ground_clearance, compose_model_xml
from physics_parity import PhysicsParityConfig, add_physics_parity_arguments, apply_physics_parity
from static_equilibrium import ground_contact_force, settle_static_equilibrium

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEPLOY_DIR = _SCRIPT_DIR.parent
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

from runtime import (  # noqa: E402
    JumpGoalRuntime,
    project_pd_position_target,
    project_position_target_to_lower_limit,
    saturate_torque_at_velocity_limit,
)

_REPO_ROOT = _SCRIPT_DIR.parents[2]
_DEFAULT_MANIFEST = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"
_DEFAULT_MODEL = _REPO_ROOT / "data_storage" / "g1_23dof_holo_compat.xml"
_DEFAULT_OVERLAY = _SCRIPT_DIR / "model_overlay.xml"
_SELF_CHECK_DURATION_S = 3.0


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


def _positive_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Manifest field {path} must be a finite number.")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"Manifest field {path} must be positive, got {result}.")
    return result


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Manifest field {path} must be a non-negative integer.")
    return value


def _float_array(value: Any, path: str, length: int) -> np.ndarray:
    values = _sequence(value, path, length)
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest field {path} must contain only numbers.") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Manifest field {path} must contain {length} finite numbers.")
    return result


def _float_pairs(value: Any, path: str, length: int) -> np.ndarray:
    values = _sequence(value, path, length)
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest field {path} must contain only numeric pairs.") from exc
    if result.shape != (length, 2) or not np.all(np.isfinite(result)):
        raise ValueError(f"Manifest field {path} must contain {length} finite [low, high] pairs.")
    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError(f"Manifest field {path} must satisfy low < high for every pair.")
    return result


def _build_name_permutations(
    policy_names: tuple[str, ...], backend_names: tuple[str, ...], backend_label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build and validate policy/backend index permutations from names.

    The first returned array maps each policy index to its backend index. The
    inverse maps each backend index to its policy index.
    """
    if len(policy_names) != len(backend_names):
        raise ValueError(f"Policy has {len(policy_names)} joints but {backend_label} has {len(backend_names)} entries.")
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("Manifest policy joint names must be unique.")
    if len(set(backend_names)) != len(backend_names):
        raise ValueError(f"{backend_label} target joint names must be unique.")
    policy_set = set(policy_names)
    backend_set = set(backend_names)
    if policy_set != backend_set:
        missing = sorted(policy_set - backend_set)
        extra = sorted(backend_set - policy_set)
        raise ValueError(f"{backend_label} joint names differ from the manifest. Missing={missing}, extra={extra}.")

    backend_index = {name: index for index, name in enumerate(backend_names)}
    policy_from_backend = np.asarray([backend_index[name] for name in policy_names], dtype=np.int32)
    expected_indices = np.arange(len(policy_names), dtype=np.int32)
    if not np.array_equal(np.sort(policy_from_backend), expected_indices):
        raise ValueError(f"Policy-to-{backend_label} permutation is not bijective: {policy_from_backend.tolist()}.")
    backend_from_policy = np.empty_like(policy_from_backend)
    backend_from_policy[policy_from_backend] = expected_indices
    if not np.array_equal(backend_from_policy[policy_from_backend], expected_indices):
        raise ValueError(f"{backend_label}-to-policy inverse permutation is invalid.")
    return policy_from_backend, backend_from_policy


def _load_reference_frame0(manifest: DeploymentManifest) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load frame-0 joints from the declared source and root pose from the manifest."""
    path = manifest.reference_source
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read reference motion CSV declared by the manifest: {path}.") from exc
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != manifest.reference_source_sha256:
        raise ValueError(
            "Reference motion CSV SHA-256 disagrees with manifest reference.source_sha256; "
            f"expected {manifest.reference_source_sha256}, got {actual_sha256}. Re-export the deployment bundle."
        )
    try:
        frame = next(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))
    except StopIteration as exc:
        raise ValueError(f"Reference motion CSV is empty: {path}.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Reference motion CSV is not valid UTF-8: {path}.") from exc

    required = manifest.joint_names
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Reference motion CSV is missing frame-0 columns: {missing}.")
    try:
        joint_pos = np.asarray([float(frame[name]) for name in manifest.joint_names], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Reference motion CSV frame 0 contains a non-numeric deployment value: {path}.") from exc
    if not np.all(np.isfinite(joint_pos)):
        raise ValueError("Reference motion CSV frame 0 contains non-finite values.")
    if not np.allclose(joint_pos, manifest.default_pos, rtol=0.0, atol=1e-6):
        maximum_error = float(np.max(np.abs(joint_pos - manifest.default_pos)))
        raise ValueError(
            "Reference frame-0 joint positions disagree with manifest joints.default_pos; "
            f"max error={maximum_error:.3e} rad."
        )
    root_quaternion_wxyz = np.roll(manifest.reference_root_quaternion_xyzw, 1)
    return joint_pos, manifest.reference_root_position.copy(), root_quaternion_wxyz


def _validate_reference_override(path: Path | None, manifest: DeploymentManifest) -> None:
    """Require the deprecated reference override to agree with the manifest."""
    if path is not None and path.resolve() != manifest.reference_source.resolve():
        raise ValueError(
            "--reference_csv must match manifest reference.source_csv; re-export instead of overriding it."
        )


class DeploymentManifest:
    """Validated view of deployment constants shared with the hardware runner."""

    _REQUIRED_TERMS = {
        "joint_pos",
        "joint_vel",
        "goal_remaining",
        "base_ang_vel",
        "projected_gravity",
        "last_action",
        "goal_command",
        "reference_preview",
        "jump_phase",
    }

    def __init__(self, path: Path):  # noqa: C901 - validates the fixed schema in one pass
        self.path = path.resolve()
        with self.path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        self.raw = _mapping(raw, "<root>")
        schema_version = self.raw.get("schema_version")
        if schema_version not in ("1.2", "1.3", "1.4", "1.5", "1.6", "1.7"):
            raise ValueError(
                f"Expected deploy manifest schema 1.2 through 1.7, got {schema_version!r}; "
                "re-export the deployment bundle."
            )
        self.schema_version = schema_version

        control = _mapping(self.raw.get("control"), "control")
        self.sim_dt = _positive_float(control.get("sim_dt"), "control.sim_dt")
        self.policy_dt = _positive_float(control.get("policy_dt"), "control.policy_dt")
        self.policy_hz = _positive_float(control.get("policy_hz"), "control.policy_hz")
        self.decimation = _nonnegative_int(control.get("decimation"), "control.decimation")
        if self.decimation == 0:
            raise ValueError("Manifest field control.decimation must be positive.")
        self.episode_steps = _nonnegative_int(control.get("episode_steps"), "control.episode_steps")
        if self.episode_steps == 0:
            raise ValueError("Manifest field control.episode_steps must be positive.")
        self.episode_duration_s = _positive_float(control.get("episode_duration_s"), "control.episode_duration_s")
        if not math.isclose(self.sim_dt, 0.002, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Schema 1.2 requires control.sim_dt=0.002, got {self.sim_dt}.")
        if self.decimation != 10:
            raise ValueError(f"Schema 1.2 requires control.decimation=10, got {self.decimation}.")
        if self.episode_steps != 152:
            raise ValueError(f"Schema 1.2 requires control.episode_steps=152, got {self.episode_steps}.")
        if not math.isclose(self.policy_dt, self.sim_dt * self.decimation, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("control.policy_dt must equal control.sim_dt * control.decimation.")
        if not math.isclose(self.policy_hz, 1.0 / self.policy_dt, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("control.policy_hz must equal 1 / control.policy_dt.")

        joints = _mapping(self.raw.get("joints"), "joints")
        joint_names = _sequence(joints.get("names"), "joints.names")
        if not joint_names or not all(isinstance(name, str) and name for name in joint_names):
            raise ValueError("Manifest field joints.names must contain non-empty strings.")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Manifest field joints.names contains duplicates.")
        self.joint_names = tuple(joint_names)
        self.joint_count = len(self.joint_names)
        _sequence(joints.get("unitree_sdk2_slots"), "joints.unitree_sdk2_slots", self.joint_count)
        self.default_pos = _float_array(joints.get("default_pos"), "joints.default_pos", self.joint_count)
        self.default_vel = _float_array(joints.get("default_vel"), "joints.default_vel", self.joint_count)
        self.joint_position_limits = (
            _float_pairs(joints.get("position_limits"), "joints.position_limits", self.joint_count)
            if self.schema_version in ("1.5", "1.6", "1.7")
            else None
        )

        observation = _mapping(self.raw.get("observation"), "observation")
        self.observation_dim = _nonnegative_int(observation.get("total_dim"), "observation.total_dim")
        if self.observation_dim != 326:
            raise ValueError(f"Schema 1.2 requires observation.total_dim=326, got {self.observation_dim}.")
        if observation.get("history_order") != "oldest_first":
            raise ValueError("Only observation.history_order='oldest_first' is supported.")
        if observation.get("history_layout") != "history_major":
            raise ValueError("Only observation.history_layout='history_major' is supported.")
        term_values = _sequence(observation.get("terms"), "observation.terms")
        self.terms: dict[str, dict[str, int | str]] = {}
        occupied = np.zeros(self.observation_dim, dtype=np.bool_)
        for index, term_value in enumerate(term_values):
            term = _mapping(term_value, f"observation.terms[{index}]")
            name = term.get("name")
            if not isinstance(name, str) or not name or name in self.terms:
                raise ValueError(f"observation.terms[{index}].name must be a unique non-empty string.")
            offset = _nonnegative_int(term.get("offset"), f"observation.terms[{index}].offset")
            step_dim = _nonnegative_int(term.get("step_dim"), f"observation.terms[{index}].step_dim")
            history = _nonnegative_int(term.get("history"), f"observation.terms[{index}].history")
            total = _nonnegative_int(term.get("total"), f"observation.terms[{index}].total")
            if step_dim == 0 or history == 0 or total != step_dim * history:
                raise ValueError(f"Observation term {name!r} has inconsistent dimensions.")
            end = offset + total
            if end > self.observation_dim or np.any(occupied[offset:end]):
                raise ValueError(f"Observation term {name!r} overlaps another term or exceeds total_dim.")
            occupied[offset:end] = True
            self.terms[name] = {
                "name": name,
                "offset": offset,
                "step_dim": step_dim,
                "history": history,
                "total": total,
            }
        if set(self.terms) != self._REQUIRED_TERMS or not np.all(occupied):
            raise ValueError("Manifest observation terms must be exactly the fixed schema and cover total_dim.")
        expected_term_shapes = {
            "joint_pos": (self.joint_count, 4),
            "joint_vel": (self.joint_count, 4),
            "goal_remaining": (3, 4),
            "base_ang_vel": (3, 4),
            "projected_gravity": (3, 4),
            "last_action": (self.joint_count, 1),
            "goal_command": (7, 1),
            "reference_preview": (70, 1),
            "jump_phase": (6, 1),
        }
        for name, (step_dim, history) in expected_term_shapes.items():
            if (self.terms[name]["step_dim"], self.terms[name]["history"]) != (step_dim, history):
                raise ValueError(f"Observation term {name!r} must have step_dim/history {step_dim}/{history}.")

        action = _mapping(self.raw.get("action"), "action")
        action_dim = _nonnegative_int(action.get("dim"), "action.dim")
        if action_dim != 23 or action_dim != self.joint_count:
            raise ValueError("Schema 1.2 requires 23 policy joints and action.dim=23.")
        self.action_scale = _float_array(action.get("scale"), "action.scale", self.joint_count)
        self.action_offset = _float_array(action.get("offset"), "action.offset", self.joint_count)
        self.filter_alpha = _float_array(action.get("filter_alpha"), "action.filter_alpha", self.joint_count)
        # A null clip is legitimate: this policy commands position targets past the joint
        # stops, and bounding them changes the dynamics rather than tidying the command.
        raw_clip = action.get("clip")
        self.action_clip = None if raw_clip is None else _float_pairs(raw_clip, "action.clip", self.joint_count)
        if not np.array_equal(self.action_offset, self.default_pos):
            raise ValueError("Manifest action.offset must exactly equal joints.default_pos.")
        if np.any(self.filter_alpha <= 0.0) or np.any(self.filter_alpha > 1.0):
            raise ValueError("Manifest action.filter_alpha values must be in (0, 1].")
        expected_formula = "q_target = alpha*clip(offset + scale*a_delayed) + (1-alpha)*q_target_prev"
        if action.get("formula") != expected_formula:
            raise ValueError("Manifest action.formula does not match the schema 1.2 transform.")
        brake_value = action.get("lower_limit_brake")
        if self.schema_version in ("1.4", "1.5", "1.6", "1.7") and brake_value is None:
            raise ValueError(f"Manifest schema {self.schema_version} requires action.lower_limit_brake.")
        self.brake_position_lower: np.ndarray | None = None
        self.brake_position_upper: np.ndarray | None = None
        self.brake_velocity_lookahead: np.ndarray | None = None
        if brake_value is not None:
            brake = _mapping(brake_value, "action.lower_limit_brake")
            if brake.get("type") != "velocity_lookahead":
                raise ValueError("Manifest action.lower_limit_brake.type must be 'velocity_lookahead'.")
            brake_period_s = _positive_float(brake.get("period_s"), "action.lower_limit_brake.period_s")
            if not math.isclose(brake_period_s, self.sim_dt, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("Manifest lower-limit braking must run at control.sim_dt.")
            self.brake_position_lower = _float_array(
                brake.get("position_lower"), "action.lower_limit_brake.position_lower", self.joint_count
            )
            self.brake_position_upper = _float_array(
                brake.get("position_upper"), "action.lower_limit_brake.position_upper", self.joint_count
            )
            self.brake_velocity_lookahead = _float_array(
                brake.get("velocity_lookahead_s"),
                "action.lower_limit_brake.velocity_lookahead_s",
                self.joint_count,
            )
            if np.any(self.brake_position_lower >= self.brake_position_upper):
                raise ValueError("Manifest lower-limit brake position bounds must be strictly increasing.")
            if np.any(self.brake_velocity_lookahead < 0.0) or not np.any(self.brake_velocity_lookahead > 0.0):
                raise ValueError("Manifest lower-limit brake lookahead must be non-negative and active.")
            if self.action_clip is None or not np.array_equal(
                self.action_clip,
                np.column_stack((self.brake_position_lower, self.brake_position_upper)),
            ):
                raise ValueError("Manifest lower-limit brake bounds must exactly equal action.clip.")
            expected_brake_formula = "q_requested = max(q_filtered, min(q_upper, q_lower + t_lookahead*max(-dq, 0)))"
            if brake.get("formula") != expected_brake_formula:
                raise ValueError("Manifest lower-limit brake formula is unsupported.")
        projection_value = action.get("torque_projection")
        if self.schema_version in ("1.3", "1.4", "1.5", "1.6", "1.7") and projection_value is None:
            raise ValueError(f"Manifest schema {self.schema_version} requires action.torque_projection.")
        self.effort_limit_ratio: np.ndarray | None = None
        if projection_value is not None:
            projection = _mapping(projection_value, "action.torque_projection")
            if projection.get("type") != "instantaneous_pd":
                raise ValueError("Manifest action.torque_projection.type must be 'instantaneous_pd'.")
            projection_period_s = _positive_float(projection.get("period_s"), "action.torque_projection.period_s")
            if not math.isclose(projection_period_s, self.sim_dt, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("Manifest torque projection must run at control.sim_dt.")
            self.effort_limit_ratio = _float_array(
                projection.get("effort_limit_ratio"),
                "action.torque_projection.effort_limit_ratio",
                self.joint_count,
            )
            if np.any(self.effort_limit_ratio <= 0.0) or np.any(self.effort_limit_ratio > 1.0):
                raise ValueError("Manifest torque-projection ratios must be in (0, 1].")
            expected_projection_formula = (
                "q_target = q + (clip(kp*(q_requested-q)-kd*dq, -ratio*effort_limit, ratio*effort_limit)+kd*dq)/kp"
            )
            if projection.get("formula") != expected_projection_formula:
                raise ValueError("Manifest torque-projection formula is unsupported.")
        delay = _mapping(action.get("delay_steps"), "action.delay_steps")
        self.delay_min = _nonnegative_int(delay.get("min"), "action.delay_steps.min")
        self.delay_max = _nonnegative_int(delay.get("max"), "action.delay_steps.max")
        if self.delay_min > self.delay_max:
            raise ValueError("action.delay_steps.min must not exceed action.delay_steps.max.")

        actuators = _mapping(self.raw.get("actuators"), "actuators")
        if actuators.get("type") != "implicit_pd":
            raise ValueError("Only actuators.type='implicit_pd' is supported.")
        self.stiffness = _float_array(actuators.get("stiffness"), "actuators.stiffness", self.joint_count)
        self.damping = _float_array(actuators.get("damping"), "actuators.damping", self.joint_count)
        self.effort_limit = _float_array(actuators.get("effort_limit"), "actuators.effort_limit", self.joint_count)
        self.armature = _float_array(actuators.get("armature"), "actuators.armature", self.joint_count)
        if np.any(self.stiffness < 0.0) or np.any(self.damping < 0.0):
            raise ValueError("Manifest stiffness and damping values must be non-negative.")
        if np.any(self.effort_limit <= 0.0) or np.any(self.armature < 0.0):
            raise ValueError("Manifest effort limits must be positive and armatures non-negative.")
        if self.effort_limit_ratio is not None and np.any(self.stiffness <= 0.0):
            raise ValueError("Manifest torque projection requires strictly positive stiffness.")
        velocity_limit = actuators.get("velocity_limit")
        self.velocity_limit = np.full(self.joint_count, np.inf, dtype=np.float64)
        if velocity_limit is not None:
            velocity_values = _sequence(velocity_limit, "actuators.velocity_limit", self.joint_count)
            for index, value in enumerate(velocity_values):
                if value is None:
                    continue
                self.velocity_limit[index] = _positive_float(value, f"actuators.velocity_limit[{index}]")

        reference = _mapping(self.raw.get("reference"), "reference")
        source_csv = reference.get("source_csv")
        if not isinstance(source_csv, str) or not source_csv or not Path(source_csv).is_absolute():
            raise ValueError("Manifest field reference.source_csv must be a non-empty absolute path.")
        self.reference_source = Path(source_csv)
        source_sha256 = reference.get("source_sha256")
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError("Manifest field reference.source_sha256 must be a lowercase SHA-256 hex digest.")
        self.reference_source_sha256 = source_sha256
        root_frame0 = _mapping(reference.get("root_frame0"), "reference.root_frame0")
        self.reference_root_position = _float_array(root_frame0.get("pos"), "reference.root_frame0.pos", 3)
        self.reference_root_quaternion_xyzw = _float_array(
            root_frame0.get("quat_xyzw"), "reference.root_frame0.quat_xyzw", 4
        )
        quaternion_norm = float(np.linalg.norm(self.reference_root_quaternion_xyzw))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("Manifest field reference.root_frame0.quat_xyzw must be a unit quaternion.")
        self.phase_names = tuple(_sequence(reference.get("phase_names"), "reference.phase_names"))
        expected_phase_names = ("IDLE", "CROUCH", "TAKEOFF", "FLIGHT", "LAND", "STAND")
        if self.phase_names != expected_phase_names:
            raise ValueError(f"reference.phase_names must equal {expected_phase_names}.")

        goal = _mapping(self.raw.get("goal"), "goal")
        if goal.get("quat_order") != "xyzw":
            raise ValueError("Only goal.quat_order='xyzw' is supported.")
        ranges = _mapping(goal.get("ranges"), "goal.ranges")
        self.goal_ranges: dict[str, tuple[float, float]] = {}
        for name in ("pos_x", "pos_y", "roll", "pitch", "yaw"):
            bounds = _float_array(ranges.get(name), f"goal.ranges.{name}", 2)
            if bounds[0] > bounds[1]:
                raise ValueError(f"goal.ranges.{name} lower bound exceeds its upper bound.")
            self.goal_ranges[name] = (float(bounds[0]), float(bounds[1]))
        freeze = _mapping(goal.get("flight_freeze"), "goal.flight_freeze")
        if not isinstance(freeze.get("enabled"), bool):
            raise ValueError("goal.flight_freeze.enabled must be a boolean.")
        self.flight_freeze_enabled = freeze["enabled"]

    def goal_value(self, name: str, requested: float | None) -> float:
        """Resolve and range-check one goal component."""
        lower, upper = self.goal_ranges[name]
        value = (lower + upper) * 0.5 if requested is None else requested
        if not lower <= value <= upper:
            raise ValueError(f"Goal {name}={value} is outside manifest range [{lower}, {upper}].")
        return value

    @property
    def sha256(self) -> str:
        """Return a stable digest of the parsed manifest contents."""
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()


class ModelInterface:
    """Indices linking policy-order values to the fixed MuJoCo model."""

    def __init__(self, model: mujoco.MjModel, manifest: DeploymentManifest):
        free_type = int(mujoco.mjtJoint.mjJNT_FREE)
        free_joint_ids = [joint_id for joint_id in range(model.njnt) if int(model.jnt_type[joint_id]) == free_type]
        if len(free_joint_ids) != 1:
            raise ValueError(f"MuJoCo model must contain exactly one floating-base joint, got {free_joint_ids}.")
        self.root_joint_id = free_joint_ids[0]
        self.root_qpos_adr = int(model.jnt_qposadr[self.root_joint_id])
        self.root_dof_adr = int(model.jnt_dofadr[self.root_joint_id])

        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        model_joint_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(model.njnt)
            if int(model.jnt_type[joint_id]) == hinge_type
        )
        if len(model_joint_names) != manifest.joint_count or any(name is None for name in model_joint_names):
            raise ValueError(
                f"MuJoCo must contain {manifest.joint_count} uniquely named hinge joints, got {model_joint_names}."
            )
        self.mujoco_joint_names = model_joint_names
        permutation_from_names, inverse_from_names = _build_name_permutations(
            manifest.joint_names, self.mujoco_joint_names, "MuJoCo hinge"
        )
        self.mujoco_joint_ids = np.asarray(
            [joint_id for joint_id in range(model.njnt) if int(model.jnt_type[joint_id]) == hinge_type],
            dtype=np.int32,
        )
        # Resolve every policy joint through MuJoCo's name API. The hinge-order
        # indices below are derived from those IDs, never from declaration position.
        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in manifest.joint_names],
            dtype=np.int32,
        )
        hinge_index_by_id = {int(joint_id): index for index, joint_id in enumerate(self.mujoco_joint_ids)}
        self.policy_from_mujoco = np.asarray(
            [hinge_index_by_id[int(joint_id)] for joint_id in self.joint_ids], dtype=np.int32
        )
        self.mujoco_from_policy = np.empty_like(self.policy_from_mujoco)
        self.mujoco_from_policy[self.policy_from_mujoco] = np.arange(manifest.joint_count, dtype=np.int32)
        if not np.array_equal(self.policy_from_mujoco, permutation_from_names) or not np.array_equal(
            self.mujoco_from_policy, inverse_from_names
        ):
            raise ValueError("MuJoCo joint name and ID permutation resolution disagree.")
        # Policy-order addresses are used only for observations, resets, and logs.
        self.qpos_adr = np.asarray(model.jnt_qposadr[self.joint_ids], dtype=np.int32)
        self.dof_adr = np.asarray(model.jnt_dofadr[self.joint_ids], dtype=np.int32)

        joint_transmission = int(mujoco.mjtTrn.mjTRN_JOINT)
        actuator_joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
        if (
            model.nu != manifest.joint_count
            or np.any(model.actuator_trntype != joint_transmission)
            or np.any(actuator_joint_ids < 0)
        ):
            raise ValueError(f"MuJoCo must contain {manifest.joint_count} single-joint actuators, got {model.nu}.")
        actuator_joint_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)) for joint_id in actuator_joint_ids
        )
        if any(name is None for name in actuator_joint_names):
            raise ValueError(f"Every MuJoCo actuator must target a named joint, got {actuator_joint_names}.")
        self.actuator_joint_names = actuator_joint_names
        actuator_permutation_from_names, actuator_inverse_from_names = _build_name_permutations(
            manifest.joint_names, self.actuator_joint_names, "MuJoCo actuator"
        )
        actuator_index_by_joint_id = {
            int(joint_id): actuator_id for actuator_id, joint_id in enumerate(actuator_joint_ids)
        }
        self.policy_from_actuator = np.asarray(
            [actuator_index_by_joint_id[int(joint_id)] for joint_id in self.joint_ids], dtype=np.int32
        )
        self.actuator_from_policy = np.empty_like(self.policy_from_actuator)
        self.actuator_from_policy[self.policy_from_actuator] = np.arange(manifest.joint_count, dtype=np.int32)
        if not np.array_equal(self.policy_from_actuator, actuator_permutation_from_names) or not np.array_equal(
            self.actuator_from_policy, actuator_inverse_from_names
        ):
            raise ValueError("MuJoCo actuator target name and ID permutation resolution disagree.")
        self.actuator_ids = np.arange(model.nu, dtype=np.int32)
        self.actuator_ids_policy = self.actuator_ids[self.policy_from_actuator]
        self.actuator_qpos_adr = np.asarray(model.jnt_qposadr[actuator_joint_ids], dtype=np.int32)
        self.actuator_dof_adr = np.asarray(model.jnt_dofadr[actuator_joint_ids], dtype=np.int32)
        expected_gear = np.zeros_like(model.actuator_gear)
        expected_gear[:, 0] = 1.0
        if not np.allclose(model.actuator_gear, expected_gear):
            raise ValueError("Every MuJoCo motor must have unit joint transmission gear.")

        self.sensors = {
            "pelvis_quaternion": SensorReader(model, "imu_quat", 4),
            "pelvis_angular_velocity": SensorReader(model, "imu_gyro", 3),
            "pelvis_position": SensorReader(model, "frame_pos", 3),
            "pelvis_linear_velocity": SensorReader(model, "frame_vel", 3),
        }
        self.ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sim2sim_ground")
        if self.ground_geom_id < 0:
            raise ValueError("Composed model is missing sim2sim_ground.")
        self.ground_height = float(model.geom_pos[self.ground_geom_id, 2])
        foot_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        self.foot_geom_ids: tuple[frozenset[int], frozenset[int]] = tuple(
            frozenset(
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
            )
            for body_name in foot_body_names
        )  # type: ignore[assignment]
        if any(not geom_ids for geom_ids in self.foot_geom_ids):
            raise ValueError("Could not resolve collidable geoms for both feet.")

        self.limited_tendon_ids = np.flatnonzero(model.tendon_limited).astype(np.int32)
        self.tendon_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, int(tendon_id)) or f"tendon_{tendon_id}"
            for tendon_id in self.limited_tendon_ids
        )

    def apply_manifest_dynamics(self, model: mujoco.MjModel, manifest: DeploymentManifest) -> None:
        """Apply manifest armatures and torque limits to the compiled model."""
        model.dof_armature[self.dof_adr] = manifest.armature
        model.actuator_ctrllimited[self.actuator_ids_policy] = 1
        model.actuator_ctrlrange[self.actuator_ids_policy, 0] = -manifest.effort_limit
        model.actuator_ctrlrange[self.actuator_ids_policy, 1] = manifest.effort_limit
        # The source MJCF also declares per-joint actuator-force limits. Keep those
        # from silently re-clamping the manifest's single source of truth.
        model.jnt_actfrclimited[self.joint_ids] = 1
        model.jnt_actfrcrange[self.joint_ids, 0] = -manifest.effort_limit
        model.jnt_actfrcrange[self.joint_ids, 1] = manifest.effort_limit

    def print_permutations(self) -> None:
        """Print all name-derived order mappings once at startup."""
        print(f"Joint permutation policy_from_mujoco={self.policy_from_mujoco.tolist()}")
        print(f"Joint permutation mujoco_from_policy={self.mujoco_from_policy.tolist()}")
        print(f"Actuator permutation policy_from_actuator={self.policy_from_actuator.tolist()}")
        print(f"Actuator permutation actuator_from_policy={self.actuator_from_policy.tolist()}")

    def tendon_state(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return limited-tendon lengths, slack, constraint force, and active flags."""
        tendon_ids = self.limited_tendon_ids
        lengths = np.asarray(data.ten_length[tendon_ids], dtype=np.float64).copy()
        ranges = np.asarray(model.tendon_range[tendon_ids], dtype=np.float64)
        slack = np.minimum(lengths - ranges[:, 0], ranges[:, 1] - lengths)
        force = np.zeros(len(tendon_ids), dtype=np.float64)
        active = np.zeros(len(tendon_ids), dtype=np.bool_)
        tendon_index = {int(tendon_id): index for index, tendon_id in enumerate(tendon_ids)}
        constraint_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
        for constraint_id in range(data.nefc):
            if int(data.efc_type[constraint_id]) != constraint_type:
                continue
            index = tendon_index.get(int(data.efc_id[constraint_id]))
            if index is not None:
                force[index] += abs(float(data.efc_force[constraint_id]))
                active[index] |= force[index] > 1e-12
        return lengths, slack, force, active


class SensorReader:
    """Address and validate a named MuJoCo sensor."""

    def __init__(self, model: mujoco.MjModel, name: str, expected_dim: int):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"MuJoCo model is missing required sensor {name!r}.")
        self.address = int(model.sensor_adr[sensor_id])
        self.dimension = int(model.sensor_dim[sensor_id])
        if self.dimension != expected_dim:
            raise ValueError(f"Sensor {name!r} must have dimension {expected_dim}, got {self.dimension}.")

    def read(self, data: mujoco.MjData) -> np.ndarray:
        """Return a copied sensor sample."""
        return np.asarray(data.sensordata[self.address : self.address + self.dimension], dtype=np.float64).copy()


class OnnxPolicy:
    """Single-input, single-output ONNX policy executor."""

    def __init__(self, policy_path: Path, observation_dim: int, action_dim: int):
        if not policy_path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            try:
                import onnx
                from onnx.reference import ReferenceEvaluator
            except ImportError:
                raise RuntimeError(
                    "Policy execution requires onnxruntime or onnx.reference; open-loop modes require neither."
                ) from exc
            onnx_model = onnx.load(str(policy_path), load_external_data=True)
            inputs = list(onnx_model.graph.input)
            outputs = list(onnx_model.graph.output)
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("ONNX policy must have exactly one input and one output.")

            def static_shape(value_info: Any) -> list[int]:
                return [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]

            input_shape = static_shape(inputs[0])
            output_shape = static_shape(outputs[0])
            if input_shape != [1, observation_dim] or output_shape != [1, action_dim]:
                raise ValueError(
                    f"ONNX policy shapes must be [1, {observation_dim}]/[1, {action_dim}], "
                    f"got {input_shape}/{output_shape}."
                )
            self.session = ReferenceEvaluator(onnx_model)
            self.input_name = inputs[0].name
            self.output_name = outputs[0].name
            self.backend = "onnx.reference"
        else:
            self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("ONNX policy must have exactly one input and one output.")
            self.input_name = inputs[0].name
            self.output_name = outputs[0].name
            if inputs[0].shape != [1, observation_dim] or outputs[0].shape != [1, action_dim]:
                raise ValueError(
                    f"ONNX policy shapes must be [1, {observation_dim}]/[1, {action_dim}], "
                    f"got {inputs[0].shape}/{outputs[0].shape}."
                )
            self.backend = "onnxruntime"
        self.action_dim = action_dim

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        action = self.session.run(
            [self.output_name], {self.input_name: observation[np.newaxis, :].astype(np.float32, copy=False)}
        )[0]
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.action_dim,) or not np.all(np.isfinite(action)):
            raise ValueError(f"ONNX policy returned invalid action shape or values: {action.shape}.")
        return action


class StepLogger:
    """Collect 500 Hz controller samples and write them without pickle data."""

    def __init__(self, output_path: Path, metadata: dict[str, Any]):
        self.output_path = output_path.resolve()
        self.metadata = metadata
        self.values: dict[str, list[np.ndarray | float | int]] = {
            "time": [],
            "phase": [],
            "qpos": [],
            "qvel": [],
            "action": [],
            "delayed_action": [],
            "q_target": [],
            "applied_tau": [],
            "pelvis_pose": [],
            "pelvis_velocity": [],
            "foot_contact_forces": [],
            "observation": [],
        }

    def append(
        self,
        *,
        sim_time: float,
        phase: int,
        qpos: np.ndarray,
        qvel: np.ndarray,
        action: np.ndarray,
        delayed_action: np.ndarray,
        q_target: np.ndarray,
        applied_tau: np.ndarray,
        pelvis_pose: np.ndarray,
        pelvis_velocity: np.ndarray,
        foot_contact_forces: np.ndarray,
        observation: np.ndarray,
        tendon_length: np.ndarray | None = None,
        tendon_limit_slack: np.ndarray | None = None,
        tendon_limit_force: np.ndarray | None = None,
        tendon_limit_active: np.ndarray | None = None,
    ) -> None:
        """Append one post-physics-step sample."""
        row = {
            "time": sim_time,
            "phase": phase,
            "qpos": qpos,
            "qvel": qvel,
            "action": action,
            "delayed_action": delayed_action,
            "q_target": q_target,
            "applied_tau": applied_tau,
            "pelvis_pose": pelvis_pose,
            "pelvis_velocity": pelvis_velocity,
            "foot_contact_forces": foot_contact_forces,
            "observation": observation,
        }
        for name, value in row.items():
            self.values[name].append(value.copy() if isinstance(value, np.ndarray) else value)
        optional = {
            "tendon_length": tendon_length,
            "tendon_limit_slack": tendon_limit_slack,
            "tendon_limit_force": tendon_limit_force,
            "tendon_limit_active": tendon_limit_active,
        }
        for name, value in optional.items():
            if value is not None:
                self.values.setdefault(name, []).append(value.copy())

    def save(self) -> None:
        """Write all collected samples to a compressed NPZ file."""
        if not self.values["time"]:
            raise RuntimeError("Cannot save an empty trajectory log.")
        arrays = {name: np.asarray(values) for name, values in self.values.items()}
        arrays["metadata_json"] = np.asarray(json.dumps(self.metadata, sort_keys=True))
        arrays["phase_names"] = np.asarray(self.metadata["phase_names"], dtype=np.str_)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_path, **arrays)


class SelfCheckMetrics:
    """Accumulate diagnostics for the three-second passive hold."""

    def __init__(self, initial_height: float):
        self.initial_height = initial_height
        self.maximum_pelvis_drop = 0.0
        self.maximum_tilt_rad = 0.0
        self.maximum_floor_penetration = 0.0
        self.nonfoot_ground_contacts: set[str] = set()
        self.finite = True

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        interface: ModelInterface,
        pelvis_position: np.ndarray,
        projected_gravity: np.ndarray,
    ) -> None:
        """Update stand-stability metrics from the current simulated state."""
        self.maximum_pelvis_drop = max(self.maximum_pelvis_drop, self.initial_height - pelvis_position[2])
        upright_cosine = float(np.clip(-projected_gravity[2], -1.0, 1.0))
        self.maximum_tilt_rad = max(self.maximum_tilt_rad, math.acos(upright_cosine))
        self.finite &= bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        all_foot_geoms = interface.foot_geom_ids[0] | interface.foot_geom_ids[1]
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if interface.ground_geom_id not in (geom1, geom2):
                continue
            self.maximum_floor_penetration = max(self.maximum_floor_penetration, max(0.0, -float(contact.dist)))
            robot_geom = geom2 if geom1 == interface.ground_geom_id else geom1
            if robot_geom not in all_foot_geoms:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
                self.nonfoot_ground_contacts.add(name or f"geom_{robot_geom}")


def _diagnostic_projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return world -Z in the pelvis frame for 500 Hz diagnostics."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps or not math.isfinite(norm):
        raise ValueError("IMU quaternion has invalid norm.")
    quaternion = quaternion / norm
    w, x, y, z = quaternion
    return np.asarray(
        (
            2.0 * (w * y - x * z),
            -2.0 * (y * z + w * x),
            2.0 * (x * x + y * y) - 1.0,
        ),
        dtype=np.float64,
    )


def _apply_contact_compliance(
    model: mujoco.MjModel,
    interface: ModelInterface,
    timeconst: float,
    dampratio: float,
) -> dict[str, Any]:
    """Retune the ground contact's solver reference to change contact compliance.

    MuJoCo's contact stiffness is set by ``solref = (timeconst, dampratio)``: the constraint
    behaves as a spring-damper whose natural frequency is ``1/timeconst``, so stiffness scales
    as ``1/timeconst^2``. The negative direct form is deliberately not used; it specifies the
    spring in constraint-acceleration units, which made this model diverge. The ground plane
    carries the higher contact priority, so its reference parameters win contact mixing against
    every foot geom.

    Args:
        model: The compiled MuJoCo model, mutated in place.
        interface: Resolved joint, actuator and geometry indices.
        timeconst: Contact solver time constant [s]; must be at least two simulation steps.
        dampratio: Contact solver damping ratio; 1.0 is critically damped.

    Returns:
        A record of what was written, for the run log.
    """
    minimum_timeconst = 2.0 * float(model.opt.timestep)
    if timeconst < minimum_timeconst:
        raise ValueError(
            f"Contact timeconst {timeconst} is below MuJoCo's stability floor of two simulation "
            f"steps ({minimum_timeconst})."
        )
    if dampratio <= 0.0:
        raise ValueError(f"Contact dampratio must be positive, got {dampratio}.")
    previous = np.asarray(model.geom_solref[interface.ground_geom_id], dtype=np.float64).copy()
    model.geom_solref[interface.ground_geom_id] = (timeconst, dampratio)
    return {
        "timeconst": float(timeconst),
        "dampratio": float(dampratio),
        "geom": "sim2sim_ground",
        "solref_before": previous.tolist(),
        "solref_after": np.asarray(model.geom_solref[interface.ground_geom_id], dtype=np.float64).tolist(),
        "solimp": np.asarray(model.geom_solimp[interface.ground_geom_id], dtype=np.float64).tolist(),
    }


def _foot_contact_forces(model: mujoco.MjModel, data: mujoco.MjData, interface: ModelInterface) -> np.ndarray:
    """Return left/right net ground contact forces in world axes [N]."""
    return np.stack(
        [ground_contact_force(model, data, interface.ground_geom_id, geom_ids) for geom_ids in interface.foot_geom_ids]
    )


def _resolve_goal_values(manifest: DeploymentManifest, args: argparse.Namespace) -> dict[str, float]:
    """Resolve simulator CLI goal overrides against manifest ranges."""
    return {
        name: manifest.goal_value(name, getattr(args, f"goal_{name}"))
        for name in ("pos_x", "pos_y", "roll", "pitch", "yaw")
    }


def _load_action_sequence(path: Path, manifest: DeploymentManifest) -> np.ndarray:
    """Load a finite policy-order action sequence from an NPY file."""
    sequence = np.load(path, allow_pickle=False)
    if sequence.ndim != 2 or sequence.shape[1] != manifest.joint_count or sequence.shape[0] == 0:
        raise ValueError(
            f"Action sequence must have shape [N, {manifest.joint_count}] with N > 0, got {sequence.shape}."
        )
    if sequence.shape[0] > manifest.episode_steps:
        raise ValueError(
            f"Action sequence has {sequence.shape[0]} rows but manifest tables have only "
            f"{manifest.episode_steps} policy steps."
        )
    sequence = np.asarray(sequence, dtype=np.float64)
    if not np.all(np.isfinite(sequence)):
        raise ValueError("Action sequence contains non-finite values.")
    return sequence


def _compute_actuator_control(
    target: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    effort_limit: np.ndarray,
    velocity_limit: np.ndarray,
    *,
    use_implicit_pd: bool,
    emulate_velocity_limit: bool,
) -> np.ndarray:
    """Compute position- or torque-actuator control with optional speed saturation."""
    if use_implicit_pd and not emulate_velocity_limit:
        return target
    torque = stiffness * (target - position) - damping * velocity
    torque = np.clip(torque, -effort_limit, effort_limit)
    if emulate_velocity_limit:
        torque = saturate_torque_at_velocity_limit(torque, velocity, velocity_limit)
    if not use_implicit_pd:
        return torque
    if np.any(stiffness <= 0.0):
        raise ValueError("Implicit-PD velocity-limit emulation requires strictly positive stiffness.")
    return position + (torque + damping * velocity) / stiffness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--overlay", type=Path, default=_DEFAULT_OVERLAY)
    parser.add_argument(
        "--reference_csv",
        type=Path,
        default=None,
        help="Deprecated compatibility option; if supplied, it must match reference.source_csv in the manifest.",
    )
    parser.add_argument("--policy", type=Path, default=None, help="Defaults to policy.onnx beside the manifest.")
    parser.add_argument("--log", type=Path, default=Path("mujoco_jump_log.npz"))
    parser.add_argument(
        "--settle_to_equilibrium",
        action="store_true",
        help=(
            "Settle to static equilibrium before step 0. Off by default: joint PD alone topples "
            "this robot, and Isaac does not settle either, so enabling it makes sim2sim less "
            "comparable rather than more."
        ),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--emulate_velocity_limit",
        action="store_true",
        help="Emulate manifest actuator velocity limits with torque-speed saturation.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-check", "--self_check", dest="self_check", action="store_true")
    mode.add_argument("--cross-check", "--cross_check", dest="cross_check", action="store_true")
    parser.add_argument(
        "--action_sequence",
        "--action-sequence",
        dest="action_sequence",
        type=Path,
        default=None,
        help="Cross-check raw actions as an NPY array shaped [policy steps, 23] in policy order.",
    )
    parser.add_argument("--delay_steps", type=int, default=None)
    parser.add_argument(
        "--contact_timeconst",
        type=float,
        default=None,
        help=(
            "Ground contact solver time constant [s]. Contact stiffness scales as 1/timeconst^2, so "
            "smaller is stiffer. MuJoCo needs at least two simulation steps here. Omit to keep the "
            "compiled model's default."
        ),
    )
    parser.add_argument(
        "--contact_dampratio",
        type=float,
        default=1.0,
        help="Ground contact solver damping ratio; 1.0 is critically damped and is MuJoCo's default.",
    )
    parser.add_argument(
        "--clamp_joint_velocity",
        action="store_true",
        help=(
            "Diagnostic: clamp joint velocities to actuators.velocity_limit after every physics step. "
            "PhysX brakes at that limit and MuJoCo has no equivalent constraint; this is a coarse A/B "
            "probe of that asymmetry, not a fidelity-grade implementation of solver-side braking."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal_pos_x", type=float, default=None)
    parser.add_argument("--goal_pos_y", type=float, default=None)
    parser.add_argument("--goal_roll", type=float, default=None, help="Goal roll [rad].")
    parser.add_argument("--goal_pitch", type=float, default=None, help="Goal pitch [rad].")
    parser.add_argument("--goal_yaw", type=float, default=None, help="Goal yaw [rad].")
    parser.add_argument(
        "--no_ground_clearance",
        action="store_true",
        help=(
            "Do NOT lift the robot so its lowest foot geom touches z=0. Diagnostic only: without "
            "the lift MuJoCo starts interpenetrating and the simulation goes non-finite."
        ),
    )
    parser.add_argument(
        "--disable_tendon_limits",
        action="store_true",
        help=(
            "Disable the eight ankle parallel-linkage tendon limits. Diagnostic only: PhysX "
            "does not model them and the policy trained without them, so this isolates how "
            "much of any Isaac/MuJoCo divergence they explain. Never use for validation."
        ),
    )
    add_physics_parity_arguments(parser)
    parser.add_argument("--dump_composed_model", type=Path, default=None)
    args = parser.parse_args()
    if args.action_sequence is not None and not args.cross_check:
        parser.error("--action_sequence requires --cross-check.")
    if args.emulate_velocity_limit and args.clamp_joint_velocity:
        parser.error("--emulate_velocity_limit cannot be combined with diagnostic --clamp_joint_velocity.")
    return args


def run(args: argparse.Namespace) -> None:  # noqa: C901 - one validated simulation/logging pipeline
    settle = bool(getattr(args, "settle_to_equilibrium", False))
    # Default ON. Starting at Isaac's raw reference height drives MuJoCo to NaN, because the
    # MJCF places the foot collision geoms 7.3 mm BELOW where the USD does at the same
    # pelvis height -- the two assets disagree on foot geometry. Isaac therefore free-falls
    # 7.3 mm onto the floor at t=0 (0.00 N at t=0, 2717 N within 20 ms, 8x body weight),
    # while MuJoCo would start interpenetrating. Neither is a faithful copy of the other.
    ground_clearance = not bool(getattr(args, "no_ground_clearance", False))
    """Load, simulate, and log a policy or deterministic open-loop replay."""
    parity_config = PhysicsParityConfig.from_args(args)
    manifest = DeploymentManifest(args.manifest)
    _validate_reference_override(args.reference_csv, manifest)
    reference_joint_pos, reference_root_position, reference_root_quaternion = _load_reference_frame0(manifest)
    composed_xml, overlay_timestep = compose_model_xml(args.model, args.overlay)
    if not math.isclose(overlay_timestep, manifest.sim_dt, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Overlay timestep {overlay_timestep} does not match manifest control.sim_dt {manifest.sim_dt}."
        )
    if args.dump_composed_model is not None:
        args.dump_composed_model.parent.mkdir(parents=True, exist_ok=True)
        args.dump_composed_model.write_text(composed_xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_string(composed_xml)
    if not math.isclose(float(model.opt.timestep), manifest.sim_dt, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Compiled MuJoCo timestep does not match the deployment manifest.")
    if bool(getattr(args, "disable_tendon_limits", False)):
        model.tendon_limited[:] = 0
        print(f"[DIAGNOSTIC] Disabled {int(model.ntendon)} tendon limit(s); this run is not a validation.")
    interface = ModelInterface(model, manifest)
    interface.print_permutations()
    interface.apply_manifest_dynamics(model, manifest)
    apply_physics_parity(
        model,
        interface.actuator_ids_policy,
        manifest.stiffness,
        manifest.damping,
        manifest.effort_limit,
        parity_config,
    )
    contact_compliance = None
    if args.contact_timeconst is not None:
        contact_compliance = _apply_contact_compliance(model, interface, args.contact_timeconst, args.contact_dampratio)
        print(f"Contact compliance: {contact_compliance}")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[interface.root_qpos_adr : interface.root_qpos_adr + 3] = reference_root_position
    data.qpos[interface.root_qpos_adr + 3 : interface.root_qpos_adr + 7] = reference_root_quaternion
    data.qvel[interface.root_dof_adr : interface.root_dof_adr + 6] = 0.0
    data.qpos[interface.qpos_adr] = reference_joint_pos
    data.qvel[interface.dof_adr] = 0.0
    # Matching Isaac means matching its INITIAL PENETRATION, not removing ours. Measured:
    # Isaac spawns the robot 7.3 mm inside the floor and PhysX resolves that with a 2717 N
    # impulse in the first 20 ms -- 8x body weight -- in every episode the policy ever
    # trained on. Lifting the robot clear in MuJoCo is more physically correct but makes the
    # two engines start from different states, so it is off by default for sim2sim.
    initial_root_height_offset = (
        0.0
        if not ground_clearance
        else apply_initial_ground_clearance(
            model,
            data,
            interface.root_qpos_adr,
            interface.ground_geom_id,
            interface.foot_geom_ids[0] | interface.foot_geom_ids[1],
        )
    )
    print(f"Initial root height offset: {initial_root_height_offset:.9f} m")

    if not np.allclose(data.qpos[interface.qpos_adr], reference_joint_pos, rtol=0.0, atol=1e-12):
        raise RuntimeError("MuJoCo did not retain the written frame-0 joint positions.")
    if not np.allclose(data.qvel[interface.dof_adr], 0.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("MuJoCo did not retain zero frame-0 joint velocities.")
    if not np.allclose(
        data.qpos[interface.root_qpos_adr : interface.root_qpos_adr + 7],
        np.concatenate(
            (
                reference_root_position + np.asarray((0.0, 0.0, initial_root_height_offset)),
                reference_root_quaternion,
            )
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("MuJoCo did not retain the ground-corrected frame-0 root pose.")
    if not np.allclose(data.qvel[interface.root_dof_adr : interface.root_dof_adr + 6], 0.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("MuJoCo did not retain zero frame-0 root velocity.")

    # Settling to static equilibrium is OFF by default, and that is deliberate. It was tried
    # and measured: holding the reference pose under joint PD does not settle the robot, it
    # topples it -- after 2 s the geoms touching the ground are head, both hands and both
    # thighs, with only 44.6 N of the 334.8 N weight left on the feet. Joint-space PD cannot
    # balance this robot at any gain; that needs the ankle-strategy controller in
    # scripts/g1_jump_deploy/control/balance.py.
    #
    # More importantly, settling would make sim2sim WORSE rather than better. Isaac does not
    # settle: it writes reference frame 0 and starts the episode immediately, so the policy
    # has never seen a settled pose and the robot never has to stand statically. Matching
    # Isaac's initial condition is the whole point of the comparison.
    settle_result = (
        None
        if not settle
        else settle_static_equilibrium(
            model,
            data,
            root_dof_adr=interface.root_dof_adr,
            joint_qpos_adr=interface.qpos_adr,
            joint_dof_adr=interface.dof_adr,
            actuator_ids=interface.actuator_ids_policy,
            reference_joint_pos=reference_joint_pos,
            stiffness=manifest.stiffness,
            damping=manifest.damping,
            effort_limit=manifest.effort_limit,
            ground_geom_id=interface.ground_geom_id,
            foot_geom_ids=interface.foot_geom_ids[0] | interface.foot_geom_ids[1],
            use_implicit_pd=parity_config.use_implicit_pd,
        )
    )
    if settle_result is not None:
        print(
            "Static-equilibrium settle: "
            f"duration={settle_result.duration_s:.3f} s, "
            f"foot_contact_force={settle_result.foot_contact_force_n:.3f} N, "
            f"root_linear_speed={settle_result.root_linear_speed_m_s:.6f} m/s, "
            f"root_angular_speed={settle_result.root_angular_speed_rad_s:.6f} rad/s"
        )

    measured_pos = np.asarray(data.qpos[interface.qpos_adr], dtype=np.float64).copy()
    initial_position = interface.sensors["pelvis_position"].read(data)
    initial_quaternion = interface.sensors["pelvis_quaternion"].read(data)
    goal_values = _resolve_goal_values(manifest, args)

    open_loop = args.self_check or args.cross_check
    if args.cross_check and args.seed != 0:
        raise ValueError("--cross-check fixes --seed to 0; do not supply another value.")
    if args.cross_check and args.delay_steps not in (None, 0):
        raise ValueError("--cross-check fixes --delay_steps to 0.")
    rng = np.random.default_rng(0 if args.cross_check else args.seed)
    if args.cross_check:
        delay_steps = 0
    elif args.delay_steps is None:
        delay_steps = int(rng.integers(manifest.delay_min, manifest.delay_max + 1))
    else:
        delay_steps = args.delay_steps
    if not manifest.delay_min <= delay_steps <= manifest.delay_max:
        raise ValueError(
            f"delay_steps={delay_steps} is outside manifest range [{manifest.delay_min}, {manifest.delay_max}]."
        )

    policy_path = args.policy.resolve() if args.policy else manifest.path.parent / "policy.onnx"
    policy = None if open_loop else OnnxPolicy(policy_path, manifest.observation_dim, manifest.joint_count)
    goal_runtime = JumpGoalRuntime(manifest.path, freeze_during_flight=manifest.flight_freeze_enabled)
    goal_runtime.arm(
        goal_values["pos_x"],
        goal_values["pos_y"],
        goal_values["yaw"],
        roll=goal_values["roll"],
        pitch=goal_values["pitch"],
    )
    # MuJoCo framequat and JumpGoalRuntime sensor boundaries are both WXYZ.
    # The runtime alone converts the manifest goal command to policy XYZW.
    goal_runtime.trigger(
        initial_position,
        initial_quaternion,
        measured_pos,
        action_delay_steps=delay_steps,
        goal_pos_z_w=interface.ground_height,
    )
    goal_world = goal_runtime.goal_position_w

    action_sequence = None
    if args.cross_check:
        action_sequence = (
            _load_action_sequence(args.action_sequence, manifest)
            if args.action_sequence is not None
            else np.zeros((manifest.episode_steps, manifest.joint_count), dtype=np.float64)
        )
    total_sim_steps = (
        int(round(_SELF_CHECK_DURATION_S / manifest.sim_dt))
        if args.self_check
        else (
            len(action_sequence) * manifest.decimation
            if action_sequence is not None
            else manifest.episode_steps * manifest.decimation
        )
    )
    if args.self_check and not math.isclose(
        total_sim_steps * manifest.sim_dt, _SELF_CHECK_DURATION_S, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Manifest sim_dt cannot represent the required three-second self-check exactly.")
    metadata = {
        "schema_version": manifest.raw["schema_version"],
        "task": manifest.raw.get("task"),
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "simulator": "mujoco",
        "model": str(args.model.resolve()),
        "overlay": str(args.overlay.resolve()),
        "reference_csv": str(manifest.reference_source),
        "reference_sha256": manifest.reference_source_sha256,
        "reference_frame": 0,
        "initial_root_height_offset_m": initial_root_height_offset,
        "settle_duration_s": None if settle_result is None else settle_result.duration_s,
        "settle_static_weight_n": None if settle_result is None else settle_result.static_weight_n,
        "settle_foot_contact_force_n": None if settle_result is None else settle_result.foot_contact_force_n,
        "settle_root_linear_velocity_m_s": None
        if settle_result is None
        else settle_result.root_linear_velocity_m_s.tolist(),
        "settle_root_angular_velocity_rad_s": None
        if settle_result is None
        else settle_result.root_angular_velocity_rad_s.tolist(),
        "settle_root_linear_speed_m_s": None if settle_result is None else settle_result.root_linear_speed_m_s,
        "settle_root_angular_speed_rad_s": None if settle_result is None else settle_result.root_angular_speed_rad_s,
        "policy": None if open_loop else str(policy_path),
        "policy_backend": None if policy is None else policy.backend,
        "self_check": args.self_check,
        "cross_check": args.cross_check,
        "action_sequence": None if args.action_sequence is None else str(args.action_sequence.resolve()),
        "sim_dt": manifest.sim_dt,
        "policy_dt": manifest.policy_dt,
        "decimation": manifest.decimation,
        "delay_steps": delay_steps,
        "emulate_velocity_limit": args.emulate_velocity_limit,
        "joint_velocity_clamped": bool(args.clamp_joint_velocity),
        "contact_compliance": contact_compliance,
        "seed": 0 if args.cross_check else args.seed,
        "goal": goal_values,
        "goal_world": goal_world.tolist(),
        "phase_names": manifest.phase_names,
        "qpos_qvel_order": manifest.joint_names,
        "policy_from_mujoco": interface.policy_from_mujoco.tolist(),
        "mujoco_from_policy": interface.mujoco_from_policy.tolist(),
        "policy_from_actuator": interface.policy_from_actuator.tolist(),
        "actuator_from_policy": interface.actuator_from_policy.tolist(),
        "mujoco_joint_names": interface.mujoco_joint_names,
        "actuator_joint_names": interface.actuator_joint_names,
        "limited_tendon_names": interface.tendon_names,
        "physics_parity": parity_config.metadata(),
        "applied_tau_source": "data.actuator_force sampled post-mj_step before mj_forward",
        "pelvis_pose_convention": "position_world_xyz[m], quaternion_world_from_body_wxyz",
        "pelvis_velocity_convention": "linear_world_xyz[m/s], angular_body_xyz[rad/s]",
        "foot_contact_force_convention": "left_then_right, world_xyz[N]",
        "sample_convention": (
            "sample 0 is the initialized frame-0 state (or post-settle state when requested) with zero "
            "applied_tau because it has no preceding integration interval; later samples are post-physics "
            "states and observation is held from the latest policy tick"
        ),
    }
    logger = StepLogger(args.log, metadata)
    self_check = SelfCheckMetrics(initial_position[2]) if args.self_check else None

    raw_action = np.zeros(manifest.joint_count, dtype=np.float64)
    delayed_action = np.zeros(manifest.joint_count, dtype=np.float64)
    q_target = measured_pos.copy()
    observation = np.zeros(manifest.observation_dim, dtype=np.float32)
    phase = 0
    stiffness_actuator = manifest.stiffness[interface.actuator_from_policy]
    damping_actuator = manifest.damping[interface.actuator_from_policy]
    effort_limit_actuator = manifest.effort_limit[interface.actuator_from_policy]
    velocity_limit_actuator = manifest.velocity_limit[interface.actuator_from_policy]

    def process_policy_tick(policy_step: int) -> None:
        nonlocal delayed_action, observation, phase, q_target, raw_action
        if goal_runtime.done:
            raise RuntimeError("JumpGoalRuntime finished before the MuJoCo policy loop.")
        pelvis_position = interface.sensors["pelvis_position"].read(data)
        pelvis_quaternion = interface.sensors["pelvis_quaternion"].read(data)
        # Both IMU and contact odometry use MuJoCo's WXYZ framequat here. The
        # runtime owns projected gravity, goal remaining, histories, and tables.
        observation = goal_runtime.step(
            np.asarray(data.qpos[interface.qpos_adr], dtype=np.float64),
            np.asarray(data.qvel[interface.dof_adr], dtype=np.float64),
            interface.sensors["pelvis_angular_velocity"].read(data),
            pelvis_quaternion,
            pelvis_position,
            pelvis_quaternion,
        )
        phase_term = manifest.terms["jump_phase"]
        phase_offset = int(phase_term["offset"])
        phase_end = phase_offset + int(phase_term["step_dim"])
        phase = int(np.argmax(observation[phase_offset:phase_end]))
        if policy is not None:
            raw_action = policy(observation)
            q_target = goal_runtime.transform_action(raw_action)
            delayed_action = goal_runtime.delayed_action
        elif action_sequence is not None:
            raw_action = action_sequence[policy_step].copy()
            q_target = goal_runtime.transform_action(raw_action)
            delayed_action = goal_runtime.delayed_action
        else:
            # The diagnostic performs no inference or action transform: it holds
            # reference frame 0 directly under the same inner PD loop.
            raw_action.fill(0.0)
            delayed_action.fill(0.0)
            q_target = reference_joint_pos.copy()

    def append_current_state(applied_tau_policy: np.ndarray) -> None:
        pelvis_position = interface.sensors["pelvis_position"].read(data)
        pelvis_quaternion = interface.sensors["pelvis_quaternion"].read(data)
        pelvis_linear_velocity = interface.sensors["pelvis_linear_velocity"].read(data)
        pelvis_angular_velocity = interface.sensors["pelvis_angular_velocity"].read(data)
        foot_forces = _foot_contact_forces(model, data, interface)
        tendon_length, tendon_limit_slack, tendon_limit_force, tendon_limit_active = interface.tendon_state(model, data)
        if self_check is not None:
            self_check.update(
                model,
                data,
                interface,
                pelvis_position,
                _diagnostic_projected_gravity(pelvis_quaternion),
            )
        logger.append(
            sim_time=float(data.time),
            phase=phase,
            qpos=np.asarray(data.qpos[interface.qpos_adr], dtype=np.float64),
            qvel=np.asarray(data.qvel[interface.dof_adr], dtype=np.float64),
            action=raw_action,
            delayed_action=delayed_action,
            q_target=q_target,
            applied_tau=applied_tau_policy,
            pelvis_pose=np.concatenate((pelvis_position, pelvis_quaternion)),
            pelvis_velocity=np.concatenate((pelvis_linear_velocity, pelvis_angular_velocity)),
            foot_contact_forces=foot_forces,
            observation=observation,
            tendon_length=tendon_length,
            tendon_limit_slack=tendon_limit_slack,
            tendon_limit_force=tendon_limit_force,
            tendon_limit_active=tendon_limit_active,
        )

    process_policy_tick(0)
    append_current_state(np.zeros(manifest.joint_count, dtype=np.float64))

    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)
    try:
        for sim_step in range(total_sim_steps):
            step_start = time.monotonic()
            if sim_step > 0 and sim_step % manifest.decimation == 0:
                policy_step = sim_step // manifest.decimation
                process_policy_tick(policy_step)

            # Scatter policy-order targets into actual MuJoCo actuator order. In
            # parity mode ctrl is the position target and MuJoCo solves PD. The
            # diagnostic legacy mode retains explicit Python torque PD.
            applied_q_target = q_target
            measured_pos = np.asarray(data.qpos[interface.qpos_adr], dtype=np.float64)
            measured_vel = np.asarray(data.qvel[interface.dof_adr], dtype=np.float64)
            if manifest.brake_velocity_lookahead is not None:
                applied_q_target = project_position_target_to_lower_limit(
                    applied_q_target,
                    measured_vel,
                    manifest.brake_position_lower,
                    manifest.brake_position_upper,
                    manifest.brake_velocity_lookahead,
                )
            if manifest.effort_limit_ratio is not None:
                applied_q_target = project_pd_position_target(
                    applied_q_target,
                    measured_pos,
                    measured_vel,
                    manifest.stiffness,
                    manifest.damping,
                    manifest.effort_limit,
                    manifest.effort_limit_ratio,
                )
            q_target_actuator = applied_q_target[interface.actuator_from_policy]
            qpos_actuator = np.asarray(data.qpos[interface.actuator_qpos_adr], dtype=np.float64)
            qvel_actuator = np.asarray(data.qvel[interface.actuator_dof_adr], dtype=np.float64)
            data.ctrl[interface.actuator_ids] = _compute_actuator_control(
                q_target_actuator,
                qpos_actuator,
                qvel_actuator,
                stiffness_actuator,
                damping_actuator,
                effort_limit_actuator,
                velocity_limit_actuator,
                use_implicit_pd=parity_config.use_implicit_pd,
                emulate_velocity_limit=args.emulate_velocity_limit,
            )
            mujoco.mj_step(model, data)
            # actuator_force is the force MuJoCo actually applied during the
            # completed step. Capture it before mj_forward recomputes forces at
            # the post-integration state.
            applied_tau_actuator = np.asarray(data.actuator_force, dtype=np.float64).copy()
            applied_tau_policy = applied_tau_actuator[interface.policy_from_actuator]
            if args.clamp_joint_velocity:
                data.qvel[interface.dof_adr] = np.clip(
                    data.qvel[interface.dof_adr], -manifest.velocity_limit, manifest.velocity_limit
                )
            # mj_step integrates qpos/qvel after its forward-dynamics stages. Refresh
            # frame sensors and end-of-step contacts before logging or the next policy tick.
            mujoco.mj_forward(model, data)
            append_current_state(applied_tau_policy)

            if viewer is not None:
                if not viewer.is_running():
                    raise RuntimeError("Viewer closed before the requested simulation completed.")
                viewer.sync()
                remaining = manifest.sim_dt - (time.monotonic() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()

    if not args.self_check and (action_sequence is None or len(action_sequence) == manifest.episode_steps):
        if not goal_runtime.done:
            raise RuntimeError("MuJoCo completed without consuming the full JumpGoalRuntime episode.")
    logger.save()
    print(f"Wrote {total_sim_steps + 1} controller samples to {logger.output_path}")
    print(f"Selected action delay: {delay_steps} policy step(s)")
    if policy is not None:
        print(f"ONNX backend: {policy.backend}")
    if self_check is not None:
        print(
            "Self-check metrics: "
            f"pelvis_drop={self_check.maximum_pelvis_drop:.6f} m, "
            f"tilt={math.degrees(self_check.maximum_tilt_rad):.6f} deg, "
            f"floor_penetration={self_check.maximum_floor_penetration:.6f} m, "
            f"nonfoot_contacts={sorted(self_check.nonfoot_ground_contacts)}, "
            f"finite={self_check.finite}"
        )
        # This is deliberately diagnostic-only. The nominal 20 N.m/rad ankle PD produced
        # a slow passive topple in the audited run (z 0.7930 -> 0.1004 m, 94.8 deg tilt,
        # 0.023 m penetration) without saturation. An actively controlled jump task never
        # required a three-second open-loop stand, so those values are not fidelity limits.
        print(f"Self-check diagnostic completed for {_SELF_CHECK_DURATION_S:.1f} s (no pass/fail thresholds).")
    elif args.cross_check:
        print(f"Cross-check replay completed deterministically with {len(action_sequence)} policy steps.")


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo implementations of the jump FSM robot and operator protocols."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from scripts.g1_jump_deploy.fsm.jump_fsm import JumpGoal
from scripts.g1_jump_deploy.mujoco.model_overlay import apply_initial_ground_clearance, compose_model_xml
from scripts.g1_jump_deploy.runtime import (
    project_pd_position_target,
    project_position_target_to_lower_limit,
    saturate_torque_at_velocity_limit,
)

_GANTRY_POSITION_STIFFNESS_N_M = 1_000.0
_GANTRY_POSITION_DAMPING_N_S_M = 300.0
_GANTRY_ATTITUDE_STIFFNESS_N_M_RAD = 300.0
_GANTRY_ATTITUDE_DAMPING_N_M_S_RAD = 30.0
_GANTRY_FORCE_LIMIT_N = 750.0
_GANTRY_TORQUE_LIMIT_N_M = 200.0


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {name} must be an object.")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Manifest field {name} must be a finite positive number.")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"Manifest field {name} must be a finite positive number.")
    return result


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest field {name} must contain {length} numeric values.") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Manifest field {name} must contain {length} finite values.")
    return result


def _name_permutations(
    manifest_names: tuple[str, ...], backend_names: tuple[str, ...], backend_label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return manifest-to-backend and backend-to-manifest permutations."""
    if len(manifest_names) != len(backend_names):
        raise ValueError(
            f"Manifest has {len(manifest_names)} joints but {backend_label} has {len(backend_names)} entries."
        )
    if len(set(manifest_names)) != len(manifest_names):
        raise ValueError("Manifest joint names must be unique.")
    if len(set(backend_names)) != len(backend_names):
        raise ValueError(f"{backend_label} joint names must be unique.")
    if set(manifest_names) != set(backend_names):
        missing = sorted(set(manifest_names) - set(backend_names))
        extra = sorted(set(backend_names) - set(manifest_names))
        raise ValueError(f"{backend_label} names differ from the manifest. Missing={missing}, extra={extra}.")

    backend_index = {name: index for index, name in enumerate(backend_names)}
    manifest_to_backend = np.asarray([backend_index[name] for name in manifest_names], dtype=np.int32)
    expected = np.arange(len(manifest_names), dtype=np.int32)
    if not np.array_equal(np.sort(manifest_to_backend), expected):
        raise ValueError(f"Manifest-to-{backend_label} permutation is not bijective: {manifest_to_backend.tolist()}.")
    backend_to_manifest = np.empty_like(manifest_to_backend)
    backend_to_manifest[manifest_to_backend] = expected
    if not np.array_equal(backend_to_manifest[manifest_to_backend], expected):
        raise ValueError(f"{backend_label}-to-manifest inverse permutation is invalid.")
    return manifest_to_backend, backend_to_manifest


@dataclass(frozen=True)
class _Manifest:
    joint_names: tuple[str, ...]
    default_position: np.ndarray
    effort_limit: np.ndarray
    velocity_limit: np.ndarray
    effort_limit_ratio: np.ndarray | None
    brake_position_lower: np.ndarray | None
    brake_position_upper: np.ndarray | None
    brake_velocity_lookahead: np.ndarray | None
    armature: np.ndarray
    sim_dt: float
    policy_dt: float
    decimation: int
    episode_steps: int
    observation_dim: int
    flight_start_step: int
    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray
    goal_ranges: dict[str, tuple[float, float]]

    @classmethod
    def load(cls, path: Path) -> _Manifest:  # noqa: C901
        try:
            with path.resolve().open(encoding="utf-8") as stream:
                raw = json.load(stream)
        except OSError as exc:
            raise FileNotFoundError(f"Cannot read deployment manifest: {path.resolve()}.") from exc
        root = _mapping(raw, "<root>")
        control = _mapping(root.get("control"), "control")
        joints = _mapping(root.get("joints"), "joints")
        actuators = _mapping(root.get("actuators"), "actuators")
        action = _mapping(root.get("action"), "action")
        observation = _mapping(root.get("observation"), "observation")
        reference = _mapping(root.get("reference"), "reference")
        goal = _mapping(root.get("goal"), "goal")
        tables = _mapping(root.get("tables"), "tables")

        names_value = joints.get("names")
        if not isinstance(names_value, list) or not names_value:
            raise ValueError("Manifest field joints.names must be a non-empty array.")
        if not all(isinstance(name, str) and name for name in names_value):
            raise ValueError("Manifest field joints.names must contain non-empty strings.")
        joint_names = tuple(names_value)
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("Manifest field joints.names must contain unique strings.")
        joint_count = len(joint_names)
        velocity_limit = _finite_vector(actuators.get("velocity_limit"), joint_count, "actuators.velocity_limit")
        if np.any(velocity_limit <= 0.0):
            raise ValueError("Manifest actuator velocity limits must be strictly positive.")
        torque_projection = action.get("torque_projection")
        if torque_projection is None:
            effort_limit_ratio = None
        else:
            projection = _mapping(torque_projection, "action.torque_projection")
            if projection.get("type") != "instantaneous_pd":
                raise ValueError("Manifest action.torque_projection.type must be 'instantaneous_pd'.")
            effort_limit_ratio = _finite_vector(
                projection.get("effort_limit_ratio"),
                joint_count,
                "action.torque_projection.effort_limit_ratio",
            )
            if np.any(effort_limit_ratio <= 0.0) or np.any(effort_limit_ratio > 1.0):
                raise ValueError("Manifest torque-projection ratios must be in (0, 1].")

        lower_limit_brake = action.get("lower_limit_brake")
        if lower_limit_brake is None:
            brake_position_lower = None
            brake_position_upper = None
            brake_velocity_lookahead = None
        else:
            brake = _mapping(lower_limit_brake, "action.lower_limit_brake")
            if brake.get("type") != "velocity_lookahead":
                raise ValueError("Manifest action.lower_limit_brake.type must be 'velocity_lookahead'.")
            brake_period_s = _positive_float(brake.get("period_s"), "action.lower_limit_brake.period_s")
            if not math.isclose(brake_period_s, 0.002, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("Manifest lower-limit braking must run at 500 Hz.")
            brake_position_lower = _finite_vector(
                brake.get("position_lower"), joint_count, "action.lower_limit_brake.position_lower"
            )
            brake_position_upper = _finite_vector(
                brake.get("position_upper"), joint_count, "action.lower_limit_brake.position_upper"
            )
            brake_velocity_lookahead = _finite_vector(
                brake.get("velocity_lookahead_s"),
                joint_count,
                "action.lower_limit_brake.velocity_lookahead_s",
            )
            if np.any(brake_position_lower >= brake_position_upper):
                raise ValueError("Manifest lower-limit brake position bounds must be strictly increasing.")
            if np.any(brake_velocity_lookahead < 0.0) or not np.any(brake_velocity_lookahead > 0.0):
                raise ValueError("Manifest lower-limit brake lookahead must be non-negative and active.")
            action_clip = np.asarray(action.get("clip"), dtype=np.float64)
            expected_clip = np.column_stack((brake_position_lower, brake_position_upper))
            if action_clip.shape != (joint_count, 2) or not np.array_equal(action_clip, expected_clip):
                raise ValueError("Manifest lower-limit brake bounds must exactly equal action.clip.")
            expected_brake_formula = "q_requested = max(q_filtered, min(q_upper, q_lower + t_lookahead*max(-dq, 0)))"
            if brake.get("formula") != expected_brake_formula:
                raise ValueError("Manifest lower-limit brake formula is unsupported.")

        sim_dt = _positive_float(control.get("sim_dt"), "control.sim_dt")
        policy_dt = _positive_float(control.get("policy_dt"), "control.policy_dt")
        decimation_value = control.get("decimation")
        episode_steps_value = control.get("episode_steps")
        observation_dim_value = observation.get("total_dim")
        for value, name in (
            (decimation_value, "control.decimation"),
            (episode_steps_value, "control.episode_steps"),
            (observation_dim_value, "observation.total_dim"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Manifest field {name} must be a positive integer.")
        if not math.isclose(sim_dt, 0.002, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"The FSM MuJoCo backend requires 500 Hz physics, got sim_dt={sim_dt}.")
        if not math.isclose(policy_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"The FSM MuJoCo backend requires 50 Hz control, got policy_dt={policy_dt}.")
        if decimation_value != 10 or not math.isclose(
            policy_dt, sim_dt * decimation_value, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("Manifest control rates must use ten 500 Hz physics steps per 50 Hz FSM step.")

        root_frame0 = _mapping(reference.get("root_frame0"), "reference.root_frame0")
        root_quaternion_xyzw = _finite_vector(root_frame0.get("quat_xyzw"), 4, "reference.root_frame0.quat_xyzw")
        quaternion_norm = float(np.linalg.norm(root_quaternion_xyzw))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("Manifest reference.root_frame0.quat_xyzw must be a unit quaternion.")

        ranges = _mapping(goal.get("ranges"), "goal.ranges")
        goal_ranges = {}
        for name in ("pos_x", "pos_y", "roll", "pitch", "yaw"):
            bounds = _finite_vector(ranges.get(name), 2, f"goal.ranges.{name}")
            if bounds[0] > bounds[1]:
                raise ValueError(f"Manifest goal.ranges.{name} lower bound exceeds its upper bound.")
            goal_ranges[name] = (float(bounds[0]), float(bounds[1]))

        phase_names_value = reference.get("phase_names")
        if not isinstance(phase_names_value, list) or "FLIGHT" not in phase_names_value:
            raise ValueError("Manifest reference.phase_names must contain FLIGHT.")
        phase_filename = tables.get("jump_phase")
        if not isinstance(phase_filename, str) or not phase_filename or Path(phase_filename).name != phase_filename:
            raise ValueError("Manifest tables.jump_phase must name a table beside the manifest.")
        phase_path = path.resolve().parent / phase_filename
        try:
            phase_table = np.load(phase_path, allow_pickle=False)
        except OSError as exc:
            raise FileNotFoundError(f"Cannot read jump phase table: {phase_path}.") from exc
        expected_phase_shape = (episode_steps_value, len(phase_names_value))
        if phase_table.shape != expected_phase_shape or not np.all(np.isfinite(phase_table)):
            raise ValueError(f"Jump phase table must be finite with shape {expected_phase_shape}.")
        flight_steps = np.flatnonzero(np.argmax(phase_table, axis=1) == phase_names_value.index("FLIGHT"))
        if flight_steps.size == 0:
            raise ValueError("Jump phase table contains no FLIGHT samples.")

        return cls(
            joint_names=joint_names,
            default_position=_finite_vector(joints.get("default_pos"), joint_count, "joints.default_pos"),
            effort_limit=_finite_vector(actuators.get("effort_limit"), joint_count, "actuators.effort_limit"),
            velocity_limit=velocity_limit,
            effort_limit_ratio=effort_limit_ratio,
            brake_position_lower=brake_position_lower,
            brake_position_upper=brake_position_upper,
            brake_velocity_lookahead=brake_velocity_lookahead,
            armature=_finite_vector(actuators.get("armature"), joint_count, "actuators.armature"),
            sim_dt=sim_dt,
            policy_dt=policy_dt,
            decimation=decimation_value,
            episode_steps=episode_steps_value,
            observation_dim=observation_dim_value,
            flight_start_step=int(flight_steps[0]),
            root_position=_finite_vector(root_frame0.get("pos"), 3, "reference.root_frame0.pos"),
            root_quaternion_wxyz=np.roll(root_quaternion_xyzw, 1),
            goal_ranges=goal_ranges,
        )


class _Sensor:
    def __init__(self, model: mujoco.MjModel, name: str, dimension: int):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"MuJoCo model is missing sensor {name!r}.")
        actual_dimension = int(model.sensor_dim[sensor_id])
        if actual_dimension != dimension:
            raise ValueError(f"MuJoCo sensor {name!r} has dimension {actual_dimension}, expected {dimension}.")
        self.address = int(model.sensor_adr[sensor_id])
        self.dimension = dimension

    def read(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.sensordata[self.address : self.address + self.dimension], dtype=np.float64).copy()


class MujocoRobot:
    """Run the FSM's manifest-order robot boundary against MuJoCo.

    The latest 50 Hz base-target/gain command is held while
    :meth:`step_physics` adds a freshly evaluated balance correction and
    evaluates saturated PD torque at 500 Hz.

    Args:
        manifest_path: Deployment manifest path.
        model_path: Source G1 MJCF path.
        overlay_path: Sim2sim overlay path.
        feedback_timeout_s: Maximum wall-clock feedback age [s].
        effort_scale: Fraction of manifest effort limits available to control.
        target_rate_limit_rad_s: Optional joint-target slew limit [rad/s].
        emulate_velocity_limit: Whether to emulate the actuator velocity limit
            with torque-speed saturation.
        gantry_support_fraction: Fraction of robot weight supported upward at
            the pelvis.
        ground_contact_enabled: Whether the compiled ground geom can collide.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        model_path: str | Path,
        overlay_path: str | Path,
        *,
        feedback_timeout_s: float = 0.01,
        effort_scale: float = 1.0,
        target_rate_limit_rad_s: float | None = None,
        emulate_velocity_limit: bool = False,
        gantry_support_fraction: float = 0.0,
        ground_contact_enabled: bool = True,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.model_path = Path(model_path).resolve()
        self.overlay_path = Path(overlay_path).resolve()
        self._manifest = _Manifest.load(self.manifest_path)
        if not math.isfinite(feedback_timeout_s) or feedback_timeout_s <= 0.0:
            raise ValueError("feedback_timeout_s must be a positive finite duration.")
        self._feedback_timeout_s = float(feedback_timeout_s)
        if not math.isfinite(effort_scale) or not 0.0 < effort_scale <= 1.0:
            raise ValueError("effort_scale must be finite and in (0, 1].")
        self._effort_scale = float(effort_scale)
        if target_rate_limit_rad_s is not None and (
            not math.isfinite(target_rate_limit_rad_s) or target_rate_limit_rad_s <= 0.0
        ):
            raise ValueError("target_rate_limit_rad_s must be a positive finite velocity.")
        self._target_rate_limit_rad_s = target_rate_limit_rad_s
        if not isinstance(emulate_velocity_limit, bool):
            raise ValueError("emulate_velocity_limit must be a boolean.")
        self._emulate_velocity_limit = emulate_velocity_limit
        if not math.isfinite(gantry_support_fraction) or not 0.0 <= gantry_support_fraction <= 1.0:
            raise ValueError("gantry_support_fraction must be finite and in [0, 1].")
        self._gantry_support_fraction = float(gantry_support_fraction)
        if not isinstance(ground_contact_enabled, bool):
            raise ValueError("ground_contact_enabled must be a boolean.")
        self._ground_contact_enabled = ground_contact_enabled

        model_xml, overlay_timestep = compose_model_xml(self.model_path, self.overlay_path)
        if not math.isclose(overlay_timestep, self._manifest.sim_dt, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"Overlay timestep {overlay_timestep} does not match manifest sim_dt {self._manifest.sim_dt}."
            )
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        if not math.isclose(float(self.model.opt.timestep), self._manifest.sim_dt, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Compiled MuJoCo timestep does not match the deployment manifest.")

        self._resolve_model_indices()
        self._apply_manifest_dynamics()
        self._gantry_support_force_world = (
            -self._gantry_support_fraction
            * float(mujoco.mj_getTotalmass(self.model))
            * np.asarray(self.model.opt.gravity, dtype=np.float64)
        )
        self.data = mujoco.MjData(self.model)
        self._reset_to_reference()
        if not self._ground_contact_enabled:
            self.model.geom_contype[self._ground_geom_id] = 0
            self.model.geom_conaffinity[self._ground_geom_id] = 0
            mujoco.mj_forward(self.model, self.data)

        self._base_target = self._manifest.default_position.copy()
        self._balance_offset = np.zeros(self.joint_count, dtype=np.float64)
        self._stiffness = np.zeros(self.joint_count, dtype=np.float64)
        self._damping = np.zeros(self.joint_count, dtype=np.float64)
        self._applied_torque = np.zeros(self.joint_count, dtype=np.float64)
        self._applied_target = self._manifest.default_position.copy()
        self._feedback_monotonic = time.monotonic()
        self._control_deadline_missed = False
        self._last_control_duration_s = 0.0
        self._maximum_control_duration_s = 0.0
        self._control_deadline_miss_count = 0

    def _resolve_model_indices(self) -> None:  # noqa: C901
        free_type = int(mujoco.mjtJoint.mjJNT_FREE)
        free_joint_ids = [
            joint_id for joint_id in range(self.model.njnt) if int(self.model.jnt_type[joint_id]) == free_type
        ]
        if len(free_joint_ids) != 1:
            raise ValueError(f"MuJoCo model must contain exactly one free joint, got {free_joint_ids}.")
        self._root_joint_id = free_joint_ids[0]
        self._root_body_id = int(self.model.jnt_bodyid[self._root_joint_id])
        self._root_qpos_address = int(self.model.jnt_qposadr[self._root_joint_id])
        self._root_dof_address = int(self.model.jnt_dofadr[self._root_joint_id])

        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        mujoco_joint_ids = np.asarray(
            [joint_id for joint_id in range(self.model.njnt) if int(self.model.jnt_type[joint_id]) == hinge_type],
            dtype=np.int32,
        )
        self.mujoco_joint_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)) for joint_id in mujoco_joint_ids
        )
        if any(name is None for name in self.mujoco_joint_names):
            raise ValueError(f"Every MuJoCo hinge must be named, got {self.mujoco_joint_names}.")
        self.policy_from_mujoco, self.mujoco_from_policy = _name_permutations(
            self.joint_names, self.mujoco_joint_names, "MuJoCo hinge"
        )
        self._joint_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names],
            dtype=np.int32,
        )
        if np.any(self._joint_ids < 0) or len(set(self._joint_ids.tolist())) != self.joint_count:
            raise ValueError("Manifest joints did not resolve to unique MuJoCo joint IDs.")
        hinge_index_by_id = {int(joint_id): index for index, joint_id in enumerate(mujoco_joint_ids)}
        id_permutation = np.asarray([hinge_index_by_id[int(joint_id)] for joint_id in self._joint_ids], dtype=np.int32)
        if not np.array_equal(id_permutation, self.policy_from_mujoco):
            raise ValueError("MuJoCo joint name and ID permutations disagree.")
        self._qpos_addresses = np.asarray(self.model.jnt_qposadr[self._joint_ids], dtype=np.int32)
        self._dof_addresses = np.asarray(self.model.jnt_dofadr[self._joint_ids], dtype=np.int32)

        actuator_joint_ids = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        joint_transmission = int(mujoco.mjtTrn.mjTRN_JOINT)
        if (
            self.model.nu != self.joint_count
            or np.any(self.model.actuator_trntype != joint_transmission)
            or np.any(actuator_joint_ids < 0)
        ):
            raise ValueError(
                f"MuJoCo must contain exactly {self.joint_count} single-joint actuators, got {self.model.nu}."
            )
        self.actuator_joint_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)) for joint_id in actuator_joint_ids
        )
        if any(name is None for name in self.actuator_joint_names):
            raise ValueError(f"Every MuJoCo actuator must target a named joint, got {self.actuator_joint_names}.")
        self.policy_from_actuator, self.actuator_from_policy = _name_permutations(
            self.joint_names, self.actuator_joint_names, "MuJoCo actuator"
        )
        actuator_by_joint_id = {int(joint_id): actuator_id for actuator_id, joint_id in enumerate(actuator_joint_ids)}
        id_permutation = np.asarray(
            [actuator_by_joint_id[int(joint_id)] for joint_id in self._joint_ids], dtype=np.int32
        )
        if not np.array_equal(id_permutation, self.policy_from_actuator):
            raise ValueError("MuJoCo actuator name and ID permutations disagree.")
        expected = np.arange(self.joint_count, dtype=np.int32)
        if not np.array_equal(np.sort(self.policy_from_actuator), expected) or not np.array_equal(
            np.sort(self.actuator_from_policy), expected
        ):
            raise ValueError("MuJoCo actuator permutation is not a genuine bijection.")
        self._actuator_joint_qpos_addresses = np.asarray(self.model.jnt_qposadr[actuator_joint_ids], dtype=np.int32)
        self._actuator_joint_dof_addresses = np.asarray(self.model.jnt_dofadr[actuator_joint_ids], dtype=np.int32)
        expected_gear = np.zeros_like(self.model.actuator_gear)
        expected_gear[:, 0] = 1.0
        if not np.allclose(self.model.actuator_gear, expected_gear):
            raise ValueError("Every MuJoCo actuator must use unit joint transmission gear.")

        self._pelvis_quaternion_sensor = _Sensor(self.model, "imu_quat", 4)
        self._pelvis_angular_velocity_sensor = _Sensor(self.model, "imu_gyro", 3)
        self._pelvis_position_sensor = _Sensor(self.model, "frame_pos", 3)
        self._pelvis_linear_velocity_sensor = _Sensor(self.model, "frame_vel", 3)
        self._ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sim2sim_ground")
        if self._ground_geom_id < 0:
            raise ValueError("Composed MuJoCo model is missing sim2sim_ground.")
        foot_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        foot_geom_ids = []
        for body_name in foot_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"MuJoCo model is missing foot body {body_name!r}.")
            geom_ids = frozenset(
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) == body_id
                and (int(self.model.geom_contype[geom_id]) != 0 or int(self.model.geom_conaffinity[geom_id]) != 0)
            )
            if not geom_ids:
                raise ValueError(f"MuJoCo foot body {body_name!r} has no collidable geoms.")
            foot_geom_ids.append(geom_ids)
        self._foot_geom_ids = (foot_geom_ids[0], foot_geom_ids[1])

    def _apply_manifest_dynamics(self) -> None:
        effort_limit = self._effort_scale * self._manifest.effort_limit
        self.model.dof_armature[self._dof_addresses] = self._manifest.armature
        actuator_ids_policy = self.policy_from_actuator
        self.model.actuator_ctrllimited[actuator_ids_policy] = 1
        self.model.actuator_ctrlrange[actuator_ids_policy, 0] = -effort_limit
        self.model.actuator_ctrlrange[actuator_ids_policy, 1] = effort_limit
        self.model.jnt_actfrclimited[self._joint_ids] = 1
        self.model.jnt_actfrcrange[self._joint_ids, 0] = -effort_limit
        self.model.jnt_actfrcrange[self._joint_ids, 1] = effort_limit

    def _reset_to_reference(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        root_qpos = self._root_qpos_address
        root_dof = self._root_dof_address
        self.data.qpos[root_qpos : root_qpos + 3] = self._manifest.root_position
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = self._manifest.root_quaternion_wxyz
        self.data.qvel[root_dof : root_dof + 6] = 0.0
        self.data.qpos[self._qpos_addresses] = self._manifest.default_position
        self.data.qvel[self._dof_addresses] = 0.0
        if self._ground_contact_enabled:
            self.initial_root_height_offset = apply_initial_ground_clearance(
                self.model,
                self.data,
                self._root_qpos_address,
                self._ground_geom_id,
                self._foot_geom_ids[0] | self._foot_geom_ids[1],
            )
        else:
            self.initial_root_height_offset = 0.0
            mujoco.mj_forward(self.model, self.data)
        self._set_gantry_reference()
        if not np.allclose(
            self.data.qpos[self._qpos_addresses], self._manifest.default_position, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError("MuJoCo did not retain the manifest frame-0 joint positions.")

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Manifest-order joint names."""
        return self._manifest.joint_names

    @property
    def joint_count(self) -> int:
        """Number of controlled joints."""
        return len(self._manifest.joint_names)

    @property
    def sim_dt(self) -> float:
        """Physics and PD period [s]."""
        return self._manifest.sim_dt

    @property
    def policy_dt(self) -> float:
        """FSM and policy period [s]."""
        return self._manifest.policy_dt

    @property
    def decimation(self) -> int:
        """Number of physics steps per FSM step."""
        return self._manifest.decimation

    @property
    def episode_steps(self) -> int:
        """Number of policy steps in the manifest episode."""
        return self._manifest.episode_steps

    @property
    def observation_dim(self) -> int:
        """Policy observation dimension."""
        return self._manifest.observation_dim

    @property
    def flight_start_step(self) -> int:
        """First manifest policy step in the FLIGHT phase."""
        return self._manifest.flight_start_step

    @property
    def goal_ranges(self) -> dict[str, tuple[float, float]]:
        """Copy of the manifest goal envelope."""
        return dict(self._manifest.goal_ranges)

    @property
    def effort_limits(self) -> np.ndarray:
        """Manifest effort limits [N·m], in manifest order."""
        return self._manifest.effort_limit.copy()

    @property
    def command_effort_limits(self) -> np.ndarray:
        """Scaled command effort limits [N·m], in manifest order."""
        ratio = np.full(self.joint_count, self._effort_scale)
        if self._manifest.effort_limit_ratio is not None:
            ratio = np.minimum(ratio, self._manifest.effort_limit_ratio)
        return ratio * self._manifest.effort_limit

    @property
    def gantry_support_force_world(self) -> np.ndarray:
        """Constant pelvis support force in world axes [N]."""
        return self._gantry_support_force_world.copy()

    @property
    def ground_contact_enabled(self) -> bool:
        """Whether the compiled ground geom participates in collision."""
        return self._ground_contact_enabled

    @property
    def joint_positions(self) -> np.ndarray:
        """Measured joint positions [rad], in manifest order."""
        return np.asarray(self.data.qpos[self._qpos_addresses], dtype=np.float64).copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        """Measured joint velocities [rad/s], in manifest order."""
        return np.asarray(self.data.qvel[self._dof_addresses], dtype=np.float64).copy()

    @property
    def base_angular_velocity(self) -> np.ndarray:
        """Pelvis angular velocity in the body frame [rad/s]."""
        return self._pelvis_angular_velocity_sensor.read(self.data)

    @property
    def imu_quaternion(self) -> np.ndarray:
        """World-from-pelvis quaternion in WXYZ order."""
        return self._pelvis_quaternion_sensor.read(self.data)

    @property
    def odometry_position(self) -> np.ndarray:
        """Pelvis position in world coordinates [m]."""
        return self._pelvis_position_sensor.read(self.data)

    @property
    def odometry_quaternion(self) -> np.ndarray:
        """World-from-pelvis odometry quaternion in WXYZ order."""
        return self._pelvis_quaternion_sensor.read(self.data)

    @property
    def pelvis_linear_velocity(self) -> np.ndarray:
        """Pelvis linear velocity in world coordinates [m/s]."""
        return self._pelvis_linear_velocity_sensor.read(self.data)

    @property
    def foot_contact_force_vectors(self) -> np.ndarray:
        """Left/right net ground forces in world XYZ [N]."""
        forces = np.zeros((2, 3), dtype=np.float64)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self._ground_geom_id not in (geom1, geom2):
                continue
            wrench_contact = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_id, wrench_contact)
            force_world = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3).T @ wrench_contact[:3]
            for foot_index, geom_ids in enumerate(self._foot_geom_ids):
                if geom2 in geom_ids:
                    forces[foot_index] += force_world
                elif geom1 in geom_ids:
                    forces[foot_index] -= force_world
        return forces

    @property
    def foot_contact_forces(self) -> np.ndarray:
        """Left/right supporting vertical ground forces [N]."""
        return np.maximum(self.foot_contact_force_vectors[:, 2], 0.0)

    @property
    def joint_limit_violations(self) -> np.ndarray:
        """Flags for joints at or outside their position limits."""
        positions = self.joint_positions
        limited = np.asarray(self.model.jnt_limited[self._joint_ids], dtype=np.bool_)
        ranges = np.asarray(self.model.jnt_range[self._joint_ids], dtype=np.float64)
        return limited & ((positions <= ranges[:, 0]) | (positions >= ranges[:, 1]))

    @property
    def feedback_stale(self) -> bool:
        """Whether the last post-physics feedback sample is too old."""
        return time.monotonic() - self._feedback_monotonic > self._feedback_timeout_s

    @property
    def control_deadline_missed(self) -> bool:
        """Whether the preceding 50 Hz FSM call exceeded its period."""
        return self._control_deadline_missed

    @property
    def command_target(self) -> np.ndarray:
        """Most recently applied rate-limited joint target [rad]."""
        return self._applied_target.copy()

    @property
    def command_base_target(self) -> np.ndarray:
        """Held 50 Hz base joint-position target [rad], in manifest order."""
        return self._base_target.copy()

    @property
    def balance_offset(self) -> np.ndarray:
        """Most recently applied 500 Hz balance correction [rad]."""
        return self._balance_offset.copy()

    @property
    def command_stiffness(self) -> np.ndarray:
        """Held position gains [N·m/rad], in manifest order."""
        return self._stiffness.copy()

    @property
    def command_damping(self) -> np.ndarray:
        """Held velocity gains [N·m·s/rad], in manifest order."""
        return self._damping.copy()

    @property
    def applied_torque(self) -> np.ndarray:
        """Most recent saturated PD torque [N·m], in manifest order."""
        return self._applied_torque.copy()

    @property
    def last_control_duration_s(self) -> float:
        """Most recent FSM execution duration [s]."""
        return self._last_control_duration_s

    @property
    def maximum_control_duration_s(self) -> float:
        """Maximum measured FSM execution duration [s]."""
        return self._maximum_control_duration_s

    @property
    def control_deadline_miss_count(self) -> int:
        """Number of measured FSM calls exceeding the 50 Hz deadline."""
        return self._control_deadline_miss_count

    def print_permutations(self) -> None:
        """Print the verified manifest/backend order mappings."""
        print(f"Joint permutation policy_from_mujoco={self.policy_from_mujoco.tolist()}")
        print(f"Joint permutation mujoco_from_policy={self.mujoco_from_policy.tolist()}")
        print(f"Actuator permutation policy_from_actuator={self.policy_from_actuator.tolist()}")
        print(f"Actuator permutation actuator_from_policy={self.actuator_from_policy.tolist()}")

    def reset_state(
        self,
        joint_positions: np.ndarray,
        root_quaternion_wxyz: np.ndarray | None = None,
    ) -> None:
        """Reset to a measured joint posture and optional pelvis attitude.

        Args:
            joint_positions: Manifest-order joint positions [rad].
            root_quaternion_wxyz: Optional world-from-pelvis quaternion in WXYZ
                order. When omitted, the manifest reference attitude is used.
        """
        positions = np.asarray(joint_positions, dtype=np.float64)
        if positions.shape != (self.joint_count,) or not np.all(np.isfinite(positions)):
            raise ValueError(f"joint_positions must contain {self.joint_count} finite values.")
        quaternion = (
            self._manifest.root_quaternion_wxyz.copy()
            if root_quaternion_wxyz is None
            else np.asarray(root_quaternion_wxyz, dtype=np.float64)
        )
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("root_quaternion_wxyz must contain four finite values.")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= np.finfo(np.float64).eps:
            raise ValueError("root_quaternion_wxyz must be non-zero.")
        quaternion = quaternion / quaternion_norm

        mujoco.mj_resetData(self.model, self.data)
        root_qpos = self._root_qpos_address
        root_dof = self._root_dof_address
        self.data.qpos[root_qpos : root_qpos + 3] = self._manifest.root_position
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = quaternion
        self.data.qvel[root_dof : root_dof + 6] = 0.0
        self.data.qpos[self._qpos_addresses] = positions
        self.data.qvel[self._dof_addresses] = 0.0
        if self._ground_contact_enabled:
            self.initial_root_height_offset = apply_initial_ground_clearance(
                self.model,
                self.data,
                self._root_qpos_address,
                self._ground_geom_id,
                self._foot_geom_ids[0] | self._foot_geom_ids[1],
            )
        else:
            self.initial_root_height_offset = 0.0
            mujoco.mj_forward(self.model, self.data)
        self._set_gantry_reference()
        self._base_target = positions.copy()
        self._balance_offset.fill(0.0)
        self._stiffness.fill(0.0)
        self._damping.fill(0.0)
        self._applied_torque.fill(0.0)
        self._applied_target = positions.copy()
        self._feedback_monotonic = time.monotonic()
        self._control_deadline_missed = False
        self._last_control_duration_s = 0.0
        self._maximum_control_duration_s = 0.0
        self._control_deadline_miss_count = 0

    def _set_gantry_reference(self) -> None:
        root_qpos = self._root_qpos_address
        self._gantry_reference_position = np.asarray(self.data.qpos[root_qpos : root_qpos + 3], dtype=np.float64).copy()
        self._gantry_reference_quaternion = np.asarray(
            self.data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float64
        ).copy()

    def _apply_gantry_wrench(self) -> None:
        self.data.xfrc_applied[self._root_body_id, :] = 0.0
        if self._gantry_support_fraction == 0.0:
            return
        root_qpos = self._root_qpos_address
        root_dof = self._root_dof_address
        position = np.asarray(self.data.qpos[root_qpos : root_qpos + 3], dtype=np.float64)
        linear_velocity = np.asarray(self.data.qvel[root_dof : root_dof + 3], dtype=np.float64)
        force = self._gantry_support_force_world.copy()
        # A load-bearing overhead harness supports height but must leave horizontal
        # travel free so commanded displacement remains observable. Roll and pitch
        # restraint are applied separately below.
        force[2] -= _GANTRY_POSITION_STIFFNESS_N_M * (position[2] - self._gantry_reference_position[2])
        force[2] -= _GANTRY_POSITION_DAMPING_N_S_M * linear_velocity[2]
        force = np.clip(force, -_GANTRY_FORCE_LIMIT_N, _GANTRY_FORCE_LIMIT_N)

        quaternion = np.asarray(self.data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float64)
        attitude_error = np.zeros(3, dtype=np.float64)
        mujoco.mju_subQuat(attitude_error, quaternion, self._gantry_reference_quaternion)
        attitude_error[2] = 0.0
        angular_velocity = np.asarray(self.data.qvel[root_dof + 3 : root_dof + 6], dtype=np.float64)
        torque = -_GANTRY_ATTITUDE_STIFFNESS_N_M_RAD * attitude_error
        torque -= _GANTRY_ATTITUDE_DAMPING_N_M_S_RAD * angular_velocity
        torque[2] = 0.0
        torque = np.clip(torque, -_GANTRY_TORQUE_LIMIT_N_M, _GANTRY_TORQUE_LIMIT_N_M)

        self.data.xfrc_applied[self._root_body_id, :3] = force
        self.data.xfrc_applied[self._root_body_id, 3:] = torque

    def command_joint_position_target(
        self,
        target: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ) -> None:
        """Hold a manifest-order base target and gain set for inner-loop PD.

        Args:
            target: Base joint position targets [rad].
            stiffness: Position gains [N·m/rad].
            damping: Velocity gains [N·m·s/rad].
        """
        values = []
        for value, name in (
            (target, "target"),
            (stiffness, "stiffness"),
            (damping, "damping"),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (self.joint_count,) or not np.all(np.isfinite(array)):
                raise ValueError(f"MuJoCo command {name} must contain {self.joint_count} finite values.")
            values.append(array.copy())
        if np.any(values[1] < 0.0) or np.any(values[2] < 0.0):
            raise ValueError("MuJoCo command gains must be non-negative.")
        self._base_target, self._stiffness, self._damping = values

    def set_target_rate_limit(self, target_rate_limit_rad_s: float | None) -> None:
        """Set the fast-loop joint-target slew limit.

        Args:
            target_rate_limit_rad_s: Joint-target slew limit [rad/s], or
                ``None`` to retain the full target dynamics.
        """
        if target_rate_limit_rad_s is not None and (
            not math.isfinite(target_rate_limit_rad_s) or target_rate_limit_rad_s <= 0.0
        ):
            raise ValueError("target_rate_limit_rad_s must be a positive finite velocity or None.")
        self._target_rate_limit_rad_s = target_rate_limit_rad_s

    def record_control_duration(self, duration_s: float) -> None:
        """Record one FSM call duration for deadline reporting.

        Args:
            duration_s: Wall-clock FSM execution duration [s].
        """
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("Control duration must be finite and non-negative.")
        self._last_control_duration_s = float(duration_s)
        self._maximum_control_duration_s = max(self._maximum_control_duration_s, self._last_control_duration_s)
        self._control_deadline_missed = self._last_control_duration_s > self.policy_dt
        self._control_deadline_miss_count += int(self._control_deadline_missed)

    def step_physics(self, balance_offset: np.ndarray) -> None:
        """Apply fast balance feedback and advance MuJoCo by one 500 Hz step.

        Args:
            balance_offset: Gated joint-target correction [rad], in manifest
                order, freshly evaluated for this physics step.
        """
        offset = np.asarray(balance_offset, dtype=np.float64)
        if offset.shape != (self.joint_count,) or not np.all(np.isfinite(offset)):
            raise ValueError(f"Balance offset must contain {self.joint_count} finite values.")
        self._balance_offset = offset.copy()
        desired_target = self._base_target + self._balance_offset
        if not np.all(np.isfinite(desired_target)):
            raise ValueError("Base target plus balance offset must be finite.")
        if self._target_rate_limit_rad_s is None:
            target = desired_target
        else:
            maximum_step = self._target_rate_limit_rad_s * self.sim_dt
            target = self._applied_target + np.clip(
                desired_target - self._applied_target,
                -maximum_step,
                maximum_step,
            )
        if self._manifest.brake_velocity_lookahead is not None:
            target = project_position_target_to_lower_limit(
                target,
                self.joint_velocities,
                self._manifest.brake_position_lower,
                self._manifest.brake_position_upper,
                self._manifest.brake_velocity_lookahead,
            )
        if self._manifest.effort_limit_ratio is not None and np.all(self._stiffness > 0.0):
            available_ratio = np.minimum(self._manifest.effort_limit_ratio, self._effort_scale)
            target = project_pd_position_target(
                target,
                self.joint_positions,
                self.joint_velocities,
                self._stiffness,
                self._damping,
                self._manifest.effort_limit,
                available_ratio,
            )
        self._applied_target = target.copy()
        target_actuator = target[self.actuator_from_policy]
        stiffness_actuator = self._stiffness[self.actuator_from_policy]
        damping_actuator = self._damping[self.actuator_from_policy]
        effort_limit_actuator = self.command_effort_limits[self.actuator_from_policy]
        positions_actuator = np.asarray(self.data.qpos[self._actuator_joint_qpos_addresses], dtype=np.float64)
        velocities_actuator = np.asarray(self.data.qvel[self._actuator_joint_dof_addresses], dtype=np.float64)
        torque_actuator = stiffness_actuator * (target_actuator - positions_actuator)
        torque_actuator -= damping_actuator * velocities_actuator
        torque_actuator = np.clip(torque_actuator, -effort_limit_actuator, effort_limit_actuator)
        if self._emulate_velocity_limit:
            velocity_limit_actuator = self._manifest.velocity_limit[self.actuator_from_policy]
            torque_actuator = saturate_torque_at_velocity_limit(
                torque_actuator,
                velocities_actuator,
                velocity_limit_actuator,
            )
        self.data.ctrl[:] = torque_actuator
        self._applied_torque = torque_actuator[self.policy_from_actuator].copy()
        self._apply_gantry_wrench()
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._feedback_monotonic = time.monotonic()


@dataclass(frozen=True)
class OperatorTimelineEntry:
    """One reproducible operator intent pulse.

    Args:
        time_s: Pulse start in simulation time [s].
        duration_s: Pulse duration [s].
        goal: Goal made pending from this entry onward.
        request_start: Whether the start intent is active during the pulse.
        confirm: Whether the confirmation intent is active during the pulse.
        abort: Whether the abort intent is active during the pulse.
        label: Human-readable event name.
    """

    time_s: float
    duration_s: float = 0.02
    goal: JumpGoal | None = None
    request_start: bool = False
    confirm: bool = False
    abort: bool = False
    label: str = ""


class ScriptedOperator:
    """Expose operator intents from a fixed simulation-time timeline.

    Args:
        timeline: Ordered intent pulses.
    """

    def __init__(self, timeline: tuple[OperatorTimelineEntry, ...]):
        previous_time = -math.inf
        for entry in timeline:
            if not math.isfinite(entry.time_s) or entry.time_s < 0.0:
                raise ValueError("Operator event times must be finite and non-negative.")
            if not math.isfinite(entry.duration_s) or entry.duration_s <= 0.0:
                raise ValueError("Operator event durations must be finite and positive.")
            if entry.time_s < previous_time:
                raise ValueError("Operator timeline entries must be ordered by time.")
            if entry.goal is not None and not isinstance(entry.goal, JumpGoal):
                raise TypeError("Operator timeline goals must be JumpGoal instances.")
            previous_time = entry.time_s
        self.timeline = timeline
        self._time_s = 0.0
        self._pending_goal: JumpGoal | None = None

    def update(self, time_s: float) -> None:
        """Advance the operator timeline to a simulation time.

        Args:
            time_s: Current simulation time [s].
        """
        if not math.isfinite(time_s) or time_s < self._time_s - 1.0e-12:
            raise ValueError("Operator timeline time must be finite and monotonic.")
        self._time_s = float(time_s)
        for entry in self.timeline:
            if entry.time_s <= self._time_s + 1.0e-12 and entry.goal is not None:
                self._pending_goal = entry.goal

    def _active(self, attribute: str) -> bool:
        return any(
            getattr(entry, attribute)
            and entry.time_s <= self._time_s + 1.0e-12
            and self._time_s < entry.time_s + entry.duration_s - 1.0e-12
            for entry in self.timeline
        )

    @property
    def pending_goal(self) -> JumpGoal | None:
        """Most recently offered goal."""
        return self._pending_goal

    @property
    def request_start(self) -> bool:
        """Whether a start pulse is active."""
        return self._active("request_start")

    @property
    def confirm(self) -> bool:
        """Whether a confirmation pulse is active."""
        return self._active("confirm")

    @property
    def abort(self) -> bool:
        """Whether an abort pulse is active."""
        return self._active("abort")

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the G1 IMU-feedback ankle strategy in MuJoCo at 500 Hz."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_MUJOCO_DIR = _SCRIPT_DIR.parent / "mujoco"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(_MUJOCO_DIR))

from model_overlay import apply_initial_ground_clearance, compose_model_xml  # noqa: E402
from physics_parity import (  # noqa: E402
    PhysicsParityConfig,
    add_physics_parity_arguments,
    apply_physics_parity,
    configure_implicit_pd,
)
from static_equilibrium import StaticEquilibriumResult, settle_static_equilibrium  # noqa: E402

from scripts.g1_jump_deploy.control.balance import (  # noqa: E402
    BalanceController,
    BalanceControllerConfig,
    project_ankle_target,
    quaternion_to_roll_pitch,
)

_DEFAULT_MANIFEST = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"
_DEFAULT_MODEL = _REPO_ROOT / "data_storage" / "g1_23dof_holo_compat.xml"
_DEFAULT_OVERLAY = _MUJOCO_DIR / "model_overlay.xml"
_DEFAULT_DURATION = 10.0
_PUSH_TIME = 5.0
_PUSH_DURATION = 0.05
_TARGET_ROLL = math.radians(0.43)
_TARGET_PITCH = math.radians(7.49)
_GAIN_SWEEP = ((1.75, 0.2), (2.4, 0.16), (2.8, 0.16), (3.2, 0.16), (3.5, 0.2), (4.0, 0.2))
_PUSH_SWEEP = (-19.0, -18.0, -15.0, -10.0, 10.0, 15.0, 18.0, 20.0, 21.0, 22.0)
_STAND_ANKLE_KP = 80.0
_STAND_ANKLE_KD = 5.0
_STAND_ANKLE_GAIN_SWEEP = (
    (60.0, 4.3),
    (70.0, 4.7),
    (80.0, 5.0),
    (90.0, 5.3),
    (100.0, 5.6),
    (110.0, 5.9),
    (120.0, 6.1),
)
_TABLE_HEADER = (
    "label              final tilt final drift  max drift  max tilt  peak N.m non-foot contacts        verdict"
)


def _as_float_array(value: Any, path: str, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Manifest field {path} must contain {length} finite numbers.")
    return result


def _resolve_reference_path(declared_path: str) -> Path:
    path = Path(declared_path)
    if path.is_file():
        return path
    workspace_root = Path("/workspace/isaaclab")
    try:
        relative_path = path.relative_to(workspace_root)
    except ValueError as exc:
        raise FileNotFoundError(f"Manifest reference CSV does not exist: {path}.") from exc
    relocated_path = _REPO_ROOT / relative_path
    if not relocated_path.is_file():
        raise FileNotFoundError(f"Manifest reference CSV does not exist at {path} or relocated path {relocated_path}.")
    return relocated_path


@dataclass(frozen=True)
class StandManifest:
    """Validated manifest fields needed by the standalone stand measurement."""

    joint_names: tuple[str, ...]
    reference_joint_pos: np.ndarray
    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limit: np.ndarray
    armature: np.ndarray
    sim_dt: float

    @classmethod
    def load(cls, path: Path) -> StandManifest:
        """Load the reference state and actuator configuration from a manifest."""
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        control = raw["control"]
        sim_dt = float(control["sim_dt"])
        if not math.isclose(sim_dt, 0.002, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"Stand validation requires a 0.002 s timestep, got {sim_dt}.")

        joints = raw["joints"]
        joint_names = tuple(joints["names"])
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("Manifest joint names must be non-empty and unique.")
        joint_count = len(joint_names)
        default_pos = _as_float_array(joints["default_pos"], "joints.default_pos", joint_count)

        reference = raw["reference"]
        source_path = _resolve_reference_path(reference["source_csv"])
        source_bytes = source_path.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        if digest != reference["source_sha256"]:
            raise ValueError(
                "Reference CSV SHA-256 disagrees with the manifest: "
                f"expected {reference['source_sha256']}, got {digest}."
            )
        try:
            frame_zero = next(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))
            reference_joint_pos = np.asarray([float(frame_zero[name]) for name in joint_names], dtype=np.float64)
        except (KeyError, StopIteration, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Could not read the declared frame-0 joint state from {source_path}.") from exc
        if not np.allclose(reference_joint_pos, default_pos, rtol=0.0, atol=1.0e-6):
            raise ValueError("Reference frame-0 joint positions disagree with joints.default_pos.")

        root_frame = reference["root_frame0"]
        root_position = _as_float_array(root_frame["pos"], "reference.root_frame0.pos", 3)
        root_quaternion_xyzw = _as_float_array(root_frame["quat_xyzw"], "reference.root_frame0.quat_xyzw", 4)
        if not math.isclose(float(np.linalg.norm(root_quaternion_xyzw)), 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("Reference root quaternion must have unit norm.")

        actuators = raw["actuators"]
        return cls(
            joint_names=joint_names,
            reference_joint_pos=reference_joint_pos,
            root_position=root_position,
            root_quaternion_wxyz=np.roll(root_quaternion_xyzw, 1),
            stiffness=_as_float_array(actuators["stiffness"], "actuators.stiffness", joint_count),
            damping=_as_float_array(actuators["damping"], "actuators.damping", joint_count),
            effort_limit=_as_float_array(actuators["effort_limit"], "actuators.effort_limit", joint_count),
            armature=_as_float_array(actuators["armature"], "actuators.armature", joint_count),
            sim_dt=sim_dt,
        )


class StandModel:
    """Name-resolved MuJoCo addresses used by the stand measurement."""

    def __init__(self, model: mujoco.MjModel, manifest: StandManifest):
        free_joint_ids = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
        if len(free_joint_ids) != 1:
            raise ValueError(f"Expected one floating-base joint, got {free_joint_ids.tolist()}.")
        root_joint_id = int(free_joint_ids[0])
        self.root_qpos_adr = int(model.jnt_qposadr[root_joint_id])
        self.root_dof_adr = int(model.jnt_dofadr[root_joint_id])

        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in manifest.joint_names],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0) or len(set(self.joint_ids.tolist())) != len(manifest.joint_names):
            raise ValueError("Could not uniquely resolve every manifest joint in MuJoCo.")
        self.qpos_adr = np.asarray(model.jnt_qposadr[self.joint_ids], dtype=np.int32)
        self.dof_adr = np.asarray(model.jnt_dofadr[self.joint_ids], dtype=np.int32)

        actuator_joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
        actuator_by_joint = {int(joint_id): actuator_id for actuator_id, joint_id in enumerate(actuator_joint_ids)}
        try:
            self.actuator_ids = np.asarray(
                [actuator_by_joint[int(joint_id)] for joint_id in self.joint_ids], dtype=np.int32
            )
        except KeyError as exc:
            raise ValueError("Every manifest joint must have one MuJoCo actuator.") from exc
        if len(set(self.actuator_ids.tolist())) != len(manifest.joint_names):
            raise ValueError("MuJoCo actuator mapping is not one-to-one.")

        self.quaternion_slice = self._sensor_slice(model, "imu_quat", 4)
        self.gyroscope_slice = self._sensor_slice(model, "imu_gyro", 3)
        self.position_slice = self._sensor_slice(model, "frame_pos", 3)
        self.ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sim2sim_ground")
        self.pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.ground_geom_id < 0 or self.pelvis_body_id < 0:
            raise ValueError("Composed model is missing the ground or pelvis.")
        foot_body_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        }
        if -1 in foot_body_ids:
            raise ValueError("Could not resolve both foot bodies.")
        self.foot_geom_ids = frozenset(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in foot_body_ids
            and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
        )
        if not self.foot_geom_ids:
            raise ValueError("Could not resolve collidable foot geoms.")

        model.dof_armature[self.dof_adr] = manifest.armature
        model.actuator_ctrllimited[self.actuator_ids] = 1
        model.actuator_ctrlrange[self.actuator_ids, 0] = -manifest.effort_limit
        model.actuator_ctrlrange[self.actuator_ids, 1] = manifest.effort_limit
        model.jnt_actfrclimited[self.joint_ids] = 1
        model.jnt_actfrcrange[self.joint_ids, 0] = -manifest.effort_limit
        model.jnt_actfrcrange[self.joint_ids, 1] = manifest.effort_limit

    @staticmethod
    def _sensor_slice(model: mujoco.MjModel, name: str, dimension: int) -> slice:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0 or int(model.sensor_dim[sensor_id]) != dimension:
            raise ValueError(f"MuJoCo sensor {name!r} must have dimension {dimension}.")
        address = int(model.sensor_adr[sensor_id])
        return slice(address, address + dimension)


@dataclass(frozen=True)
class RunResult:
    """End-to-end measurements from one stand simulation."""

    duration: float
    final_roll_deg: float
    final_pitch_deg: float
    final_tilt_error_deg: float
    maximum_tilt_error_deg: float
    horizontal_drift: float
    maximum_horizontal_drift: float
    nonfoot_contacts: tuple[str, ...]
    peak_torque: float
    finite: bool
    settle: StaticEquilibriumResult

    @property
    def passed(self) -> bool:
        """Whether this run meets all ten-second stand acceptance thresholds."""
        return (
            self.duration >= 10.0 - 1.0e-12
            and self.final_tilt_error_deg < 5.0
            and self.maximum_horizontal_drift < 0.05
            and not self.nonfoot_contacts
            and self.finite
        )

    @property
    def recovered(self) -> bool:
        """Whether a push run finishes balanced without a fall."""
        return (
            self.final_tilt_error_deg < 5.0
            and self.horizontal_drift < 0.05
            and not self.nonfoot_contacts
            and self.finite
        )


def _contact_names(model: mujoco.MjModel, data: mujoco.MjData, indices: StandModel) -> set[str]:
    names: set[str] = set()
    for contact_id in range(data.ncon):
        geom1 = int(data.contact[contact_id].geom1)
        geom2 = int(data.contact[contact_id].geom2)
        if indices.ground_geom_id not in (geom1, geom2):
            continue
        robot_geom = geom2 if geom1 == indices.ground_geom_id else geom1
        if robot_geom not in indices.foot_geom_ids:
            names.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom) or f"geom_{robot_geom}")
    return names


def _simulate(
    model_xml: str,
    manifest: StandManifest,
    config: BalanceControllerConfig,
    duration: float,
    *,
    response_sign: int = 1,
    push_impulse: float = 0.0,
    stand_ankle_kp: float = _STAND_ANKLE_KP,
    stand_ankle_kd: float = _STAND_ANKLE_KD,
    parity_config: PhysicsParityConfig = PhysicsParityConfig(),
    log_initial_offset: bool = False,
) -> RunResult:
    if not math.isfinite(push_impulse):
        raise ValueError("Push impulse must be finite.")
    if push_impulse != 0.0 and duration < _PUSH_TIME + _PUSH_DURATION:
        raise ValueError("A non-zero push requires the simulation to include the complete push interval.")
    if response_sign not in (-1, 1):
        raise ValueError("Response sign must be +1 or -1.")
    if not math.isfinite(stand_ankle_kp) or stand_ankle_kp <= 0.0:
        raise ValueError("Stand ankle stiffness must be positive and finite.")
    if not math.isfinite(stand_ankle_kd) or stand_ankle_kd < 0.0:
        raise ValueError("Stand ankle damping must be non-negative and finite.")
    model = mujoco.MjModel.from_xml_string(model_xml)
    if not math.isclose(float(model.opt.timestep), manifest.sim_dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Composed MuJoCo timestep disagrees with the manifest.")
    indices = StandModel(model, manifest)
    apply_physics_parity(
        model,
        indices.actuator_ids,
        manifest.stiffness,
        manifest.damping,
        manifest.effort_limit,
        parity_config,
        print_status=log_initial_offset,
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[indices.root_qpos_adr : indices.root_qpos_adr + 3] = manifest.root_position
    data.qpos[indices.root_qpos_adr + 3 : indices.root_qpos_adr + 7] = manifest.root_quaternion_wxyz
    data.qvel[indices.root_dof_adr : indices.root_dof_adr + 6] = 0.0
    data.qpos[indices.qpos_adr] = manifest.reference_joint_pos
    data.qvel[indices.dof_adr] = 0.0
    initial_root_height_offset = apply_initial_ground_clearance(
        model,
        data,
        indices.root_qpos_adr,
        indices.ground_geom_id,
        indices.foot_geom_ids,
    )
    if log_initial_offset:
        print(f"Initial root height offset: {initial_root_height_offset:.9f} m")

    settle_result = settle_static_equilibrium(
        model,
        data,
        root_dof_adr=indices.root_dof_adr,
        joint_qpos_adr=indices.qpos_adr,
        joint_dof_adr=indices.dof_adr,
        actuator_ids=indices.actuator_ids,
        reference_joint_pos=manifest.reference_joint_pos,
        stiffness=manifest.stiffness,
        damping=manifest.damping,
        effort_limit=manifest.effort_limit,
        ground_geom_id=indices.ground_geom_id,
        foot_geom_ids=indices.foot_geom_ids,
        use_implicit_pd=parity_config.use_implicit_pd,
    )
    if log_initial_offset:
        print(
            "Static-equilibrium settle: "
            f"duration={settle_result.duration_s:.3f} s, "
            f"foot_contact_force={settle_result.foot_contact_force_n:.3f} N, "
            f"root_linear_speed={settle_result.root_linear_speed_m_s:.6f} m/s, "
            f"root_angular_speed={settle_result.root_angular_speed_rad_s:.6f} rad/s"
        )

    initial_position = np.asarray(data.sensordata[indices.position_slice], dtype=np.float64).copy()
    controller = BalanceController(manifest.joint_names, config)
    stand_target = manifest.reference_joint_pos.copy()
    steps = int(round(duration / manifest.sim_dt))
    if not math.isclose(steps * manifest.sim_dt, duration, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Duration {duration} s is not an integer number of 0.002 s steps.")

    maximum_tilt_error = 0.0
    maximum_drift = 0.0
    peak_torque = 0.0
    nonfoot_contacts: set[str] = set()
    finite = True
    stiffness = manifest.stiffness.copy()
    damping = manifest.damping.copy()
    ankle_indices = [
        manifest.joint_names.index(f"{side}_ankle_{axis}_joint")
        for side in ("left", "right")
        for axis in ("pitch", "roll")
    ]
    stiffness[ankle_indices] = stand_ankle_kp
    damping[ankle_indices] = stand_ankle_kd
    if parity_config.use_implicit_pd:
        configure_implicit_pd(model, indices.actuator_ids, stiffness, damping, manifest.effort_limit)
    push_applied_steps = 0
    for step in range(steps):
        quaternion = np.asarray(data.sensordata[indices.quaternion_slice], dtype=np.float64).copy()
        angular_velocity = np.asarray(data.sensordata[indices.gyroscope_slice], dtype=np.float64).copy()
        joint_positions = np.asarray(data.qpos[indices.qpos_adr], dtype=np.float64).copy()
        joint_velocities = np.asarray(data.qvel[indices.dof_adr], dtype=np.float64).copy()
        joint_target = controller.compute(
            stand_target,
            quaternion,
            angular_velocity,
            joint_positions,
            joint_velocities,
            manifest.sim_dt,
        )
        if response_sign == -1:
            pitch_offset, roll_offset = controller.last_ankle_offset
            for side in ("left", "right"):
                pitch_index = manifest.joint_names.index(f"{side}_ankle_pitch_joint")
                roll_index = manifest.joint_names.index(f"{side}_ankle_roll_joint")
                joint_target[pitch_index], joint_target[roll_index] = project_ankle_target(
                    stand_target[pitch_index] - pitch_offset,
                    stand_target[roll_index] - roll_offset,
                )
        if parity_config.use_implicit_pd:
            data.ctrl[indices.actuator_ids] = joint_target
        else:
            torque = stiffness * (joint_target - joint_positions) - damping * joint_velocities
            data.ctrl[indices.actuator_ids] = np.clip(torque, -manifest.effort_limit, manifest.effort_limit)

        sim_time = step * manifest.sim_dt
        data.xfrc_applied[indices.pelvis_body_id].fill(0.0)
        if push_impulse != 0.0 and _PUSH_TIME <= sim_time < _PUSH_TIME + _PUSH_DURATION:
            data.xfrc_applied[indices.pelvis_body_id, 1] = push_impulse / _PUSH_DURATION
            push_applied_steps += 1

        mujoco.mj_step(model, data)
        realised_torque = np.asarray(data.actuator_force[indices.actuator_ids], dtype=np.float64)
        peak_torque = max(peak_torque, float(np.max(np.abs(realised_torque))))
        mujoco.mj_forward(model, data)
        actual_quaternion = np.asarray(data.sensordata[indices.quaternion_slice], dtype=np.float64)
        roll, pitch = quaternion_to_roll_pitch(actual_quaternion)
        tilt_error = math.hypot(roll - config.target_roll, pitch - config.target_pitch)
        position = np.asarray(data.sensordata[indices.position_slice], dtype=np.float64)
        drift = float(np.linalg.norm(position[:2] - initial_position[:2]))
        maximum_tilt_error = max(maximum_tilt_error, tilt_error)
        maximum_drift = max(maximum_drift, drift)
        nonfoot_contacts.update(_contact_names(model, data, indices))
        finite &= bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))

    if push_impulse != 0.0:
        expected_push_steps = int(round(_PUSH_DURATION / manifest.sim_dt))
        if push_applied_steps != expected_push_steps:
            raise RuntimeError(
                f"Push was applied for {push_applied_steps} steps; expected exactly {expected_push_steps}."
            )

    final_quaternion = np.asarray(data.sensordata[indices.quaternion_slice], dtype=np.float64)
    final_roll, final_pitch = quaternion_to_roll_pitch(final_quaternion)
    final_tilt_error = math.hypot(final_roll - config.target_roll, final_pitch - config.target_pitch)
    final_position = np.asarray(data.sensordata[indices.position_slice], dtype=np.float64)
    horizontal_drift = float(np.linalg.norm(final_position[:2] - initial_position[:2]))
    return RunResult(
        duration=duration,
        final_roll_deg=math.degrees(final_roll),
        final_pitch_deg=math.degrees(final_pitch),
        final_tilt_error_deg=math.degrees(final_tilt_error),
        maximum_tilt_error_deg=math.degrees(maximum_tilt_error),
        horizontal_drift=horizontal_drift,
        maximum_horizontal_drift=maximum_drift,
        nonfoot_contacts=tuple(sorted(nonfoot_contacts)),
        peak_torque=peak_torque,
        finite=finite,
        settle=settle_result,
    )


def _print_result(label: str, result: RunResult, verdict: str) -> None:
    contacts = "none" if not result.nonfoot_contacts else ";".join(result.nonfoot_contacts)
    print(
        f"{label:<18} {result.final_tilt_error_deg:10.3f} {100.0 * result.horizontal_drift:11.3f} "
        f"{100.0 * result.maximum_horizontal_drift:10.3f} "
        f"{result.maximum_tilt_error_deg:10.3f} {result.peak_torque:10.3f} {contacts:<24} {verdict}"
    )


def _config(kp: float, kd: float, ki: float, integral_enabled: bool) -> BalanceControllerConfig:
    return BalanceControllerConfig(
        target_roll=_TARGET_ROLL,
        target_pitch=_TARGET_PITCH,
        roll_kp=kp,
        pitch_kp=kp,
        roll_kd=kd,
        pitch_kd=kd,
        roll_ki=ki,
        pitch_ki=ki,
        integral_enabled=integral_enabled,
        initial_pitch_integral=0.2 if integral_enabled else 0.0,
    )


def _parse_push_impulses(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of numbers.") from exc
    if not result or not all(math.isfinite(item) and item != 0.0 for item in result):
        raise argparse.ArgumentTypeError("Push impulses must be finite and non-zero.")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--overlay", type=Path, default=_DEFAULT_OVERLAY)
    parser.add_argument("--duration", type=float, default=_DEFAULT_DURATION)
    parser.add_argument("--kp", type=float, default=BalanceControllerConfig.roll_kp)
    parser.add_argument("--kd", type=float, default=BalanceControllerConfig.roll_kd)
    parser.add_argument("--ki", type=float, default=BalanceControllerConfig.roll_ki)
    parser.add_argument("--no_integral", action="store_true")
    parser.add_argument("--push_impulses", type=_parse_push_impulses, default=_PUSH_SWEEP)
    parser.add_argument("--skip_sweep", action="store_true", help="Skip gain sweep for development-only quick runs.")
    parser.add_argument(
        "--skip_push_sweep", action="store_true", help="Skip push sweep for development-only quick runs."
    )
    add_physics_parity_arguments(parser)
    return parser.parse_args()


def main() -> int:
    """Run the measured stand acceptance, gain sweep, sign check, and push sweep."""
    args = _parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError("Duration must be positive and finite.")
    manifest = StandManifest.load(args.manifest.resolve())
    model_xml, timestep = compose_model_xml(args.model.resolve(), args.overlay.resolve())
    if not math.isclose(timestep, manifest.sim_dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Overlay and manifest timesteps disagree.")
    chosen_config = _config(args.kp, args.kd, args.ki, not args.no_integral)
    parity_config = PhysicsParityConfig.from_args(args)

    print("G1 stand balance measurement at 500 Hz")
    print(
        f"target roll={math.degrees(_TARGET_ROLL):.2f} deg, pitch={math.degrees(_TARGET_PITCH):.2f} deg; "
        f"chosen outer kp={args.kp:g}, kd={args.kd:g}, ki={args.ki:g}, integral={not args.no_integral}; "
        f"stand ankle joint kp={_STAND_ANKLE_KP:g}, kd={_STAND_ANKLE_KD:g}"
    )
    print(_TABLE_HEADER)
    chosen_result = _simulate(
        model_xml,
        manifest,
        chosen_config,
        args.duration,
        parity_config=parity_config,
        log_initial_offset=True,
    )
    _print_result("chosen", chosen_result, "PASS" if chosen_result.passed else "FAIL")

    sign_duration = min(3.0, args.duration)
    print("\nEmpirical response-sign check (same state feedback, opposite target-offset sign)")
    print("Selected convention: positive body attitude error -> positive ankle target offset.")
    positive_sign = _simulate(
        model_xml, manifest, chosen_config, sign_duration, response_sign=1, parity_config=parity_config
    )
    negative_sign = _simulate(
        model_xml, manifest, chosen_config, sign_duration, response_sign=-1, parity_config=parity_config
    )
    print(_TABLE_HEADER)
    _print_result("positive error -> +", positive_sign, "SELECTED")
    _print_result("positive error -> -", negative_sign, "REJECTED")

    robustness_passed = True
    if not args.skip_sweep:
        print("\nBalance gain sweep")
        print(_TABLE_HEADER)
        for kp, kd in _GAIN_SWEEP:
            result = _simulate(
                model_xml,
                manifest,
                _config(kp, kd, args.ki, not args.no_integral),
                args.duration,
                parity_config=parity_config,
            )
            _print_result(f"kp={kp:g} kd={kd:g}", result, "PASS" if result.passed else "FAIL")

        print("\nStand ankle joint-gain sweep with balance feedback")
        print(_TABLE_HEADER)
        for ankle_kp, ankle_kd in _STAND_ANKLE_GAIN_SWEEP:
            result = _simulate(
                model_xml,
                manifest,
                chosen_config,
                args.duration,
                stand_ankle_kp=ankle_kp,
                stand_ankle_kd=ankle_kd,
                parity_config=parity_config,
            )
            _print_result(f"joint {ankle_kp:g}/{ankle_kd:g}", result, "PASS" if result.passed else "FAIL")
            robustness_passed &= result.passed
        print(f"Gain robustness [60, 120]: {'PASS' if robustness_passed else 'FAIL'}")

    if not args.skip_push_sweep:
        print(f"\nSigned lateral pelvis push sweep ({_PUSH_DURATION:.3f} s impulse at t={_PUSH_TIME:.1f} s)")
        print(_TABLE_HEADER)
        survived = {-1: [], 1: []}
        failed = {-1: [], 1: []}
        for impulse in args.push_impulses:
            result = _simulate(
                model_xml,
                manifest,
                chosen_config,
                max(args.duration, _PUSH_TIME + 5.0),
                push_impulse=impulse,
                parity_config=parity_config,
            )
            recovered = result.recovered
            direction = 1 if impulse > 0.0 else -1
            if recovered:
                survived[direction].append(abs(impulse))
            else:
                failed[direction].append(abs(impulse))
            _print_result(f"push={impulse:+g} N.s", result, "RECOVERED" if recovered else "NOT RECOVERED")
        for direction, label in ((-1, "-Y"), (1, "+Y")):
            if not survived[direction] and not failed[direction]:
                print(f"{label}: not tested.")
                continue
            if not survived[direction]:
                print(f"{label}: did not recover the smallest tested impulse.")
                continue
            largest_recovered = max(survived[direction])
            larger_failures = [magnitude for magnitude in failed[direction] if magnitude > largest_recovered]
            if larger_failures:
                print(
                    f"{label}: recovered {largest_recovered:g} N.s; "
                    f"smallest larger tested failure {min(larger_failures):g} N.s."
                )
            else:
                print(f"{label}: recovered every tested impulse through {largest_recovered:g} N.s.")

    print("\nAcceptance thresholds: duration >= 10 s, final tilt < 5 deg, max drift < 5 cm, no non-foot contact.")
    return 0 if chosen_result.passed and robustness_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

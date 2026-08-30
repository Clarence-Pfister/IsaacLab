# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an RSL-RL jump policy in Isaac and record a MuJoCo-compatible trajectory."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import math
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tensordict import TensorDict

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from deploy_mujoco import DeploymentManifest, OnnxPolicy, StepLogger, _load_reference_frame0
from isaac_openloop_replay import (
    _foot_forces,
    _numpy,
    _require_close,
    _require_reference_source,
    _write_and_verify_frame0,
)

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

_CONTACT_FORCE_THRESHOLD_N = 1.0


def _create_parser() -> argparse.ArgumentParser:
    """Create the Isaac policy-rollout argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--policy_onnx",
        type=Path,
        default=None,
        help=(
            "Drive the rollout with an exported ONNX actor instead of an RSL-RL checkpoint, so the "
            "run is independent of the training task's current policy architecture."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=Path("isaac_policy_rollout.npz"))
    parser.add_argument(
        "--goal_pos_x",
        type=float,
        default=None,
        help="Goal x displacement [m]. Defaults to the midpoint of the manifest range.",
    )
    parser.add_argument("--goal_pos_y", type=float, default=None, help="Goal y displacement [m].")
    parser.add_argument("--goal_yaw", type=float, default=None, help="Goal yaw [rad].")
    parser.add_argument(
        "--delay_steps",
        type=int,
        default=0,
        help="Fixed raw-action delay [policy steps]. Must lie inside the manifest range.",
    )
    parser.add_argument(
        "--force_manifest_dynamics",
        action="store_true",
        help=(
            "Overwrite the resolved task's action pipeline and actuator gains with the deployment "
            "manifest's, so an exported bundle can be replayed after the training task moves on. "
            "The run then reproduces the bundle rather than the current training configuration."
        ),
    )
    parser.add_argument(
        "--contact_stiffness",
        type=float,
        default=None,
        help=(
            "Per-shape PhysX compliant contact stiffness [N/m] for every robot collision shape. "
            "Zero restores rigid contacts. Omit to leave the task's contact model untouched."
        ),
    )
    parser.add_argument(
        "--contact_damping",
        type=float,
        default=None,
        help=(
            "Per-shape PhysX compliant contact damping [N.s/m]. Defaults to critical damping for "
            "the robot's total mass, 2*sqrt(stiffness*mass)."
        ),
    )
    parser.add_argument(
        "--joint_velocity_limit",
        type=float,
        default=None,
        help=(
            "Override every actuator's solver-side joint velocity limit [rad/s]. PhysX brakes at this "
            "limit and the MuJoCo harness has no equivalent constraint; raise it to test what the "
            "constraint is worth."
        ),
    )
    parser.add_argument(
        "--refresh_frame0_state",
        action="store_true",
        help=(
            "Recompute cached derived state after the frame-0 write so the first observation "
            "reports the written orientation instead of the pre-write one."
        ),
    )
    add_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    return setup_preset_cli(_create_parser())


def _goal_values(manifest: DeploymentManifest, args: argparse.Namespace) -> dict[str, float]:
    requested = {
        "pos_x": args.goal_pos_x,
        "pos_y": args.goal_pos_y,
        "roll": None,
        "pitch": None,
        "yaw": args.goal_yaw,
    }
    return {name: manifest.goal_value(name, value) for name, value in requested.items()}


def _range_pair(value: Any, name: str) -> tuple[float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 2 or not np.all(np.isfinite(values)) or values[0] > values[1]:
        raise ValueError(f"Resolved task goal range {name!r} is invalid: {value}.")
    return values


def _configure_rollout(
    env_cfg: Any, manifest: DeploymentManifest, args: argparse.Namespace, goal_values: dict[str, float]
) -> dict[str, list[float]]:
    """Make one deterministic environment while preserving the selected task's action pipeline."""
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    if args.device is not None:
        env_cfg.sim.device = args.device
    if not np.isclose(env_cfg.sim.dt, manifest.sim_dt) or env_cfg.decimation != manifest.decimation:
        raise ValueError(
            "Resolved task control rates disagree with the manifest: "
            f"task={env_cfg.sim.dt}/{env_cfg.decimation}, manifest={manifest.sim_dt}/{manifest.decimation}."
        )

    # Keep the initial state and action offset aligned with the comparison manifest. The
    # task's motion loader still uses its configured CSV for reference-preview observations.
    env_cfg.scene.robot.init_state.joint_pos = dict(zip(manifest.joint_names, manifest.default_pos))

    for event_name in (
        "physics_material",
        "robot_mass",
        "pelvis_com",
        "actuator_gains",
        "push_robot",
        "base_external_force_torque",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    env_cfg.events.reset_to_reference.params["init_start_prob"] = 0.0

    if args.force_manifest_dynamics:
        _pin_to_manifest(env_cfg, manifest)

    delay_steps = int(args.delay_steps)
    if not manifest.delay_min <= delay_steps <= manifest.delay_max:
        raise ValueError(
            f"delay_steps={delay_steps} is outside manifest range [{manifest.delay_min}, {manifest.delay_max}]."
        )
    if args.joint_velocity_limit is not None:
        if not np.isfinite(args.joint_velocity_limit) or args.joint_velocity_limit <= 0.0:
            raise ValueError(f"joint_velocity_limit must be finite and positive, got {args.joint_velocity_limit}.")
        for actuator_cfg in env_cfg.scene.robot.actuators.values():
            actuator_cfg.velocity_limit_sim = args.joint_velocity_limit

    action_cfg = env_cfg.actions.joint_pos
    if hasattr(action_cfg, "min_delay_steps"):
        action_cfg.min_delay_steps = delay_steps
        action_cfg.max_delay_steps = delay_steps
    elif delay_steps != 0:
        raise ValueError(
            f"The resolved task's action term cannot apply the requested delay of {delay_steps} policy step(s)."
        )

    env_cfg.observations.policy.enable_corruption = False
    for term_name in ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "goal_remaining"):
        term = getattr(env_cfg.observations.policy, term_name)
        term.noise = None
    goal_remaining_params = env_cfg.observations.policy.goal_remaining.params
    if "freeze_prob" in goal_remaining_params or "drift_std" in goal_remaining_params:
        # Preserve the deployment flight latch but remove its stochastic inputs.
        env_cfg.observations.policy.goal_remaining.params = {"freeze_prob": 1.0, "drift_std": 0.0}

    ranges = env_cfg.commands.jump_goal.ranges
    resolved_ranges: dict[str, list[float]] = {}
    for name, goal_value in goal_values.items():
        lower, upper = _range_pair(getattr(ranges, name), name)
        resolved_ranges[name] = [lower, upper]
        if not lower <= goal_value <= upper:
            raise ValueError(f"Goal {name}={goal_value} is outside the resolved task range [{lower}, {upper}].")
        setattr(ranges, name, (goal_value, goal_value))
    return resolved_ranges


def _apply_contact_compliance(robot: Any, stiffness: float, damping: float | None) -> tuple[np.ndarray, float]:
    """Write a compliant contact spring onto every collision shape of the robot.

    PhysX stores the compliant spring in the material, and the tensor setter takes friction and
    the spring together, so the current friction is read back and rewritten unchanged. A stiffness
    of zero disables the compliant model and restores rigid contacts.

    Args:
        robot: The articulation whose collision shapes are updated.
        stiffness: Compliant contact stiffness per shape [N/m].
        damping: Compliant contact damping per shape [N.s/m], or None for critical damping.

    Returns:
        The stiffness/damping actually reported back by PhysX, and the damping that was written.
    """
    import warp as wp

    view = robot.root_view
    materials = wp.to_torch(view.get_material_properties())
    compliant, _ = view.get_compliant_material_properties()
    compliant_torch = wp.to_torch(compliant)
    count, shapes = compliant_torch.shape[0], compliant_torch.shape[1]

    mass = float(_numpy(robot.data.default_mass).sum(axis=1)[0])
    if damping is None:
        damping = 2.0 * math.sqrt(stiffness * mass)

    data = torch.zeros((count, shapes, 4), dtype=torch.float32, device=compliant_torch.device)
    data[..., 0:2] = materials[..., 0:2]
    data[..., 2] = stiffness
    data[..., 3] = damping
    # Minimum keeps the softer of the two contacting materials in charge, so the robot's
    # spring governs against the rigid ground plane. Friction keeps PhysX's default average.
    combine = torch.ones((count, shapes, 3), dtype=torch.uint8, device=compliant_torch.device)
    combine[..., 0] = 0
    indices = torch.arange(count, dtype=torch.int32, device=compliant_torch.device)
    view.set_compliant_material_properties(
        wp.from_torch(data.contiguous(), dtype=wp.float32),
        wp.from_torch(combine.contiguous(), dtype=wp.uint8),
        wp.from_torch(indices, dtype=wp.int32),
    )
    readback = _numpy(view.get_compliant_material_properties()[0])
    friction_error = float(
        np.max(np.abs(_numpy(view.get_material_properties())[..., 0:2] - _numpy(materials)[..., 0:2]))
    )
    if friction_error > 0.0:
        raise RuntimeError(f"Writing the contact spring changed friction by {friction_error}.")
    return readback, damping


def _pin_to_manifest(env_cfg: Any, manifest: DeploymentManifest) -> None:
    """Rewrite the task's action pipeline and actuator gains to the manifest's values.

    An exported bundle is defined by its manifest, so replaying one after the training task has
    moved on means driving the manifest's dynamics rather than the task's current tuning.

    Args:
        env_cfg: Environment configuration, mutated in place before construction.
        manifest: Deployment manifest that defines the exported bundle.
    """
    from isaaclab.utils.string import resolve_matching_names

    joint_names = list(manifest.joint_names)
    manifest_action = manifest.raw["action"]
    action_cfg = env_cfg.actions.joint_pos
    action_cfg.scale = dict(zip(joint_names, [float(value) for value in manifest_action["scale"]]))
    action_cfg.clip = (
        None
        if manifest_action.get("clip") is None
        else dict(zip(joint_names, [tuple(pair) for pair in manifest_action["clip"]]))
    )
    if hasattr(action_cfg, "alpha"):
        action_cfg.alpha = dict(zip(joint_names, [float(value) for value in manifest_action["filter_alpha"]]))
    torque_projection = manifest_action.get("torque_projection")
    if hasattr(action_cfg, "effort_limit_ratio"):
        action_cfg.effort_limit_ratio = (
            None
            if torque_projection is None
            else dict(
                zip(
                    joint_names,
                    [float(value) for value in torque_projection["effort_limit_ratio"]],
                )
            )
        )
    lower_limit_brake = manifest_action.get("lower_limit_brake")
    if hasattr(action_cfg, "lower_limit_velocity_lookahead"):
        action_cfg.lower_limit_velocity_lookahead = (
            None
            if lower_limit_brake is None
            else {
                name: float(value)
                for name, value in zip(joint_names, lower_limit_brake["velocity_lookahead_s"])
                if float(value) > 0.0
            }
        )

    for actuator_cfg in env_cfg.scene.robot.actuators.values():
        joint_ids, matched_names = resolve_matching_names(actuator_cfg.joint_names_expr, joint_names)
        actuator_cfg.stiffness = {name: float(manifest.stiffness[i]) for i, name in zip(joint_ids, matched_names)}
        actuator_cfg.damping = {name: float(manifest.damping[i]) for i, name in zip(joint_ids, matched_names)}
        actuator_cfg.armature = {name: float(manifest.armature[i]) for i, name in zip(joint_ids, matched_names)}
        actuator_cfg.effort_limit_sim = {
            name: float(manifest.effort_limit[i]) for i, name in zip(joint_ids, matched_names)
        }
    print("Pinned the action pipeline and actuator gains to the deployment manifest.")


def _runtime_action_alpha(action_term: Any, joint_count: int) -> np.ndarray:
    if not hasattr(action_term, "_alpha"):
        return np.ones(joint_count, dtype=np.float64)
    return np.asarray(_numpy(action_term._alpha), dtype=np.float64).reshape(-1)


def _validate_runtime(env: Any, manifest: DeploymentManifest) -> tuple[Any, Any, bool]:
    """Validate joint/dynamics contracts and report whether action filtering matches the manifest."""
    _require_reference_source(env, manifest)
    manifest_action = manifest.raw["action"]
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    runtime_names = tuple(robot.joint_names)
    action_names = tuple(action_term._joint_names)
    if runtime_names != manifest.joint_names or action_names != manifest.joint_names:
        raise ValueError(
            "Isaac runtime/action joint order must equal manifest policy order: "
            f"runtime={runtime_names}, action={action_names}, manifest={manifest.joint_names}."
        )
    _require_close("default joint position", robot.data.default_joint_pos, manifest.default_pos)
    _require_close("action scale", action_term._scale, np.asarray(manifest_action["scale"], dtype=np.float64))
    _require_close("action offset", action_term._offset, np.asarray(manifest_action["offset"], dtype=np.float64))
    # Only require a runtime clip when the manifest declares one. A null clip is legitimate:
    # this policy deliberately commands position targets past the joint stops.
    manifest_clip = manifest.raw["action"].get("clip")
    runtime_has_clip = action_term.cfg.clip is not None and hasattr(action_term, "_clip")
    if manifest_clip is not None and not runtime_has_clip:
        raise ValueError("Isaac runtime is missing the action clip declared by the deployment manifest.")
    if manifest_clip is None and runtime_has_clip:
        raise ValueError("Isaac runtime applies an action clip that the deployment manifest does not declare.")
    if manifest_clip is not None:
        _require_close("action clip", action_term._clip, np.asarray(manifest_clip, dtype=np.float64).reshape(-1))
    runtime_effort_ratio = getattr(action_term, "_effort_limit_ratio", None)
    if (manifest.effort_limit_ratio is None) != (runtime_effort_ratio is None):
        raise ValueError("Isaac runtime torque projection presence disagrees with the deployment manifest.")
    if manifest.effort_limit_ratio is not None:
        _require_close("torque projection ratio", runtime_effort_ratio, manifest.effort_limit_ratio)
    runtime_brake_lookahead = getattr(action_term, "_lower_limit_velocity_lookahead", None)
    if (manifest.brake_velocity_lookahead is None) != (runtime_brake_lookahead is None):
        raise ValueError("Isaac runtime lower-limit braking presence disagrees with the deployment manifest.")
    if manifest.brake_velocity_lookahead is not None:
        _require_close(
            "lower-limit brake lookahead",
            runtime_brake_lookahead,
            manifest.brake_velocity_lookahead,
        )
    _require_close("joint stiffness", robot.data.joint_stiffness, manifest.stiffness)
    _require_close("joint damping", robot.data.joint_damping, manifest.damping)
    _require_close("joint armature", robot.data.joint_armature, manifest.armature)
    _require_close("joint effort limit", robot.data.joint_effort_limits, manifest.effort_limit)
    observation_names = tuple(env.observation_manager.active_terms["policy"])
    manifest_names = tuple(manifest.terms)
    if observation_names != manifest_names:
        raise ValueError(
            "Isaac policy observation order disagrees with the manifest: "
            f"runtime={observation_names}, manifest={manifest_names}."
        )
    observation_sizes = tuple(
        int(np.prod(term_shape)) for term_shape in env.observation_manager.group_obs_term_dim["policy"]
    )
    manifest_sizes = tuple(int(term["total"]) for term in manifest.terms.values())
    if observation_sizes != manifest_sizes:
        raise ValueError(
            "Isaac policy observation term sizes disagree with the manifest: "
            f"runtime={observation_sizes}, manifest={manifest_sizes}."
        )
    if env._physics_handles_decimation:
        raise RuntimeError("Rollout logging requires a physics backend that exposes every simulation substep.")
    if not all(actuator.is_implicit_model for actuator in robot.actuators.values()):
        raise RuntimeError("This logger's torque interpretation requires every robot actuator to be implicit.")

    runtime_alpha = _runtime_action_alpha(action_term, manifest.joint_count)
    filter_matches = bool(
        np.allclose(
            runtime_alpha,
            np.asarray(manifest_action["filter_alpha"], dtype=np.float64),
            rtol=0.0,
            atol=1.0e-6,
        )
    )
    if not filter_matches:
        print(
            "WARNING: The selected task's action filter differs from the deployment manifest; "
            "the log records the task's actual delayed_action and q_target."
        )
    return robot, action_term, filter_matches


class _OnnxActor:
    """Adapt the deployment bundle's ONNX actor to the rollout's policy call signature."""

    def __init__(self, policy_path: Path, manifest: DeploymentManifest, device: str):
        self._policy = OnnxPolicy(policy_path, manifest.observation_dim, manifest.joint_count)
        self._device = device
        self.backend = self._policy.backend

    def __call__(self, observation: TensorDict) -> torch.Tensor:
        values = np.asarray(_numpy(observation["policy"])[0], dtype=np.float32)
        action = self._policy(values)
        return torch.as_tensor(action, dtype=torch.float32, device=self._device).unsqueeze(0)


def _policy_observation(observation_dict: dict[str, torch.Tensor], num_envs: int) -> TensorDict:
    return TensorDict(observation_dict, batch_size=[num_envs])


def _infer_action(policy: Any, observation: TensorDict, joint_count: int) -> torch.Tensor:
    action = policy(observation)
    if not isinstance(action, torch.Tensor) or action.shape != (1, joint_count):
        raise ValueError(f"RSL-RL policy returned {type(action).__name__} with shape {getattr(action, 'shape', None)}.")
    if not bool(torch.all(torch.isfinite(action))):
        raise ValueError("RSL-RL policy returned a non-finite action.")
    return action


def _append_state(
    logger: StepLogger,
    env: Any,
    robot: Any,
    *,
    sim_time: float,
    phase: int,
    action: np.ndarray,
    delayed_action: np.ndarray,
    q_target: np.ndarray,
    applied_tau: np.ndarray,
    observation: np.ndarray,
) -> None:
    root_quaternion_wxyz = np.roll(_numpy(robot.data.root_link_quat_w)[0], 1)
    logger.append(
        sim_time=sim_time,
        phase=phase,
        qpos=np.asarray(_numpy(robot.data.joint_pos)[0], dtype=np.float64),
        qvel=np.asarray(_numpy(robot.data.joint_vel)[0], dtype=np.float64),
        action=np.asarray(action, dtype=np.float64),
        delayed_action=np.asarray(delayed_action, dtype=np.float64),
        q_target=np.asarray(q_target, dtype=np.float64),
        applied_tau=np.asarray(applied_tau, dtype=np.float64),
        pelvis_pose=np.asarray(
            np.concatenate((_numpy(robot.data.root_link_pos_w)[0], root_quaternion_wxyz)), dtype=np.float64
        ),
        pelvis_velocity=np.asarray(
            np.concatenate((_numpy(robot.data.root_link_lin_vel_w)[0], _numpy(robot.data.root_link_ang_vel_b)[0])),
            dtype=np.float64,
        ),
        foot_contact_forces=np.asarray(_foot_forces(env), dtype=np.float64),
        observation=np.asarray(observation, dtype=np.float32),
    )


def _tilt_deg(quaternion_wxyz: np.ndarray) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    _, x, y, _ = quaternion
    body_up_dot_world_up = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(float(np.clip(body_up_dot_world_up, -1.0, 1.0))))


def _airborne_window(times: np.ndarray, contact_forces: np.ndarray) -> str:
    contact = np.any(np.linalg.norm(contact_forces, axis=2) > _CONTACT_FORCE_THRESHOLD_N, axis=1)
    supported = np.flatnonzero(contact)
    if not len(supported):
        return "unavailable (no supported-contact sample)"

    start_search = int(supported[0]) + 1
    windows: list[tuple[int, int]] = []
    index = start_search
    while index < len(contact):
        if contact[index]:
            index += 1
            continue
        start = index
        while index < len(contact) and not contact[index]:
            index += 1
        windows.append((start, index))
    if not windows:
        return "none"
    start, stop = max(windows, key=lambda window: window[1] - window[0])
    if stop < len(times):
        end_time = times[stop]
        end_label = f"{end_time:.3f} s"
    else:
        end_time = times[-1]
        end_label = f">={end_time:.3f} s (end of log)"
    return f"{times[start]:.3f} s to {end_label}, duration {end_time - times[start]:.3f} s"


def _print_summary(logger: StepLogger, manifest: DeploymentManifest) -> None:
    arrays = {name: np.asarray(values) for name, values in logger.values.items()}
    # Sample zero has no preceding integration interval and is deliberately logged as zero torque.
    applied_tau = np.abs(np.asarray(arrays["applied_tau"][1:], dtype=np.float64))
    tolerance = np.maximum(1.0e-6, manifest.effort_limit * 1.0e-5)
    saturated = applied_tau >= manifest.effort_limit[None, :] - tolerance[None, :]
    peaks = np.max(applied_tau, axis=0)

    print("\nIsaac implicit-actuator torque estimate (post-limit)")
    print(f"{'joint':30s} {'saturated':>11s} {'peak (N.m)':>13s} {'peak/limit':>12s}")
    for name, fraction, peak, limit in zip(
        manifest.joint_names, np.mean(saturated, axis=0), peaks, manifest.effort_limit
    ):
        print(f"{name:30s} {100.0 * fraction:10.1f}% {peak:13.2f} {100.0 * peak / limit:11.1f}%")

    pelvis_z = np.asarray(arrays["pelvis_pose"][:, 2], dtype=np.float64)
    print(
        "Pelvis z [m]: "
        f"start={pelvis_z[0]:.3f}, min={np.min(pelvis_z):.3f}, "
        f"max={np.max(pelvis_z):.3f}, end={pelvis_z[-1]:.3f}"
    )
    print(
        f"Airborne window (both feet below {_CONTACT_FORCE_THRESHOLD_N:g} N): "
        f"{_airborne_window(arrays['time'], arrays['foot_contact_forces'])}"
    )
    print(f"Final tilt: {_tilt_deg(arrays['pelvis_pose'][-1, 3:]):.1f} deg")


def _verify_saved_log(path: Path, manifest: DeploymentManifest) -> None:
    sample_count = manifest.episode_steps * manifest.decimation + 1
    expected_shapes = {
        "time": (sample_count,),
        "phase": (sample_count,),
        "qpos": (sample_count, manifest.joint_count),
        "qvel": (sample_count, manifest.joint_count),
        "action": (sample_count, manifest.joint_count),
        "delayed_action": (sample_count, manifest.joint_count),
        "q_target": (sample_count, manifest.joint_count),
        "applied_tau": (sample_count, manifest.joint_count),
        "pelvis_pose": (sample_count, 7),
        "pelvis_velocity": (sample_count, 6),
        "foot_contact_forces": (sample_count, 2, 3),
        "observation": (sample_count, manifest.observation_dim),
        "metadata_json": (),
        "phase_names": (len(manifest.phase_names),),
    }
    expected_dtypes = {
        "time": np.dtype(np.float64),
        "phase": np.dtype(np.int64),
        "qpos": np.dtype(np.float64),
        "qvel": np.dtype(np.float64),
        "action": np.dtype(np.float64),
        "delayed_action": np.dtype(np.float64),
        "q_target": np.dtype(np.float64),
        "applied_tau": np.dtype(np.float64),
        "pelvis_pose": np.dtype(np.float64),
        "pelvis_velocity": np.dtype(np.float64),
        "foot_contact_forces": np.dtype(np.float64),
        "observation": np.dtype(np.float32),
    }
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != tuple(expected_shapes):
            raise RuntimeError(
                f"Saved log field order disagrees with deploy_mujoco.py: {archive.files} != {list(expected_shapes)}."
            )
        for name, shape in expected_shapes.items():
            if archive[name].shape != shape:
                raise RuntimeError(f"Saved log {name} has shape {archive[name].shape}; expected {shape}.")
        for name, dtype in expected_dtypes.items():
            if archive[name].dtype != dtype:
                raise RuntimeError(f"Saved log {name} has dtype {archive[name].dtype}; expected {dtype}.")


def _record_rollout(
    env: Any,
    policy: Any,
    manifest: DeploymentManifest,
    args: argparse.Namespace,
    checkpoint_path: Path,
    goal_values: dict[str, float],
    resolved_goal_ranges: dict[str, list[float]],
    reference_joint_pos: np.ndarray,
    reference_root_position: np.ndarray,
    reference_root_quaternion: np.ndarray,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    robot, action_term, filter_matches = _validate_runtime(env, manifest)
    contact_compliance: dict[str, Any] | None = None
    if args.contact_stiffness is not None:
        readback, applied_damping = _apply_contact_compliance(robot, args.contact_stiffness, args.contact_damping)
        contact_compliance = {
            "stiffness_requested": float(args.contact_stiffness),
            "damping_applied": float(applied_damping),
            "stiffness_readback": sorted({float(value) for value in readback[..., 0].reshape(-1)}),
            "damping_readback": sorted({float(value) for value in readback[..., 1].reshape(-1)}),
            "combine_mode": "minimum",
        }
        print(f"Contact compliance: {contact_compliance}")
    # The phase table lives beside the manifest; DeploymentManifest stopped exposing it
    # when the action pipeline moved into the shared runtime.
    jump_phase_table = np.load(manifest.path.parent / manifest.raw["tables"]["jump_phase"])
    observation_dict = _write_and_verify_frame0(
        env,
        robot,
        reference_joint_pos,
        reference_root_position,
        reference_root_quaternion,
        refresh_derived_state=args.refresh_frame0_state,
    )

    goal_command = env.command_manager.get_term("jump_goal")
    goal_world = _numpy(goal_command.pose_command_w)[0, :3]
    runtime_alpha = _runtime_action_alpha(action_term, manifest.joint_count)
    motion_loader = getattr(env, "motion_loader", None)
    task_reference_csv = getattr(motion_loader, "csv_path", None)
    metadata = {
        "schema_version": manifest.raw["schema_version"],
        "task": manifest.raw.get("task"),
        "runtime_task": args.task,
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "simulator": "isaac",
        "model": None,
        "overlay": None,
        "reference_csv": str(manifest.reference_source),
        "reference_sha256": manifest.reference_source_sha256,
        "task_reference_csv": task_reference_csv,
        "reference_frame": 0,
        "policy": str(checkpoint_path),
        "policy_backend": "onnxruntime" if args.policy_onnx is not None else "rsl_rl_eager",
        "self_check": False,
        "cross_check": False,
        "closed_loop": True,
        "action_sequence": None,
        "sim_dt": manifest.sim_dt,
        "policy_dt": manifest.policy_dt,
        "decimation": manifest.decimation,
        "delay_steps": int(args.delay_steps),
        "frame0_derived_state_refreshed": bool(args.refresh_frame0_state),
        "joint_velocity_limits": np.asarray(_numpy(robot.data.joint_velocity_limits), dtype=np.float64)
        .reshape(-1)
        .tolist(),
        "contact_compliance": contact_compliance,
        "dynamics_pinned_to_manifest": bool(args.force_manifest_dynamics),
        "seed": 0,
        "goal": goal_values,
        "goal_world": goal_world.tolist(),
        "resolved_task_goal_ranges": resolved_goal_ranges,
        "phase_names": manifest.phase_names,
        "qpos_qvel_order": manifest.joint_names,
        "runtime_filter_alpha": runtime_alpha.tolist(),
        "action_filter_matches_manifest": filter_matches,
        "applied_tau_source": "robot.data.applied_torque",
        "applied_tau_semantics": (
            "ImplicitActuator PD estimate clipped to effort_limit_sim before the logged physics interval; "
            "not a measured or PhysX-solver-realized joint effort"
        ),
        "computed_torque_semantics": (
            "robot.data.computed_torque is the corresponding unclipped PD estimate and is not logged as an array"
        ),
        "pelvis_pose_convention": "position_world_xyz[m], quaternion_world_from_body_wxyz",
        "pelvis_velocity_convention": "linear_world_xyz[m/s], angular_body_xyz[rad/s]",
        "foot_contact_force_convention": "left_then_right, world_xyz[N]",
        "sample_convention": (
            "sample 0 is pre-physics frame-0 state; later samples are post-physics states; "
            "applied_tau is the actuator estimate captured before and associated with that physics step; "
            "observation is held from policy tick"
        ),
    }
    logger = StepLogger(args.log, metadata)
    observation = np.asarray(_numpy(observation_dict["policy"])[0], dtype=np.float32)
    policy_input = _policy_observation(observation_dict, env.num_envs)

    with torch.inference_mode():
        for policy_step in range(manifest.episode_steps):
            raw_action = _infer_action(policy, policy_input, manifest.joint_count)
            raw_action_np = np.asarray(_numpy(raw_action)[0], dtype=np.float64)
            env.action_manager.process_action(raw_action)
            delayed_action = np.asarray(_numpy(action_term.raw_actions)[0], dtype=np.float64)
            q_target = np.asarray(_numpy(action_term.processed_actions)[0], dtype=np.float64)
            phase = int(np.argmax(jump_phase_table[policy_step]))

            if policy_step == 0:
                _append_state(
                    logger,
                    env,
                    robot,
                    sim_time=0.0,
                    phase=phase,
                    action=raw_action_np,
                    delayed_action=delayed_action,
                    q_target=q_target,
                    applied_tau=np.zeros(manifest.joint_count, dtype=np.float64),
                    observation=observation,
                )

            for inner_step in range(manifest.decimation):
                env._sim_step_counter += 1
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                # For implicit actuators this is a clipped PD estimate produced by Isaac Lab,
                # not torque feedback from PhysX. Capture it before stepping so its timing is exact.
                applied_tau = np.asarray(_numpy(robot.data.applied_torque)[0], dtype=np.float64)
                env.sim.step(render=False)
                env.scene.update(dt=manifest.sim_dt)

                _append_state(
                    logger,
                    env,
                    robot,
                    sim_time=(policy_step * manifest.decimation + inner_step + 1) * manifest.sim_dt,
                    phase=phase,
                    action=raw_action_np,
                    delayed_action=delayed_action,
                    q_target=q_target,
                    applied_tau=applied_tau,
                    observation=observation,
                )

            if policy_step + 1 < manifest.episode_steps:
                env.episode_length_buf += 1
                env.common_step_counter += 1
                env.command_manager.compute(dt=manifest.policy_dt)
                observation_dict = env.observation_manager.compute(update_history=True)
                observation = np.asarray(_numpy(observation_dict["policy"])[0], dtype=np.float32)
                policy_input = _policy_observation(observation_dict, env.num_envs)

    logger.save()
    _verify_saved_log(logger.output_path, manifest)
    print(f"Wrote {manifest.episode_steps * manifest.decimation + 1} Isaac policy samples to {logger.output_path}")
    print(f"Selected action delay: {int(args.delay_steps)} policy step(s)")
    print(
        "applied_tau: robot.data.applied_torque, the clipped implicit-actuator PD estimate; "
        "PhysX solver-realized torque is unavailable."
    )
    _print_summary(logger, manifest)


def main() -> None:
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0]] + hydra_args
    manifest = DeploymentManifest(args.manifest)
    goal_values = _goal_values(manifest, args)
    reference_joint_pos, reference_root_position, reference_root_quaternion = _load_reference_frame0(manifest)

    if (args.checkpoint is None) == (args.policy_onnx is None):
        raise ValueError("Pass exactly one of --checkpoint or --policy_onnx.")

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    resolved_goal_ranges = _configure_rollout(env_cfg, manifest, args, goal_values)
    installed_rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    if agent_cfg.clip_actions is not None:
        raise ValueError("The RSL-RL runner config adds raw-action clipping not represented by deployment schema 1.2.")
    checkpoint_path = (
        args.policy_onnx.resolve()
        if args.policy_onnx is not None
        else Path(retrieve_file_path(args.checkpoint)).resolve()
    )
    env_cfg.log_dir = str(checkpoint_path.parent)

    with launch_simulation(env_cfg, args):
        gym_env = gym.make(args.task, cfg=env_cfg)
        rl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        try:
            print(f"[INFO]: Loading policy from: {checkpoint_path}")
            if args.policy_onnx is not None:
                policy = _OnnxActor(checkpoint_path, manifest, rl_env.unwrapped.device)
                print(f"ONNX backend: {policy.backend}")
            elif agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                runner.load(str(checkpoint_path))
                policy = runner.get_inference_policy(device=rl_env.unwrapped.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                runner.load(str(checkpoint_path))
                policy = runner.get_inference_policy(device=rl_env.unwrapped.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            _record_rollout(
                rl_env.unwrapped,
                policy,
                manifest,
                args,
                checkpoint_path,
                goal_values,
                resolved_goal_ranges,
                reference_joint_pos,
                reference_root_position,
                reference_root_quaternion,
            )
            if version.parse(installed_rsl_rl_version) < version.parse("4.0.0"):
                print("WARNING: Recurrent-policy reset behavior was not exercised because the rollout never resets.")
        finally:
            rl_env.close()


if __name__ == "__main__":
    main()

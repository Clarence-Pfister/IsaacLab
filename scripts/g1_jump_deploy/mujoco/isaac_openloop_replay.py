# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a deterministic Isaac open-loop trajectory for the MuJoCo cross-check."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from deploy_mujoco import (
    DeploymentManifest,
    StepLogger,
    _load_action_sequence,
    _load_reference_frame0,
    _validate_reference_override,
)

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--reference_csv",
        type=Path,
        default=None,
        help="Deprecated compatibility option; if supplied, it must match reference.source_csv in the manifest.",
    )
    parser.add_argument("--action_sequence", "--action-sequence", dest="action_sequence", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=Path("isaac_openloop_log.npz"))
    parser.add_argument("--goal_pos_x", type=float, default=None)
    parser.add_argument("--goal_pos_y", type=float, default=None)
    parser.add_argument("--goal_roll", type=float, default=None, help="Goal roll [rad].")
    parser.add_argument("--goal_pitch", type=float, default=None, help="Goal pitch [rad].")
    parser.add_argument("--goal_yaw", type=float, default=None, help="Goal yaw [rad].")
    add_launcher_args(parser)
    parser.set_defaults(headless=True)
    return setup_preset_cli(parser)


def _as_torch(value: Any) -> torch.Tensor:
    """Return an Isaac tensor/proxy as a torch tensor without changing its device."""
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "torch", None)
    if tensor is not None:
        return tensor
    import warp

    return warp.to_torch(value)


def _numpy(value: Any) -> np.ndarray:
    """Copy an Isaac tensor/proxy to a NumPy array."""
    return _as_torch(value).detach().cpu().numpy().copy()


def _goal_values(manifest: DeploymentManifest, args: argparse.Namespace) -> dict[str, float]:
    return {
        name: manifest.goal_value(name, getattr(args, f"goal_{name}"))
        for name in ("pos_x", "pos_y", "roll", "pitch", "yaw")
    }


def _configure_deterministic_replay(
    env_cfg: Any, manifest: DeploymentManifest, args: argparse.Namespace, goal_values: dict[str, float]
) -> None:
    """Remove every stochastic Stage 3 input before environment construction."""
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    if args.device is not None:
        env_cfg.sim.device = args.device
    if not np.isclose(env_cfg.sim.dt, manifest.sim_dt) or env_cfg.decimation != manifest.decimation:
        raise ValueError(
            "Resolved task control rates disagree with the manifest: "
            f"task={env_cfg.sim.dt}/{env_cfg.decimation}, "
            f"manifest={manifest.sim_dt}/{manifest.decimation}."
        )

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

    env_cfg.actions.joint_pos.min_delay_steps = 0
    env_cfg.actions.joint_pos.max_delay_steps = 0
    env_cfg.observations.policy.enable_corruption = False
    for term_name in ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "goal_remaining"):
        getattr(env_cfg.observations.policy, term_name).noise = None
    goal_remaining = env_cfg.observations.policy.goal_remaining
    if "freeze_prob" in goal_remaining.params or "drift_std" in goal_remaining.params:
        # Preserve a legacy deployment flight latch while removing its stochastic inputs.
        # A fully latched policy has no parameters and already is deterministic.
        goal_remaining.params = {"freeze_prob": 1.0, "drift_std": 0.0}

    ranges = env_cfg.commands.jump_goal.ranges
    for name, value in goal_values.items():
        setattr(ranges, name, (value, value))


def _require_close(name: str, actual: Any, expected: np.ndarray) -> None:
    actual_array = np.asarray(_numpy(actual), dtype=np.float64).reshape(-1)
    if actual_array.shape != expected.shape or not np.allclose(actual_array, expected, rtol=0.0, atol=1e-6):
        maximum_error = float(np.max(np.abs(actual_array - expected))) if actual_array.shape == expected.shape else None
        raise ValueError(
            f"Isaac runtime {name} disagrees with the deployment manifest: "
            f"shape={actual_array.shape}/{expected.shape}, max_error={maximum_error}."
        )


def _require_reference_source(env: Any, manifest: DeploymentManifest) -> None:
    """Require the task motion loader to use the source declared by the manifest."""
    loader = getattr(env, "motion_loader", None)
    runtime_source = getattr(loader, "csv_path", None)
    if not isinstance(runtime_source, str):
        raise ValueError("Isaac runtime did not resolve a reference motion CSV.")
    if Path(runtime_source).resolve() != manifest.reference_source.resolve():
        raise ValueError(
            "Isaac runtime reference motion disagrees with manifest reference.source_csv: "
            f"runtime={runtime_source}, manifest={manifest.reference_source}."
        )


def _validate_runtime(env: Any, manifest: DeploymentManifest) -> tuple[Any, Any]:
    _require_reference_source(env, manifest)
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
    _require_close("action scale", action_term._scale, manifest.action_scale)
    _require_close("action offset", action_term._offset, manifest.action_offset)
    _require_close("filter alpha", action_term._alpha, manifest.filter_alpha)
    # Only require a runtime clip when the manifest actually declares one. A null clip is
    # legitimate for this task: the policy deliberately commands targets past the joint stops.
    manifest_clip = manifest.raw["action"].get("clip")
    runtime_has_clip = action_term.cfg.clip is not None and hasattr(action_term, "_clip")
    if manifest_clip is not None and not runtime_has_clip:
        raise ValueError("Isaac runtime is missing the action clip declared by the deployment manifest.")
    if manifest_clip is None and runtime_has_clip:
        raise ValueError("Isaac runtime applies an action clip that the deployment manifest does not declare.")
    if manifest_clip is not None:
        _require_close("action clip", action_term._clip, np.asarray(manifest_clip, dtype=np.float64).reshape(-1))
    _require_close("joint stiffness", robot.data.joint_stiffness, manifest.stiffness)
    _require_close("joint damping", robot.data.joint_damping, manifest.damping)
    _require_close("joint armature", robot.data.joint_armature, manifest.armature)
    _require_close("joint effort limit", robot.data.joint_effort_limits, manifest.effort_limit)
    if env._physics_handles_decimation:
        raise RuntimeError("Cross-check logging requires a physics backend that exposes every 2 ms substep.")
    return robot, action_term


def _write_and_verify_frame0(
    env: Any,
    robot: Any,
    joint_pos: np.ndarray,
    root_position: np.ndarray,
    root_quaternion_wxyz: np.ndarray,
    refresh_derived_state: bool = False,
) -> dict[str, torch.Tensor]:
    """Write the exact frame-0 state, read it back, and rebuild manager histories.

    Args:
        env: The manager-based environment being driven.
        robot: The articulation whose frame-0 state is written.
        joint_pos: Frame-0 joint positions in policy order [rad].
        root_position: Frame-0 root position in world coordinates [m].
        root_quaternion_wxyz: Frame-0 root orientation as a world-from-body quaternion.
        refresh_derived_state: Advance the articulation data timestamp so cached derived
            quantities such as ``projected_gravity_b`` are recomputed from the written pose.
            ``sim.forward()`` does not invalidate them, so the first observation otherwise
            reports the pre-write orientation, exactly as the training reset path does.
    """
    env_ids = torch.zeros(1, dtype=torch.int32, device=env.device)
    joint_position_tensor = torch.as_tensor(joint_pos, dtype=torch.float32, device=env.device).unsqueeze(0)
    joint_velocity_tensor = torch.zeros_like(joint_position_tensor)
    root_quaternion_xyzw = np.roll(root_quaternion_wxyz, -1)
    root_pose = torch.as_tensor(
        np.concatenate((root_position, root_quaternion_xyzw)), dtype=torch.float32, device=env.device
    ).unsqueeze(0)
    root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=env.device)

    robot.write_joint_position_to_sim_index(position=joint_position_tensor, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_velocity_tensor, env_ids=env_ids)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    _require_close("written frame-0 joint position", robot.data.joint_pos, joint_pos)
    _require_close("written frame-0 joint velocity", robot.data.joint_vel, np.zeros_like(joint_pos))
    _require_close("written frame-0 root position", robot.data.root_link_pos_w, root_position)
    _require_close("written frame-0 root quaternion XYZW", robot.data.root_link_quat_w, root_quaternion_xyzw)
    _require_close("written frame-0 root linear velocity", robot.data.root_link_lin_vel_w, np.zeros(3))
    _require_close("written frame-0 root angular velocity", robot.data.root_link_ang_vel_w, np.zeros(3))

    if refresh_derived_state:
        # Timestamped caches only recompute once the articulation clock moves past them.
        robot.update(dt=env.physics_dt)

    env.episode_length_buf.zero_()
    env.common_step_counter = 0
    if not hasattr(env, "start_times"):
        env.start_times = torch.zeros(env.num_envs, device=env.device)
    else:
        env.start_times.zero_()
    env.action_manager.reset(env_ids)
    env.command_manager.reset(env_ids)
    env.observation_manager.reset(env_ids)
    observation_dict = env.observation_manager.compute(update_history=True)

    # Manager resets are not allowed to undo the explicit state write.
    _require_close("post-manager frame-0 joint position", robot.data.joint_pos, joint_pos)
    _require_close("post-manager frame-0 joint velocity", robot.data.joint_vel, np.zeros_like(joint_pos))
    _require_close("post-manager frame-0 root position", robot.data.root_link_pos_w, root_position)
    _require_close("post-manager frame-0 root quaternion XYZW", robot.data.root_link_quat_w, root_quaternion_xyzw)
    return observation_dict


def _foot_forces(env: Any) -> np.ndarray:
    result = np.zeros((2, 3), dtype=np.float64)
    for foot_index, sensor_name in enumerate(("contact_forces_left_foot", "contact_forces_right_foot")):
        forces = env.scene[sensor_name].data.net_forces_w
        if forces is None:
            raise RuntimeError(f"Contact sensor {sensor_name!r} does not expose net forces.")
        result[foot_index] = _numpy(forces)[0].sum(axis=0)
    return result


def _record_replay(
    env: Any,
    manifest: DeploymentManifest,
    args: argparse.Namespace,
    sequence: np.ndarray,
    goal_values: dict[str, float],
    reference_joint_pos: np.ndarray,
    reference_root_position: np.ndarray,
    reference_root_quaternion: np.ndarray,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    robot, action_term = _validate_runtime(env, manifest)
    # The phase table ships beside the manifest; DeploymentManifest stopped exposing it
    # when the action pipeline moved into the shared runtime.
    _jump_phase_table = np.load(manifest.path.parent / manifest.raw["tables"]["jump_phase"])
    observation_dict = _write_and_verify_frame0(
        env,
        robot,
        reference_joint_pos,
        reference_root_position,
        reference_root_quaternion,
    )

    goal_command = env.command_manager.get_term("jump_goal")
    goal_world = _numpy(goal_command.pose_command_w)[0, :3]
    metadata = {
        "schema_version": manifest.raw["schema_version"],
        "task": manifest.raw.get("task"),
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "simulator": "isaac",
        "model": None,
        "overlay": None,
        "reference_csv": str(manifest.reference_source),
        "reference_sha256": manifest.reference_source_sha256,
        "reference_frame": 0,
        "policy": None,
        "policy_backend": None,
        "self_check": False,
        "cross_check": True,
        "action_sequence": None if args.action_sequence is None else str(args.action_sequence.resolve()),
        "sim_dt": manifest.sim_dt,
        "policy_dt": manifest.policy_dt,
        "decimation": manifest.decimation,
        "delay_steps": 0,
        "seed": 0,
        "goal": goal_values,
        "goal_world": goal_world.tolist(),
        "phase_names": manifest.phase_names,
        "qpos_qvel_order": manifest.joint_names,
        "pelvis_pose_convention": "position_world_xyz[m], quaternion_world_from_body_wxyz",
        "pelvis_velocity_convention": "linear_world_xyz[m/s], angular_body_xyz[rad/s]",
        "foot_contact_force_convention": "left_then_right, world_xyz[N]",
        "sample_convention": (
            "sample 0 is pre-physics frame-0 state; later samples are post-physics states; "
            "observation is held from policy tick"
        ),
    }
    logger = StepLogger(args.log, metadata)
    observation = _numpy(observation_dict["policy"])[0]

    with torch.inference_mode():
        for policy_step, raw_action_np in enumerate(sequence):
            raw_action = torch.as_tensor(raw_action_np, dtype=torch.float32, device=env.device).unsqueeze(0)
            env.action_manager.process_action(raw_action)
            delayed_action = _numpy(action_term.raw_actions)[0]
            q_target = _numpy(action_term.processed_actions)[0]
            phase = int(np.argmax(_jump_phase_table[policy_step]))

            if policy_step == 0:
                root_quaternion_xyzw = _numpy(robot.data.root_link_quat_w)[0]
                logger.append(
                    sim_time=0.0,
                    phase=phase,
                    qpos=_numpy(robot.data.joint_pos)[0],
                    qvel=_numpy(robot.data.joint_vel)[0],
                    action=np.asarray(raw_action_np, dtype=np.float64),
                    delayed_action=delayed_action,
                    q_target=q_target,
                    applied_tau=np.zeros(manifest.joint_count, dtype=np.float64),
                    pelvis_pose=np.concatenate(
                        (_numpy(robot.data.root_link_pos_w)[0], np.roll(root_quaternion_xyzw, 1))
                    ),
                    pelvis_velocity=np.zeros(6, dtype=np.float64),
                    foot_contact_forces=np.zeros((2, 3), dtype=np.float64),
                    observation=observation,
                )

            for inner_step in range(manifest.decimation):
                env._sim_step_counter += 1
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=manifest.sim_dt)

                root_quaternion_xyzw = _numpy(robot.data.root_link_quat_w)[0]
                root_quaternion_wxyz = np.roll(root_quaternion_xyzw, 1)
                logger.append(
                    sim_time=(policy_step * manifest.decimation + inner_step + 1) * manifest.sim_dt,
                    phase=phase,
                    qpos=_numpy(robot.data.joint_pos)[0],
                    qvel=_numpy(robot.data.joint_vel)[0],
                    action=np.asarray(raw_action_np, dtype=np.float64),
                    delayed_action=delayed_action,
                    q_target=q_target,
                    applied_tau=_numpy(robot.data.applied_torque)[0],
                    pelvis_pose=np.concatenate((_numpy(robot.data.root_link_pos_w)[0], root_quaternion_wxyz)),
                    pelvis_velocity=np.concatenate(
                        (_numpy(robot.data.root_link_lin_vel_w)[0], _numpy(robot.data.root_link_ang_vel_b)[0])
                    ),
                    foot_contact_forces=_foot_forces(env),
                    observation=observation,
                )

            if policy_step + 1 < len(sequence):
                env.episode_length_buf += 1
                env.common_step_counter += 1
                env.command_manager.compute(dt=manifest.policy_dt)
                observation_dict = env.observation_manager.compute(update_history=True)
                observation = _numpy(observation_dict["policy"])[0]

    logger.save()
    print(f"Wrote {len(sequence) * manifest.decimation + 1} deterministic Isaac samples to {logger.output_path}")
    print("Selected action delay: 0 policy step(s)")


def main() -> None:
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0]] + hydra_args
    manifest = DeploymentManifest(args.manifest)
    _validate_reference_override(args.reference_csv, manifest)
    reference_joint_pos, reference_root_position, reference_root_quaternion = _load_reference_frame0(manifest)
    task_name = manifest.raw.get("task")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("Manifest task must be a non-empty string.")
    goal_values = _goal_values(manifest, args)
    sequence = (
        _load_action_sequence(args.action_sequence, manifest)
        if args.action_sequence is not None
        else np.zeros((manifest.episode_steps, manifest.joint_count), dtype=np.float64)
    )

    env_cfg, _ = resolve_task_config(task_name, "")
    _configure_deterministic_replay(env_cfg, manifest, args, goal_values)
    with launch_simulation(env_cfg, args):
        env = gym.make(task_name, cfg=env_cfg).unwrapped
        try:
            _record_replay(
                env,
                manifest,
                args,
                sequence,
                goal_values,
                reference_joint_pos,
                reference_root_position,
                reference_root_quaternion,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()

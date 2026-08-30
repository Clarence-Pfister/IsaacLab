# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporarily probe exact chained jump commands in Isaac."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.motion import warp_to_torch
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


def _record_pre_reset_state(
    env: Any,
    env_ids: torch.Tensor,
    base_event: Callable,
    base_params: dict[str, Any],
) -> None:
    robot = env.scene[base_params["asset_cfg"].name]
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    env._repeat_probe_pre_reset_pose = warp_to_torch(robot.data.root_link_pose_w)[ids].clone()
    env._repeat_probe_pre_reset_velocity = warp_to_torch(robot.data.root_link_vel_w)[ids].clone()
    env._repeat_probe_pre_reset_joint_pos = warp_to_torch(robot.data.joint_pos)[ids].clone()
    env._repeat_probe_pre_reset_joint_velocity = warp_to_torch(robot.data.joint_vel)[ids].clone()
    joint_limit_data = (
        robot.data.soft_joint_pos_limits
        if base_params.get("use_soft_joint_limits", True)
        else robot.data.joint_pos_limits
    )
    env._repeat_probe_pre_reset_joint_limits = warp_to_torch(joint_limit_data)[ids].clone()
    default_reset_flag = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env._repeat_probe_pre_reset_timeout = getattr(env, "reset_time_outs", default_reset_flag)[ids].clone()
    env._repeat_probe_pre_reset_terminated = getattr(env, "reset_terminated", default_reset_flag)[ids].clone()
    base_event(env, env_ids, **base_params)


def _body_displacement(start_pose: torch.Tensor, end_pose: torch.Tensor) -> tuple[float, float]:
    x, y, z, w = start_pose[3:].tolist()
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    delta_x = float(end_pose[0] - start_pose[0])
    delta_y = float(end_pose[1] - start_pose[1])
    return (
        math.cos(yaw) * delta_x + math.sin(yaw) * delta_y,
        -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--goal_pos_x", type=float, nargs="+", required=True)
    parser.add_argument("--retrigger_value", type=float, default=0.25)
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.commands.jump_goal.debug_vis = False
    env_cfg.commands.jump_goal.zero_goal_probability = 0.0
    env_cfg.commands.jump_goal.boundary_goal_probability = 0.0
    env_cfg.commands.jump_goal.retrigger_cycle_goal_probability = 0.0
    env_cfg.actions.joint_pos.min_delay_steps = 0
    env_cfg.actions.joint_pos.max_delay_steps = 0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.policy.goal_command.params["retrigger_value"] = args.retrigger_value
    env_cfg.observations.critic.goal_command.params["retrigger_value"] = args.retrigger_value
    for term_name in ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "goal_remaining"):
        getattr(env_cfg.observations.policy, term_name).noise = None

    reset_cfg = env_cfg.events.reset_to_reference
    reset_cfg.params["retrigger_probability"] = 1.0
    reset_cfg.params["init_start_prob"] = 0.0
    base_event = reset_cfg.func
    base_params = dict(reset_cfg.params)
    # The reset event observes the global counter before the final increment.
    # Give this diagnostic a two-policy-step bookkeeping tolerance so a normal
    # timeout is classified as the full episode that it is intended to be.
    reset_cfg.func = _record_pre_reset_state
    reset_cfg.params = {"base_event": base_event, "base_params": base_params}

    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.seed = 0
    checkpoint = Path(retrieve_file_path(args.checkpoint)).resolve()
    env_cfg.log_dir = str(checkpoint.parent)

    print("probe: launching Isaac", flush=True)
    with launch_simulation(env_cfg, args):
        print("probe: creating environment", flush=True)
        gym_env = gym.make(args.task, cfg=env_cfg)
        print("probe: environment ready", flush=True)
        rl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        try:
            print(f"probe: loading checkpoint {checkpoint}", flush=True)
            runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=rl_env.unwrapped.device)
            policy_reset = policy.reset if version.parse(installed_version) >= version.parse("4.0.0") else None

            env = rl_env.unwrapped
            command_term = env.command_manager.get_term("jump_goal")
            original_resample = command_term._resample_command
            resample_count = 0

            def resample_fixed(_self: Any, env_ids: Any) -> None:
                nonlocal resample_count
                goal = args.goal_pos_x[min(resample_count, len(args.goal_pos_x) - 1)]
                _self.cfg.ranges.pos_x = (goal, goal)
                original_resample(env_ids)
                resample_count += 1

            command_term._resample_command = types.MethodType(resample_fixed, command_term)
            observation, _ = rl_env.reset()
            if policy_reset is not None:
                policy_reset(torch.ones(1, device=env.device, dtype=torch.bool))

            robot = env.scene["robot"]
            start_pose = warp_to_torch(robot.data.root_link_pose_w)[0].clone()
            episode_retriggered = bool(env.retrigger_reset_mask[0])
            completed = 0
            maximum_steps = len(args.goal_pos_x) * math.ceil(env_cfg.episode_length_s / env.step_dt + 2.0)
            with torch.no_grad():
                for _ in range(maximum_steps):
                    action = policy(observation)
                    observation, _, dones, _ = rl_env.step(action)
                    if not bool(dones[0]):
                        continue
                    end_pose = env._repeat_probe_pre_reset_pose[0]
                    dx, dy = _body_displacement(start_pose, end_pose)
                    root_velocity = env._repeat_probe_pre_reset_velocity[0]
                    joint_pos = env._repeat_probe_pre_reset_joint_pos[0]
                    joint_velocity = env._repeat_probe_pre_reset_joint_velocity[0]
                    joint_limits = env._repeat_probe_pre_reset_joint_limits[0]
                    quaternion = end_pose[3:]
                    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
                    tilt = torch.acos(torch.clamp(1.0 - 2.0 * torch.sum(torch.square(quaternion[:2])), -1.0, 1.0))
                    joint_margin = torch.minimum(
                        joint_pos - joint_limits[:, 0],
                        joint_limits[:, 1] - joint_pos,
                    ).min()
                    next_retrigger = bool(env.retrigger_reset_mask[0])
                    print(
                        f"episode={completed + 1} goal={args.goal_pos_x[completed]:+.3f} "
                        f"dx={dx:+.4f} dy={dy:+.4f} retrigger={episode_retriggered} "
                        f"next_retrigger={next_retrigger} timeout={bool(env._repeat_probe_pre_reset_timeout[0])} "
                        f"terminated={bool(env._repeat_probe_pre_reset_terminated[0])} "
                        f"height={float(end_pose[2]):.3f} tilt_deg={math.degrees(float(tilt)):.2f} "
                        f"root_speed={float(torch.linalg.vector_norm(root_velocity[:3])):.3f} "
                        f"root_rate={float(torch.linalg.vector_norm(root_velocity[3:])):.3f} "
                        f"joint_speed={float(torch.max(torch.abs(joint_velocity))):.3f} "
                        f"joint_margin={float(joint_margin):+.4f}",
                        flush=True,
                    )
                    completed += 1
                    if policy_reset is not None:
                        policy_reset(dones.bool())
                    if completed == len(args.goal_pos_x):
                        break
                    start_pose = warp_to_torch(robot.data.root_link_pose_w)[0].clone()
                    episode_retriggered = bool(env.retrigger_reset_mask[0])
            if completed != len(args.goal_pos_x):
                raise RuntimeError(f"Completed {completed}/{len(args.goal_pos_x)} requested episodes.")
        finally:
            rl_env.close()


if __name__ == "__main__":
    main()

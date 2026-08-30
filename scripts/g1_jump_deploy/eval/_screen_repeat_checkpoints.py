# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporarily screen repeated-jump checkpoints in one Isaac launch."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import math
import sys
import types
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.motion import warp_to_torch
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
from scripts.g1_jump_deploy.eval._probe_repeat_sequence import _body_displacement, _record_pre_reset_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--goal_pos_x", type=float, nargs="+", required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--max_goal_error", type=float, default=0.08)
    parser.add_argument("--retrigger_goal_pos_x_scale", type=float, default=1.0)
    parser.add_argument("--ankle_roll_damping", type=float)
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    if not math.isfinite(args.max_goal_error) or args.max_goal_error <= 0.0:
        parser.error("--max_goal_error must be positive and finite")
    if (
        not math.isfinite(args.retrigger_goal_pos_x_scale)
        or args.retrigger_goal_pos_x_scale <= 0.0
    ):
        parser.error("--retrigger_goal_pos_x_scale must be positive and finite")
    if args.ankle_roll_damping is not None and args.ankle_roll_damping <= 0.0:
        parser.error("--ankle_roll_damping must be positive")

    checkpoints = tuple(Path(retrieve_file_path(value)).resolve() for value in args.checkpoint)
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = len(checkpoints) * args.replicates
    env_cfg.seed = 0
    env_cfg.commands.jump_goal.debug_vis = False
    env_cfg.commands.jump_goal.zero_goal_probability = 0.0
    env_cfg.commands.jump_goal.boundary_goal_probability = 0.0
    env_cfg.commands.jump_goal.retrigger_cycle_goal_probability = 0.0
    env_cfg.observations.policy.goal_command.params["retrigger_goal_pos_x_scale"] = (
        args.retrigger_goal_pos_x_scale
    )
    env_cfg.observations.critic.goal_command.params["retrigger_goal_pos_x_scale"] = (
        args.retrigger_goal_pos_x_scale
    )
    if args.ankle_roll_damping is not None:
        env_cfg.scene.robot.actuators["feet"].damping[".*_ankle_roll_joint"] = (
            args.ankle_roll_damping
        )
    env_cfg.actions.joint_pos.min_delay_steps = 0
    env_cfg.actions.joint_pos.max_delay_steps = 0
    env_cfg.observations.policy.enable_corruption = False
    for term_name in ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "goal_remaining"):
        getattr(env_cfg.observations.policy, term_name).noise = None

    reset_cfg = env_cfg.events.reset_to_reference
    reset_cfg.params["retrigger_probability"] = 1.0
    reset_cfg.params["init_start_prob"] = 0.0
    base_event = reset_cfg.func
    base_params = dict(reset_cfg.params)
    required_joint_margin = max(
        0.0,
        float(base_params["joint_limit_margin"])
        - float(base_params["joint_limit_tolerance"]),
    )
    reset_cfg.func = _record_pre_reset_state
    reset_cfg.params = {"base_event": base_event, "base_params": base_params}

    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.seed = 0
    env_cfg.log_dir = str(checkpoints[0].parent)

    with launch_simulation(env_cfg, args):
        gym_env = gym.make(args.task, cfg=env_cfg)
        rl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        try:
            runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            policies = []
            for checkpoint in checkpoints:
                runner.load(str(checkpoint))
                policies.append(copy.deepcopy(runner.get_inference_policy(device=rl_env.unwrapped.device)))

            env = rl_env.unwrapped
            command_term = env.command_manager.get_term("jump_goal")
            original_resample = command_term._resample_command
            resample_count = torch.zeros(env.num_envs, dtype=torch.int64)

            def resample_fixed(_self: Any, env_ids: Any) -> None:
                ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1).cpu()
                original_range = _self.cfg.ranges.pos_x
                for env_id in ids.tolist():
                    goal_index = min(int(resample_count[env_id]), len(args.goal_pos_x) - 1)
                    goal = args.goal_pos_x[goal_index]
                    _self.cfg.ranges.pos_x = (goal, goal)
                    original_resample([env_id])
                    resample_count[env_id] += 1
                _self.cfg.ranges.pos_x = original_range

            command_term._resample_command = types.MethodType(resample_fixed, command_term)
            observation, _ = rl_env.reset()
            robot = env.scene["robot"]
            start_pose = warp_to_torch(robot.data.root_link_pose_w).clone()
            episode_retrigger = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            completed = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
            failed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            maximum_steps = len(args.goal_pos_x) * math.ceil(env_cfg.episode_length_s / env.step_dt + 2.0)

            with torch.no_grad():
                for _ in range(maximum_steps):
                    action_parts = []
                    for policy_index, policy in enumerate(policies):
                        start = policy_index * args.replicates
                        stop = start + args.replicates
                        action_parts.append(policy(observation[start:stop]))
                    action = torch.cat(action_parts, dim=0)
                    observation, _, dones, _ = rl_env.step(action)
                    done_ids = torch.nonzero(dones, as_tuple=False).flatten()
                    for offset, env_id_tensor in enumerate(done_ids):
                        env_id = int(env_id_tensor)
                        episode_index = int(completed[env_id])
                        if episode_index >= len(args.goal_pos_x):
                            continue
                        end_pose = env._repeat_probe_pre_reset_pose[offset]
                        dx, dy = _body_displacement(start_pose[env_id], end_pose)
                        joint_pos = env._repeat_probe_pre_reset_joint_pos[offset]
                        joint_limits = env._repeat_probe_pre_reset_joint_limits[offset]
                        joint_margins = torch.minimum(
                            joint_pos - joint_limits[:, 0],
                            joint_limits[:, 1] - joint_pos,
                        )
                        joint_index = int(torch.argmin(joint_margins))
                        joint_margin = joint_margins[joint_index]
                        terminated = bool(env._repeat_probe_pre_reset_terminated[offset])
                        expected_retrigger = episode_index > 0
                        retrigger_matches = bool(episode_retrigger[env_id]) == expected_retrigger
                        goal_error = math.hypot(dx - args.goal_pos_x[episode_index], dy)
                        failed[env_id] |= (
                            terminated
                            or not retrigger_matches
                            or goal_error > args.max_goal_error
                            or float(joint_margin) < required_joint_margin
                        )
                        policy_index = env_id // args.replicates
                        replicate = env_id % args.replicates
                        print(
                            f"checkpoint={checkpoints[policy_index].stem} replicate={replicate} "
                            f"episode={episode_index + 1} goal={args.goal_pos_x[episode_index]:+.3f} "
                            f"dx={dx:+.4f} dy={dy:+.4f} error={goal_error:.4f} "
                            f"retrigger={bool(episode_retrigger[env_id])} "
                            f"terminated={terminated} joint_margin={float(joint_margin):+.4f} "
                            f"joint={robot.joint_names[joint_index]} "
                            f"q={float(joint_pos[joint_index]):+.4f} "
                            f"limits=({float(joint_limits[joint_index, 0]):+.4f},"
                            f"{float(joint_limits[joint_index, 1]):+.4f})",
                            flush=True,
                        )
                        completed[env_id] += 1
                        start_pose[env_id] = warp_to_torch(robot.data.root_link_pose_w)[env_id]
                        episode_retrigger[env_id] = env.retrigger_reset_mask[env_id]
                    if bool(torch.all(completed >= len(args.goal_pos_x))):
                        break

            for policy_index, checkpoint in enumerate(checkpoints):
                start = policy_index * args.replicates
                stop = start + args.replicates
                completed_count = int(torch.min(completed[start:stop]))
                failed_count = int(torch.count_nonzero(failed[start:stop]))
                print(
                    f"SUMMARY checkpoint={checkpoint.stem} completed={completed_count}/{len(args.goal_pos_x)} "
                    f"failed_replicates={failed_count}/{args.replicates}",
                    flush=True,
                )
        finally:
            rl_env.close()


if __name__ == "__main__":
    main()

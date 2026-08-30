# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporarily probe a vectorized second-jump command sweep in Isaac."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import sys
import types
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
from scripts.g1_jump_deploy.eval._probe_repeat_sequence import _body_displacement, _record_pre_reset_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--first_goal_pos_x", type=float, required=True)
    parser.add_argument("--second_goal_pos_x", type=float, nargs="+", required=True)
    parser.add_argument("--retrigger_value", type=float, default=0.25)
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = len(args.second_goal_pos_x)
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
    reset_cfg.func = _record_pre_reset_state
    reset_cfg.params = {"base_event": base_event, "base_params": base_params}

    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.seed = 0
    checkpoint = Path(retrieve_file_path(args.checkpoint)).resolve()
    env_cfg.log_dir = str(checkpoint.parent)

    print("sweep: launching Isaac", flush=True)
    with launch_simulation(env_cfg, args):
        gym_env = gym.make(args.task, cfg=env_cfg)
        rl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        try:
            runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=rl_env.unwrapped.device)
            policy_reset = policy.reset if version.parse(installed_version) >= version.parse("4.0.0") else None

            env = rl_env.unwrapped
            command_term = env.command_manager.get_term("jump_goal")
            original_resample = command_term._resample_command
            resample_count = torch.zeros(env.num_envs, dtype=torch.int64)

            def resample_fixed(_self: Any, env_ids: Any) -> None:
                ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1).cpu()
                original_range = _self.cfg.ranges.pos_x
                for env_id in ids.tolist():
                    goal = (
                        args.first_goal_pos_x
                        if int(resample_count[env_id]) == 0
                        else args.second_goal_pos_x[env_id]
                    )
                    _self.cfg.ranges.pos_x = (goal, goal)
                    original_resample([env_id])
                    resample_count[env_id] += 1
                _self.cfg.ranges.pos_x = original_range

            command_term._resample_command = types.MethodType(resample_fixed, command_term)
            observation, _ = rl_env.reset()
            if policy_reset is not None:
                policy_reset(torch.ones(env.num_envs, device=env.device, dtype=torch.bool))

            robot = env.scene["robot"]
            first_start = warp_to_torch(robot.data.root_link_pose_w).clone()
            second_start = torch.zeros_like(first_start)
            first_complete = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            second_complete = torch.zeros_like(first_complete)
            second_retrigger = torch.zeros_like(first_complete)
            first_terminal_joint_pos = torch.zeros(
                (env.num_envs, robot.num_joints),
                device=env.device,
                dtype=warp_to_torch(robot.data.joint_pos).dtype,
            )
            maximum_steps = 2 * math.ceil(env_cfg.episode_length_s / env.step_dt + 2.0)

            with torch.no_grad():
                for _ in range(maximum_steps):
                    action = policy(observation)
                    observation, _, dones, _ = rl_env.step(action)
                    done_ids = torch.nonzero(dones, as_tuple=False).flatten()
                    for offset, env_id_tensor in enumerate(done_ids):
                        env_id = int(env_id_tensor)
                        if not bool(first_complete[env_id]):
                            end_pose = env._repeat_probe_pre_reset_pose[offset]
                            dx, dy = _body_displacement(first_start[env_id], end_pose)
                            joint_pos = env._repeat_probe_pre_reset_joint_pos[offset]
                            joint_limits = env._repeat_probe_pre_reset_joint_limits[offset]
                            joint_margin = torch.minimum(
                                joint_pos - joint_limits[:, 0],
                                joint_limits[:, 1] - joint_pos,
                            ).min()
                            root_velocity = env._repeat_probe_pre_reset_velocity[offset]
                            first_terminal_joint_pos[env_id] = joint_pos
                            first_complete[env_id] = True
                            second_start[env_id] = warp_to_torch(robot.data.root_link_pose_w)[env_id]
                            second_retrigger[env_id] = env.retrigger_reset_mask[env_id]
                            joint_delta = torch.max(
                                torch.abs(first_terminal_joint_pos[env_id] - first_terminal_joint_pos[0])
                            )
                            print(
                                f"first env={env_id} goal={args.first_goal_pos_x:+.3f} "
                                f"dx={dx:+.4f} dy={dy:+.4f} "
                                f"terminated={bool(env._repeat_probe_pre_reset_terminated[offset])} "
                                f"height={float(end_pose[2]):.3f} "
                                f"root_speed={float(torch.linalg.vector_norm(root_velocity[:3])):.3f} "
                                f"joint_speed={float(torch.max(torch.abs(env._repeat_probe_pre_reset_joint_velocity[offset]))):.3f} "
                                f"joint_margin={float(joint_margin):+.4f} "
                                f"joint_delta_env0={float(joint_delta):.6f} "
                                f"next_retrigger={bool(second_retrigger[env_id])}",
                                flush=True,
                            )
                            continue
                        if bool(second_complete[env_id]):
                            continue
                        end_pose = env._repeat_probe_pre_reset_pose[offset]
                        dx, dy = _body_displacement(second_start[env_id], end_pose)
                        joint_pos = env._repeat_probe_pre_reset_joint_pos[offset]
                        joint_limits = env._repeat_probe_pre_reset_joint_limits[offset]
                        joint_margin = torch.minimum(
                            joint_pos - joint_limits[:, 0],
                            joint_limits[:, 1] - joint_pos,
                        ).min()
                        root_velocity = env._repeat_probe_pre_reset_velocity[offset]
                        second_complete[env_id] = True
                        print(
                            f"internal_goal={args.second_goal_pos_x[env_id]:+.3f} dx={dx:+.4f} dy={dy:+.4f} "
                            f"retrigger={bool(second_retrigger[env_id])} "
                            f"terminated={bool(env._repeat_probe_pre_reset_terminated[offset])} "
                            f"height={float(end_pose[2]):.3f} "
                            f"root_speed={float(torch.linalg.vector_norm(root_velocity[:3])):.3f} "
                            f"joint_speed={float(torch.max(torch.abs(env._repeat_probe_pre_reset_joint_velocity[offset]))):.3f} "
                            f"joint_margin={float(joint_margin):+.4f}",
                            flush=True,
                        )
                    if policy_reset is not None:
                        policy_reset(dones.bool())
                    if bool(torch.all(second_complete)):
                        break
            if not bool(torch.all(second_complete)):
                raise RuntimeError(
                    f"Completed {int(torch.count_nonzero(second_complete))}/{env.num_envs} second jumps."
                )
        finally:
            rl_env.close()


if __name__ == "__main__":
    main()

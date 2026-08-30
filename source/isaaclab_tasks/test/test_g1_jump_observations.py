# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.jump_env_cfg import G1JumpStage2EnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.observations import (
    obs_goal_command_remaining_orientation,
    obs_goal_command_remaining_orientation_retrigger,
    obs_goal_command_remaining_orientation_retrigger_goal,
    obs_goal_remaining,
    obs_goal_remaining_latched,
)


class _CommandManager:
    def __init__(self, pose_command_w: torch.Tensor):
        self.term = SimpleNamespace(pose_command_w=pose_command_w)

    def get_term(self, name: str):
        assert name == "jump_goal"
        return self.term


def _make_env() -> SimpleNamespace:
    root_pos_w = torch.tensor(((0.0, 0.0, 0.8), (1.0, 2.0, 0.7)))
    root_quat_w = torch.tensor(((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    pose_command_w = torch.tensor(((0.4, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0), (1.1, 2.3, 0.0, 0.0, 0.0, 0.0, 1.0)))
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos_w, root_quat_w=root_quat_w))
    return SimpleNamespace(
        scene={"robot": robot},
        command_manager=_CommandManager(pose_command_w),
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        num_envs=2,
        device="cpu",
    )


def test_goal_remaining_latched_ignores_motion_and_resets_per_environment() -> None:
    env = _make_env()
    initial = obs_goal_remaining_latched(env)
    expected_initial = torch.tensor(((0.4, -0.2, -0.8), (0.1, 0.3, -0.7)))
    torch.testing.assert_close(initial, expected_initial)

    env.episode_length_buf[:] = 1
    env.scene["robot"].data.root_pos_w[:, :2] += torch.tensor(((0.3, -0.1), (-0.2, 0.4)))
    torch.testing.assert_close(obs_goal_remaining_latched(env), expected_initial)

    env.episode_length_buf[0] = 0
    env.command_manager.term.pose_command_w[0, :3] = torch.tensor((0.8, 0.1, 0.0))
    expected_after_reset = expected_initial.clone()
    expected_after_reset[0] = torch.tensor((0.5, 0.2, -0.8))
    torch.testing.assert_close(obs_goal_remaining_latched(env), expected_after_reset)


def test_goal_command_remaining_orientation_reports_current_attitude_error() -> None:
    command_yaw = torch.deg2rad(torch.tensor(30.0))
    current_yaw = torch.deg2rad(torch.tensor(10.0))
    command_quat = torch.tensor((0.0, 0.0, torch.sin(command_yaw / 2.0), torch.cos(command_yaw / 2.0)))
    current_quat = torch.tensor((0.0, 0.0, torch.sin(current_yaw / 2.0), torch.cos(current_yaw / 2.0)))
    pose_command_b = torch.tensor(((0.2, -0.1, 0.0, *command_quat),))
    pose_command_w = pose_command_b.clone()
    robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=current_quat.unsqueeze(0)))
    command_manager = _CommandManager(pose_command_w)
    command_manager.term.pose_command_b = pose_command_b
    env = SimpleNamespace(scene={"robot": robot}, command_manager=command_manager)

    command = obs_goal_command_remaining_orientation(env)

    remaining_yaw = command_yaw - current_yaw
    expected_quat = torch.tensor((0.0, 0.0, torch.sin(remaining_yaw / 2.0), torch.cos(remaining_yaw / 2.0)))
    torch.testing.assert_close(command[0, :3], pose_command_b[0, :3])
    torch.testing.assert_close(command[0, 3:], expected_quat)


def test_retrigger_goal_command_marks_carried_state_without_mutating_physical_goal() -> None:
    pose_command_b = torch.tensor(
        (
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
    )
    pose_command_w = pose_command_b.clone()
    robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2)))
    command_manager = _CommandManager(pose_command_w)
    command_manager.term.pose_command_b = pose_command_b
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=command_manager,
        retrigger_reset_mask=torch.tensor((False, True)),
    )

    command = obs_goal_command_remaining_orientation_retrigger(env)

    torch.testing.assert_close(command[:, 2], torch.tensor((0.0, 0.25)))
    torch.testing.assert_close(command_manager.term.pose_command_b[:, 2], torch.zeros(2))


def test_retrigger_goal_command_affine_channel_is_zero_for_fresh_episode() -> None:
    pose_command_b = torch.tensor(
        (
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
    )
    pose_command_w = pose_command_b.clone()
    robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2)))
    command_manager = _CommandManager(pose_command_w)
    command_manager.term.pose_command_b = pose_command_b
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=command_manager,
        retrigger_reset_mask=torch.tensor((False, True)),
    )

    command = obs_goal_command_remaining_orientation_retrigger_goal(
        env,
        retrigger_value=0.25,
        retrigger_goal_pos_x_scale=2.0,
    )

    torch.testing.assert_close(command[:, 2], torch.tensor((0.0, 0.05)))
    torch.testing.assert_close(command_manager.term.pose_command_b[:, 2], torch.zeros(2))


def test_stage2_starts_deployable_goal_feedback_curriculum() -> None:
    cfg = G1JumpStage2EnvCfg()

    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining_latched
    assert cfg.observations.critic.goal_remaining.func is obs_goal_remaining

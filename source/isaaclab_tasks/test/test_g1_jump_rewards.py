# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-space G1 jump rewards."""

import math
from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.constants import (
    JUMP_PHASES,
    REFERENCE_MOTION_FPS,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp import rewards


def _yaw_quaternion(yaw: float) -> torch.Tensor:
    return torch.tensor((0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)))


def test_target_angular_rate_scores_yaw_without_penalizing_jump_pitch_and_roll(monkeypatch) -> None:
    phase_weights = (1.0,) * len(JUMP_PHASES)
    active_frames = sum(end - start for start, end in JUMP_PHASES.values())
    active_duration_s = active_frames / REFERENCE_MOTION_FPS
    target_yaw_rate = 0.2
    command_term = SimpleNamespace(target_yaw_displacement_w=torch.full((2,), target_yaw_rate * active_duration_s))
    command_manager = SimpleNamespace(get_term=lambda name: command_term)
    root_ang_vel_w = torch.tensor(((4.0, -3.0, target_yaw_rate), (0.0, 0.0, target_yaw_rate)))
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_ang_vel_w=root_ang_vel_w))},
        command_manager=command_manager,
        num_envs=2,
        device="cpu",
    )
    monkeypatch.setattr(rewards, "get_phase_weight", lambda _env, _weights: torch.ones(2))

    score = rewards.target_angular_rate(env, gradient=7.0, phase_weights=phase_weights)

    torch.testing.assert_close(score, torch.ones(2))


def test_target_velocity_error_remains_unsaturated_away_from_command(monkeypatch) -> None:
    phase_weights = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    active_frames = sum(
        end - start for weight, (start, end) in zip(phase_weights, JUMP_PHASES.values()) if weight != 0.0
    )
    active_duration_s = active_frames / REFERENCE_MOTION_FPS
    target_velocity_xy = torch.tensor(((0.0, 0.2), (0.0, -0.2)))
    command_term = SimpleNamespace(target_displacement_w=target_velocity_xy * active_duration_s)
    current_velocity_xy = torch.tensor(((1.0, 0.0), (-0.5, -0.2)))
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=current_velocity_xy))},
        command_manager=SimpleNamespace(get_term=lambda name: command_term),
        num_envs=2,
        device="cpu",
    )
    monkeypatch.setattr(rewards, "get_phase_weight", lambda _env, _weights: torch.ones(2))

    error = rewards.target_velocity_error(env, phase_weights=phase_weights)

    torch.testing.assert_close(error, torch.tensor((1.04, 0.25)))


def test_target_position_error_can_select_retrigger_episodes(monkeypatch) -> None:
    current_position_xy = torch.tensor(((0.2, -0.1), (0.4, 0.3)))
    target_position_xy = torch.tensor(((0.1, 0.1), (0.1, -0.1)))
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=current_position_xy))},
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.column_stack((target_position_xy, torch.zeros((2, 5))))
        ),
        retrigger_reset_mask=torch.tensor((True, False)),
    )
    monkeypatch.setattr(rewards, "get_phase_weight", lambda _env, _weights: torch.tensor((2.0, 3.0)))

    error = rewards.target_position_error(
        env,
        phase_weights=(1.0,) * len(JUMP_PHASES),
        retrigger_only=True,
    )

    torch.testing.assert_close(error, torch.tensor((0.1, 0.0)))


def test_target_heading_uses_wrapped_yaw_error(monkeypatch) -> None:
    current_quat = torch.stack((_yaw_quaternion(math.pi - 0.05), _yaw_quaternion(0.0)))
    target_quat = torch.stack((_yaw_quaternion(-math.pi + 0.05), _yaw_quaternion(0.0)))
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_quat_w=current_quat))},
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.column_stack((torch.zeros((2, 3)), target_quat))
        ),
    )
    monkeypatch.setattr(rewards, "get_phase_weight", lambda _env, _weights: torch.ones(2))

    score = rewards.target_heading(env, gradient=30.0, phase_weights=(1.0,) * len(JUMP_PHASES))

    torch.testing.assert_close(score, torch.tensor((math.exp(-30.0 * 0.1**2), 1.0)))


def test_joint_target_lower_limit_penalizes_only_normalized_shortfall(monkeypatch) -> None:
    targets = torch.tensor(((0.05, 0.8, 0.09), (0.10, 0.7, 0.30)))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos_target=targets))})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 2])
    monkeypatch.setattr(rewards, "get_phase_weight", lambda _env, _weights: torch.tensor((2.0, 3.0)))

    penalty = rewards.joint_target_lower_limit(
        env,
        lower_limit=0.1,
        normalization=0.1,
        phase_weights=(1.0,) * len(JUMP_PHASES),
        asset_cfg=asset_cfg,
    )

    torch.testing.assert_close(penalty, torch.tensor((0.52, 0.0)))


def test_joint_position_limit_margin_penalizes_proximity_on_both_sides(monkeypatch) -> None:
    joint_pos = torch.tensor(((-0.255, 0.0, 0.250), (-0.200, 0.0, 0.260)))
    limits = torch.tensor((((-0.262, 0.262),) * 3, ((-0.262, 0.262),) * 3))
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=joint_pos, soft_joint_pos_limits=limits))}
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 2])
    monkeypatch.setattr(rewards, "get_phase_weight", lambda _env, _weights: torch.tensor((2.0, 3.0)))

    penalty = rewards.joint_position_limit_margin(
        env,
        margin=0.01,
        phase_weights=(1.0,) * len(JUMP_PHASES),
        asset_cfg=asset_cfg,
    )

    torch.testing.assert_close(penalty, torch.tensor((0.006, 0.024)), atol=1.0e-6, rtol=0.0)


def test_joint_position_limit_margin_can_score_only_retriggers_against_physical_limits(
    monkeypatch,
) -> None:
    joint_pos = torch.tensor(((-0.230,), (-0.230,)))
    soft_limits = torch.tensor((((-0.235, 0.235),), ((-0.235, 0.235),)))
    physical_limits = torch.tensor((((-0.262, 0.262),), ((-0.262, 0.262),)))
    env = SimpleNamespace(
        retrigger_reset_mask=torch.tensor((False, True)),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=joint_pos,
                    soft_joint_pos_limits=soft_limits,
                    joint_pos_limits=physical_limits,
                )
            )
        },
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0])
    monkeypatch.setattr(rewards, "get_phase_weight", lambda _env, _weights: torch.ones(2))

    penalty = rewards.joint_position_limit_margin(
        env,
        margin=0.05,
        phase_weights=(1.0,) * len(JUMP_PHASES),
        asset_cfg=asset_cfg,
        retrigger_only=True,
        use_soft_joint_limits=False,
    )

    torch.testing.assert_close(penalty, torch.tensor((0.0, 0.018)), atol=1.0e-6, rtol=0.0)

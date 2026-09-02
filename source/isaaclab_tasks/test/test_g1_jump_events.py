# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.constants import REFERENCE_DURATION_S
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.jump_env_cfg import (
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatDenseEnvCfg,
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatEnvCfg,
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerAwareEnvCfg,
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerChainEnvCfg,
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerGoalEnvCfg,
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatStrongEnvCfg,
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowDampedEnvCfg,
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowDirectRepeatEnvCfg,
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowRepeatEnvCfg,
    G1JumpStage2DeployLongitudinalOdometrySmoothNarrowRepeatEnvCfg,
    G1JumpStage2DeployLongitudinalSmoothNarrowRepeatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp import motion, terminations
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.events import (
    _full_episode_mask,
    _sample_retrigger_mask,
    _terminal_state_is_retriggerable,
    perturb_trigger_state,
    reference_or_terminal_state_initialization,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.observations import (
    obs_goal_command_remaining_orientation_retrigger,
)


def test_trigger_state_perturbation_updates_only_the_selected_environments() -> None:
    joint_names = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
    )

    class FakeArticulation:
        def __init__(self):
            root_pose = torch.zeros((3, 7))
            root_pose[:, 2] = 0.75
            root_pose[:, 6] = 1.0
            self.data = SimpleNamespace(
                joint_pos=SimpleNamespace(torch=torch.zeros((3, len(joint_names)))),
                joint_vel=SimpleNamespace(torch=torch.zeros((3, len(joint_names)))),
                root_link_pose_w=SimpleNamespace(torch=root_pose),
            )
            self.writes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def find_joints(self, pattern: str) -> tuple[list[int], list[str]]:
            indices = [index for index, name in enumerate(joint_names) if re.fullmatch(pattern, name)]
            return indices, [joint_names[index] for index in indices]

        def write_joint_position_to_sim_index(self, *, position: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.writes["joint_pos"] = (position.clone(), env_ids.clone())

        def write_joint_velocity_to_sim_index(self, *, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.writes["joint_vel"] = (velocity.clone(), env_ids.clone())

        def write_root_pose_to_sim_index(self, *, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
            self.writes["root_pose"] = (root_pose.clone(), env_ids.clone())

    torch.manual_seed(7)
    asset = FakeArticulation()
    env = SimpleNamespace(scene={"robot": asset})
    env_ids = torch.tensor((0, 2))
    perturb_trigger_state(env, env_ids)

    joint_pos, joint_pos_env_ids = asset.writes["joint_pos"]
    joint_vel, joint_vel_env_ids = asset.writes["joint_vel"]
    root_pose, root_pose_env_ids = asset.writes["root_pose"]
    assert torch.equal(joint_pos_env_ids, env_ids)
    assert torch.equal(joint_vel_env_ids, env_ids)
    assert torch.equal(root_pose_env_ids, env_ids)

    leg_joint_ids = [*range(4), *range(6, 10)]
    ankle_pitch_joint_ids = [4, 10]
    ankle_roll_joint_ids = [5, 11]
    assert torch.all(torch.abs(joint_pos[:, leg_joint_ids]) <= 0.05)
    assert torch.count_nonzero(joint_pos[:, leg_joint_ids]) == 2 * len(leg_joint_ids)
    torch.testing.assert_close(joint_pos[:, ankle_pitch_joint_ids[0]], joint_pos[:, ankle_pitch_joint_ids[1]])
    assert torch.all(torch.abs(joint_pos[:, ankle_pitch_joint_ids]) <= 0.15)
    assert torch.all(torch.abs(joint_pos[:, ankle_roll_joint_ids]) <= 0.03)
    assert torch.count_nonzero(joint_pos[:, ankle_roll_joint_ids]) == 2 * len(ankle_roll_joint_ids)
    torch.testing.assert_close(joint_pos[:, 12], torch.zeros(2))
    assert torch.all(torch.abs(joint_vel) <= 0.1)
    assert torch.count_nonzero(joint_vel) == joint_vel.numel()

    assert torch.all((root_pose[:, 2] >= 0.75) & (root_pose[:, 2] <= 0.76))
    torch.testing.assert_close(torch.linalg.vector_norm(root_pose[:, 3:7], dim=-1), torch.ones(2))
    roll = torch.atan2(
        2.0 * (root_pose[:, 6] * root_pose[:, 3] + root_pose[:, 4] * root_pose[:, 5]),
        1.0 - 2.0 * (root_pose[:, 3].square() + root_pose[:, 4].square()),
    )
    pitch = torch.asin(
        torch.clamp(2.0 * (root_pose[:, 6] * root_pose[:, 4] - root_pose[:, 5] * root_pose[:, 3]), -1.0, 1.0)
    )
    yaw = torch.atan2(
        2.0 * (root_pose[:, 6] * root_pose[:, 5] + root_pose[:, 3] * root_pose[:, 4]),
        1.0 - 2.0 * (root_pose[:, 4].square() + root_pose[:, 5].square()),
    )
    assert torch.all(torch.abs(roll) <= math.radians(1.5))
    assert torch.all(torch.abs(pitch) <= math.radians(3.0))
    torch.testing.assert_close(yaw, torch.zeros(2), atol=1.0e-7, rtol=0.0)


def test_terminal_retrigger_filter_rejects_unsafe_landing_states() -> None:
    sample_count = 7
    root_pos = torch.zeros((sample_count, 3))
    root_pos[:, 2] = 0.78
    root_quat = torch.zeros((sample_count, 4))
    root_quat[:, 3] = 1.0
    root_velocity = torch.zeros((sample_count, 6))
    joint_pos = torch.zeros((sample_count, 2))
    joint_velocity = torch.zeros_like(joint_pos)
    joint_pos_limits = torch.empty((sample_count, 2, 2))
    joint_pos_limits[..., 0] = -1.0
    joint_pos_limits[..., 1] = 1.0

    root_pos[1, 2] = 0.5
    root_quat[2] = torch.tensor((math.sin(0.2 / 2.0), 0.0, 0.0, math.cos(0.2 / 2.0)))
    root_velocity[3, 0] = 0.31
    root_velocity[4, 3] = 0.51
    joint_pos[5, 0] = 0.99
    joint_velocity[6, 1] = 2.01

    safe = _terminal_state_is_retriggerable(
        root_pos,
        root_quat,
        root_velocity,
        joint_pos,
        joint_velocity,
        joint_pos_limits,
        root_height_range=(0.65, 0.9),
        max_tilt_rad=0.15,
        max_root_linear_speed=0.3,
        max_root_angular_speed=0.5,
        max_joint_speed=2.0,
        joint_limit_margin=0.02,
        joint_limit_tolerance=0.0,
    )

    torch.testing.assert_close(safe, torch.tensor((True, False, False, False, False, False, False)))


def test_terminal_retrigger_filter_allows_only_numerical_joint_limit_tolerance() -> None:
    root_pos = torch.tensor(((0.0, 0.0, 0.78),) * 3)
    root_quat = torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 3)
    root_velocity = torch.zeros((3, 6))
    joint_pos = torch.tensor(((-1.0005,), (1.0005,), (1.002,)))
    joint_velocity = torch.zeros_like(joint_pos)
    joint_pos_limits = torch.tensor((((-1.0, 1.0),),) * 3)

    safe = _terminal_state_is_retriggerable(
        root_pos,
        root_quat,
        root_velocity,
        joint_pos,
        joint_velocity,
        joint_pos_limits,
        root_height_range=(0.65, 0.9),
        max_tilt_rad=0.15,
        max_root_linear_speed=0.3,
        max_root_angular_speed=0.5,
        max_joint_speed=2.0,
        joint_limit_margin=0.0,
        joint_limit_tolerance=0.001,
    )

    torch.testing.assert_close(safe, torch.tensor((True, True, False)))


def test_full_episode_gate_uses_elapsed_real_steps() -> None:
    complete = _full_episode_mask(
        elapsed_steps=torch.tensor((351, 352, 352)),
        previous_retrigger=torch.tensor((False, False, True)),
        step_dt=0.02,
        terminal_hold_duration_s=4.0,
        retrigger_prepare_duration_s=0.0,
        fresh_prepare_duration_s=0.0,
    )

    assert complete.tolist() == [False, True, True]


def test_retrigger_sampling_can_force_fresh_carried_alternation() -> None:
    sampled = _sample_retrigger_mask(
        previous_retrigger=torch.tensor((False, True, False, True)),
        retrigger_probability=1.0,
        retrigger_after_retrigger_probability=0.0,
    )

    assert sampled.tolist() == [True, False, True, False]


def test_repeat_task_carries_only_safe_timeout_states_into_phase_zero() -> None:
    cfg = G1JumpStage2DeployLongitudinalOdometrySmoothNarrowRepeatEnvCfg()
    deploy_cfg = G1JumpStage2DeployLongitudinalSmoothNarrowRepeatEnvCfg()

    assert cfg.events.reset_to_reference.func is reference_or_terminal_state_initialization
    assert cfg.events.reset_to_reference.params["retrigger_probability"] == 1.0
    assert cfg.events.reset_to_reference.params["init_start_prob"] == 0.0
    assert cfg.events.reset_to_reference.params["joint_limit_margin"] == 0.0
    assert cfg.events.reset_to_reference.params["joint_limit_tolerance"] == 0.001
    assert cfg.events.reset_to_reference.params["zero_retrigger_velocity"] is True
    assert cfg.commands.jump_goal.ranges.pos_x == (-0.1, 0.1)
    assert cfg.observations.policy.goal_remaining.scale == 4.0
    assert cfg.actions.joint_pos.lower_limit_velocity_lookahead == {".*_knee_joint": 0.032}
    assert cfg.rewards.ankle_roll_position_limit_margin.weight == -20.0
    assert cfg.rewards.ankle_roll_position_limit_margin.params["margin"] == 0.01
    assert deploy_cfg.observations.policy.goal_remaining.func is not cfg.observations.policy.goal_remaining.func
    assert deploy_cfg.actions.joint_pos.lower_limit_velocity_lookahead == {".*_knee_joint": 0.032}


def test_repeat_handoff_clock_clamps_during_negative_start_time() -> None:
    env = type("FakeEnv", (), {})()
    env.start_times = torch.tensor((-0.26, -0.26, -0.26))
    env.episode_length_buf = torch.tensor((0, 13, 14))
    env.step_dt = 0.02

    torch.testing.assert_close(motion.get_env_time(env), torch.tensor((0.0, 0.0, 0.02)))


def test_reference_motion_timeout_can_include_policy_stand_hold(monkeypatch) -> None:
    monkeypatch.setattr(
        terminations,
        "get_env_time",
        lambda _env: torch.tensor((4.02, 4.04)),
    )

    done = terminations.reference_motion_complete(object(), hold_duration_s=1.0)

    torch.testing.assert_close(done, torch.tensor((False, True)))


def test_latched_repeat_task_models_stand_hold_and_phase_zero_handoff() -> None:
    cfg = G1JumpStage2DeployLongitudinalLatchedSmoothNarrowRepeatEnvCfg()

    assert cfg.events.reset_to_reference.params["terminal_hold_duration_s"] == 1.0
    assert cfg.events.reset_to_reference.params["retrigger_prepare_duration_s"] == 0.26
    assert cfg.events.reset_to_reference.params["fresh_prepare_duration_s"] == 0.26
    assert cfg.events.reset_to_reference.params["retrigger_probability"] == 0.25
    assert cfg.terminations.time_out.params["hold_duration_s"] == 1.0
    assert math.isclose(cfg.episode_length_s, 4.293333333333333)
    assert cfg.observations.policy.goal_remaining.func is not cfg.observations.critic.goal_remaining.func
    assert cfg.commands.jump_goal.zero_goal_probability == 0.25
    assert cfg.commands.jump_goal.boundary_goal_probability == 0.5
    assert cfg.rewards.target_position.weight == 16.0
    assert cfg.rewards.target_velocity_error.weight == -75.0
    assert cfg.rewards.reference_joint_target_deviation.weight == -20.0
    assert cfg.rewards.reference_joint_target_deviation.params["phase_weights"] == (
        8.0,
        2.0,
        0.25,
        0.25,
        2.0,
        10.0,
    )


def test_direct_repeat_task_matches_policy_stand_confirmation_handoff() -> None:
    cfg = G1JumpStage2DeployLongitudinalLatchedSmoothNarrowDirectRepeatEnvCfg()

    assert cfg.events.reset_to_reference.params["terminal_hold_duration_s"] == 1.0
    assert cfg.events.reset_to_reference.params["retrigger_prepare_duration_s"] == 0.0
    assert cfg.events.reset_to_reference.params["fresh_prepare_duration_s"] == 0.0
    assert cfg.events.reset_to_reference.params["retrigger_probability"] == 1.0
    assert cfg.terminations.time_out.params["hold_duration_s"] == 1.0
    assert math.isclose(cfg.episode_length_s, 4.033333333333333)
    assert cfg.rewards.ankle_roll_position_limit_margin.weight == -100.0
    assert cfg.rewards.ankle_roll_position_limit_margin.params["margin"] == 0.04


def test_commandable_repeat_task_preserves_fresh_starts_and_models_settled_retriggers() -> None:
    cfg = G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatEnvCfg()
    deploy_cfg = G1JumpStage2DeployLongitudinalLatchedSmoothNarrowDampedEnvCfg()

    assert cfg.events.reset_to_reference.params["terminal_hold_duration_s"] == 4.0
    assert cfg.events.reset_to_reference.params["retrigger_prepare_duration_s"] == 0.0
    assert cfg.events.reset_to_reference.params["fresh_prepare_duration_s"] == 0.0
    assert cfg.events.reset_to_reference.params["retrigger_probability"] == 0.5
    assert cfg.terminations.time_out.params["hold_duration_s"] == 4.0
    assert math.isclose(cfg.episode_length_s, 7.033333333333333)
    assert cfg.rewards.target_position_error.weight == -200.0
    assert cfg.rewards.target_position_error.params["retrigger_only"] is True
    assert cfg.rewards.target_position_error.params["phase_weights"] == (0.0, 0.0, 0.0, 0.0, 4.0, 12.0)
    assert cfg.rewards.ankle_roll_position_limit_margin.weight == -50.0
    assert cfg.rewards.ankle_roll_position_limit_margin.params["margin"] == 0.03
    assert cfg.scene.robot.actuators["feet"].damping == {
        ".*_ankle_pitch_joint": 2.0,
        ".*_ankle_roll_joint": 4.0,
    }
    assert deploy_cfg.scene.robot.actuators["feet"].damping == cfg.scene.robot.actuators["feet"].damping
    assert math.isclose(deploy_cfg.episode_length_s, REFERENCE_DURATION_S)


def test_strong_commandable_repeat_task_prioritizes_retrigger_accuracy_and_ankle_margin() -> None:
    cfg = G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatStrongEnvCfg()

    assert cfg.events.reset_to_reference.params["retrigger_probability"] == 0.75
    assert cfg.commands.jump_goal.zero_goal_probability == 0.2
    assert cfg.commands.jump_goal.boundary_goal_probability == 0.6
    assert cfg.rewards.target_position.weight == 0.0
    assert cfg.rewards.target_position_error.weight == -2000.0
    assert cfg.rewards.target_position_error.params["retrigger_only"] is True
    assert cfg.rewards.target_velocity_error.weight == -150.0
    assert cfg.rewards.ankle_roll_position_limit_margin.weight == -200.0
    assert cfg.rewards.ankle_roll_position_limit_margin.params["margin"] == 0.04
    assert cfg.rewards.reference_joint_target_deviation.params["phase_weights"] == (
        8.0,
        2.0,
        0.05,
        0.05,
        2.0,
        10.0,
    )


def test_dense_commandable_repeat_task_supplies_command_credit_during_jump() -> None:
    cfg = G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatDenseEnvCfg()

    assert cfg.events.reset_to_reference.params["retrigger_probability"] == 0.75
    assert cfg.rewards.target_position_error.weight == -500.0
    assert cfg.rewards.target_position_error.params["retrigger_only"] is False
    assert cfg.rewards.target_position_error.params["phase_weights"] == (0.0, 0.0, 2.0, 4.0, 12.0, 2.0)
    assert cfg.rewards.target_velocity_error.weight == -1000.0


def test_retrigger_aware_task_exposes_same_mode_marker_to_actor_and_critic() -> None:
    cfg = G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerAwareEnvCfg()

    assert cfg.observations.policy.goal_command.func is obs_goal_command_remaining_orientation_retrigger
    assert cfg.observations.critic.goal_command.func is obs_goal_command_remaining_orientation_retrigger
    assert cfg.rewards.target_position.weight == 16.0
    assert cfg.rewards.target_position_error.weight == -300.0
    assert cfg.rewards.target_position_error.params["retrigger_only"] is True
    assert cfg.rewards.target_velocity_error.weight == -150.0
    assert cfg.rewards.ankle_roll_position_limit_margin.weight == -1000.0
    assert cfg.rewards.ankle_roll_position_limit_margin.params["margin"] == 0.05


def test_retrigger_goal_task_carries_only_states_inside_physical_joint_limits() -> None:
    cfg = G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerGoalEnvCfg()

    params = cfg.events.reset_to_reference.params
    assert params["use_soft_joint_limits"] is False
    assert params["joint_limit_margin"] == 0.01
    assert params["joint_limit_tolerance"] == 0.001


def test_retrigger_chain_task_trains_long_safe_signed_command_sequences() -> None:
    cfg = G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerChainEnvCfg()

    assert cfg.events.reset_to_reference.params["retrigger_probability"] == 1.0
    assert cfg.events.reset_to_reference.params["retrigger_after_retrigger_probability"] == 1.0
    assert cfg.commands.jump_goal.retrigger_cycle_goal_probability == 1.0
    assert cfg.commands.jump_goal.zero_goal_probability == 0.2
    assert cfg.commands.jump_goal.boundary_goal_probability == 0.8
    assert cfg.rewards.target_position.weight == 0.0
    assert cfg.rewards.target_position_error.weight == -2000.0
    assert cfg.rewards.target_position_error.params["retrigger_only"] is True
    assert cfg.rewards.target_position_error.params["phase_weights"] == (
        0.0,
        0.0,
        2.0,
        4.0,
        16.0,
        4.0,
    )
    assert cfg.rewards.target_velocity_error.weight == -1000.0
    assert cfg.rewards.termination_penalty.weight == -500.0
    assert cfg.rewards.ankle_roll_position_limit_margin.params["retrigger_only"] is True
    assert cfg.rewards.ankle_roll_position_limit_margin.params["use_soft_joint_limits"] is False

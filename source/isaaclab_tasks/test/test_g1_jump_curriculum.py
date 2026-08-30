# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import math

import pytest

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1JumpFineTunePPORunnerCfg,
    G1JumpPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.constants import (
    CSV_MOTION_PATH,
    JUMP_PHASES,
    REFERENCE_MOTION_FPS,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.jump_env_cfg import (
    G1JumpStage1DeployEnvCfg,
    G1JumpStage2DeployEnvCfg,
    G1JumpStage2DeployLongitudinalContactEnvCfg,
    G1JumpStage2DeployLongitudinalEnvCfg,
    G1JumpStage2DeployLongitudinalOdometryEnvCfg,
    G1JumpStage2DeployLongitudinalOdometryRobustEnvCfg,
    G1JumpStage2DeployLongitudinalOdometrySmoothEnvCfg,
    G1JumpStage2DeployLongitudinalOdometrySmoothNarrowEnvCfg,
    G1JumpStage2DeployLongitudinalOdometrySmoothTargetSafeEnvCfg,
    G1JumpStage2DeployLongitudinalSmoothEnvCfg,
    G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg,
    G1JumpStage2DeployLongitudinalUniformEnvCfg,
    G1JumpStage2DeployTranslationEnvCfg,
    G1JumpStage2EnvCfg,
    G1JumpStage3DeployLongitudinalEnvCfg,
    G1JumpStage3DeployTranslationEnvCfg,
    G1JumpStage3NarrowEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.events import randomize_contact_compliance
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.observations import (
    obs_goal_command_remaining_orientation,
    obs_goal_remaining,
    obs_goal_remaining_latched,
)


def test_jump_reset_attitude_matches_reference_frame_zero() -> None:
    with open(CSV_MOTION_PATH, newline="", encoding="utf-8") as stream:
        frame0 = next(csv.DictReader(stream))
    expected_xyzw = tuple(
        float(frame0[name])
        for name in (
            "root_quaternionX",
            "root_quaternionY",
            "root_quaternionZ",
            "root_quaternionW",
        )
    )

    cfg = G1JumpStage2DeployLongitudinalEnvCfg()

    assert cfg.scene.robot.init_state.rot == pytest.approx(expected_xyzw, abs=1.0e-7)


def test_jump_training_preserves_intermediate_deployment_candidates() -> None:
    cfg = G1JumpPPORunnerCfg()

    assert cfg.save_interval == 100


def test_jump_fine_tuning_uses_a_fixed_conservative_learning_rate() -> None:
    cfg = G1JumpFineTunePPORunnerCfg()

    assert cfg.save_interval == 25
    assert cfg.algorithm.schedule == "fixed"
    assert cfg.algorithm.learning_rate == 1.0e-5


def test_stage2_yaw_rewards_distinguish_ignoring_the_largest_command() -> None:
    cfg = G1JumpStage2EnvCfg()
    maximum_yaw = cfg.commands.jump_goal.ranges.yaw[1]

    orientation_gradient = cfg.rewards.target_orientation.params["gradient"]
    no_turn_orientation_score = math.exp(-orientation_gradient * math.sin(maximum_yaw / 2.0) ** 2)
    assert no_turn_orientation_score <= 0.15

    angular_rate = cfg.rewards.target_angular_rate
    phase_weights = angular_rate.params["phase_weights"]
    active_frames = sum(
        end - start for weight, (start, end) in zip(phase_weights, JUMP_PHASES.values()) if weight != 0.0
    )
    maximum_target_rate = maximum_yaw / (active_frames / REFERENCE_MOTION_FPS)
    no_turn_rate_score = math.exp(-angular_rate.params["gradient"] * maximum_target_rate**2)
    assert no_turn_rate_score <= 0.60


def test_deploy_curriculum_uses_filtered_targets_and_relaxes_reference_prior() -> None:
    stage1 = G1JumpStage1DeployEnvCfg()
    stage2 = G1JumpStage2DeployEnvCfg()
    stage3 = G1JumpStage3NarrowEnvCfg()

    for cfg in (stage1, stage2, stage3):
        action = cfg.actions.joint_pos
        assert action.alpha
        assert action.min_delay_steps == 0

    assert stage1.actions.joint_pos.max_delay_steps == 0
    assert stage2.actions.joint_pos.max_delay_steps == 0
    assert stage3.actions.joint_pos.max_delay_steps == 2
    assert stage1.rewards.reference_joint_target_deviation.weight == -50.0
    assert stage2.rewards.reference_joint_target_deviation.weight == -20.0
    assert stage3.rewards.reference_joint_target_deviation.weight == -10.0
    assert stage1.rewards.action_rate.weight == -0.5
    assert stage2.rewards.action_rate.weight == -0.1
    assert stage3.rewards.action_rate.weight == -0.1


def test_stage3_narrow_separates_robustness_from_range_expansion() -> None:
    cfg = G1JumpStage3NarrowEnvCfg()

    ranges = cfg.commands.jump_goal.ranges
    assert ranges.pos_x == (-0.4, 0.4)
    assert ranges.pos_y == (-0.3, 0.3)
    assert ranges.yaw == (-math.pi / 6.0, math.pi / 6.0)
    assert cfg.rewards.target_position.params["gradient"] == 21.07
    assert cfg.rewards.target_orientation.params["gradient"] == 30.0

    action = cfg.actions.joint_pos
    assert action.min_delay_steps == 0
    assert action.max_delay_steps == 2
    assert cfg.observations.policy.enable_corruption
    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining_latched
    assert cfg.observations.critic.goal_remaining.func is obs_goal_remaining

    assert cfg.events.physics_material is not None
    assert cfg.events.robot_mass is not None
    assert cfg.events.pelvis_com is not None
    assert cfg.events.actuator_gains is not None
    assert cfg.events.push_robot is not None


def test_translation_curriculum_adds_deployable_attitude_feedback_before_turns() -> None:
    cfg = G1JumpStage2DeployTranslationEnvCfg()

    ranges = cfg.commands.jump_goal.ranges
    assert ranges.pos_x == (-0.2, 0.2)
    assert ranges.pos_y == (-0.15, 0.15)
    assert ranges.yaw == (0.0, 0.0)
    assert cfg.observations.policy.goal_command.func is obs_goal_command_remaining_orientation
    assert cfg.observations.critic.goal_command.func is obs_goal_command_remaining_orientation
    assert cfg.rewards.target_heading.params["gradient"] == 30.0
    assert cfg.rewards.target_heading.params["phase_weights"] == (0.0, 0.0, 0.0, 0.0, 8.0, 12.0)


def test_longitudinal_curriculum_matches_the_deployment_checkpoint_contract() -> None:
    stage2 = G1JumpStage2DeployLongitudinalEnvCfg()
    stage3 = G1JumpStage3DeployLongitudinalEnvCfg()

    for cfg in (stage2, stage3):
        ranges = cfg.commands.jump_goal.ranges
        assert ranges.pos_x == (-0.2, 0.2)
        assert ranges.pos_y == (0.0, 0.0)
        assert ranges.yaw == (0.0, 0.0)
        assert cfg.commands.jump_goal.zero_goal_probability == 0.25
        assert cfg.commands.jump_goal.boundary_goal_probability == 0.5
        assert cfg.observations.policy.goal_remaining.scale == 4.0
        assert cfg.observations.critic.goal_remaining.scale == 4.0
        assert cfg.observations.policy.goal_command.scale == 4.0
        assert cfg.observations.critic.goal_command.scale == 4.0
        assert cfg.rewards.target_position.weight == 8.0
        assert cfg.rewards.target_velocity.weight == 0.0
        assert cfg.rewards.target_velocity_error.weight == -50.0
        assert cfg.rewards.target_heading.weight == 3.0
        assert cfg.rewards.reference_joint_target_deviation.weight == -5.0
        assert cfg.actions.joint_pos.effort_limit_ratio == 0.6

    assert stage2.actions.joint_pos.max_delay_steps == 0
    assert stage3.actions.joint_pos.max_delay_steps == 2
    assert stage3.observations.policy.enable_corruption


def test_longitudinal_contact_curriculum_randomizes_only_contact_compliance() -> None:
    cfg = G1JumpStage2DeployLongitudinalContactEnvCfg()

    compliance = cfg.events.contact_compliance
    assert compliance.func is randomize_contact_compliance
    assert compliance.params["stiffness_range"] == (1.0e5, 1.0e6)
    assert compliance.params["damping_ratio_range"] == (0.8, 1.2)
    assert compliance.params["rigid_probability"] == 0.25
    assert not hasattr(cfg.events, "robot_mass")
    assert not hasattr(cfg.events, "pelvis_com")
    assert not hasattr(cfg.events, "actuator_gains")
    assert not hasattr(cfg.events, "push_robot")
    assert cfg.actions.joint_pos.max_delay_steps == 0
    assert not cfg.observations.policy.enable_corruption


def test_longitudinal_uniform_curriculum_emphasizes_interior_commands() -> None:
    cfg = G1JumpStage2DeployLongitudinalUniformEnvCfg()

    assert cfg.commands.jump_goal.zero_goal_probability == 0.1
    assert cfg.commands.jump_goal.boundary_goal_probability == 0.1
    assert cfg.commands.jump_goal.ranges.pos_x == (-0.2, 0.2)
    assert cfg.commands.jump_goal.ranges.pos_y == (0.0, 0.0)


def test_longitudinal_odometry_curriculum_closes_the_position_feedback_loop() -> None:
    cfg = G1JumpStage2DeployLongitudinalOdometryEnvCfg()

    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining
    assert cfg.observations.critic.goal_remaining.func is obs_goal_remaining
    assert cfg.commands.jump_goal.zero_goal_probability == 0.1
    assert cfg.commands.jump_goal.boundary_goal_probability == 0.1
    assert cfg.commands.jump_goal.ranges.pos_x == (-0.2, 0.2)
    assert cfg.commands.jump_goal.ranges.pos_y == (0.0, 0.0)


def test_longitudinal_odometry_robust_curriculum_adds_noise_and_contact_only() -> None:
    cfg = G1JumpStage2DeployLongitudinalOdometryRobustEnvCfg()

    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining
    assert cfg.observations.policy.enable_corruption
    assert cfg.observations.policy.joint_pos.noise.n_min == -0.01
    assert cfg.observations.policy.joint_pos.noise.n_max == 0.01
    assert cfg.observations.policy.joint_vel.noise.n_min == -1.0
    assert cfg.observations.policy.joint_vel.noise.n_max == 1.0
    assert cfg.observations.policy.goal_remaining.noise.n_min == -0.02
    assert cfg.observations.policy.goal_remaining.noise.n_max == 0.02
    assert cfg.events.contact_compliance.func is randomize_contact_compliance
    assert not hasattr(cfg.events, "robot_mass")
    assert not hasattr(cfg.events, "pelvis_com")
    assert not hasattr(cfg.events, "actuator_gains")
    assert not hasattr(cfg.events, "push_robot")
    assert cfg.actions.joint_pos.max_delay_steps == 0


def test_longitudinal_odometry_smooth_curriculum_limits_leg_target_bandwidth() -> None:
    cfg = G1JumpStage2DeployLongitudinalOdometrySmoothEnvCfg()

    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining
    assert cfg.actions.joint_pos.alpha == {
        ".*_hip_.*": 0.3,
        ".*_knee_joint": 0.3,
        ".*_ankle_.*": 0.3,
    }
    assert cfg.actions.joint_pos.max_delay_steps == 0
    assert not cfg.observations.policy.enable_corruption


def test_longitudinal_smooth_curriculum_preserves_latched_hardware_feedback() -> None:
    cfg = G1JumpStage2DeployLongitudinalSmoothEnvCfg()

    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining_latched
    assert cfg.observations.critic.goal_remaining.func is obs_goal_remaining
    assert cfg.actions.joint_pos.alpha == {
        ".*_hip_.*": 0.3,
        ".*_knee_joint": 0.3,
        ".*_ankle_.*": 0.3,
    }


def test_longitudinal_smooth_narrow_curriculum_enforces_validated_command_range() -> None:
    cfg = G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg()

    assert cfg.commands.jump_goal.ranges.pos_x == (-0.1, 0.1)
    assert cfg.commands.jump_goal.ranges.pos_y == (0.0, 0.0)
    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining_latched
    assert cfg.actions.joint_pos.clip["left_knee_joint"][0] == 0.1
    assert cfg.actions.joint_pos.clip["right_knee_joint"][0] == 0.1
    assert cfg.actions.joint_pos.lower_limit_velocity_lookahead == {".*_knee_joint": 0.028}
    assert cfg.actions.joint_pos.alpha == {
        ".*_hip_.*": 0.3,
        ".*_knee_joint": 0.3,
        ".*_ankle_.*": 0.3,
    }


def test_longitudinal_odometry_smooth_narrow_curriculum_retains_live_feedback() -> None:
    cfg = G1JumpStage2DeployLongitudinalOdometrySmoothNarrowEnvCfg()

    assert cfg.commands.jump_goal.ranges.pos_x == (-0.1, 0.1)
    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining
    assert cfg.actions.joint_pos.clip["left_knee_joint"][0] == 0.1
    assert cfg.actions.joint_pos.clip["right_knee_joint"][0] == 0.1
    assert cfg.actions.joint_pos.lower_limit_velocity_lookahead == {".*_knee_joint": 0.028}


def test_longitudinal_target_safe_curriculum_penalizes_knee_stop_requests() -> None:
    cfg = G1JumpStage2DeployLongitudinalOdometrySmoothTargetSafeEnvCfg()

    assert cfg.commands.jump_goal.ranges.pos_x == (-0.1, 0.1)
    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining
    assert cfg.actions.joint_pos.clip["left_knee_joint"][0] == pytest.approx(-0.087267)
    assert cfg.actions.joint_pos.clip["right_knee_joint"][0] == pytest.approx(-0.087267)
    assert cfg.rewards.knee_target_lower_limit.weight == -2.0
    assert cfg.rewards.knee_target_lower_limit.params["lower_limit"] == 0.1


def test_stage3_translation_randomizes_contact_and_enforces_a_standard_g1_torque_margin() -> None:
    cfg = G1JumpStage3DeployTranslationEnvCfg()

    ranges = cfg.commands.jump_goal.ranges
    assert ranges.pos_x == (-0.2, 0.2)
    assert ranges.pos_y == (-0.15, 0.15)
    assert ranges.yaw == (0.0, 0.0)
    assert cfg.observations.policy.goal_command.func is obs_goal_command_remaining_orientation
    assert cfg.observations.critic.goal_command.func is obs_goal_command_remaining_orientation

    compliance = cfg.events.contact_compliance
    assert compliance.func is randomize_contact_compliance
    assert compliance.params["stiffness_range"] == (1.0e4, 1.0e6)
    assert compliance.params["damping_ratio_range"] == (0.7, 1.4)
    assert compliance.params["rigid_probability"] == 0.25

    torque_limit = cfg.rewards.joint_torque_demand_limit
    assert torque_limit.weight == -50.0
    assert torque_limit.params["soft_ratio"] == 0.6

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import pytest

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1JumpFineTunePPORunnerCfg,
    G1JumpPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.constants import (
    CSV_MOTION_PATH,
    CSV_MOTION_PATH_EXTENDED,
    JUMP_PHASES,
    JUMP_PHASES_EXTENDED,
    REFERENCE_DURATION_S,
    REFERENCE_DURATION_S_EXTENDED,
    REFERENCE_MOTION_FPS,
    REFERENCE_NUM_FRAMES,
    REFERENCE_NUM_FRAMES_EXTENDED,
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
    G1JumpStage2DeployLongitudinalSmoothNarrowExtendedEnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRange020EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRange040EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRange060EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRange080EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRange100EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRangeContact020EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRangeContact040EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRangeContact060EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRangeContact080EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRangeContact100EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRangeContactTrigger020EnvCfg,
    G1JumpStage2DeployLongitudinalSmoothRangeContactTrigger040EnvCfg,
    G1JumpStage2DeployLongitudinalUniformEnvCfg,
    G1JumpStage2DeployTranslationEnvCfg,
    G1JumpStage2EnvCfg,
    G1JumpStage3DeployLongitudinalEnvCfg,
    G1JumpStage3DeployTranslationEnvCfg,
    G1JumpStage3NarrowEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.events import (
    perturb_trigger_state,
    randomize_contact_compliance,
)
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


def test_extended_reference_curriculum_preserves_narrow_deployment_contract() -> None:
    narrow = G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg()
    extended = G1JumpStage2DeployLongitudinalSmoothNarrowExtendedEnvCfg()

    assert extended.commands.jump_goal.ranges.pos_x == narrow.commands.jump_goal.ranges.pos_x == (-0.1, 0.1)
    assert extended.commands.jump_goal.ranges.pos_y == narrow.commands.jump_goal.ranges.pos_y == (0.0, 0.0)
    assert extended.commands.jump_goal.ranges.yaw == narrow.commands.jump_goal.ranges.yaw == (0.0, 0.0)
    assert extended.observations.policy.goal_remaining.func is obs_goal_remaining_latched
    assert extended.actions.joint_pos.effort_limit_ratio == narrow.actions.joint_pos.effort_limit_ratio == 0.6
    assert (
        extended.actions.joint_pos.alpha
        == narrow.actions.joint_pos.alpha
        == {
            ".*_hip_.*": 0.3,
            ".*_knee_joint": 0.3,
            ".*_ankle_.*": 0.3,
        }
    )
    assert extended.actions.joint_pos.clip == narrow.actions.joint_pos.clip
    assert extended.actions.joint_pos.clip["left_knee_joint"][0] == 0.1
    assert extended.actions.joint_pos.clip["right_knee_joint"][0] == 0.1
    assert extended.actions.joint_pos.lower_limit_velocity_lookahead == {".*_knee_joint": 0.028}

    assert narrow.reference_motion_path == CSV_MOTION_PATH
    assert narrow.reference_num_frames == REFERENCE_NUM_FRAMES
    assert narrow.reference_duration_s == REFERENCE_DURATION_S
    assert narrow.jump_phases == JUMP_PHASES
    assert extended.reference_motion_path == CSV_MOTION_PATH_EXTENDED
    assert extended.reference_num_frames == REFERENCE_NUM_FRAMES_EXTENDED
    assert extended.reference_duration_s == REFERENCE_DURATION_S_EXTENDED
    assert extended.jump_phases == JUMP_PHASES_EXTENDED
    assert narrow.episode_length_s == REFERENCE_DURATION_S
    assert extended.episode_length_s == REFERENCE_DURATION_S_EXTENDED


def test_extended_reference_phase_table_covers_every_frame() -> None:
    previous_end = 0
    covered_frames = []
    for start, end in JUMP_PHASES_EXTENDED.values():
        assert start == previous_end
        assert start < end
        covered_frames.extend(range(start, end))
        previous_end = end

    assert covered_frames == list(range(REFERENCE_NUM_FRAMES_EXTENDED))


def test_extended_reference_task_uses_fine_tune_runner() -> None:
    spec = gym.spec("Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-NarrowExtended-v0")

    assert spec.kwargs["env_cfg_entry_point"].endswith(":G1JumpStage2DeployLongitudinalSmoothNarrowExtendedEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":G1JumpFineTunePPORunnerCfg")


@pytest.mark.parametrize(
    ("cfg_type", "half_range", "expected_gradient", "expected_threshold", "wide_land_weights"),
    (
        (G1JumpStage2DeployLongitudinalSmoothRange020EnvCfg, 0.2, 21.07, 0.20, False),
        (G1JumpStage2DeployLongitudinalSmoothRange040EnvCfg, 0.4, 21.07, 0.35 * 0.4 / 0.65, False),
        (G1JumpStage2DeployLongitudinalSmoothRange060EnvCfg, 0.6, 21.07, 0.35 * 0.6 / 0.65, True),
        (G1JumpStage2DeployLongitudinalSmoothRange080EnvCfg, 0.8, 21.07, 0.35, True),
        (G1JumpStage2DeployLongitudinalSmoothRange100EnvCfg, 1.0, 21.07, 0.35, True),
    ),
)
def test_goal_range_curriculum_preserves_deployment_contract_and_narrow_position_kernel(
    cfg_type: type,
    half_range: float,
    expected_gradient: float,
    expected_threshold: float,
    wide_land_weights: bool,
) -> None:
    narrow = G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg()
    cfg = cfg_type()

    assert cfg.commands.jump_goal.ranges.pos_x == (-half_range, half_range)
    assert cfg.commands.jump_goal.ranges.pos_y == (0.0, 0.0)
    assert cfg.commands.jump_goal.ranges.yaw == (0.0, 0.0)
    assert cfg.commands.jump_goal.zero_goal_probability == 0.1
    assert cfg.commands.jump_goal.boundary_goal_probability == 0.1
    assert cfg.observations.policy.goal_remaining.func is obs_goal_remaining_latched
    assert cfg.actions.joint_pos.effort_limit_ratio == 0.6
    assert cfg.actions.joint_pos.alpha == narrow.actions.joint_pos.alpha
    assert cfg.actions.joint_pos.clip == narrow.actions.joint_pos.clip
    assert cfg.actions.joint_pos.clip["left_knee_joint"][0] >= 0.1
    assert cfg.actions.joint_pos.clip["right_knee_joint"][0] >= 0.1
    assert cfg.actions.joint_pos.lower_limit_velocity_lookahead == {".*_knee_joint": 0.028}
    assert cfg.rewards.target_position.params["gradient"] == pytest.approx(expected_gradient)
    assert cfg.terminations.task_completion_error.params["pos_threshold"] == pytest.approx(expected_threshold)

    if wide_land_weights:
        assert cfg.rewards.target_position.params["phase_weights"] == (0.0, 1.0, 2.0, 4.0, 12.0, 10.0)
        assert cfg.rewards.target_velocity.params["phase_weights"] == (0.0, 0.0, 3.0, 3.0, 6.0, 2.0)
        assert cfg.rewards.track_root_pos_z.params["phase_weights"] == (4.0, 8.0, 12.0, 8.0, 6.0, 6.0)
    else:
        assert (
            cfg.rewards.target_position.params["phase_weights"]
            == narrow.rewards.target_position.params["phase_weights"]
        )
        assert (
            cfg.rewards.target_velocity.params["phase_weights"]
            == narrow.rewards.target_velocity.params["phase_weights"]
        )
        assert (
            cfg.rewards.track_root_pos_z.params["phase_weights"]
            == narrow.rewards.track_root_pos_z.params["phase_weights"]
        )


@pytest.mark.parametrize("range_variant", ("", "Contact"))
@pytest.mark.parametrize("range_code", ("020", "040", "060", "080", "100"))
def test_goal_range_curriculum_tasks_use_fine_tune_runner(range_code: str, range_variant: str) -> None:
    task_id = f"Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-Range{range_variant}{range_code}-v0"
    spec = gym.spec(task_id)

    assert spec.kwargs["env_cfg_entry_point"].endswith(
        f":G1JumpStage2DeployLongitudinalSmoothRange{range_variant}{range_code}EnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":G1JumpFineTunePPORunnerCfg")


@pytest.mark.parametrize("range_code", ("020", "040"))
def test_goal_range_contact_trigger_tasks_use_fine_tune_runner(range_code: str) -> None:
    task_id = f"Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-RangeContactTrigger{range_code}-v0"
    spec = gym.spec(task_id)

    assert spec.kwargs["env_cfg_entry_point"].endswith(
        f":G1JumpStage2DeployLongitudinalSmoothRangeContactTrigger{range_code}EnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":G1JumpFineTunePPORunnerCfg")


@pytest.mark.parametrize(
    ("variant_args", "range_variant", "range_codes", "selection_key", "minimum_response_gain"),
    (
        ([], "", ("020", "040", "060", "080", "100"), "success_rate_0p10", "0.85"),
        (
            ["--variant", "contact", "--selection_tolerance_m", "0.20", "--minimum_response_gain", "0.90"],
            "Contact",
            ("020", "040", "060", "080", "100"),
            "success_rate_0p20",
            "0.9",
        ),
        (["--variant", "contact_trigger"], "ContactTrigger", ("020", "040"), "success_rate_0p10", "0.85"),
    ),
)
def test_goal_range_curriculum_driver_dry_run_prints_every_stage(
    variant_args: list[str],
    range_variant: str,
    range_codes: tuple[str, ...],
    selection_key: str,
    minimum_response_gain: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    driver = repo_root / "scripts" / "g1_jump_deploy" / "train" / "run_goal_range_curriculum.py"

    result = subprocess.run(
        [sys.executable, str(driver), "--dry_run", *variant_args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    for range_code in range_codes:
        task_id = f"Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-Range{range_variant}{range_code}-v0"
        assert result.stdout.count(task_id) == 2
    assert result.stdout.count("scripts/reinforcement_learning/rsl_rl/train.py") == len(range_codes)
    assert result.stdout.count("scripts/g1_jump_deploy/eval/eval_success_rate.py") == len(range_codes)
    assert "--load_checkpoint model_825.pt" in result.stdout
    assert (
        f"# Select by {selection_key} with upright_rate >= 0.99, "
        f"response_gain_xx >= {minimum_response_gain}, correlation_x >= 0.95"
    ) in result.stdout
    if range_variant:
        assert f"--run_name range{range_variant.lower()}020_from825" in result.stdout


def test_goal_range_curriculum_driver_selects_requested_tolerance_and_later_tie(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    driver = repo_root / "scripts" / "g1_jump_deploy" / "train" / "run_goal_range_curriculum.py"
    driver_namespace = runpy.run_path(str(driver))
    parse_evaluation = driver_namespace["_parse_evaluation"]
    select_checkpoint = driver_namespace["_select_checkpoint"]
    write_summary = driver_namespace["_write_summary"]

    def evaluation(
        iteration: int,
        success_0p10: int,
        success_0p20: int,
        upright: int,
        response_gain: float,
        correlation: float,
    ) -> dict[str, object]:
        output = f"""
            Overall success (95% Wilson binomial confidence interval):
            0.10 {success_0p10} 100 {success_0p10:.2f}% [0.0%, 100.0%]
            0.20 {success_0p20} 100 {success_0p20:.2f}% [0.0%, 100.0%]

            response matrix = [[{response_gain:+.3f}, n/a], [n/a, n/a]]
            offset [m] = [+0.0110, +0.0000]
            same-axis Pearson correlation = [x {correlation:+.3f}, y n/a]
            upright at episode end (height > 0.60 m, tilt <= 30 deg): {upright}/100 ({upright:.2f}%)
        """
        return parse_evaluation(output, Path(f"model_{iteration}.pt"))

    evaluations = [
        evaluation(900, 50, 100, 100, 0.90, 0.96),
        evaluation(925, 60, 100, 98, 1.20, 0.99),
        evaluation(950, 50, 90, 100, 0.90, 0.96),
        evaluation(975, 80, 100, 100, 1.00, 0.94),
    ]

    assert evaluations[0]["success_rate_0p10"] == 0.5
    assert evaluations[0]["success_rate_0p20"] == 1.0
    assert evaluations[0]["response_gain_xx"] == 0.9
    assert evaluations[0]["response_offset_x"] == 0.011
    assert evaluations[0]["correlation_x"] == 0.96
    assert select_checkpoint(evaluations, 0.10, 0.85)[0]["iteration"] == 950
    assert select_checkpoint(evaluations, 0.20, 0.85)[0]["iteration"] == 900
    fallback, fallback_note = select_checkpoint(evaluations, 0.10, 1.10)
    assert fallback["iteration"] == 975
    assert fallback_note == "no checkpoint met the gain criterion"

    selected, selection_note = select_checkpoint(evaluations, 0.10, 0.85)
    write_summary.__globals__["_SUMMARY_ROOT"] = tmp_path
    write_summary(
        driver_namespace["Stage"]("range020", "020", "test-task"),
        driver_namespace["CheckpointRef"](Path("/logs/source/model_825.pt"), "source", "model_825.pt"),
        Path("/logs/run"),
        300,
        0.10,
        0.85,
        evaluations,
        selected,
        selection_note,
    )
    summary = json.loads((tmp_path / "range020" / "summary.json").read_text(encoding="utf-8"))
    assert summary["selection"]["goal_tolerance_m"] == 0.10
    assert summary["selection"]["success_rate_0p10"] == selected["success_rate_0p10"]
    assert summary["selection"]["success_rate_0p20"] == selected["success_rate_0p20"]
    assert summary["selection"]["response_gain_xx"] == 0.9
    assert summary["selection"]["response_offset_x"] == 0.011
    assert summary["selection"]["correlation_x"] == 0.96
    assert summary["evaluations"][0]["upright_rate"] == 1.0

    write_summary(
        driver_namespace["Stage"]("fallback", "020", "test-task"),
        driver_namespace["CheckpointRef"](Path("/logs/source/model_825.pt"), "source", "model_825.pt"),
        Path("/logs/run"),
        300,
        0.10,
        1.10,
        evaluations,
        fallback,
        fallback_note,
    )
    fallback_summary = json.loads((tmp_path / "fallback" / "summary.json").read_text(encoding="utf-8"))
    assert fallback_summary["selection"]["note"] == "no checkpoint met the gain criterion"


@pytest.mark.parametrize(
    ("plain_type", "contact_type"),
    (
        (G1JumpStage2DeployLongitudinalSmoothRange020EnvCfg, G1JumpStage2DeployLongitudinalSmoothRangeContact020EnvCfg),
        (G1JumpStage2DeployLongitudinalSmoothRange040EnvCfg, G1JumpStage2DeployLongitudinalSmoothRangeContact040EnvCfg),
        (G1JumpStage2DeployLongitudinalSmoothRange060EnvCfg, G1JumpStage2DeployLongitudinalSmoothRangeContact060EnvCfg),
        (G1JumpStage2DeployLongitudinalSmoothRange080EnvCfg, G1JumpStage2DeployLongitudinalSmoothRangeContact080EnvCfg),
        (G1JumpStage2DeployLongitudinalSmoothRange100EnvCfg, G1JumpStage2DeployLongitudinalSmoothRangeContact100EnvCfg),
    ),
)
def test_goal_range_contact_curriculum_adds_only_contact_and_requested_actor_noise(
    plain_type: type, contact_type: type
) -> None:
    reference_contact = G1JumpStage2DeployLongitudinalContactEnvCfg()
    robust = G1JumpStage2DeployLongitudinalOdometryRobustEnvCfg()
    plain = plain_type()
    contact = contact_type()

    assert contact.events.contact_compliance.func is reference_contact.events.contact_compliance.func
    assert contact.events.contact_compliance.mode == reference_contact.events.contact_compliance.mode
    assert contact.events.contact_compliance.params == reference_contact.events.contact_compliance.params
    contact_event_names = list(contact.events.__dataclass_fields__)
    reference_event_names = list(reference_contact.events.__dataclass_fields__)
    assert contact_event_names == reference_event_names
    if "physics_material" in contact_event_names:
        assert contact_event_names.index("physics_material") < contact_event_names.index("contact_compliance")
    else:
        assert "physics_material" not in reference_event_names

    assert contact.commands.to_dict() == plain.commands.to_dict()
    assert contact.rewards.to_dict() == plain.rewards.to_dict()
    assert contact.terminations.to_dict() == plain.terminations.to_dict()
    assert contact.actions.to_dict() == plain.actions.to_dict()
    assert contact.observations.policy.enable_corruption
    for term_name in ("joint_vel", "base_ang_vel", "projected_gravity"):
        contact_noise = getattr(contact.observations.policy, term_name).noise
        robust_noise = getattr(robust.observations.policy, term_name).noise
        assert contact_noise.to_dict() == robust_noise.to_dict()
    assert contact.observations.policy.joint_pos.noise is plain.observations.policy.joint_pos.noise
    assert contact.observations.policy.goal_remaining.noise is plain.observations.policy.goal_remaining.noise


@pytest.mark.parametrize(
    ("contact_type", "trigger_type"),
    (
        (
            G1JumpStage2DeployLongitudinalSmoothRangeContact020EnvCfg,
            G1JumpStage2DeployLongitudinalSmoothRangeContactTrigger020EnvCfg,
        ),
        (
            G1JumpStage2DeployLongitudinalSmoothRangeContact040EnvCfg,
            G1JumpStage2DeployLongitudinalSmoothRangeContactTrigger040EnvCfg,
        ),
    ),
)
def test_goal_range_contact_trigger_curriculum_adds_only_the_post_reference_reset_event(
    contact_type: type, trigger_type: type
) -> None:
    contact = contact_type()
    trigger = trigger_type()

    contact_dict = contact.to_dict()
    trigger_dict = trigger.to_dict()
    contact_events = contact_dict.pop("events")
    trigger_events = trigger_dict.pop("events")
    perturb_event = trigger_events.pop("perturb_trigger_state")

    assert trigger_dict == contact_dict
    assert trigger_events == contact_events
    assert trigger.events.perturb_trigger_state.func is perturb_trigger_state
    assert perturb_event["mode"] == "reset"
    assert perturb_event["params"]["leg_joint_pos_noise_rad"] == 0.05
    assert perturb_event["params"]["ankle_pitch_offset_range_rad"] == (-0.15, 0.15)
    assert perturb_event["params"]["ankle_roll_noise_rad"] == 0.03
    assert perturb_event["params"]["root_pitch_noise_rad"] == pytest.approx(math.radians(3.0))
    assert perturb_event["params"]["root_roll_noise_rad"] == pytest.approx(math.radians(1.5))
    assert perturb_event["params"]["root_height_offset_range_m"] == (0.0, 0.01)
    assert perturb_event["params"]["joint_vel_noise_rad_s"] == 0.1
    event_names = list(trigger.events.__dict__)
    assert event_names.index("reset_to_reference") < event_names.index("perturb_trigger_state")


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

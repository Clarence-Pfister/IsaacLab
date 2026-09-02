# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the MuJoCo FSM scenario definitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.g1_jump_deploy.fsm.jump_fsm import JumpControllerState, JumpGoal
from scripts.g1_jump_deploy.fsm.run_fsm_mujoco import (
    InactivePolicy,
    _apply_attitude_offset,
    _command_tracking_result,
    _hardware_margin_result,
    _load_initial_state,
    _parse_args,
    _prejump_hold_upright_result,
    _repeat_goals,
    _scenario_result,
    _scenario_timeline,
    _terminal_state_reached,
    _unmeasured_ground_contact_result,
    run,
)


def test_stand_scenario_has_no_operator_events() -> None:
    timeline = _scenario_timeline(
        "stand",
        JumpGoal(0.4, 0.0, 0.0),
        {"pos_x": (-0.3, 1.0)},
        policy_dt=0.02,
        flight_start_step=43,
    )

    assert timeline == ()


def test_velocity_limit_emulation_flag_defaults_off_and_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_fsm_mujoco.py"])
    assert not _parse_args().emulate_velocity_limit

    monkeypatch.setattr(sys, "argv", ["run_fsm_mujoco.py", "--emulate_velocity_limit"])
    assert _parse_args().emulate_velocity_limit


def test_joint_limit_abort_margin_cli_is_non_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_fsm_mujoco.py", "--joint_limit_abort_margin_rad", "0.02"])
    assert _parse_args().joint_limit_abort_margin_rad == pytest.approx(0.02)

    monkeypatch.setattr(sys, "argv", ["run_fsm_mujoco.py", "--joint_limit_abort_margin_rad", "-0.01"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_late_abort_is_sampled_at_first_flight_step() -> None:
    policy_dt = 0.02
    flight_start_step = 43

    timeline = _scenario_timeline(
        "abort_late",
        JumpGoal(0.4, 0.0, 0.0),
        {"pos_x": (-0.3, 1.0)},
        policy_dt=policy_dt,
        flight_start_step=flight_start_step,
    )

    confirm, abort = timeline[-2:]
    assert abort.time_s - confirm.time_s == pytest.approx(flight_start_step * policy_dt)


def test_operator_timeline_accepts_delayed_rehearsal_edges() -> None:
    timeline = _scenario_timeline(
        "nominal",
        JumpGoal(0.0, 0.0, 0.0),
        {"pos_x": (-0.3, 1.0)},
        policy_dt=0.02,
        flight_start_step=43,
        start_time_s=4.5,
        confirm_time_s=9.0,
    )

    assert [entry.time_s for entry in timeline] == pytest.approx([4.5, 9.0])


def test_repeat_timeline_uses_a_separate_start_and_confirmation_for_each_goal() -> None:
    goals = (
        JumpGoal(-0.1, 0.0, 0.0),
        JumpGoal(0.0, 0.0, 0.0),
        JumpGoal(0.1, 0.0, 0.0),
    )

    timeline = _scenario_timeline(
        "repeat",
        goals[0],
        {"pos_x": (-0.1, 0.1)},
        policy_dt=0.02,
        flight_start_step=43,
        start_time_s=0.5,
        confirm_time_s=2.8,
        repeat_goals=goals,
        episode_steps=152,
        settle_timeout_s=4.0,
    )

    assert [entry.goal for entry in timeline[::2]] == list(goals)
    assert all(entry.request_start for entry in timeline[::2])
    assert all(entry.confirm for entry in timeline[1::2])
    assert len(timeline) == 2 * len(goals)
    assert timeline[2].time_s > timeline[1].time_s + 152 * 0.02 + 4.0


def test_repeat_timeline_confirms_soon_after_policy_stand_preparation() -> None:
    goals = (JumpGoal(-0.1, 0.0, 0.0), JumpGoal(0.1, 0.0, 0.0))

    timeline = _scenario_timeline(
        "repeat",
        goals[0],
        {"pos_x": (-0.1, 0.1)},
        policy_dt=0.02,
        flight_start_step=43,
        start_time_s=0.5,
        confirm_time_s=2.8,
        repeat_goals=goals,
        episode_steps=152,
        settle_timeout_s=4.0,
        repeat_prepare_duration_s=0.25,
    )

    second_start, second_confirm = timeline[2:]
    assert second_confirm.time_s - second_start.time_s == pytest.approx(0.6)


def test_repeat_goals_default_to_full_longitudinal_envelope() -> None:
    primary = JumpGoal(0.02, 0.0, 0.0)

    goals = _repeat_goals(primary, {"pos_x": (-0.1, 0.1)}, None)

    assert [goal.dx for goal in goals] == pytest.approx([-0.1, 0.0, 0.1])


def test_repeat_result_requires_every_complete_jump_cycle() -> None:
    one_cycle = [
        JumpControllerState.PASSIVE,
        JumpControllerState.STAND,
        JumpControllerState.GOTO_START,
        JumpControllerState.ARMED,
        JumpControllerState.JUMP,
        JumpControllerState.SETTLE,
        JumpControllerState.STAND,
    ]
    fsm = SimpleNamespace(
        transition_history=one_cycle + one_cycle[2:] + one_cycle[2:],
        state=JumpControllerState.STAND,
        last_report="Policy-native stand settled.",
    )

    passed, result = _scenario_result(
        "repeat",
        fsm,
        [],
        final_time_s=30.0,
        requested_duration_s=30.0,
        expected_jump_count=3,
    )

    assert passed
    assert "three" in result
    assert _terminal_state_reached("repeat", fsm, 30.0, expected_jump_count=3)


def test_stand_scenario_passes_only_after_requested_duration() -> None:
    fsm = SimpleNamespace(
        transition_history=[JumpControllerState.PASSIVE, JumpControllerState.STAND],
        state=JumpControllerState.STAND,
        last_report="Standing enabled.",
    )

    passed, result = _scenario_result("stand", fsm, [], final_time_s=10.0, requested_duration_s=10.0)
    early_passed, early_result = _scenario_result("stand", fsm, [], final_time_s=4.0, requested_duration_s=10.0)

    assert passed
    assert "requested duration" in result
    assert not early_passed
    assert "4.000 of 10.000 s" in early_result


def test_upright_hold_audit_rejects_soft_hips_and_accepts_ground_overrides(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsm_mujoco.py",
            "--scenario",
            "stand",
            "--headless",
            "--max_duration",
            "2.0",
            "--stand_ankle_stiffness",
            "80.0",
            "--stand_ankle_damping",
            "7.0",
            "--balance_disable_integral",
            "--balance_initial_pitch_integral",
            "0.0",
            "--log",
            str(tmp_path / "soft_hips.npz"),
        ],
    )
    run(_parse_args())
    soft_output = capsys.readouterr().out

    assert "Upright hold audit: FAIL" in soft_output
    assert "Scenario result: INCOMPLETE — upright hold failed" in soft_output

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsm_mujoco.py",
            "--scenario",
            "nominal",
            "--unmeasured_ground_validation",
            "--headless",
            "--max_duration",
            "6.0",
            "--start_time_s",
            "6.0",
            "--confirm_time_s",
            "8.8",
            "--stand_ankle_stiffness",
            "80.0",
            "--stand_ankle_damping",
            "7.0",
            "--log",
            str(tmp_path / "prepared_legs.npz"),
        ],
    )
    run(_parse_args())
    prepared_output = capsys.readouterr().out

    assert "Upright hold audit: PASS" in prepared_output


@pytest.mark.parametrize("state", ["STAND", "GOTO_START", "ARMED"])
def test_upright_hold_audit_checks_every_prejump_state(state: str) -> None:
    arrays = {
        "fsm_state": np.asarray([state]),
        "time": np.asarray([1.25]),
        "tilt": np.asarray([np.deg2rad(31.0)]),
        "pelvis_pose": np.asarray([[0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]]),
    }

    passed, result = _prejump_hold_upright_result(arrays)

    assert not passed
    assert state in result
    assert "31.00 deg" in result


def test_inactive_policy_rejects_inference() -> None:
    policy = InactivePolicy(observation_dim=326, action_dim=23)

    np.testing.assert_array_equal(policy.last_observation, np.zeros(326, dtype=np.float32))
    np.testing.assert_array_equal(policy.last_action, np.zeros(23, dtype=np.float64))
    with pytest.raises(RuntimeError, match="must not invoke"):
        policy(np.zeros(326, dtype=np.float32))


def test_load_initial_state_requires_exact_manifest_order(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    capture = {
        "schema_version": 1,
        "label": "test capture",
        "joint_names": ["first", "second"],
        "joint_positions_rad": [0.1, -0.2],
        "root_quaternion_wxyz": [2.0, 0.0, 0.0, 0.0],
    }
    path.write_text(json.dumps(capture), encoding="utf-8")

    state = _load_initial_state(path, ("first", "second"))

    np.testing.assert_allclose(state.joint_positions, [0.1, -0.2])
    np.testing.assert_allclose(state.root_quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])
    capture["joint_names"].reverse()
    path.write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest order"):
        _load_initial_state(path, ("first", "second"))


def test_apply_attitude_offset_changes_roll_and_pitch() -> None:
    quaternion = _apply_attitude_offset(
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        roll_rad=np.deg2rad(2.0),
        pitch_rad=np.deg2rad(-3.0),
    )

    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert quaternion[1] > 0.0
    assert quaternion[2] < 0.0


def test_stand_return_cli_requires_complete_blend_duration(monkeypatch) -> None:
    base = [
        "run_fsm_mujoco.py",
        "--scenario",
        "stand",
        "--stand_return_state",
        "native.json",
        "--stand_return_start_s",
        "8",
        "--stand_return_duration_s",
        "4",
    ]
    monkeypatch.setattr(sys, "argv", base + ["--max_duration", "10"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--max_duration", "14"])

    args = _parse_args()

    assert args.stand_return_start_s == pytest.approx(8.0)
    assert args.stand_return_duration_s == pytest.approx(4.0)


def test_contactless_gantry_rehearsal_cli_requires_exact_hardware_envelope(monkeypatch) -> None:
    valid = [
        "run_fsm_mujoco.py",
        "--scenario",
        "nominal",
        "--contactless_gantry_rehearsal",
        "--gantry_support_fraction",
        "1.0",
        "--effort_scale",
        "0.1",
        "--target_rate_limit_rad_s",
        "1.2",
        "--max_duration",
        "15",
        "--start_time_s",
        "4.5",
        "--confirm_time_s",
        "9.0",
        "--stand_entry_duration_s",
        "4.0",
        "--stand_ankle_stiffness",
        "80.0",
        "--stand_ankle_damping",
        "7.0",
        "--balance_disable_integral",
        "--balance_initial_roll_integral",
        "0.0",
        "--balance_initial_pitch_integral",
        "0.0",
        "--goal_pos_x",
        "0.0",
        "--goal_pos_y",
        "0.0",
        "--goal_roll",
        "0.0",
        "--goal_pitch",
        "0.0",
        "--goal_yaw",
        "0.0",
    ]
    monkeypatch.setattr(sys, "argv", valid)

    args = _parse_args()

    assert args.contactless_gantry_rehearsal
    assert args.gantry_support_fraction == pytest.approx(1.0)
    assert args.effort_scale == pytest.approx(0.1)
    assert args.target_rate_limit_rad_s == pytest.approx(1.2)


def _contactless_rehearsal_escalation_args() -> list[str]:
    return [
        "run_fsm_mujoco.py",
        "--scenario",
        "nominal",
        "--contactless_gantry_rehearsal",
        "--gantry_support_fraction",
        "1.0",
        "--rehearsal_effort_scale_override",
        "0.3",
        "--rehearsal_unlimited_slew",
        "--max_duration",
        "15",
        "--start_time_s",
        "4.5",
        "--confirm_time_s",
        "9.0",
        "--stand_entry_duration_s",
        "4.0",
        "--stand_ankle_stiffness",
        "80.0",
        "--stand_ankle_damping",
        "7.0",
        "--balance_disable_integral",
        "--balance_initial_roll_integral",
        "0.0",
        "--balance_initial_pitch_integral",
        "0.0",
        "--goal_pos_x",
        "0.0",
        "--goal_pos_y",
        "0.0",
        "--goal_roll",
        "0.0",
        "--goal_pitch",
        "0.0",
        "--goal_yaw",
        "0.0",
    ]


def test_contactless_gantry_rehearsal_escalation_is_fail_closed(monkeypatch) -> None:
    arguments = _contactless_rehearsal_escalation_args()
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit):
        _parse_args()

    arguments.append("--acknowledge_rehearsal_escalation")
    monkeypatch.setattr(sys, "argv", arguments)
    args = _parse_args()

    assert args.effort_scale == pytest.approx(0.3)
    assert args.target_rate_limit_rad_s is None


@pytest.mark.parametrize("override", ["0.1", "0.61", "nan"])
def test_contactless_gantry_rehearsal_escalation_rejects_invalid_effort(monkeypatch, override: str) -> None:
    arguments = _contactless_rehearsal_escalation_args()
    arguments[arguments.index("--rehearsal_effort_scale_override") + 1] = override
    arguments.append("--acknowledge_rehearsal_escalation")
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        _parse_args()


def test_contactless_gantry_rehearsal_rejects_escalation_ack_without_option(monkeypatch) -> None:
    arguments = _contactless_rehearsal_escalation_args()
    override_index = arguments.index("--rehearsal_effort_scale_override")
    del arguments[override_index : override_index + 2]
    arguments.remove("--rehearsal_unlimited_slew")
    arguments.extend(("--effort_scale", "0.1", "--target_rate_limit_rad_s", "1.2"))
    arguments.append("--acknowledge_rehearsal_escalation")
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        _parse_args()


def test_mujoco_rehearsal_escalation_options_require_contactless_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsm_mujoco.py",
            "--rehearsal_effort_scale_override",
            "0.3",
            "--acknowledge_rehearsal_escalation",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_unmeasured_ground_cli_keeps_ground_envelope(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsm_mujoco.py",
            "--scenario",
            "nominal",
            "--unmeasured_ground_validation",
            "--gantry_support_fraction",
            "0.0",
        ],
    )

    args = _parse_args()

    assert args.unmeasured_ground_validation
    assert not args.contactless_gantry_rehearsal
    assert args.policy_prepare_duration_s == pytest.approx(0.0)
    assert args.confirm_time_s == pytest.approx(2.8)


def test_unmeasured_ground_cli_accepts_repeat_goal_sequence(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsm_mujoco.py",
            "--scenario",
            "repeat",
            "--unmeasured_ground_validation",
            "--repeat_goal_pos_x",
            "-0.1",
            "0.0",
            "0.1",
            "--max_duration",
            "30",
        ],
    )

    args = _parse_args()

    assert args.repeat_goal_pos_x == pytest.approx([-0.1, 0.0, 0.1])


def test_jump_handoff_cli_accepts_independent_step_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsm_mujoco.py",
            "--jump_target_blend_steps",
            "3",
            "--jump_gain_blend_steps",
            "0",
            "--jump_balance_blend_steps",
            "7",
        ],
    )

    args = _parse_args()

    assert args.jump_target_blend_steps == 3
    assert args.jump_gain_blend_steps == 0
    assert args.jump_balance_blend_steps == 7


def test_goto_start_cli_accepts_slow_recovery_duration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsm_mujoco.py",
            "--goto_start_duration_s",
            "6.0",
            "--confirm_time_s",
            "6.8",
        ],
    )

    args = _parse_args()

    assert args.goto_start_duration_s == pytest.approx(6.0)


def test_terminal_return_cli_accepts_policy_step_count(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsm_mujoco.py",
            "--policy_terminal_return_steps",
            "32",
        ],
    )

    args = _parse_args()

    assert args.policy_terminal_return_steps == 32


def test_direct_policy_stand_retrigger_cli_requires_policy_stand(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsm_mujoco.py",
            "--policy_stand_after_jump",
            "--policy_stand_direct_retrigger",
        ],
    )

    args = _parse_args()

    assert args.policy_stand_direct_retrigger


def test_unmeasured_ground_hidden_contact_audit_requires_support_flight_and_touchdown() -> None:
    arrays = {
        "time": np.asarray((0.0, 0.1, 0.2, 0.3, 0.4)),
        "fsm_state": np.asarray(("ARMED", "JUMP", "JUMP", "JUMP", "SETTLE")),
        "fsm_episode_step": np.asarray((0, 0, 43, 80, 152)),
        "foot_contact_forces": np.asarray(((100.0, 100.0), (100.0, 100.0), (0.0, 0.0), (80.0, 90.0), (80.0, 90.0))),
    }

    passed, result = _unmeasured_ground_contact_result(arrays, flight_start_step=43)

    assert passed
    assert "support, flight, and bilateral touchdown" in result


def test_unmeasured_ground_hidden_contact_audit_checks_every_repeat_episode() -> None:
    arrays = {
        "time": np.arange(10, dtype=np.float64) * 0.1,
        "fsm_state": np.asarray(("ARMED", "JUMP", "JUMP", "JUMP", "SETTLE", "ARMED", "JUMP", "JUMP", "JUMP", "SETTLE")),
        "fsm_episode_step": np.asarray((0, 0, 43, 80, 152, 0, 0, 43, 80, 152)),
        "foot_contact_forces": np.asarray(
            (
                (100.0, 100.0),
                (100.0, 100.0),
                (0.0, 0.0),
                (80.0, 90.0),
                (80.0, 90.0),
                (100.0, 100.0),
                (100.0, 100.0),
                (0.0, 0.0),
                (80.0, 90.0),
                (80.0, 90.0),
            )
        ),
    }

    passed, result = _unmeasured_ground_contact_result(arrays, flight_start_step=43)

    assert passed
    assert "2/2 episodes" in result

    arrays["foot_contact_forces"][8] = 0.0
    passed, result = _unmeasured_ground_contact_result(arrays, flight_start_step=43)

    assert not passed
    assert "episode 2" in result


def test_command_tracking_audit_checks_every_relative_goal() -> None:
    states = np.asarray(("ARMED", "JUMP", "JUMP", "SETTLE", "ARMED", "JUMP", "JUMP", "SETTLE"))
    poses = np.zeros((len(states), 7), dtype=np.float64)
    poses[:, 3] = 1.0
    poses[:, 0] = (0.0, -0.04, -0.09, -0.09, -0.09, -0.04, 0.01, 0.01)
    goals = (JumpGoal(-0.1, 0.0, 0.0), JumpGoal(0.1, 0.0, 0.0))

    passed, result = _command_tracking_result(
        {"fsm_state": states, "pelvis_pose": poses},
        goals,
    )

    assert passed
    assert "2/2" in result

    poses[6:, 0] = -0.09
    passed, result = _command_tracking_result(
        {"fsm_state": states, "pelvis_pose": poses},
        goals,
    )

    assert not passed
    assert "episode 2" in result


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--gantry_support_fraction", "0.9"),
        ("--effort_scale", "0.2"),
        ("--target_rate_limit_rad_s", "1.3"),
        ("--max_duration", "14"),
        ("--start_time_s", "4.0"),
        ("--confirm_time_s", "6.0"),
        ("--goal_pos_x", "0.01"),
    ],
)
def test_contactless_gantry_rehearsal_cli_rejects_relaxed_envelope(
    monkeypatch,
    option: str,
    value: str,
) -> None:
    arguments = [
        "run_fsm_mujoco.py",
        "--scenario",
        "nominal",
        "--contactless_gantry_rehearsal",
        "--gantry_support_fraction",
        "1.0",
        "--effort_scale",
        "0.1",
        "--target_rate_limit_rad_s",
        "1.2",
        "--max_duration",
        "15",
        "--start_time_s",
        "4.5",
        "--confirm_time_s",
        "9.0",
        "--stand_entry_duration_s",
        "4.0",
        "--stand_ankle_damping",
        "7.0",
        "--balance_disable_integral",
        "--balance_initial_roll_integral",
        "0.0",
        "--balance_initial_pitch_integral",
        "0.0",
        "--goal_pos_x",
        "0.0",
        "--goal_pos_y",
        "0.0",
        "--goal_roll",
        "0.0",
        "--goal_pitch",
        "0.0",
        "--goal_yaw",
        "0.0",
    ]
    option_index = arguments.index(option)
    arguments[option_index + 1] = value
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        _parse_args()


def test_hardware_margin_rejects_scaled_torque_saturation() -> None:
    arrays = {
        "applied_tau": np.asarray(((0.0, 0.0), (9.5, 0.0))),
        "tilt": np.zeros(2),
        "qvel": np.zeros((2, 2)),
        "pelvis_pose": np.asarray(((0.0, 0.0, 0.8), (0.0, 0.0, 0.8))),
        "q_target": np.zeros((2, 2)),
        "qpos": np.zeros((2, 2)),
        "joint_limit_violations": np.zeros((2, 2), dtype=bool),
    }
    robot = SimpleNamespace(command_effort_limits=np.full(2, 10.0))

    passed, result = _hardware_margin_result(arrays, robot)

    assert not passed
    assert "command effort 95.0%" in result

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline tests for the guarded G1 stand and gantry-rehearsal boundary."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.g1_jump_deploy.fsm import JumpControllerState, JumpGoal
from scripts.g1_jump_deploy.hardware import validate_shadow_logs as shadow_validator
from scripts.g1_jump_deploy.hardware.run_fsm_g1 import (
    _FAST_DT,
    _GANTRY_STAND_ENTRY_LEG_ERROR_LIMIT_RAD,
    _TARGET_RATE_LIMIT_RAD_S,
    FeedbackLimits,
    FeedbackSnapshot,
    HardwareManifest,
    SafetyFault,
    _active_feedback_limits,
    _active_target_rate_limit,
    _bridge_native_stand_to_passive,
    _feedback_fault,
    _G1Robot,
    _GantryRehearsalOperator,
    _ground_stand_configuration,
    _GroundJumpRecorder,
    _InactivePolicy,
    _InteractiveGoalReader,
    _load_hardware_manifest,
    _lock_ground_session_after_abort,
    _native_walkrun_handoff_fault,
    _parse_args,
    _parse_ground_goal,
    _play_audio_cue,
    _project_shadow_target,
    _read_audio_cue,
    _RehearsalRecorder,
    _remote_a_pressed,
    _remote_activation_pressed,
    _remote_b_pressed,
    _remote_y_pressed,
    _restore_internal_control,
    _run_control,
    _run_shadow_policy_episode,
    _stand_entry_pose_fault,
    _StateBuffer,
    _verify_shadow_admission,
    _verify_validated_bundle,
    _wait_for_remote_activation,
)


def _manifest() -> HardwareManifest:
    return HardwareManifest(
        joint_names=tuple(f"joint_{index}" for index in range(23)),
        sdk_slots=tuple(range(23)),
        default_position=np.zeros(23),
        joint_position_lower=np.full(23, -1.1),
        joint_position_upper=np.full(23, 1.1),
        target_position_lower=np.full(23, -1.0),
        target_position_upper=np.full(23, 1.0),
        effort_limit=np.full(23, 10.0),
        velocity_limit=np.full(23, 20.0),
        stiffness=np.full(23, 40.0),
        damping=np.full(23, 2.0),
        initial_root_height=0.8,
        reference_root_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
        policy_dt=0.02,
    )


def _leg_manifest() -> HardwareManifest:
    manifest = _manifest()
    leg_names = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    )
    return HardwareManifest(
        joint_names=leg_names + tuple(f"arm_joint_{index}" for index in range(11)),
        sdk_slots=manifest.sdk_slots,
        default_position=manifest.default_position,
        joint_position_lower=manifest.joint_position_lower,
        joint_position_upper=manifest.joint_position_upper,
        target_position_lower=manifest.target_position_lower,
        target_position_upper=manifest.target_position_upper,
        effort_limit=manifest.effort_limit,
        velocity_limit=manifest.velocity_limit,
        stiffness=manifest.stiffness,
        damping=manifest.damping,
        initial_root_height=manifest.initial_root_height,
        reference_root_quaternion_wxyz=manifest.reference_root_quaternion_wxyz,
        policy_dt=manifest.policy_dt,
    )


def _snapshot(now: float) -> FeedbackSnapshot:
    return FeedbackSnapshot(
        received_at=now,
        tick=1,
        mode_pr=0,
        mode_machine=4,
        joint_positions=np.zeros(23),
        joint_velocities=np.zeros(23),
        joint_torque_estimates=np.zeros(23),
        imu_quaternion=np.asarray((1.0, 0.0, 0.0, 0.0)),
        imu_gyroscope=np.zeros(3),
        wireless_remote=bytes(40),
        maximum_temperature_c=30,
    )


def _hardware_manifest_dict() -> dict:
    return {
        "schema_version": "1.5",
        "joints": {
            "names": [f"joint_{index}" for index in range(23)],
            "unitree_sdk2_slots": list(range(23)),
            "default_pos": [0.0] * 23,
            "position_limits": [[-1.1, 1.1]] * 23,
        },
        "action": {
            "clip": [[-1.0, 1.0]] * 23,
            "torque_projection": {
                "type": "instantaneous_pd",
                "period_s": 0.002,
                "effort_limit_ratio": [0.6] * 23,
                "formula": (
                    "q_target = q + (clip(kp*(q_requested-q)-kd*dq, -ratio*effort_limit, ratio*effort_limit)+kd*dq)/kp"
                ),
            },
            "lower_limit_brake": {
                "type": "velocity_lookahead",
                "period_s": 0.002,
                "position_lower": [-1.0] * 23,
                "position_upper": [1.0] * 23,
                "velocity_lookahead_s": [0.028] * 23,
                "formula": ("q_requested = max(q_filtered, min(q_upper, q_lower + t_lookahead*max(-dq, 0)))"),
            },
        },
        "actuators": {
            "effort_limit": [10.0] * 23,
            "velocity_limit": [20.0] * 23,
            "stiffness": [40.0] * 23,
            "damping": [2.0] * 23,
        },
        "reference": {"root_frame0": {"pos": [0.0, 0.0, 0.8], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "control": {"policy_dt": 0.02, "sim_dt": 0.002},
    }


def test_load_hardware_manifest(tmp_path: Path) -> None:
    manifest = _hardware_manifest_dict()
    manifest["reference"]["root_frame0"]["quat_xyzw"] = [0.0, 0.1, 0.0, math.sqrt(0.99)]
    path = tmp_path / "deploy_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = _load_hardware_manifest(path)

    assert loaded.joint_count == 23
    assert loaded.sdk_slots == tuple(range(23))
    assert loaded.initial_root_height == pytest.approx(0.8)
    np.testing.assert_allclose(
        loaded.reference_root_quaternion_wxyz,
        (math.sqrt(0.99), 0.0, 0.1, 0.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(loaded.lower_limit_velocity_lookahead, np.full(23, 0.028))
    np.testing.assert_array_equal(loaded.effort_limit_ratio, np.full(23, 0.6))
    np.testing.assert_array_equal(loaded.effort_limit, np.full(23, 10.0))


def test_ground_stand_configuration_uses_reference_attitude_and_prepared_leg_gains() -> None:
    pitch = math.radians(7.49)
    manifest = replace(
        _leg_manifest(),
        reference_root_quaternion_wxyz=np.asarray((math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0)),
    )

    stand_gains, balance_config = _ground_stand_configuration(manifest)

    assert stand_gains.ankle_stiffness == pytest.approx(80.0)
    assert stand_gains.ankle_damping == pytest.approx(7.0)
    assert set(stand_gains.stiffness_overrides or {}) == {
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
    }
    assert set((stand_gains.stiffness_overrides or {}).values()) == {200.0}
    assert set((stand_gains.damping_overrides or {}).values()) == {5.0}
    assert balance_config.target_roll == pytest.approx(0.0)
    assert balance_config.target_pitch == pytest.approx(pitch)
    assert balance_config.integral_enabled
    assert balance_config.initial_roll_integral == pytest.approx(0.0)
    assert balance_config.initial_pitch_integral == pytest.approx(0.2)


def test_load_hardware_manifest_accepts_retrigger_aware_schema(tmp_path: Path) -> None:
    manifest = _hardware_manifest_dict()
    manifest["schema_version"] = "1.6"
    path = tmp_path / "deploy_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = _load_hardware_manifest(path)

    assert loaded.joint_count == 23


def test_load_hardware_manifest_accepts_retrigger_goal_schema(tmp_path: Path) -> None:
    manifest = _hardware_manifest_dict()
    manifest["schema_version"] = "1.7"
    path = tmp_path / "deploy_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = _load_hardware_manifest(path)

    assert loaded.joint_count == 23


def test_hardware_manifest_requires_safety_complete_schema(tmp_path: Path) -> None:
    manifest = _hardware_manifest_dict()
    manifest["schema_version"] = "1.4"
    path = tmp_path / "deploy_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema 1.5"):
        _load_hardware_manifest(path)


@pytest.mark.parametrize(
    ("field", "expected"),
    (("torque_projection", "torque_projection"), ("lower_limit_brake", "lower_limit_brake")),
)
def test_hardware_manifest_requires_command_guards(tmp_path: Path, field: str, expected: str) -> None:
    manifest = _hardware_manifest_dict()
    del manifest["action"][field]
    path = tmp_path / "deploy_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        _load_hardware_manifest(path)


def test_hardware_artifacts_must_match_accepted_digest_record(tmp_path: Path) -> None:
    manifest_path = tmp_path / "deploy_manifest.json"
    policy_path = tmp_path / "policy.onnx"
    preview_path = tmp_path / "reference_preview.npy"
    phase_path = tmp_path / "jump_phase.npy"
    manifest_path.write_text(
        json.dumps(
            {
                "tables": {
                    "reference_preview": preview_path.name,
                    "jump_phase": phase_path.name,
                }
            }
        ),
        encoding="utf-8",
    )
    policy_path.write_bytes(b"policy")
    preview_path.write_bytes(b"preview")
    phase_path.write_bytes(b"phase")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    validation_record = tmp_path / "validated_bundle.toml"
    validation_record.write_text(
        "\n".join(
            (
                'schema_version = "1.0"',
                f'[artifacts."{manifest_path.name}"]',
                f'sha256 = "{digest(manifest_path)}"',
                f'[artifacts."{policy_path.name}"]',
                f'sha256 = "{digest(policy_path)}"',
                f'[artifacts."{preview_path.name}"]',
                f'sha256 = "{digest(preview_path)}"',
                f'[artifacts."{phase_path.name}"]',
                f'sha256 = "{digest(phase_path)}"',
            )
        ),
        encoding="utf-8",
    )

    _verify_validated_bundle(manifest_path, validation_record, policy_path)

    policy_path.write_bytes(b"different policy")
    with pytest.raises(ValueError, match="does not match"):
        _verify_validated_bundle(manifest_path, validation_record, policy_path)


def test_hardware_placeholder_odometry_uses_manifest_root_height() -> None:
    snapshot = _snapshot(time.monotonic())
    robot = _G1Robot(
        _manifest(),
        _FakeStateBuffer(snapshot),
        _FakePublisher(),
        _low_command(),
        _FakeCrc(),
    )

    np.testing.assert_array_equal(robot.odometry_position, np.asarray((0.0, 0.0, 0.8)))


def test_stand_entry_pose_gate_checks_only_leg_joints() -> None:
    manifest = _leg_manifest()
    safe = _snapshot(time.monotonic())
    arm_offset = FeedbackSnapshot(**{**safe.__dict__, "joint_positions": np.r_[np.zeros(12), np.ones(11)]})
    leg_positions = np.zeros(23)
    leg_positions[8] = 0.36
    leg_offset = FeedbackSnapshot(**{**safe.__dict__, "joint_positions": leg_positions})

    assert _stand_entry_pose_fault(safe, manifest) is None
    assert _stand_entry_pose_fault(arm_offset, manifest) is None
    assert "left_ankle_pitch_joint" in (_stand_entry_pose_fault(leg_offset, manifest) or "")
    assert _stand_entry_pose_fault(leg_offset, manifest, _GANTRY_STAND_ENTRY_LEG_ERROR_LIMIT_RAD) is None


def test_native_walkrun_handoff_requires_slow_stand_pose() -> None:
    now = time.monotonic()
    manifest = _leg_manifest()
    safe = _snapshot(now)
    moving = FeedbackSnapshot(**{**safe.__dict__, "joint_velocities": np.r_[0.6, np.zeros(22)]})

    assert _native_walkrun_handoff_fault(safe, manifest, now) is None
    assert "joint speed" in (_native_walkrun_handoff_fault(moving, manifest, now) or "")


def test_gantry_standup_cli_requires_duration_and_reduced_effort(monkeypatch) -> None:
    base = [
        "run_fsm_g1.py",
        "enp131s0",
        "--entry_mode",
        "gantry_standup",
        "--enable_control",
    ]
    monkeypatch.setattr(sys, "argv", base + ["--duration", "1", "--effort_scale", "0.5"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--duration", "4", "--effort_scale", "0.75"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--duration", "4", "--effort_scale", "0.5"])

    args = _parse_args()

    assert args.entry_mode == "gantry_standup"
    assert args.duration == pytest.approx(4.0)
    assert args.effort_scale == pytest.approx(0.5)


def test_policy_check_is_strictly_read_only(monkeypatch) -> None:
    base = ["run_fsm_g1.py", "enp131s0", "--check_policy"]
    monkeypatch.setattr(sys, "argv", base + ["--enable_control"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--query_fsm"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--goal_pos_x", "0.1"])

    args = _parse_args()

    assert args.check_policy
    assert not args.enable_control
    assert not args.query_fsm
    assert args.goal_pos_x == pytest.approx(0.1)


def test_policy_shadow_requires_log_and_is_strictly_read_only(monkeypatch, tmp_path: Path) -> None:
    base = ["run_fsm_g1.py", "enp131s0", "--shadow_policy"]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--shadow_log", str(tmp_path / "shadow.npz"), "--enable_control"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", ["run_fsm_g1.py", "enp131s0", "--shadow_log", str(tmp_path / "shadow.npz")])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--shadow_log", str(tmp_path / "shadow.npz")])

    args = _parse_args()

    assert args.shadow_policy
    assert args.shadow_log == tmp_path / "shadow.npz"
    assert not args.enable_control
    assert not args.query_fsm


def test_gantry_policy_rehearsal_cli_is_fail_closed(monkeypatch, tmp_path: Path) -> None:
    admission_path = tmp_path / "admission.json"
    log_path = tmp_path / "rehearsal.npz"
    base = [
        "run_fsm_g1.py",
        "enp131s0",
        "--gantry_policy_rehearsal",
        "--entry_mode",
        "gantry_standup",
        "--exit_mode",
        "passive",
        "--duration",
        "15",
        "--effort_scale",
        "0.1",
        "--shadow_admission",
        str(admission_path),
        "--rehearsal_log",
        str(log_path),
    ]
    for required_flag in ("--enable_control", "--acknowledge_contactless_rehearsal"):
        monkeypatch.setattr(sys, "argv", base)
        with pytest.raises(SystemExit):
            _parse_args()
        base.append(required_flag)

    monkeypatch.setattr(sys, "argv", base)
    args = _parse_args()

    assert args.gantry_policy_rehearsal
    assert args.enable_control
    assert args.acknowledge_contactless_rehearsal
    assert args.entry_mode == "gantry_standup"
    assert args.exit_mode == "passive"
    assert args.duration == pytest.approx(15.0)
    assert args.effort_scale == pytest.approx(0.1)
    assert args.shadow_admission == admission_path
    assert args.rehearsal_log == log_path


def test_gantry_rehearsal_escalation_requires_acknowledgement(monkeypatch, tmp_path: Path) -> None:
    arguments = [
        "run_fsm_g1.py",
        "enp131s0",
        "--gantry_policy_rehearsal",
        "--enable_control",
        "--acknowledge_contactless_rehearsal",
        "--entry_mode",
        "gantry_standup",
        "--duration",
        "15",
        "--shadow_admission",
        str(tmp_path / "admission.json"),
        "--rehearsal_log",
        str(tmp_path / "rehearsal.npz"),
        "--rehearsal_effort_scale_override",
        "0.3",
        "--rehearsal_unlimited_slew",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit):
        _parse_args()

    arguments.append("--acknowledge_rehearsal_escalation")
    monkeypatch.setattr(sys, "argv", arguments)
    args = _parse_args()

    assert args.rehearsal_escalated
    assert args.effort_scale == pytest.approx(0.3)
    assert args.rehearsal_unlimited_slew


@pytest.mark.parametrize("override", ["0.1", "0.61", "nan"])
def test_gantry_rehearsal_escalation_rejects_invalid_effort(monkeypatch, tmp_path: Path, override: str) -> None:
    arguments = [
        "run_fsm_g1.py",
        "enp131s0",
        "--gantry_policy_rehearsal",
        "--enable_control",
        "--acknowledge_contactless_rehearsal",
        "--acknowledge_rehearsal_escalation",
        "--entry_mode",
        "gantry_standup",
        "--duration",
        "15",
        "--shadow_admission",
        str(tmp_path / "admission.json"),
        "--rehearsal_log",
        str(tmp_path / "rehearsal.npz"),
        "--rehearsal_effort_scale_override",
        override,
    ]
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.parametrize(
    "option",
    [
        "--rehearsal_effort_scale_override",
        "--rehearsal_unlimited_slew",
        "--acknowledge_rehearsal_escalation",
    ],
)
def test_rehearsal_escalation_options_require_rehearsal(monkeypatch, option: str) -> None:
    arguments = ["run_fsm_g1.py", "enp131s0", option]
    if option == "--rehearsal_effort_scale_override":
        arguments.append("0.3")
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        _parse_args()


def _ground_jump_arguments(tmp_path: Path) -> list[str]:
    return [
        "run_fsm_g1.py",
        "enp131s0",
        "--ground_jump",
        "--enable_control",
        "--acknowledge_unmeasured_ground_jump",
        "--shadow_admission",
        str(tmp_path / "admission.json"),
        "--ground_log",
        str(tmp_path / "ground.npz"),
        "--effort_scale",
        "0.3",
        "--duration",
        "20",
        "--entry_mode",
        "native_stand",
        "--exit_mode",
        "passive",
        "--goal_sequence",
        "0.0,0.1,-0.1",
    ]


@pytest.mark.parametrize(
    ("option", "takes_value"),
    [
        ("--acknowledge_unmeasured_ground_jump", False),
        ("--shadow_admission", True),
        ("--ground_log", True),
        ("--effort_scale", True),
    ],
)
def test_ground_jump_cli_requires_explicit_safety_options(
    monkeypatch, tmp_path: Path, option: str, takes_value: bool
) -> None:
    arguments = _ground_jump_arguments(tmp_path)
    index = arguments.index(option)
    del arguments[index : index + (2 if takes_value else 1)]
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.parametrize(
    "replacement",
    [
        ["--effort_scale", "0.61"],
        ["--entry_mode", "gantry_standup"],
        ["--duration", "19"],
    ],
)
def test_ground_jump_cli_rejects_relaxed_contract(monkeypatch, tmp_path: Path, replacement: list[str]) -> None:
    arguments = _ground_jump_arguments(tmp_path)
    index = arguments.index(replacement[0])
    arguments[index : index + 2] = replacement
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        _parse_args()


def test_ground_jump_cli_is_mutually_exclusive_with_rehearsal(monkeypatch, tmp_path: Path) -> None:
    arguments = _ground_jump_arguments(tmp_path) + ["--gantry_policy_rehearsal"]
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        _parse_args()


def test_ground_jump_goal_sequence_uses_manifest_range(monkeypatch, tmp_path: Path) -> None:
    arguments = _ground_jump_arguments(tmp_path)
    monkeypatch.setattr(sys, "argv", arguments)
    args = _parse_args()

    assert [goal.dx for goal in args.ground_goals] == pytest.approx([0.0, 0.1, -0.1])
    assert all((goal.dy, goal.dyaw, goal.roll, goal.pitch) == (0.0, 0.0, 0.0, 0.0) for goal in args.ground_goals)

    arguments[arguments.index("--goal_sequence") + 1] = "0.1001"
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit):
        _parse_args()


def test_ground_jump_allows_two_jump_session_duration_but_keeps_a_ceiling(monkeypatch, tmp_path: Path) -> None:
    arguments = _ground_jump_arguments(tmp_path)
    arguments[arguments.index("--duration") + 1] = "45"
    monkeypatch.setattr(sys, "argv", arguments)

    assert _parse_args().duration == pytest.approx(45.0)

    arguments[arguments.index("--duration") + 1] = "61"
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit):
        _parse_args()


def test_interactive_goal_reader_never_blocks_control_thread() -> None:
    release_read = threading.Event()

    class _BlockingInput:
        def readline(self) -> str:
            release_read.wait(timeout=1.0)
            return "0.1\n"

    reader = _InteractiveGoalReader(_BlockingInput())
    started_at = time.monotonic()
    reader.request()

    assert time.monotonic() - started_at < 0.1
    assert reader.poll() is None
    release_read.set()
    deadline = time.monotonic() + 1.0
    result = None
    while result is None and time.monotonic() < deadline:
        result = reader.poll()
        time.sleep(0.001)

    assert result == "0.1"
    assert _parse_ground_goal(result, (-0.1, 0.1)) == JumpGoal(0.1, 0.0, 0.0)


def test_interactive_goal_reader_treats_eof_as_unavailable_not_q(capsys) -> None:
    reader = _InteractiveGoalReader(SimpleNamespace(readline=lambda: ""))
    reader.request()
    deadline = time.monotonic() + 1.0
    while not reader.eof and time.monotonic() < deadline:
        assert reader.poll() is None
        time.sleep(0.001)

    assert reader.eof
    assert capsys.readouterr().out.count("STDIN EOF: no interactive input available") == 1
    with pytest.raises(RuntimeError, match="unavailable"):
        reader.request()


def test_interactive_goal_reader_returns_explicit_q() -> None:
    reader = _InteractiveGoalReader(SimpleNamespace(readline=lambda: "q\n"))
    reader.request()
    deadline = time.monotonic() + 1.0
    result = None
    while result is None and time.monotonic() < deadline:
        result = reader.poll()
        time.sleep(0.001)

    assert result == "q"
    assert not reader.eof


def test_ground_session_locks_and_clears_pending_goal_after_latched_abort(capsys) -> None:
    operator = _GantryRehearsalOperator(JumpGoal(0.1, 0.0, 0.0))
    fsm = SimpleNamespace(abort_latched=False)

    assert not _lock_ground_session_after_abort(fsm, operator, False)
    assert operator.pending_goal == JumpGoal(0.1, 0.0, 0.0)
    fsm.abort_latched = True
    assert _lock_ground_session_after_abort(fsm, operator, False)
    assert _lock_ground_session_after_abort(fsm, operator, True)

    assert operator.pending_goal is None
    assert capsys.readouterr().out.count("SESSION LOCKED after latched abort; B to exit") == 1


@pytest.mark.parametrize(
    "invalid_arguments",
    [
        ["--entry_mode", "passive"],
        ["--exit_mode", "native_walkrun"],
        ["--duration", "14"],
        ["--effort_scale", "0.11"],
        ["--goal_pos_x", "0.001"],
    ],
)
def test_gantry_policy_rehearsal_cli_rejects_relaxed_contract(
    monkeypatch,
    tmp_path: Path,
    invalid_arguments: list[str],
) -> None:
    arguments = [
        "run_fsm_g1.py",
        "enp131s0",
        "--gantry_policy_rehearsal",
        "--enable_control",
        "--acknowledge_contactless_rehearsal",
        "--entry_mode",
        "gantry_standup",
        "--exit_mode",
        "passive",
        "--duration",
        "15",
        "--effort_scale",
        "0.1",
        "--shadow_admission",
        str(tmp_path / "admission.json"),
        "--rehearsal_log",
        str(tmp_path / "rehearsal.npz"),
    ]
    option = invalid_arguments[0]
    if option in arguments:
        option_index = arguments.index(option)
        del arguments[option_index : option_index + 2]
    arguments.extend(invalid_arguments)
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        _parse_args()


def test_shadow_admission_requires_exact_logs_and_artifact_digests(tmp_path: Path) -> None:
    manifest_path = tmp_path / "deploy_manifest.json"
    policy_path = tmp_path / "policy.onnx"
    admission_path = tmp_path / "admission.json"
    manifest_path.write_bytes(b"accepted manifest")
    policy_path.write_bytes(b"accepted policy")

    logs = []
    for index, goal_pos_x in enumerate((-0.1, 0.0, 0.1)):
        log_path = tmp_path / f"shadow_{index}.npz"
        log_path.write_bytes(f"shadow {goal_pos_x}".encode())
        logs.append(
            {
                "path": str(log_path),
                "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                "steps": 152,
                "unique_feedback_ticks": 152,
                "goal_pos_x_m": goal_pos_x,
                "inference_maximum_ms": 2.0,
                "feedback_age_maximum_ms": 3.0,
                "body_tilt_maximum_deg": 4.0,
                "joint_speed_maximum_rad_s": 0.1,
                "projected_torque_maximum_fraction": 0.6,
            }
        )
    admission = {
        "schema_version": "1.0",
        "read_only_shadow_admission": True,
        "authorizes_motor_control": False,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "logs": logs,
    }
    admission_path.write_text(json.dumps(admission), encoding="utf-8")

    _verify_shadow_admission(admission_path, manifest_path, policy_path, _manifest())

    Path(logs[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from its admitted digest"):
        _verify_shadow_admission(admission_path, manifest_path, policy_path, _manifest())


def test_native_walkrun_exit_requires_gantry_and_hold_duration(monkeypatch) -> None:
    base = [
        "run_fsm_g1.py",
        "enp131s0",
        "--enable_control",
        "--exit_mode",
        "native_walkrun",
        "--effort_scale",
        "0.5",
    ]
    monkeypatch.setattr(sys, "argv", base + ["--entry_mode", "passive", "--duration", "8"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--entry_mode", "gantry_standup", "--duration", "4"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--entry_mode", "gantry_standup", "--duration", "8"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--entry_mode", "native_walkrun_gantry", "--duration", "4"])
    with pytest.raises(SystemExit):
        _parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--entry_mode", "native_walkrun_gantry", "--duration", "8"])

    args = _parse_args()

    assert args.exit_mode == "native_walkrun"


def test_remote_b_bit_decoding() -> None:
    neutral = bytearray(40)
    pressed = neutral.copy()
    pressed[3] = 0x02
    different_button = neutral.copy()
    different_button[3] = 0x01

    assert not _remote_b_pressed(neutral)
    assert _remote_b_pressed(pressed)
    assert not _remote_b_pressed(different_button)
    assert not _remote_b_pressed(bytes(39))


def test_remote_gantry_rehearsal_button_decoding_and_operator_mapping() -> None:
    neutral = bytes(40)
    goal = JumpGoal(0.0, 0.0, 0.0)
    operator = _GantryRehearsalOperator(goal)

    for mask, decoder, intent_name in (
        (0x01, _remote_a_pressed, "request_start"),
        (0x08, _remote_y_pressed, "confirm"),
        (0x02, _remote_b_pressed, "abort"),
    ):
        remote = bytearray(neutral)
        remote[3] = mask
        assert decoder(remote)
        operator.update(bytes(remote))
        assert getattr(operator, intent_name)
        operator.update(neutral)
        assert not getattr(operator, intent_name)

    assert operator.pending_goal is goal
    assert not _remote_a_pressed(bytes(39))
    assert not _remote_y_pressed(bytes(39))


def test_remote_activation_chord_requires_l1_and_r1() -> None:
    neutral = bytearray(40)
    l1_only = neutral.copy()
    l1_only[2] = 0x02
    r1_only = neutral.copy()
    r1_only[2] = 0x01
    both = neutral.copy()
    both[2] = 0x03

    assert not _remote_activation_pressed(neutral)
    assert not _remote_activation_pressed(l1_only)
    assert not _remote_activation_pressed(r1_only)
    assert _remote_activation_pressed(both)
    assert not _remote_activation_pressed(bytes(39))


def test_read_and_play_audio_cue(tmp_path: Path, monkeypatch) -> None:
    wav_path = tmp_path / "jump_mode.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(bytes(40_000))
    pcm_data, duration_s = _read_audio_cue(wav_path)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.sleep", lambda duration: None)

    class _AudioClient:
        def __init__(self):
            self.chunks = []
            self.stopped = False

        def PlayStream(self, app_name: str, stream_id: str, chunk: bytes):
            self.chunks.append((app_name, stream_id, chunk))
            return 0, None

        def PlayStop(self, app_name: str) -> int:
            self.stopped = True
            return 0

    audio_client = _AudioClient()
    _play_audio_cue(audio_client, pcm_data, duration_s)

    assert duration_s == pytest.approx(1.25)
    assert [len(chunk) for _, _, chunk in audio_client.chunks] == [32_000, 8_000]
    assert audio_client.stopped


def test_inactive_policy_always_refuses_inference() -> None:
    with pytest.raises(RuntimeError, match="must not invoke"):
        _InactivePolicy()(np.zeros(326))


def test_remote_activation_requires_hold_then_release(monkeypatch) -> None:
    class _Clock:
        def __init__(self):
            self.current = 100.0

        def monotonic(self) -> float:
            self.current += 0.005
            return self.current

    clock = _Clock()

    class _SequenceStateBuffer:
        def __init__(self):
            self.calls = 0

        def snapshot(self) -> FeedbackSnapshot:
            self.calls += 1
            snapshot = _snapshot(clock.current)
            remote = bytearray(40)
            if 2 <= self.calls <= 4:
                remote[2] = 0x03
            object.__setattr__(snapshot, "wireless_remote", bytes(remote))
            return snapshot

    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", clock.monotonic)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.sleep", lambda duration: None)
    state_buffer = _SequenceStateBuffer()

    _wait_for_remote_activation(state_buffer, _manifest(), hold_s=0.01, timeout_s=10.0)

    assert state_buffer.calls == 5


def test_b_cancels_remote_activation() -> None:
    snapshot = _snapshot(time.monotonic())
    remote = bytearray(40)
    remote[3] = 0x02
    object.__setattr__(snapshot, "wireless_remote", bytes(remote))

    with pytest.raises(SafetyFault, match="cancelled by B"):
        _wait_for_remote_activation(_FakeStateBuffer(snapshot), _manifest(), hold_s=1.0, timeout_s=1.0)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda snapshot: setattr(snapshot, "received_at", snapshot.received_at - 0.1), "feedback stale"),
        (
            lambda snapshot: setattr(
                snapshot,
                "imu_quaternion",
                np.asarray((math.cos(math.radians(15.0)), math.sin(math.radians(15.0)), 0.0, 0.0)),
            ),
            "body tilt",
        ),
        (lambda snapshot: snapshot.imu_gyroscope.__setitem__(0, 6.1), "base angular speed"),
        (lambda snapshot: snapshot.joint_velocities.__setitem__(0, 4.1), "joint speed"),
        (lambda snapshot: snapshot.joint_positions.__setitem__(0, 1.11), "joint position"),
        (lambda snapshot: setattr(snapshot, "maximum_temperature_c", 81), "motor temperature"),
    ],
)
def test_feedback_faults(mutation, expected: str) -> None:
    now = 100.0
    mutable_snapshot = SimpleNamespace(**_snapshot(now).__dict__)
    mutation(mutable_snapshot)

    reason = _feedback_fault(mutable_snapshot, _manifest(), now)

    assert reason is not None
    assert expected in reason


def test_feedback_accepts_nominal_snapshot() -> None:
    now = 100.0
    assert _feedback_fault(_snapshot(now), _manifest(), now) is None


def test_feedback_accepts_measurement_outside_narrower_target_clip() -> None:
    now = 100.0
    snapshot = _snapshot(now)
    snapshot.joint_positions[0] = 1.05

    assert _feedback_fault(snapshot, _manifest(), now) is None


def test_ground_dynamic_feedback_limits_allow_jump_envelope_only() -> None:
    now = 100.0
    manifest = _manifest()
    snapshot = _snapshot(now)
    snapshot.joint_velocities[0] = 24.9
    snapshot.joint_positions[1] = 1.119
    snapshot.imu_gyroscope[0] = 11.9
    object.__setattr__(
        snapshot,
        "imu_quaternion",
        np.asarray((math.cos(math.radians(22.0)), math.sin(math.radians(22.0)), 0.0, 0.0)),
    )

    assert (
        _feedback_fault(
            snapshot,
            manifest,
            now,
            _active_feedback_limits(manifest, JumpControllerState.JUMP, ground_jump=True, rehearsal_escalated=False),
        )
        is None
    )
    assert "body tilt" in (_feedback_fault(snapshot, manifest, now, FeedbackLimits()) or "")
    assert (
        _active_feedback_limits(manifest, JumpControllerState.ARMED, ground_jump=True, rehearsal_escalated=False)
        == FeedbackLimits()
    )
    assert _active_target_rate_limit(
        JumpControllerState.ARMED, ground_jump=True, rehearsal_unlimited_slew=False
    ) == pytest.approx(_TARGET_RATE_LIMIT_RAD_S)
    assert _active_target_rate_limit(JumpControllerState.JUMP, ground_jump=True, rehearsal_unlimited_slew=False) is None
    assert (
        _active_target_rate_limit(JumpControllerState.SETTLE, ground_jump=True, rehearsal_unlimited_slew=False) is None
    )
    assert _active_target_rate_limit(
        JumpControllerState.GOTO_START, ground_jump=True, rehearsal_unlimited_slew=False
    ) == pytest.approx(_TARGET_RATE_LIMIT_RAD_S)
    assert _active_target_rate_limit(
        JumpControllerState.STAND, ground_jump=True, rehearsal_unlimited_slew=False
    ) == pytest.approx(_TARGET_RATE_LIMIT_RAD_S)
    rehearsal_limits = _active_feedback_limits(
        manifest, JumpControllerState.JUMP, ground_jump=False, rehearsal_escalated=True
    )
    ground_limits = _active_feedback_limits(
        manifest, JumpControllerState.JUMP, ground_jump=True, rehearsal_escalated=False
    )
    assert rehearsal_limits.body_tilt_limit_rad == pytest.approx(ground_limits.body_tilt_limit_rad)
    assert rehearsal_limits.base_angular_speed_limit_rad_s == pytest.approx(
        ground_limits.base_angular_speed_limit_rad_s
    )
    assert rehearsal_limits.joint_position_margin_rad == pytest.approx(ground_limits.joint_position_margin_rad)
    np.testing.assert_array_equal(rehearsal_limits.joint_speed_limit_rad_s, ground_limits.joint_speed_limit_rad_s)
    assert (
        _active_feedback_limits(manifest, JumpControllerState.SETTLE, ground_jump=False, rehearsal_escalated=True)
        == FeedbackLimits()
    )
    assert _active_target_rate_limit(JumpControllerState.JUMP, ground_jump=False, rehearsal_unlimited_slew=True) is None
    assert (
        _active_target_rate_limit(JumpControllerState.SETTLE, ground_jump=False, rehearsal_unlimited_slew=True) is None
    )


class _FakeStateBuffer:
    def __init__(self, snapshot: FeedbackSnapshot):
        self._snapshot_value = snapshot

    def snapshot(self) -> FeedbackSnapshot:
        return self._snapshot_value


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def Write(self, message) -> bool:
        self.messages.append(message)
        return True


class _FakeCrc:
    def Crc(self, message) -> int:
        return 123


def _low_state() -> SimpleNamespace:
    motors = [
        SimpleNamespace(q=float(index), dq=0.1 * index, tau_est=-0.2 * index, temperature=[30, 31])
        for index in range(35)
    ]
    return SimpleNamespace(
        crc=123,
        motor_state=motors,
        imu_state=SimpleNamespace(
            quaternion=[1.0, 0.0, 0.0, 0.0],
            gyroscope=[0.0, 0.0, 0.0],
        ),
        wireless_remote=bytes(40),
        tick=42,
        mode_pr=0,
        mode_machine=4,
    )


def test_state_buffer_maps_joint_torque_estimates_and_rejects_bad_packets() -> None:
    state_buffer = _StateBuffer(_manifest(), _FakeCrc())
    state = _low_state()

    state_buffer.update(state)
    snapshot = state_buffer.snapshot()

    assert snapshot is not None
    np.testing.assert_array_equal(snapshot.joint_positions, np.arange(23, dtype=np.float64))
    np.testing.assert_allclose(snapshot.joint_torque_estimates, -0.2 * np.arange(23), rtol=0.0, atol=1.0e-12)
    assert snapshot.maximum_temperature_c == 31
    assert state_buffer.valid_packets == 1

    state.crc = 0
    state_buffer.update(state)
    assert state_buffer.crc_errors == 1
    state.crc = 123
    state.motor_state[0].tau_est = math.nan
    state_buffer.update(state)
    assert state_buffer.invalid_packets == 1


def _low_command():
    motors = [SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0) for _ in range(35)]
    return SimpleNamespace(mode_pr=0, mode_machine=0, motor_cmd=motors, crc=0)


def test_shadow_target_applies_torque_projection_without_publishing() -> None:
    class _Runtime:
        def transform_action(self, raw_action: np.ndarray) -> np.ndarray:
            return raw_action.copy()

    manifest = replace(_manifest(), effort_limit_ratio=np.full(23, 0.6))
    snapshot = _snapshot(time.monotonic())

    requested, projected, unprojected_torque, projected_torque, effort_ratio = _project_shadow_target(
        _Runtime(),
        np.ones(23),
        manifest,
        snapshot,
        effort_scale=1.0,
    )

    np.testing.assert_array_equal(requested, np.ones(23))
    np.testing.assert_array_equal(unprojected_torque, np.full(23, 40.0))
    np.testing.assert_allclose(projected, np.full(23, 0.15), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(projected_torque, np.full(23, 6.0), rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(effort_ratio, np.full(23, 0.6))


def test_complete_policy_shadow_creates_diagnostic_log_without_publisher(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Clock:
        def __init__(self):
            self.current = 100.0

        def monotonic(self) -> float:
            self.current += 1.0e-5
            return self.current

        def perf_counter(self) -> float:
            self.current += 5.0e-5
            return self.current

        def sleep(self, duration_s: float) -> None:
            self.current += max(duration_s, 0.0)

    clock = _Clock()

    class _LiveStateBuffer:
        valid_packets = 100
        crc_errors = 0
        invalid_packets = 0

        def __init__(self):
            self.tick = 0

        def snapshot(self) -> FeedbackSnapshot:
            self.tick += 1
            snapshot = _snapshot(clock.current)
            object.__setattr__(snapshot, "tick", self.tick)
            object.__setattr__(snapshot, "joint_positions", np.full(23, self.tick * 1.0e-4))
            return snapshot

    class _Runtime:
        def __init__(self, *args, **kwargs):
            self.steps = 0
            self._delayed_action = np.zeros(23)
            self.trigger_joint_positions = None

        @property
        def done(self) -> bool:
            return self.steps >= 3

        @property
        def delayed_action(self) -> np.ndarray:
            return self._delayed_action.copy()

        def arm(self, *args, **kwargs) -> None:
            pass

        def trigger(self, *args, **kwargs) -> None:
            self.trigger_joint_positions = np.asarray(args[2]).copy()

        def step(self, *args, **kwargs) -> np.ndarray:
            if self.steps == 0:
                np.testing.assert_array_equal(args[0], self.trigger_joint_positions)
            self.steps += 1
            return np.full(326, self.steps, dtype=np.float32)

        def transform_action(self, raw_action: np.ndarray) -> np.ndarray:
            self._delayed_action = raw_action.copy()
            return np.full(23, 0.5)

    class _Policy:
        def __init__(self, *args, **kwargs):
            pass

        def warm_up(self) -> None:
            pass

        def __call__(self, observation: np.ndarray) -> np.ndarray:
            return np.full(23, 0.25)

    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.JumpGoalRuntime", _Runtime)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.OnnxPolicy", _Policy)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", clock.monotonic)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.perf_counter", clock.perf_counter)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.sleep", clock.sleep)
    manifest_path = tmp_path / "deploy_manifest.json"
    policy_path = tmp_path / "policy.onnx"
    manifest_path.write_text("{}", encoding="utf-8")
    policy_path.write_bytes(b"policy")
    log_path = tmp_path / "shadow.npz"

    report = _run_shadow_policy_episode(
        manifest_path,
        policy_path,
        _manifest(),
        _LiveStateBuffer(),
        log_path,
        goal_pos_x=0.0,
        goal_pos_y=0.0,
        goal_yaw=0.0,
        goal_roll=0.0,
        goal_pitch=0.0,
        effort_scale=0.7,
    )

    assert report.steps == 3
    assert report.torque_projection_steps == 3
    assert report.maximum_torque_fraction == pytest.approx(0.7)
    assert report.log_path == log_path
    with np.load(log_path, allow_pickle=False) as log:
        assert log["observation"].shape == (3, 326)
        assert log["projected_target"].shape == (3, 23)
        assert log["tick"].tolist() == [2, 3, 4]
        metadata = json.loads(log["metadata_json"].item())
    assert metadata["read_only"] is True
    assert metadata["command_publisher_created"] is False

    monkeypatch.setattr(shadow_validator, "JumpGoalRuntime", _Runtime)
    summary = shadow_validator.validate_shadow_log(
        log_path,
        manifest_path,
        policy_path,
        _manifest(),
        _Policy(),
        expected_steps=3,
    )
    assert summary.steps == 3
    assert summary.goal_pos_x_m == pytest.approx(0.0)
    assert summary.projected_torque_maximum_fraction == pytest.approx(0.7)

    with np.load(log_path, allow_pickle=False) as source_log:
        tampered_values = {name: source_log[name].copy() for name in source_log.files}
    tampered_values["projected_target"][0, 0] += 0.01
    tampered_path = tmp_path / "tampered_shadow.npz"
    np.savez_compressed(tampered_path, **tampered_values)
    with pytest.raises(ValueError, match="projected_torque replay mismatch"):
        shadow_validator.validate_shadow_log(
            tampered_path,
            manifest_path,
            policy_path,
            _manifest(),
            _Policy(),
            expected_steps=3,
        )

    with pytest.raises(ValueError, match="will not be overwritten"):
        _run_shadow_policy_episode(
            manifest_path,
            policy_path,
            _manifest(),
            _LiveStateBuffer(),
            log_path,
            goal_pos_x=0.0,
            goal_pos_y=0.0,
            goal_yaw=0.0,
            goal_roll=0.0,
            goal_pitch=0.0,
            effort_scale=0.7,
        )


def test_publish_rate_limits_first_position_command(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", lambda: now)
    snapshot = _snapshot(now)
    publisher = _FakePublisher()
    command = _low_command()
    robot = _G1Robot(_manifest(), _FakeStateBuffer(snapshot), publisher, command, _FakeCrc())
    robot.command_joint_position_target(np.full(23, 0.5), np.full(23, 100.0), np.full(23, 2.0))

    robot.publish(np.zeros(23), effort_scale=1.0)

    assert len(publisher.messages) == 1
    assert command.crc == 123
    assert command.mode_machine == 4
    assert command.motor_cmd[0].mode == 1
    assert command.motor_cmd[0].q == pytest.approx(_TARGET_RATE_LIMIT_RAD_S * _FAST_DT)
    assert command.motor_cmd[0].kp == pytest.approx(100.0)


def test_publish_timestamps_after_copying_concurrent_feedback(monkeypatch) -> None:
    class _Clock:
        def __init__(self):
            self.current = 100.0

        def monotonic(self) -> float:
            return self.current

    clock = _Clock()

    class _ConcurrentStateBuffer:
        def snapshot(self) -> FeedbackSnapshot:
            clock.current += 0.001
            return _snapshot(clock.current)

    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", clock.monotonic)
    command = _low_command()
    publisher = _FakePublisher()
    robot = _G1Robot(_manifest(), _ConcurrentStateBuffer(), publisher, command, _FakeCrc())
    robot.command_joint_position_target(np.zeros(23), np.ones(23), np.ones(23))

    robot.publish(np.zeros(23), effort_scale=1.0)

    assert len(publisher.messages) == 1


def test_rehearsal_recorder_logs_only_accepted_command_and_never_overwrites(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Clock:
        def __init__(self):
            self.current = 100.0

        def monotonic(self) -> float:
            self.current += _FAST_DT
            return self.current

    clock = _Clock()
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", clock.monotonic)
    snapshot = _snapshot(clock.current)
    state_buffer = _FakeStateBuffer(snapshot)
    state_buffer.valid_packets = 1
    state_buffer.crc_errors = 0
    state_buffer.invalid_packets = 0
    robot = _G1Robot(_manifest(), state_buffer, _FakePublisher(), _low_command(), _FakeCrc())
    robot.command_joint_position_target(np.zeros(23), np.full(23, 4.0), np.full(23, 0.5))
    manifest_path = tmp_path / "deploy_manifest.json"
    policy_path = tmp_path / "policy.onnx"
    admission_path = tmp_path / "admission.json"
    manifest_path.write_bytes(b"manifest")
    policy_path.write_bytes(b"policy")
    admission_path.write_bytes(b"admission")
    log_path = tmp_path / "rehearsal.npz"
    recorder = _RehearsalRecorder(
        log_path,
        manifest_path,
        policy_path,
        admission_path,
        _manifest(),
        JumpGoal(0.0, 0.0, 0.0),
        0.1,
    )

    robot.publish(np.zeros(23), effort_scale=0.1)
    recorder.record(
        robot,
        SimpleNamespace(state=JumpControllerState.JUMP, episode_step=4),
        np.zeros(23),
    )
    recorder.write(True, "completed", state_buffer)

    with np.load(log_path, allow_pickle=False) as log:
        metadata = json.loads(log["metadata_json"].item())
        assert log["command_target"].shape == (1, 23)
        assert log["fsm_state"].tolist() == ["JUMP"]
        assert log["episode_step"].tolist() == [4]
        assert metadata["mode"] == "contactless_gantry_policy_rehearsal"
        assert metadata["ground_jump_authorized"] is False
        assert metadata["contact_sensor_available"] is False
        assert metadata["effort_scale"] == pytest.approx(0.1)
        assert metadata["ratio_envelope_exceeded_by_position_bound"] == {
            "total_count": 0,
            "per_joint_count": {},
            "per_joint_maximum_excess_nm": {},
        }

    with pytest.raises(ValueError, match="Cannot create rehearsal log"):
        recorder.write(True, "completed", state_buffer)


def test_ground_recorder_writes_authorization_and_per_jump_outcome(monkeypatch, tmp_path: Path) -> None:
    class _Clock:
        def __init__(self):
            self.current = 100.0

        def monotonic(self) -> float:
            self.current += _FAST_DT
            return self.current

    clock = _Clock()
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", clock.monotonic)
    state_buffer = _FakeStateBuffer(_snapshot(clock.current))
    state_buffer.valid_packets = 1
    state_buffer.crc_errors = 0
    state_buffer.invalid_packets = 0
    manifest = _leg_manifest()
    stand_gains, balance_config = _ground_stand_configuration(manifest)
    robot = _G1Robot(manifest, state_buffer, _FakePublisher(), _low_command(), _FakeCrc())
    robot.command_joint_position_target(np.zeros(23), np.full(23, 4.0), np.full(23, 0.5))
    manifest_path = tmp_path / "deploy_manifest.json"
    policy_path = tmp_path / "policy.onnx"
    admission_path = tmp_path / "admission.json"
    manifest_path.write_bytes(b"manifest")
    policy_path.write_bytes(b"policy")
    admission_path.write_bytes(b"admission")
    recorder = _GroundJumpRecorder(
        tmp_path / "ground.npz",
        manifest_path,
        policy_path,
        admission_path,
        manifest,
        0.3,
        stand_gains,
        balance_config,
        1.0,
        True,
        0.5,
        4.0,
        0.05,
        0.02,
    )
    fsm = SimpleNamespace(
        state=JumpControllerState.JUMP,
        episode_step=1,
        latched_goal=JumpGoal(0.1, 0.0, 0.0),
        abort_latched=True,
        latched_abort_reason="joint limit exceeded",
        latched_abort_reasons={"joint limit exceeded"},
        joint_limit_touches={"left_knee_joint": 0.0003},
    )
    for state, step in (
        (JumpControllerState.JUMP, 1),
        (JumpControllerState.SETTLE, 152),
        (JumpControllerState.STAND, 152),
    ):
        fsm.state = state
        fsm.episode_step = step
        robot.publish(np.zeros(23), effort_scale=0.3)
        recorder.record(robot, fsm, np.zeros(23))
    recorder.write(True, "completed", state_buffer)

    with np.load(recorder.log_path, allow_pickle=False) as log:
        metadata = json.loads(log["metadata_json"].item())
        np.testing.assert_array_equal(log["maximum_joint_speed_fraction"], 0.0)
        np.testing.assert_array_equal(log["body_tilt_rad"], 0.0)
        assert log["maximum_joint_speed_fraction"].shape == (3,)
    assert metadata["mode"] == "unmeasured_ground_jump"
    assert metadata["ground_jump_authorized"] is True
    assert metadata["contact_sensor_available"] is False
    assert metadata["ratio_envelope_exceeded_by_position_bound"] == {
        "total_count": 0,
        "per_joint_count": {},
        "per_joint_maximum_excess_nm": {},
    }
    assert metadata["ground_stand_controller"] == {
        "ankle_damping_nm_s_per_rad": 7.0,
        "ankle_stiffness_nm_per_rad": 80.0,
        "balance_initial_pitch_integral_rad_s": 0.2,
        "balance_initial_roll_integral_rad_s": 0.0,
        "balance_integral_enabled": True,
        "balance_target_pitch_rad": 0.0,
        "balance_target_roll_rad": 0.0,
        "balance_target_entry_duration_s": 1.0,
        "joint_limit_abort_margin_rad": 0.02,
        "policy_stand_after_jump": True,
        "settle_duration_s": 0.5,
        "settle_timeout_s": 4.0,
        "settle_joint_velocity_tolerance_rad_s": 0.05,
        "damping_overrides_nm_s_per_rad": {
            "left_hip_pitch_joint": 5.0,
            "left_knee_joint": 5.0,
            "right_hip_pitch_joint": 5.0,
            "right_knee_joint": 5.0,
        },
        "stand_entry_duration_s": 1.0,
        "stiffness_overrides_nm_per_rad": {
            "left_hip_pitch_joint": 200.0,
            "left_knee_joint": 200.0,
            "right_hip_pitch_joint": 200.0,
            "right_knee_joint": 200.0,
        },
    }
    assert metadata["jumps"] == [
        {
            "goal": {"pitch": 0.0, "pos_x": 0.1, "pos_y": 0.0, "roll": 0.0, "yaw": 0.0},
            "start_policy_step": 0,
            "end_policy_step": 152,
            "outcome": "latched_abort_settled",
            "latched_abort_reason": "joint limit exceeded",
            "latched_abort_reasons": ["joint limit exceeded"],
            "max_tilt_rad": 0.0,
            "max_estimated_torque_fraction": 0.0,
            "joint_limit_touches_rad": {"left_knee_joint": 0.0003},
        }
    ]


def test_publish_can_use_unrated_torque_projected_dynamic_target(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", lambda: now)
    snapshot = _snapshot(now)
    command = _low_command()
    publisher = _FakePublisher()
    robot = _G1Robot(_manifest(), _FakeStateBuffer(snapshot), publisher, command, _FakeCrc())
    robot.command_joint_position_target(np.full(23, 0.5), np.full(23, 10.0), np.zeros(23))

    robot.publish(np.zeros(23), effort_scale=1.0, target_rate_limit_rad_s=None)

    assert len(publisher.messages) == 1
    assert command.motor_cmd[0].q == pytest.approx(0.5)


def test_publish_refuses_estimated_torque_above_scaled_limit(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", lambda: now)
    snapshot = _snapshot(now)
    snapshot.joint_velocities[0] = 3.0
    publisher = _FakePublisher()
    robot = _G1Robot(_manifest(), _FakeStateBuffer(snapshot), publisher, _low_command(), _FakeCrc())
    robot.command_joint_position_target(np.zeros(23), np.zeros(23), np.full(23, 10.0))

    with pytest.raises(SafetyFault, match=r"position=.*damping=.*error=.*velocity="):
        robot.publish(np.zeros(23), effort_scale=0.7)

    assert publisher.messages == []
    assert robot.maximum_estimated_torque[0] == pytest.approx(30.0)
    assert robot.peak_position_torque[0] == pytest.approx(0.0)
    assert robot.peak_damping_torque[0] == pytest.approx(-30.0)


def test_publish_projects_position_target_to_manifest_torque_envelope(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", lambda: now)
    manifest = replace(_manifest(), effort_limit_ratio=np.full(23, 0.6))
    snapshot = _snapshot(now)
    command = _low_command()
    publisher = _FakePublisher()
    robot = _G1Robot(manifest, _FakeStateBuffer(snapshot), publisher, command, _FakeCrc())
    robot.command_joint_position_target(np.full(23, 0.5), np.full(23, 5_000.0), np.zeros(23))

    robot.publish(np.zeros(23), effort_scale=1.0)

    assert len(publisher.messages) == 1
    assert command.motor_cmd[0].q == pytest.approx(0.6 * 10.0 / 5_000.0)
    assert robot.maximum_estimated_torque[0] == pytest.approx(6.0)


def test_publish_allows_position_bound_ratio_excess_below_physical_limit(monkeypatch, capsys) -> None:
    now = 100.0
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", lambda: now)
    manifest = replace(_manifest(), effort_limit_ratio=np.full(23, 0.6))
    snapshot = _snapshot(now)
    snapshot.joint_positions[0] = 0.95
    snapshot.joint_velocities[0] = 2.0
    publisher = _FakePublisher()
    robot = _G1Robot(manifest, _FakeStateBuffer(snapshot), publisher, _low_command(), _FakeCrc())
    damping = np.zeros(23)
    damping[0] = 4.0
    robot.command_joint_position_target(np.ones(23), np.full(23, 10.0), damping)

    robot.publish(np.zeros(23), effort_scale=1.0, target_rate_limit_rad_s=None)
    robot.publish(np.zeros(23), effort_scale=1.0, target_rate_limit_rad_s=None)

    assert len(publisher.messages) == 2
    assert robot.ratio_envelope_position_bound_counts[0] == 2
    assert robot.ratio_envelope_position_bound_maximum_excess[0] == pytest.approx(1.5)
    warning = capsys.readouterr().out
    assert warning.count("ratio envelope exceeded by position bound for joint_0") == 1
    assert "excess=1.50 N m" in warning


def test_publish_refuses_when_target_bounds_exceed_physical_torque_limit(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", lambda: now)
    manifest = replace(_manifest(), effort_limit_ratio=np.full(23, 0.6))
    snapshot = _snapshot(now)
    snapshot.joint_positions[0] = 0.95
    snapshot.joint_velocities[0] = 2.0
    publisher = _FakePublisher()
    robot = _G1Robot(manifest, _FakeStateBuffer(snapshot), publisher, _low_command(), _FakeCrc())
    damping = np.zeros(23)
    damping[0] = 6.0
    robot.command_joint_position_target(np.ones(23), np.full(23, 10.0), damping)

    with pytest.raises(SafetyFault, match="target bounds exceed the physical torque limit"):
        robot.publish(np.zeros(23), effort_scale=1.0, target_rate_limit_rad_s=None)

    assert publisher.messages == []


class _FakeControlRobot:
    def __init__(self):
        self._manifest = _manifest()
        self._base_target = np.zeros(23)
        self._stiffness = np.ones(23)
        self._damping = np.ones(23)
        self.takeover_writes = 0
        self.damping_writes = 0

    def publish_takeover_damping(self) -> None:
        self.takeover_writes += 1

    @property
    def command_base_target(self) -> np.ndarray:
        return self._base_target.copy()

    @property
    def command_stiffness(self) -> np.ndarray:
        return self._stiffness.copy()

    @property
    def command_damping(self) -> np.ndarray:
        return self._damping.copy()

    def set_damping(self) -> None:
        pass

    def publish(self, balance_offset: np.ndarray, effort_scale: float) -> None:
        self.damping_writes += 1

    def command_joint_position_target(
        self,
        target: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ) -> None:
        self._base_target = target.copy()
        self._stiffness = stiffness.copy()
        self._damping = damping.copy()


class _FakeControlStateBuffer:
    crc_errors = 0
    invalid_packets = 0

    def __init__(self, snapshot: FeedbackSnapshot):
        self._snapshot_value = snapshot

    def snapshot(self) -> FeedbackSnapshot:
        return self._snapshot_value


class _FakeBridgeLocoClient:
    def __init__(self, fsm_ids: list[int]):
        self.fsm_ids = list(fsm_ids)
        self.velocity_requests = []
        self.fsm_requests = []

    def GetFsmId(self) -> tuple[int, int]:
        if len(self.fsm_ids) > 1:
            return 0, self.fsm_ids.pop(0)
        return 0, self.fsm_ids[0]

    def SetVelocity(self, velocity_x: float, velocity_y: float, yaw_velocity: float, duration: float) -> int:
        self.velocity_requests.append((velocity_x, velocity_y, yaw_velocity, duration))
        return 0

    def SetFsmId(self, fsm_id: int) -> int:
        self.fsm_requests.append(fsm_id)
        return 0


@pytest.mark.parametrize("native_fsm_id", [500, 801])
def test_bridge_native_stand_to_passive_preloads_command(native_fsm_id: int) -> None:
    now = time.monotonic()
    robot = _FakeControlRobot()
    state_buffer = _FakeControlStateBuffer(_snapshot(now))
    loco_client = _FakeBridgeLocoClient([native_fsm_id, 1])

    _bridge_native_stand_to_passive(loco_client, robot, state_buffer, _manifest(), native_fsm_id)

    assert robot.takeover_writes == 1
    assert loco_client.velocity_requests == [(0.0, 0.0, 0.0, 1.0)]
    assert loco_client.fsm_requests == [1]


def test_bridge_native_stand_to_passive_refuses_wrong_fsm() -> None:
    now = time.monotonic()
    robot = _FakeControlRobot()
    state_buffer = _FakeControlStateBuffer(_snapshot(now))
    loco_client = _FakeBridgeLocoClient([1])

    with pytest.raises(SafetyFault, match="expected FSM ID 500"):
        _bridge_native_stand_to_passive(loco_client, robot, state_buffer, _manifest(), 500)

    assert robot.takeover_writes == 0
    assert loco_client.velocity_requests == []
    assert loco_client.fsm_requests == []


def test_bridge_native_stand_to_passive_honors_b_abort() -> None:
    now = time.monotonic()
    snapshot = _snapshot(now)
    remote = bytearray(40)
    remote[3] = 0x02
    object.__setattr__(snapshot, "wireless_remote", bytes(remote))
    robot = _FakeControlRobot()
    state_buffer = _FakeControlStateBuffer(snapshot)
    loco_client = _FakeBridgeLocoClient([500])

    with pytest.raises(SafetyFault, match="cancelled by B"):
        _bridge_native_stand_to_passive(loco_client, robot, state_buffer, _manifest(), 500)

    assert robot.takeover_writes == 0
    assert loco_client.velocity_requests == []
    assert loco_client.fsm_requests == []


def test_bridge_native_stand_to_passive_honors_b_during_transition() -> None:
    now = time.monotonic()
    neutral_snapshot = _snapshot(now)
    abort_snapshot = _snapshot(now)
    remote = bytearray(40)
    remote[3] = 0x02
    object.__setattr__(abort_snapshot, "wireless_remote", bytes(remote))

    class _SequenceStateBuffer:
        def __init__(self):
            self.snapshots = [neutral_snapshot, abort_snapshot]

        def snapshot(self) -> FeedbackSnapshot:
            if len(self.snapshots) > 1:
                return self.snapshots.pop(0)
            return self.snapshots[0]

    robot = _FakeControlRobot()
    loco_client = _FakeBridgeLocoClient([500, 1])

    with pytest.raises(SafetyFault, match="reached PASSIVE but was cancelled by B"):
        _bridge_native_stand_to_passive(loco_client, robot, _SequenceStateBuffer(), _manifest(), 500)

    assert robot.takeover_writes == 1
    assert loco_client.fsm_requests == [1]


class _FakeLocoClient:
    def __init__(self, switch_code: int):
        self.switch_code = switch_code
        self.restore_calls = 0
        self.restore_modes = []
        self.fsm_requests = []

    def GetFsmId(self) -> tuple[int, int]:
        return 0, 1

    def SwitchToUserCtrl(self) -> int:
        return self.switch_code

    def SwitchToInternalCtrl(self, mode) -> int:
        self.restore_calls += 1
        self.restore_modes.append(mode)
        return 0

    def SetFsmId(self, fsm_id: int) -> int:
        self.fsm_requests.append(fsm_id)
        return 0


class _FakeFsm:
    policy_dt = 0.02

    def __init__(self):
        self.faults = []

    def report_fault(self, reason: str) -> None:
        self.faults.append(reason)


def test_gantry_rehearsal_control_requires_recorder() -> None:
    with pytest.raises(ValueError, match="must be enabled together"):
        _run_control(
            _FakeControlRobot(),
            _GantryRehearsalOperator(JumpGoal(0.0, 0.0, 0.0)),
            _FakeFsm(),
            _FakeControlStateBuffer(_snapshot(time.monotonic())),
            _FakeLocoClient(switch_code=0),
            internal_passive_mode=1,
            duration_s=15.0,
            effort_scale=0.1,
            gantry_policy_rehearsal=True,
        )


def test_gantry_rehearsal_refuses_a_before_stabilization(monkeypatch) -> None:
    class _Clock:
        def __init__(self):
            self.current = 100.0

        def monotonic(self) -> float:
            self.current += 0.001
            return self.current

    class _SequenceStateBuffer:
        crc_errors = 0
        invalid_packets = 0

        def __init__(self, clock: _Clock):
            self._clock = clock
            self._calls = 0

        def snapshot(self) -> FeedbackSnapshot:
            self._calls += 1
            snapshot = _snapshot(self._clock.current)
            if self._calls > 1:
                remote = bytearray(40)
                remote[3] = 0x01
                object.__setattr__(snapshot, "wireless_remote", bytes(remote))
            return snapshot

    class _StandFsm:
        policy_dt = 0.02
        episode_step = 0
        last_report = None

        def __init__(self):
            self.state = JumpControllerState.PASSIVE

        def enable(self) -> None:
            self.state = JumpControllerState.STAND

        def step(self) -> None:
            pass

        def report_fault(self, reason: str) -> None:
            self.last_report = reason

    class _Recorder:
        def record(self, robot, fsm, balance_offset: np.ndarray) -> None:
            del robot, fsm, balance_offset

    clock = _Clock()
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", clock.monotonic)
    robot = _FakeControlRobot()

    success, reason = _run_control(
        robot,
        _GantryRehearsalOperator(JumpGoal(0.0, 0.0, 0.0)),
        _StandFsm(),
        _SequenceStateBuffer(clock),
        _FakeLocoClient(switch_code=0),
        internal_passive_mode=1,
        duration_s=15.0,
        effort_scale=0.1,
        gantry_policy_rehearsal=True,
        rehearsal_recorder=_Recorder(),
    )

    assert not success
    assert "A or Y was pressed before REHEARSAL READY" in reason


def test_gantry_rehearsal_control_completes_once_and_restores_passive(monkeypatch) -> None:
    class _Clock:
        def __init__(self):
            self.current = 100.0

        def monotonic(self) -> float:
            self.current += 0.001
            return self.current

    class _LiveStateBuffer:
        crc_errors = 0
        invalid_packets = 0

        def __init__(self, clock: _Clock):
            self._clock = clock

        def snapshot(self) -> FeedbackSnapshot:
            return _snapshot(self._clock.current)

    class _RehearsalFsm:
        policy_dt = 0.02

        def __init__(self):
            self.state = JumpControllerState.PASSIVE
            self.last_report = None
            self.episode_step = 0
            self._steps = 0

        def enable(self) -> None:
            self.state = JumpControllerState.STAND

        def step(self) -> None:
            sequence = (
                JumpControllerState.STAND,
                JumpControllerState.GOTO_START,
                JumpControllerState.ARMED,
                JumpControllerState.JUMP,
                JumpControllerState.SETTLE,
                JumpControllerState.STAND,
            )
            self.state = sequence[min(self._steps, len(sequence) - 1)]
            self._steps += 1
            if self.state is JumpControllerState.JUMP:
                self.episode_step = 1

        def update_balance(self, dt: float) -> np.ndarray:
            assert dt == pytest.approx(_FAST_DT)
            return np.zeros(23)

        def report_fault(self, reason: str) -> None:
            self.last_report = reason

    class _Recorder:
        def __init__(self):
            self.samples = 0

        def record(self, robot, fsm, balance_offset: np.ndarray) -> None:
            del robot, fsm
            assert balance_offset.shape == (23,)
            self.samples += 1

    clock = _Clock()
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", clock.monotonic)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.sleep", lambda duration: None)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1._sleep_until", lambda deadline: None)
    robot = _FakeControlRobot()
    loco_client = _FakeLocoClient(switch_code=0)
    recorder = _Recorder()

    success, reason = _run_control(
        robot,
        _GantryRehearsalOperator(JumpGoal(0.0, 0.0, 0.0)),
        _RehearsalFsm(),
        _LiveStateBuffer(clock),
        loco_client,
        internal_passive_mode=1,
        duration_s=15.0,
        effort_scale=0.1,
        gantry_policy_rehearsal=True,
        rehearsal_recorder=recorder,
    )

    assert success
    assert reason == "one gantry policy rehearsal completed and settled"
    assert recorder.samples > 1
    assert loco_client.restore_modes == [1]
    assert loco_client.fsm_requests == [1]


def test_failed_user_control_switch_still_requests_native_passive(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", lambda: now)
    robot = _FakeControlRobot()
    state_buffer = _FakeControlStateBuffer(_snapshot(now))
    loco_client = _FakeLocoClient(switch_code=42)
    fsm = _FakeFsm()

    success, reason = _run_control(
        robot,
        SimpleNamespace(abort=False),
        fsm,
        state_buffer,
        loco_client,
        internal_passive_mode=1,
        duration_s=1.0,
        effort_scale=0.7,
    )

    assert not success
    assert "SwitchToUserCtrl returned code 42" in reason
    assert robot.takeover_writes == 1
    assert robot.damping_writes == 1
    assert loco_client.restore_calls == 1
    assert loco_client.restore_modes == [1]
    assert loco_client.fsm_requests == [1]


def test_restore_internal_control_verifies_expected_fsm() -> None:
    class _RestoreLocoClient:
        def __init__(self):
            self.restore_modes = []

        def SwitchToInternalCtrl(self, mode) -> int:
            self.restore_modes.append(mode)
            return 0

        def GetFsmId(self) -> tuple[int, int]:
            return 0, 801

    robot = _FakeControlRobot()
    loco_client = _RestoreLocoClient()

    restored, return_code, fsm_id = _restore_internal_control(
        robot,
        loco_client,
        internal_mode=2,
        expected_fsm_id=801,
    )

    assert restored
    assert return_code == 0
    assert fsm_id == 801
    assert loco_client.restore_modes == [2]


def test_restore_passive_rejects_success_code_when_fsm_stays_walkrun(monkeypatch) -> None:
    class _IgnoredPassiveLocoClient:
        def __init__(self):
            self.fsm_requests = []

        def SwitchToInternalCtrl(self, mode) -> int:
            del mode
            return 0

        def SetFsmId(self, fsm_id: int) -> int:
            self.fsm_requests.append(fsm_id)
            return 0

        def GetFsmId(self) -> tuple[int, int]:
            return 0, 801

    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.sleep", lambda duration: None)
    robot = _FakeControlRobot()
    loco_client = _IgnoredPassiveLocoClient()

    restored, return_code, fsm_id = _restore_internal_control(
        robot,
        loco_client,
        internal_mode=1,
        expected_fsm_id=1,
        request_fsm_id=1,
        timeout_s=0.001,
    )

    assert not restored
    assert return_code == 0
    assert fsm_id == 801
    assert loco_client.fsm_requests == [1] * 5


def test_successful_control_returns_to_verified_native_walkrun(monkeypatch) -> None:
    class _Clock:
        value = 100.0

        def monotonic(self) -> float:
            self.value += 0.001
            return self.value

    class _LiveStateBuffer:
        crc_errors = 0
        invalid_packets = 0

        def __init__(self, clock: _Clock):
            self._clock = clock

        def snapshot(self) -> FeedbackSnapshot:
            return _snapshot(self._clock.value)

    class _RunningFsm:
        policy_dt = 0.02

        def __init__(self):
            from scripts.g1_jump_deploy.fsm.jump_fsm import JumpControllerState

            self.state = JumpControllerState.STAND
            self.calibrations = []

        def set_balance_target_attitude(self, target_roll: float, target_pitch: float) -> None:
            self.calibrations.append((target_roll, target_pitch))

        def enable(self) -> None:
            pass

        def step(self) -> None:
            pass

        def update_balance(self, dt: float) -> np.ndarray:
            del dt
            return np.zeros(23)

    class _RoundTripLocoClient:
        def __init__(self):
            self.fsm_id = 1
            self.restore_modes = []
            self.velocity_requests = []

        def GetFsmId(self) -> tuple[int, int]:
            return 0, self.fsm_id

        def SwitchToUserCtrl(self) -> int:
            return 0

        def SetVelocity(self, velocity_x: float, velocity_y: float, yaw_velocity: float, duration: float) -> int:
            self.velocity_requests.append((velocity_x, velocity_y, yaw_velocity, duration))
            return 0

        def SwitchToInternalCtrl(self, mode) -> int:
            self.restore_modes.append(mode)
            self.fsm_id = 801 if mode == 2 else 1
            return 0

    clock = _Clock()
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.monotonic", clock.monotonic)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1.time.sleep", lambda duration: None)
    monkeypatch.setattr("scripts.g1_jump_deploy.hardware.run_fsm_g1._sleep_until", lambda deadline: None)
    robot = _FakeControlRobot()
    robot._manifest = _leg_manifest()
    loco_client = _RoundTripLocoClient()
    fsm = _RunningFsm()

    success, reason = _run_control(
        robot,
        SimpleNamespace(abort=False),
        fsm,
        _LiveStateBuffer(clock),
        loco_client,
        internal_passive_mode=1,
        duration_s=0.01,
        effort_scale=0.5,
        success_internal_mode=2,
        success_internal_fsm_id=801,
        success_handoff_position=np.zeros(23),
        recalibrate_balance_after_user_switch=True,
    )

    assert success
    assert reason == "requested stand duration completed"
    assert loco_client.velocity_requests == [(0.0, 0.0, 0.0, 1.0)]
    assert loco_client.restore_modes == [2]
    assert fsm.calibrations == pytest.approx([(0.0, 0.0)])

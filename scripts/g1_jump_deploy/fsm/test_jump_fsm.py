# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from scripts.g1_jump_deploy.control.balance import BalanceControllerConfig
from scripts.g1_jump_deploy.fsm import (
    JumpControllerConfig,
    JumpControllerFSM,
    JumpControllerState,
    JumpGoal,
)

_EPISODE_STEPS = 152
_FLIGHT_START_STEP = 43
_FAST_DT = 0.002
_FAST_STEPS_PER_CONTROL = 10
_TERMS = [
    {"name": "joint_pos", "offset": 0, "step_dim": 23, "history": 4, "total": 92},
    {"name": "joint_vel", "offset": 92, "step_dim": 23, "history": 4, "total": 92},
    {"name": "goal_remaining", "offset": 184, "step_dim": 3, "history": 4, "total": 12},
    {"name": "base_ang_vel", "offset": 196, "step_dim": 3, "history": 4, "total": 12},
    {"name": "projected_gravity", "offset": 208, "step_dim": 3, "history": 4, "total": 12},
    {"name": "last_action", "offset": 220, "step_dim": 23, "history": 1, "total": 23},
    {"name": "goal_command", "offset": 243, "step_dim": 7, "history": 1, "total": 7},
    {"name": "reference_preview", "offset": 250, "step_dim": 70, "history": 1, "total": 70},
    {"name": "jump_phase", "offset": 320, "step_dim": 6, "history": 1, "total": 6},
]
_JOINT_NAMES = (
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
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


def _quaternion_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    roll_half = 0.5 * roll
    pitch_half = 0.5 * pitch
    return np.asarray(
        (
            math.cos(roll_half) * math.cos(pitch_half),
            math.sin(roll_half) * math.cos(pitch_half),
            math.cos(roll_half) * math.sin(pitch_half),
            -math.sin(roll_half) * math.sin(pitch_half),
        )
    )


@dataclass
class _Command:
    target: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray


class _FakeRobot:
    def __init__(self, default_position: np.ndarray):
        self.joint_positions = default_position.copy()
        self.joint_velocities = np.zeros(23)
        self.base_angular_velocity = np.zeros(3)
        balance_config = BalanceControllerConfig()
        self.imu_quaternion = _quaternion_from_roll_pitch(balance_config.target_roll, math.radians(4.0))
        self.odometry_position = np.asarray((0.0, 0.0, 0.8))
        self.odometry_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
        self.foot_contact_forces = np.asarray((100.0, 100.0))
        self.joint_limit_violations = np.zeros(23, dtype=np.bool_)
        self.feedback_stale = False
        self.control_deadline_missed = False
        self.follow_commands = True
        self.commands: list[_Command] = []

    def command_joint_position_target(
        self,
        target: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ) -> None:
        self.commands.append(_Command(target.copy(), stiffness.copy(), damping.copy()))
        if self.follow_commands:
            self.joint_positions = target.copy()
            self.joint_velocities = np.zeros_like(target)


class _FakeOperator:
    def __init__(self):
        self.pending_goal: JumpGoal | None = None
        self.request_start = False
        self.confirm = False
        self.abort = False


class _ZeroPolicy:
    def __init__(self):
        self.observations: list[np.ndarray] = []

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(observation.copy())
        return np.zeros(23, dtype=np.float32)


class _GoalConditionedPolicy(_ZeroPolicy):
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(observation.copy())
        return np.full(23, observation[243], dtype=np.float32)


class _ForbiddenContactFeedback:
    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        raise RuntimeError("foot contact feedback was read")


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    default_position = np.linspace(-0.2, 0.2, 23, dtype=np.float32)
    jump_stiffness = np.full(23, 40.0, dtype=np.float32)
    jump_damping = np.full(23, 10.0, dtype=np.float32)
    ankle_indices = [index for index, name in enumerate(_JOINT_NAMES) if "ankle" in name]
    jump_stiffness[ankle_indices] = 20.0
    jump_damping[ankle_indices] = 2.0

    preview = np.zeros((_EPISODE_STEPS, 70), dtype=np.float32)
    phase = np.zeros((_EPISODE_STEPS, 6), dtype=np.float32)
    phase[:6, 0] = 1.0
    phase[6:19, 1] = 1.0
    phase[19:_FLIGHT_START_STEP, 2] = 1.0
    phase[_FLIGHT_START_STEP:70, 3] = 1.0
    phase[70:100, 4] = 1.0
    phase[100:, 5] = 1.0
    preview_path = tmp_path / "reference_preview_152x70.npy"
    phase_path = tmp_path / "jump_phase_152x6.npy"
    np.save(preview_path, preview, allow_pickle=False)
    np.save(phase_path, phase, allow_pickle=False)

    manifest = {
        "schema_version": "1.2",
        "task": "test",
        "checkpoint": "/tmp/test.pt",
        "exported_at": "2026-01-01T00:00:00Z",
        "joint_order_matches_constants": True,
        "control": {
            "policy_dt": 0.02,
            "policy_hz": 50.0,
            "sim_dt": 0.002,
            "decimation": 10,
            "episode_steps": _EPISODE_STEPS,
            "episode_duration_s": 3.033333333333333,
        },
        "joints": {
            "names": list(_JOINT_NAMES),
            "unitree_sdk2_slots": list(range(23)),
            "default_pos": default_position.tolist(),
            "default_vel": [0.0] * 23,
        },
        "observation": {
            "total_dim": 326,
            "history_order": "oldest_first",
            "history_layout": "history_major",
            "terms": _TERMS,
        },
        "action": {
            "dim": 23,
            "scale": [0.1] * 23,
            "offset": default_position.tolist(),
            "filter_alpha": [1.0] * 23,
            "delay_steps": {"min": 0, "max": 0},
            "clip": [[float(position - 1.0), float(position + 1.0)] for position in default_position],
            "formula": "q_target = alpha*clip(offset + scale*a_delayed) + (1-alpha)*q_target_prev",
        },
        "actuators": {
            "type": "implicit_pd",
            "stiffness": jump_stiffness.tolist(),
            "damping": jump_damping.tolist(),
            "effort_limit": [100.0] * 23,
            "velocity_limit": [30.0] * 23,
            "armature": [0.01] * 23,
        },
        "reference": {
            "fps": 30.0,
            "num_frames": 91,
            "phase_names": ["IDLE", "CROUCH", "TAKEOFF", "FLIGHT", "LAND", "STAND"],
            "phase_frame_ranges": [[0, 6], [6, 19], [19, 26], [26, 43], [43, 60], [60, 91]],
            "preview_offsets_frames": [1, 4, 7],
            "source_csv": "/tmp/reference.csv",
            "source_sha256": "0" * 64,
            "root_frame0": {"pos": [0.0, 0.0, 0.8], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
        },
        "goal": {
            "quat_order": "xyzw",
            "ranges": {
                "pos_x": [-0.3, 1.0],
                "pos_y": [-0.6, 0.6],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-1.1, 1.1],
            },
            "flight_freeze": {"enabled": True, "freeze_prob_trained": 0.8, "drift_std_trained": 0.005},
        },
        "tables": {"reference_preview": preview_path.name, "jump_phase": phase_path.name},
    }
    manifest_path = tmp_path / "deploy_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, default_position, jump_stiffness, jump_damping


@pytest.fixture
def controller_config() -> JumpControllerConfig:
    return JumpControllerConfig(
        stand_entry_duration_s=0.1,
        goto_start_duration_s=0.1,
        goto_start_timeout_s=0.2,
        armed_timeout_s=0.1,
        settle_duration_s=0.1,
        stiffness_slew_per_s=1_000.0,
        damping_slew_per_s=100.0,
    )


@pytest.fixture
def controller(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy]:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    policy = _ZeroPolicy()
    fsm = JumpControllerFSM(manifest_path, robot, operator, policy, config=controller_config)
    return fsm, robot, operator, policy


def _control_step(fsm: JumpControllerFSM) -> np.ndarray:
    fsm.step()
    return np.stack([fsm.update_balance(_FAST_DT) for _ in range(_FAST_STEPS_PER_CONTROL)])


def _pulse_start(fsm: JumpControllerFSM, operator: _FakeOperator, goal: JumpGoal = JumpGoal(0.4, 0.0, 0.0)):
    operator.pending_goal = goal
    operator.request_start = True
    _control_step(fsm)
    operator.request_start = False


def _advance_to(fsm: JumpControllerFSM, state: JumpControllerState, maximum_steps: int = 500) -> None:
    for _ in range(maximum_steps):
        if fsm.state is state:
            return
        _control_step(fsm)
    pytest.fail(f"Controller did not reach {state}; current state is {fsm.state}: {fsm.last_report}")


def _prepare_armed(fsm: JumpControllerFSM, operator: _FakeOperator) -> None:
    fsm.enable()
    _control_step(fsm)
    _pulse_start(fsm, operator)
    _advance_to(fsm, JumpControllerState.ARMED)


def _confirm(fsm: JumpControllerFSM, operator: _FakeOperator) -> np.ndarray:
    operator.confirm = True
    offsets = _control_step(fsm)
    operator.confirm = False
    assert fsm.state is JumpControllerState.JUMP
    return offsets


def test_full_nominal_path_and_explicit_stand_gains(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
):
    fsm, _, operator, policy = controller
    _, _, jump_stiffness, jump_damping = bundle
    ankle_mask = np.asarray(["ankle" in name for name in _JOINT_NAMES])
    np.testing.assert_allclose(fsm.stand_stiffness[ankle_mask], 80.0)
    np.testing.assert_allclose(fsm.stand_damping[ankle_mask], 5.0)
    np.testing.assert_array_equal(fsm.stand_stiffness[~ankle_mask], jump_stiffness[~ankle_mask])
    np.testing.assert_array_equal(fsm.stand_damping[~ankle_mask], jump_damping[~ankle_mask])

    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    _advance_to(fsm, JumpControllerState.SETTLE)
    _advance_to(fsm, JumpControllerState.STAND)

    assert fsm.transition_history == [
        JumpControllerState.PASSIVE,
        JumpControllerState.STAND,
        JumpControllerState.GOTO_START,
        JumpControllerState.ARMED,
        JumpControllerState.JUMP,
        JumpControllerState.SETTLE,
        JumpControllerState.STAND,
    ]
    assert len(policy.observations) == _EPISODE_STEPS


def test_policy_terminal_return_finishes_jump_at_validated_stand_target(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    policy = _GoalConditionedPolicy()
    config = replace(controller_config, policy_terminal_return_steps=10)
    fsm = JumpControllerFSM(manifest_path, robot, operator, policy, config=config)

    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    _advance_to(fsm, JumpControllerState.SETTLE)

    np.testing.assert_allclose(robot.commands[-1].target, default_position, atol=1.0e-7)
    np.testing.assert_allclose(robot.commands[-1].stiffness, fsm.stand_stiffness)
    np.testing.assert_allclose(robot.commands[-1].damping, fsm.stand_damping)
    assert fsm.balance_gate == pytest.approx(1.0)


def test_policy_native_stand_continues_final_reference_after_jump(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    policy = _ZeroPolicy()
    config = replace(controller_config, policy_stand_after_jump=True)
    fsm = JumpControllerFSM(manifest_path, robot, operator, policy, config=config)
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    _advance_to(fsm, JumpControllerState.SETTLE)
    jump_observation_count = len(policy.observations)
    _advance_to(fsm, JumpControllerState.STAND)

    assert jump_observation_count == _EPISODE_STEPS
    assert len(policy.observations) > jump_observation_count
    assert fsm.policy_stand_active
    assert fsm.balance_gate == 0.0
    np.testing.assert_array_equal(policy.observations[-1][320:326], (0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

    observation_count = len(policy.observations)
    offsets = _control_step(fsm)

    assert len(policy.observations) == observation_count + 1
    assert fsm.state is JumpControllerState.STAND
    assert fsm.policy_stand_active
    np.testing.assert_array_equal(offsets, 0.0)


def test_policy_native_stand_recalibrates_before_normalizing_for_next_goal(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    policy = _GoalConditionedPolicy()
    config = replace(controller_config, policy_stand_after_jump=True)
    fsm = JumpControllerFSM(manifest_path, robot, operator, policy, config=config)
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    _advance_to(fsm, JumpControllerState.SETTLE)
    _advance_to(fsm, JumpControllerState.STAND)
    robot.imu_quaternion = _quaternion_from_roll_pitch(math.radians(-2.0), 0.0)

    _pulse_start(fsm, operator, JumpGoal(-0.2, 0.0, 0.0))
    _advance_to(fsm, JumpControllerState.ARMED)

    np.testing.assert_allclose(robot.commands[-1].target, default_position)
    assert fsm.balance_gate == 1.0

    _confirm(fsm, operator)

    assert fsm.state is JumpControllerState.JUMP
    assert fsm.transition_history[-3:] == [
        JumpControllerState.GOTO_START,
        JumpControllerState.ARMED,
        JumpControllerState.JUMP,
    ]


def test_policy_native_stand_can_prepare_next_goal_without_default_normalization(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    policy = _GoalConditionedPolicy()
    config = replace(
        controller_config,
        policy_stand_after_jump=True,
        policy_stand_retrigger_prepare_duration_s=0.04,
    )
    fsm = JumpControllerFSM(manifest_path, robot, operator, policy, config=config)
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    _advance_to(fsm, JumpControllerState.SETTLE)
    _advance_to(fsm, JumpControllerState.STAND)

    _pulse_start(fsm, operator, JumpGoal(-0.2, 0.0, 0.0))
    _advance_to(fsm, JumpControllerState.ARMED)

    assert fsm.policy_prepared
    assert fsm.balance_gate == 0.0
    assert not np.allclose(robot.commands[-1].target, default_position)


def test_policy_native_stand_can_hold_until_direct_retrigger_confirmation(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.6"
    manifest["joints"]["position_limits"] = [[-2.0, 2.0]] * 23
    manifest["goal"]["retrigger_indicator"] = {
        "mode": "goal_command_z",
        "fresh_value": 0.0,
        "retrigger_value": 0.25,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    policy = _GoalConditionedPolicy()
    config = replace(
        controller_config,
        policy_stand_after_jump=True,
        policy_stand_direct_retrigger=True,
    )
    fsm = JumpControllerFSM(manifest_path, robot, operator, policy, config=config)
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    _advance_to(fsm, JumpControllerState.SETTLE)
    _advance_to(fsm, JumpControllerState.STAND)

    old_episode_observation_count = len(policy.observations)
    _pulse_start(fsm, operator, JumpGoal(-0.2, 0.0, 0.0))
    _advance_to(fsm, JumpControllerState.ARMED)

    assert not fsm.policy_prepared
    assert fsm.balance_gate == 0.0
    assert len(policy.observations) > old_episode_observation_count
    np.testing.assert_array_equal(
        policy.observations[-1][320:326],
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert policy.observations[-1][243] == pytest.approx(0.4)

    _confirm(fsm, operator)

    assert fsm.state is JumpControllerState.JUMP
    assert policy.observations[-1][243] == pytest.approx(-0.2)
    assert policy.observations[-1][245] == pytest.approx(0.25)
    np.testing.assert_allclose(robot.commands[-1].target, default_position - 0.02, atol=1.0e-7)
    np.testing.assert_array_equal(
        policy.observations[-1][320:326],
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert fsm.transition_history[-3:] == [
        JumpControllerState.GOTO_START,
        JumpControllerState.ARMED,
        JumpControllerState.JUMP,
    ]


def test_first_jump_observation_measures_goal_height_from_floor(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
) -> None:
    fsm, robot, operator, policy = controller
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)

    _control_step(fsm)

    goal_remaining = policy.observations[0][184:196].reshape(4, 3)
    expected = np.asarray((0.4, 0.0, -robot.odometry_position[2]), dtype=np.float32)
    np.testing.assert_allclose(goal_remaining, np.tile(expected, (4, 1)), rtol=0.0, atol=1.0e-7)


def test_confirmation_runs_first_policy_step_without_stand_hold(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
) -> None:
    fsm, robot, operator, policy = controller
    _prepare_armed(fsm, operator)
    command_count = len(robot.commands)

    _confirm(fsm, operator)

    assert len(robot.commands) == command_count + 1
    assert len(policy.observations) == 1
    assert fsm.episode_step == 1
    assert fsm.phase_clock_history == [0]


@pytest.mark.parametrize("goal_x", (-0.1, 0.0, 0.1))
def test_goal_conditioned_policy_preparation_preserves_command_and_clock(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
    goal_x: float,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    policy = _GoalConditionedPolicy()
    config = replace(
        controller_config,
        policy_prepare_duration_s=0.06,
        goto_start_timeout_s=0.24,
        armed_timeout_s=0.2,
    )
    fsm = JumpControllerFSM(manifest_path, robot, operator, policy, config=config)
    goal = JumpGoal(goal_x, 0.0, 0.0)

    fsm.enable()
    _control_step(fsm)
    _pulse_start(fsm, operator, goal)
    _advance_to(fsm, JumpControllerState.ARMED)

    assert fsm.policy_prepared
    assert fsm.episode_step == 0
    assert fsm.phase_clock_history == []
    assert len(policy.observations) == 3
    for observation in policy.observations:
        assert observation[243] == pytest.approx(goal_x)
        np.testing.assert_array_equal(observation[320:326], (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    _control_step(fsm)
    prepared_action = np.full(23, goal_x, dtype=np.float32)
    np.testing.assert_allclose(policy.observations[-1][220:243], prepared_action)
    observation_count = len(policy.observations)
    _confirm(fsm, operator)

    assert len(policy.observations) == observation_count + 1
    assert policy.observations[-1][243] == pytest.approx(goal_x)
    np.testing.assert_allclose(policy.observations[-1][220:243], prepared_action)
    np.testing.assert_array_equal(policy.observations[-1][320:326], (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert fsm.episode_step == 1
    assert fsm.phase_clock_history == [0]


def test_goal_conditioned_preparation_supports_two_distinct_jump_cycles(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    policy = _GoalConditionedPolicy()
    config = replace(
        controller_config,
        policy_prepare_duration_s=0.04,
        goto_start_timeout_s=0.22,
        armed_timeout_s=0.2,
    )
    fsm = JumpControllerFSM(manifest_path, robot, operator, policy, config=config)
    fsm.enable()
    _control_step(fsm)

    observation_ranges = []
    for goal_x in (0.1, -0.1):
        first_observation = len(policy.observations)
        _pulse_start(fsm, operator, JumpGoal(goal_x, 0.0, 0.0))
        _advance_to(fsm, JumpControllerState.ARMED)
        assert fsm.policy_prepared
        _confirm(fsm, operator)
        _advance_to(fsm, JumpControllerState.SETTLE)
        _advance_to(fsm, JumpControllerState.STAND)
        observation_ranges.append((first_observation, len(policy.observations), goal_x))

    for start, end, goal_x in observation_ranges:
        assert end - start > _EPISODE_STEPS
        np.testing.assert_allclose(
            [observation[243] for observation in policy.observations[start:end]],
            goal_x,
        )
    assert fsm.transition_history.count(JumpControllerState.JUMP) == 2
    assert fsm.transition_history.count(JumpControllerState.SETTLE) == 2


def test_stand_entry_installs_full_gains_and_seeds_target_from_measurement(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
):
    fsm, robot, _, _ = controller
    _, default_position, _, _ = bundle
    measured_position = default_position + np.linspace(-0.03, 0.03, len(default_position))
    robot.follow_commands = False
    robot.joint_positions = measured_position.copy()

    fsm.enable()
    first_offsets = _control_step(fsm)

    first_command = robot.commands[-1]
    np.testing.assert_array_equal(first_command.target, measured_position)
    np.testing.assert_array_equal(first_command.stiffness, fsm.stand_stiffness)
    np.testing.assert_array_equal(first_command.damping, fsm.stand_damping)
    assert np.linalg.norm(first_offsets) > 0.0
    assert fsm.balance_gate == 1.0

    _control_step(fsm)

    progress = fsm.policy_dt / controller_config.stand_entry_duration_s
    blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
    expected_target = measured_position + blend * (default_position - measured_position)
    np.testing.assert_allclose(robot.commands[-1].target, expected_target)
    np.testing.assert_array_equal(robot.commands[-1].stiffness, fsm.stand_stiffness)
    np.testing.assert_array_equal(robot.commands[-1].damping, fsm.stand_damping)
    assert fsm.balance_gate == 1.0


def test_stand_can_hold_measured_pose_with_calibrated_balance(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    measured_position = default_position + np.linspace(-0.03, 0.03, len(default_position))
    target_roll = math.radians(1.0)
    target_pitch = math.radians(-6.0)
    robot = _FakeRobot(default_position)
    robot.follow_commands = False
    robot.joint_positions = measured_position.copy()
    robot.imu_quaternion = _quaternion_from_roll_pitch(target_roll, target_pitch)
    config = replace(controller_config, stand_hold_measured_pose=True)
    balance_config = BalanceControllerConfig(
        target_roll=target_roll,
        target_pitch=target_pitch,
        initial_roll_integral=0.0,
        initial_pitch_integral=0.0,
    )
    fsm = JumpControllerFSM(
        manifest_path,
        robot,
        _FakeOperator(),
        _ZeroPolicy(),
        config=config,
        balance_config=balance_config,
    )

    fsm.enable()
    for _ in range(10):
        fsm.step()
        np.testing.assert_array_equal(robot.commands[-1].target, measured_position)
        np.testing.assert_allclose(fsm.update_balance(_FAST_DT), 0.0, rtol=0.0, atol=1.0e-12)


def test_balance_target_can_be_calibrated_only_while_passive(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
) -> None:
    fsm, robot, _, _ = controller
    target_roll = math.radians(2.0)
    target_pitch = math.radians(-3.0)
    robot.imu_quaternion = _quaternion_from_roll_pitch(target_roll, target_pitch)

    fsm.set_balance_target_attitude(target_roll, target_pitch)
    fsm.enable()
    fsm.step()

    np.testing.assert_allclose(fsm.update_balance(_FAST_DT), 0.0, rtol=0.0, atol=1.0e-12)
    with pytest.raises(RuntimeError, match="PASSIVE"):
        fsm.set_balance_target_attitude(target_roll, target_pitch)


def test_jump_gains_crossfade_during_manifest_idle_phase(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
):
    fsm, robot, operator, _ = controller
    manifest_path, _, _, _ = bundle
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase_names = manifest["reference"]["phase_names"]
    idle_frame_range = manifest["reference"]["phase_frame_ranges"][phase_names.index("IDLE")]
    idle_duration_s = (idle_frame_range[1] - idle_frame_range[0]) / manifest["reference"]["fps"]
    idle_end_step = round(idle_duration_s / manifest["control"]["policy_dt"])
    pre_jump_command_start = len(robot.commands)
    _prepare_armed(fsm, operator)

    for command in robot.commands[pre_jump_command_start:]:
        np.testing.assert_array_equal(command.stiffness, fsm.stand_stiffness)
        np.testing.assert_array_equal(command.damping, fsm.stand_damping)
    _confirm(fsm, operator)

    jump_commands = [robot.commands[-1]]
    for _ in range(idle_end_step - 1):
        _control_step(fsm)
        jump_commands.append(robot.commands[-1])

    assert idle_end_step == 10
    assert not np.array_equal(jump_commands[0].stiffness, fsm.jump_stiffness)
    assert not np.array_equal(jump_commands[0].damping, fsm.jump_damping)
    np.testing.assert_array_equal(jump_commands[-1].stiffness, fsm.jump_stiffness)
    np.testing.assert_array_equal(jump_commands[-1].damping, fsm.jump_damping)


def test_single_operator_action_cannot_reach_jump(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
):
    fsm, _, operator, policy = controller
    fsm.enable()
    _control_step(fsm)

    operator.confirm = True
    for _ in range(3):
        _control_step(fsm)
    assert fsm.state is JumpControllerState.STAND

    operator.pending_goal = JumpGoal(0.4, 0.0, 0.0)
    operator.request_start = True
    _advance_to(fsm, JumpControllerState.ARMED)
    _control_step(fsm)
    assert fsm.state is JumpControllerState.ARMED
    assert len(policy.observations) == 0


def test_armed_times_out_to_stand(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
):
    fsm, _, operator, _ = controller
    _prepare_armed(fsm, operator)

    _advance_to(fsm, JumpControllerState.STAND)

    assert "timed out" in (fsm.last_report or "")


def test_out_of_envelope_goal_is_rejected_without_clamping(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
):
    fsm, _, operator, policy = controller
    fsm.enable()
    _control_step(fsm)
    rejected_goal = JumpGoal(1.01, 0.0, 0.0)

    _pulse_start(fsm, operator, rejected_goal)

    assert fsm.state is JumpControllerState.STAND
    assert fsm.latched_goal is None
    assert "pos_x=1.01" in (fsm.last_report or "")
    assert "outside manifest range [-0.3, 1.0]" in (fsm.last_report or "")
    assert len(policy.observations) == 0


@pytest.mark.parametrize(
    ("interlock", "report_name"),
    (
        ("pose", "pose"),
        ("velocity", "joint_velocity"),
        ("feet", "feet_loaded"),
        ("tilt", "body_tilt"),
        ("reporting", "joint_reporting"),
        ("limit", "joint_limits"),
        ("balance", "balance_offset"),
    ),
)
def test_each_prearm_interlock_independently_blocks_armed(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    interlock: str,
    report_name: str,
):
    fsm, robot, operator, _ = controller
    _, default_position, _, _ = bundle
    if interlock == "pose":
        robot.follow_commands = False
        robot.joint_positions = default_position + 0.1
    elif interlock == "velocity":
        robot.follow_commands = False
        robot.joint_velocities[7] = 1.0
    elif interlock == "feet":
        robot.foot_contact_forces[0] = 0.0
    elif interlock == "tilt":
        balance_config = BalanceControllerConfig()
        robot.imu_quaternion = _quaternion_from_roll_pitch(
            balance_config.target_roll, balance_config.target_pitch - math.radians(6.0)
        )
    elif interlock == "reporting":
        robot.follow_commands = False
        robot.joint_positions = default_position[:-1]
        robot.joint_velocities = np.zeros(22)
    elif interlock == "limit":
        robot.joint_limit_violations[7] = True
    elif interlock == "balance":
        robot.follow_commands = False
        robot.imu_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))

    fsm.enable()
    _control_step(fsm)
    _pulse_start(fsm, operator)
    _advance_to(fsm, JumpControllerState.STAND)

    assert JumpControllerState.ARMED not in fsm.transition_history
    assert report_name in (fsm.last_report or "")


def test_balance_target_attitude_passes_prearm_tilt_interlock(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
):
    fsm, robot, operator, _ = controller
    balance_config = BalanceControllerConfig()
    robot.follow_commands = False
    robot.imu_quaternion = _quaternion_from_roll_pitch(balance_config.target_roll, balance_config.target_pitch)

    _prepare_armed(fsm, operator)

    assert fsm.state is JumpControllerState.ARMED
    assert "body_tilt" not in (fsm.last_report or "")


def test_gantry_rehearsal_explicitly_allows_unmeasured_foot_contact(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    robot.foot_contact_forces.fill(0.0)
    operator = _FakeOperator()
    policy = _ZeroPolicy()
    fsm = JumpControllerFSM(
        manifest_path,
        robot,
        operator,
        policy,
        config=replace(
            controller_config,
            contact_safety_mode=JumpControllerConfig.ContactSafetyMode.GANTRY_REHEARSAL,
        ),
    )

    _prepare_armed(fsm, operator)

    assert fsm.state is JumpControllerState.ARMED
    assert "feet_loaded" not in (fsm.last_report or "")

    _confirm(fsm, operator)
    _advance_to(fsm, JumpControllerState.SETTLE)
    _advance_to(fsm, JumpControllerState.STAND)

    assert len(policy.observations) == _EPISODE_STEPS
    assert fsm.transition_history == [
        JumpControllerState.PASSIVE,
        JumpControllerState.STAND,
        JumpControllerState.GOTO_START,
        JumpControllerState.ARMED,
        JumpControllerState.JUMP,
        JumpControllerState.SETTLE,
        JumpControllerState.STAND,
    ]


def test_gantry_rehearsal_post_takeoff_abort_enters_damping_immediately(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    robot.foot_contact_forces.fill(0.0)
    operator = _FakeOperator()
    policy = _ZeroPolicy()
    fsm = JumpControllerFSM(
        manifest_path,
        robot,
        operator,
        policy,
        config=replace(
            controller_config,
            contact_safety_mode=JumpControllerConfig.ContactSafetyMode.GANTRY_REHEARSAL,
        ),
    )
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    while fsm.episode_step < fsm.flight_start_step:
        _control_step(fsm)

    operator.abort = True
    _control_step(fsm)

    assert fsm.state is JumpControllerState.DAMPING
    assert fsm.episode_step == _FLIGHT_START_STEP
    assert len(policy.observations) == _FLIGHT_START_STEP
    assert "Gantry rehearsal aborted" in (fsm.last_report or "")
    np.testing.assert_array_equal(robot.commands[-1].stiffness, 0.0)


def test_unmeasured_ground_never_reads_contact_and_latches_late_abort(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    robot.foot_contact_forces = _ForbiddenContactFeedback()
    operator = _FakeOperator()
    policy = _ZeroPolicy()
    fsm = JumpControllerFSM(
        manifest_path,
        robot,
        operator,
        policy,
        config=replace(
            controller_config,
            contact_safety_mode=JumpControllerConfig.ContactSafetyMode.UNMEASURED_GROUND,
        ),
    )
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    while fsm.episode_step < fsm.flight_start_step:
        _control_step(fsm)

    operator.abort = True
    _control_step(fsm)
    operator.abort = False

    assert fsm.state is JumpControllerState.JUMP
    while fsm.state is JumpControllerState.JUMP:
        _control_step(fsm)

    assert fsm.state is JumpControllerState.DAMPING
    assert fsm.episode_step == _EPISODE_STEPS
    assert len(policy.observations) == _EPISODE_STEPS
    assert fsm.phase_clock_history == list(range(_EPISODE_STEPS))


def test_contact_safety_mode_requires_enum(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    invalid_config = replace(controller_config, contact_safety_mode="gantry_rehearsal")

    with pytest.raises(ValueError, match="ContactSafetyMode"):
        JumpControllerFSM(
            manifest_path,
            _FakeRobot(default_position),
            _FakeOperator(),
            _ZeroPolicy(),
            config=invalid_config,
        )


def test_balance_ankle_offsets_do_not_fail_start_pose_interlock(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
):
    fsm, robot, operator, _ = controller
    _, default_position, _, _ = bundle
    ankle_indices = [index for index, name in enumerate(_JOINT_NAMES) if "ankle" in name]
    robot.follow_commands = False
    robot.joint_positions = default_position.copy()
    robot.joint_positions[ankle_indices] += 0.1
    balance_config = BalanceControllerConfig()
    robot.imu_quaternion = _quaternion_from_roll_pitch(balance_config.target_roll, balance_config.target_pitch)

    _prepare_armed(fsm, operator)

    assert fsm.state is JumpControllerState.ARMED
    assert "pose" not in (fsm.last_report or "")


def test_fsm_constructs_when_manifest_disables_flight_freeze(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
):
    manifest_path, default_position, _, _ = bundle
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["goal"]["flight_freeze"]["enabled"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fsm = JumpControllerFSM(
        manifest_path,
        _FakeRobot(default_position),
        _FakeOperator(),
        _ZeroPolicy(),
        config=controller_config,
    )

    assert fsm.state is JumpControllerState.PASSIVE


def test_jump_has_exact_length_and_monotonic_phase_clock(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
):
    fsm, _, operator, policy = controller
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)

    states_before_step = []
    while fsm.state is JumpControllerState.JUMP:
        states_before_step.append(fsm.state)
        _control_step(fsm)

    assert len(states_before_step) == _EPISODE_STEPS - 1
    assert len(policy.observations) == _EPISODE_STEPS
    assert fsm.episode_step == _EPISODE_STEPS
    assert fsm.phase_clock_history == list(range(_EPISODE_STEPS))
    assert fsm.state is JumpControllerState.SETTLE


def test_balance_and_target_crossfade_together_during_idle_handoff(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
):
    fsm, robot, operator, _ = controller
    _, default_position, _, _ = bundle
    _prepare_armed(fsm, operator)
    first_jump_command = len(robot.commands)
    first_offsets = _confirm(fsm, operator)
    robot.imu_quaternion = _quaternion_from_roll_pitch(0.2, 0.2)

    fast_offsets = [first_offsets]
    while fsm.state is JumpControllerState.JUMP:
        fast_offsets.append(_control_step(fsm))

    jump_commands = robot.commands[first_jump_command : first_jump_command + _EPISODE_STEPS]
    assert len(jump_commands) == _EPISODE_STEPS
    maximum_offsets = np.max(np.abs(np.asarray(fast_offsets)), axis=(1, 2))
    assert np.all(maximum_offsets[:9] > 0.0)
    np.testing.assert_array_equal(maximum_offsets[9:], 0.0)
    for command in jump_commands:
        np.testing.assert_allclose(command.target, default_position, rtol=0.0, atol=1.0e-7)


def test_jump_handoff_blends_can_be_disabled_independently(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
) -> None:
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    policy = _ZeroPolicy()
    config = replace(
        controller_config,
        jump_target_blend_steps=0,
        jump_gain_blend_steps=0,
        jump_balance_blend_steps=0,
    )
    fsm = JumpControllerFSM(manifest_path, robot, operator, policy, config=config)
    _prepare_armed(fsm, operator)

    _confirm(fsm, operator)

    np.testing.assert_array_equal(robot.commands[-1].stiffness, fsm.jump_stiffness)
    np.testing.assert_array_equal(robot.commands[-1].damping, fsm.jump_damping)
    assert fsm.balance_gate == 0.0


def test_balance_contributes_nonzero_in_stand_with_attitude_error(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
):
    fsm, robot, _, _ = controller
    _, default_position, _, _ = bundle
    balance_config = BalanceControllerConfig()
    robot.imu_quaternion = _quaternion_from_roll_pitch(
        balance_config.target_roll + 0.02, balance_config.target_pitch - 0.01
    )
    fsm.enable()

    entry_offsets = _control_step(fsm)
    fast_offsets = _control_step(fsm)

    ankle_indices = [index for index, name in enumerate(_JOINT_NAMES) if "ankle" in name]
    assert np.any(np.abs(entry_offsets[:, ankle_indices]) > 0.0)
    np.testing.assert_allclose(robot.commands[-1].target, default_position)
    assert np.any(np.abs(fast_offsets[:, ankle_indices]) > 0.0)
    assert fsm.balance_gate == 1.0
    assert np.linalg.norm(fsm.balance_ankle_offset) > 0.0


def test_fsm_step_holds_base_target_until_fast_balance_update(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
):
    fsm, robot, _, _ = controller
    _, default_position, _, _ = bundle
    fsm.enable()
    _control_step(fsm)
    integral_before = fsm.balance_integral_error

    fsm.step()

    np.testing.assert_array_equal(fsm.balance_integral_error, integral_before)
    np.testing.assert_allclose(robot.commands[-1].target, default_position)
    balance_offset = fsm.update_balance(_FAST_DT)
    assert np.linalg.norm(balance_offset) > 0.0
    assert not np.array_equal(fsm.balance_integral_error, integral_before)


def test_goto_start_smoothly_blends_from_held_base_to_load_compensated_target(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
):
    fsm, robot, operator, _ = controller
    _, default_position, _, _ = bundle
    fsm.enable()
    _control_step(fsm)
    robot.follow_commands = False
    robot.joint_positions = default_position + 0.03

    _pulse_start(fsm, operator)
    goto_entry_balance_gate = fsm.balance_gate
    _control_step(fsm)

    assert fsm.state is JumpControllerState.GOTO_START
    progress = fsm.policy_dt / controller_config.goto_start_duration_s
    blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
    expected_target = default_position - 0.03 * blend
    ankle_mask = np.asarray(["ankle" in name for name in _JOINT_NAMES])
    expected_target[ankle_mask] = default_position[ankle_mask]
    np.testing.assert_allclose(robot.commands[-1].target, expected_target)
    assert goto_entry_balance_gate == fsm.balance_gate == 1.0


def test_goto_start_compensates_static_load_deflection_outside_balance_loop(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    fsm, robot, operator, _ = controller
    _, default_position, _, _ = bundle
    knee_index = _JOINT_NAMES.index("left_knee_joint")
    ankle_index = _JOINT_NAMES.index("left_ankle_pitch_joint")
    deflection = 0.04
    fsm.enable()
    _control_step(fsm)
    robot.follow_commands = False
    robot.joint_positions[knee_index] -= deflection
    robot.joint_positions[ankle_index] -= deflection

    _pulse_start(fsm, operator)
    _advance_to(fsm, JumpControllerState.ARMED)

    assert robot.commands[-1].target[knee_index] == pytest.approx(default_position[knee_index] + deflection)
    assert robot.commands[-1].target[ankle_index] == pytest.approx(default_position[ankle_index])
    _control_step(fsm)
    assert robot.commands[-1].target[knee_index] == pytest.approx(default_position[knee_index] + deflection)


def test_balance_integral_resets_at_boundaries_but_not_each_step(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
):
    fsm, _, operator, _ = controller
    initial_integral = np.asarray((0.0, BalanceControllerConfig().initial_pitch_integral))
    fsm.enable()
    np.testing.assert_allclose(fsm.balance_integral_error, initial_integral)

    _control_step(fsm)
    after_entry_step = fsm.balance_integral_error
    _control_step(fsm)
    after_first_step = fsm.balance_integral_error
    _control_step(fsm)
    after_second_step = fsm.balance_integral_error
    assert not np.array_equal(after_entry_step, initial_integral)
    assert not np.array_equal(after_first_step, after_entry_step)
    assert not np.array_equal(after_second_step, after_first_step)

    _pulse_start(fsm, operator)
    _advance_to(fsm, JumpControllerState.ARMED)
    _confirm(fsm, operator)
    for _ in range(10):
        _control_step(fsm)
    integral_during_jump = fsm.balance_integral_error
    _control_step(fsm)
    np.testing.assert_array_equal(fsm.balance_integral_error, integral_during_jump)

    operator.abort = True
    _control_step(fsm)
    assert fsm.state is JumpControllerState.DAMPING
    np.testing.assert_allclose(fsm.balance_integral_error, initial_integral)


def test_zero_attitude_error_keeps_steady_state_preload(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
):
    fsm, robot, _, _ = controller
    _, default_position, _, _ = bundle
    balance_config = BalanceControllerConfig()
    robot.imu_quaternion = _quaternion_from_roll_pitch(balance_config.target_roll, balance_config.target_pitch)
    fsm.enable()

    for _ in range(6):
        fast_offsets = _control_step(fsm)

    np.testing.assert_allclose(
        fsm.balance_ankle_offset, (balance_config.initial_pitch_integral, 0.0), rtol=0.0, atol=1.0e-14
    )
    for side in ("left", "right"):
        pitch_index = _JOINT_NAMES.index(f"{side}_ankle_pitch_joint")
        np.testing.assert_allclose(fast_offsets[:, pitch_index], balance_config.initial_pitch_integral)
        assert robot.commands[-1].target[pitch_index] == pytest.approx(default_position[pitch_index])


def test_abort_before_takeoff_reaches_damping_promptly(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
):
    fsm, robot, operator, policy = controller
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    operator.abort = True

    _control_step(fsm)

    assert fsm.state is JumpControllerState.DAMPING
    assert fsm.episode_step == 1
    assert len(policy.observations) == 1
    np.testing.assert_array_equal(robot.commands[-1].stiffness, 0.0)


def test_abort_after_takeoff_finishes_episode_then_applies_damping(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
):
    fsm, robot, operator, policy = controller
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    while fsm.episode_step < fsm.flight_start_step:
        _control_step(fsm)
    assert fsm.episode_step == _FLIGHT_START_STEP

    operator.abort = True
    _control_step(fsm)
    assert fsm.state is JumpControllerState.JUMP
    assert fsm.abort_latched
    while fsm.state is JumpControllerState.JUMP:
        _control_step(fsm)

    assert fsm.state is JumpControllerState.DAMPING
    assert fsm.episode_step == _EPISODE_STEPS
    assert len(policy.observations) == _EPISODE_STEPS
    assert fsm.phase_clock_history == list(range(_EPISODE_STEPS))
    np.testing.assert_array_equal(robot.commands[-1].stiffness, 0.0)
    np.testing.assert_array_equal(robot.commands[-1].damping, fsm.jump_damping)


def test_post_takeoff_abort_damps_on_touchdown_without_stopping_phase_clock(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
):
    fsm, robot, operator, policy = controller
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    while fsm.episode_step < fsm.flight_start_step:
        _control_step(fsm)
    robot.foot_contact_forces[:] = 0.0
    operator.abort = True
    _control_step(fsm)
    operator.abort = False
    while fsm.episode_step < 70:
        _control_step(fsm)

    robot.foot_contact_forces[:] = 100.0
    _control_step(fsm)

    assert fsm.state is JumpControllerState.JUMP
    np.testing.assert_array_equal(robot.commands[-1].stiffness, 0.0)
    while fsm.state is JumpControllerState.JUMP:
        _control_step(fsm)
    assert fsm.state is JumpControllerState.DAMPING
    assert len(policy.observations) == _EPISODE_STEPS
    assert fsm.phase_clock_history == list(range(_EPISODE_STEPS))


def test_gains_are_continuous_across_goto_start_and_settle(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    controller_config: JumpControllerConfig,
):
    fsm, robot, operator, _ = controller
    fsm.enable()
    _control_step(fsm)
    goto_start_index = len(robot.commands) - 1
    _pulse_start(fsm, operator)
    _advance_to(fsm, JumpControllerState.ARMED)
    goto_commands = robot.commands[goto_start_index:]

    _confirm(fsm, operator)
    while fsm.state is JumpControllerState.JUMP:
        _control_step(fsm)
    settle_start_index = len(robot.commands) - 1
    _advance_to(fsm, JumpControllerState.STAND)
    _control_step(fsm)
    settle_commands = robot.commands[settle_start_index:]

    stiffness_limit = controller_config.stiffness_slew_per_s * fsm.policy_dt
    damping_limit = controller_config.damping_slew_per_s * fsm.policy_dt
    for commands in (goto_commands, settle_commands):
        stiffness = np.stack([command.stiffness for command in commands])
        damping = np.stack([command.damping for command in commands])
        assert np.max(np.abs(np.diff(stiffness, axis=0))) <= stiffness_limit + 1.0e-12
        assert np.max(np.abs(np.diff(damping, axis=0))) <= damping_limit + 1.0e-12


def test_settle_waits_for_measured_stand_convergence(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
):
    fsm, robot, operator, _ = controller
    _, default_position, _, _ = bundle
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    _advance_to(fsm, JumpControllerState.SETTLE)

    robot.follow_commands = False
    robot.joint_positions = default_position + 0.2
    robot.joint_velocities.fill(0.8)
    nominal_settle_steps = math.ceil(controller_config.settle_duration_s / fsm.policy_dt)
    for _ in range(nominal_settle_steps):
        _control_step(fsm)

    assert fsm.state is JumpControllerState.SETTLE

    robot.joint_positions = default_position.copy()
    robot.joint_velocities.fill(0.0)
    _control_step(fsm)

    assert fsm.state is JumpControllerState.STAND


def test_settle_timeout_enters_damping(
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
):
    manifest_path, default_position, _, _ = bundle
    robot = _FakeRobot(default_position)
    operator = _FakeOperator()
    fsm = JumpControllerFSM(
        manifest_path,
        robot,
        operator,
        _ZeroPolicy(),
        config=replace(controller_config, settle_timeout_s=0.2),
    )
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    _advance_to(fsm, JumpControllerState.SETTLE)

    robot.follow_commands = False
    robot.joint_positions = default_position + 0.2
    robot.joint_velocities.fill(0.8)
    _advance_to(fsm, JumpControllerState.DAMPING)

    assert fsm.last_report is not None
    assert fsm.last_report.startswith("Settle timed out:")
    assert fsm.last_report.endswith("damping enabled.")


def test_settle_balance_offset_follows_existing_quintic_blend(
    controller: tuple[JumpControllerFSM, _FakeRobot, _FakeOperator, _ZeroPolicy],
    bundle: tuple[Path, np.ndarray, np.ndarray, np.ndarray],
    controller_config: JumpControllerConfig,
):
    fsm, robot, operator, _ = controller
    _, default_position, _, _ = bundle
    _prepare_armed(fsm, operator)
    _confirm(fsm, operator)
    balance_config = BalanceControllerConfig()
    robot.imu_quaternion = _quaternion_from_roll_pitch(balance_config.target_roll, balance_config.target_pitch)
    while fsm.state is JumpControllerState.JUMP:
        final_jump_offsets = _control_step(fsm)
    np.testing.assert_array_equal(final_jump_offsets, 0.0)
    settle_start_index = len(robot.commands)

    settle_fast_offsets = []
    while fsm.state is JumpControllerState.SETTLE:
        settle_fast_offsets.append(_control_step(fsm))

    settle_commands = robot.commands[settle_start_index:]
    pitch_index = _JOINT_NAMES.index("right_ankle_pitch_joint")
    applied_offsets = np.concatenate(settle_fast_offsets)[:, pitch_index]
    progress = np.arange(1, len(settle_commands) + 1) * fsm.policy_dt / controller_config.settle_duration_s
    progress = np.clip(progress, 0.0, 1.0)
    expected_blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
    expected_offsets = np.repeat(
        balance_config.initial_pitch_integral * expected_blend,
        _FAST_STEPS_PER_CONTROL,
    )
    expected_offsets[-_FAST_STEPS_PER_CONTROL:] = 0.0
    for command in settle_commands:
        np.testing.assert_allclose(command.target, default_position)
    np.testing.assert_allclose(applied_offsets, expected_offsets, rtol=0.0, atol=1.0e-7)
    maximum_configured_step = np.max(np.diff(np.concatenate(((0.0,), expected_offsets))))
    assert np.max(np.diff(np.concatenate(((0.0,), applied_offsets)))) <= maximum_configured_step + 1.0e-12

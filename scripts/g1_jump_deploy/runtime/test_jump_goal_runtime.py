# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.g1_jump_deploy.runtime import JumpGoalRuntime

_EPISODE_STEPS = 152
_FLOAT32_ATOL = 1.0e-6
_FLOAT32_RTOL = 1.0e-6
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


def _write_bundle(
    directory: Path,
    *,
    delay_steps: int = 0,
    flight_range: range = range(40, 70),
    land_start: int = 70,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    default_pos = np.linspace(-0.5, 0.6, 23, dtype=np.float32)
    default_vel = np.linspace(-0.2, 0.2, 23, dtype=np.float32)
    scale = np.linspace(0.1, 0.3, 23, dtype=np.float32)
    alpha = np.linspace(0.2, 1.0, 23, dtype=np.float32)
    preview = np.arange(_EPISODE_STEPS * 70, dtype=np.float32).reshape(_EPISODE_STEPS, 70)
    phase = np.zeros((_EPISODE_STEPS, 6), dtype=np.float32)
    phase[:, 0] = 1.0
    for step in flight_range:
        phase[step] = 0.0
        phase[step, 3] = 1.0
    phase[land_start:] = 0.0
    phase[land_start:, 4] = 1.0
    phase[100:] = 0.0
    phase[100:, 5] = 1.0

    preview_path = directory / "reference_preview_152x70.npy"
    phase_path = directory / "jump_phase_152x6.npy"
    np.save(preview_path, preview, allow_pickle=False)
    np.save(phase_path, phase, allow_pickle=False)
    manifest = {
        "schema_version": "1.2",
        "task": "test",
        "checkpoint": "/tmp/test.pt",
        "exported_at": "2026-01-01T00:00:00+00:00",
        "control": {
            "policy_dt": 0.02,
            "policy_hz": 50.0,
            "sim_dt": 0.002,
            "decimation": 10,
            "episode_steps": _EPISODE_STEPS,
            "episode_duration_s": 3.0333333333333333,
        },
        "joints": {
            "names": [f"joint_{index}" for index in range(23)],
            "unitree_sdk2_slots": list(range(23)),
            "default_pos": default_pos.tolist(),
            "default_vel": default_vel.tolist(),
        },
        "observation": {
            "total_dim": 326,
            "history_order": "oldest_first",
            "history_layout": "history_major",
            "terms": _TERMS,
        },
        "action": {
            "dim": 23,
            "scale": scale.tolist(),
            "offset": default_pos.tolist(),
            "filter_alpha": alpha.tolist(),
            "delay_steps": {"min": delay_steps, "max": delay_steps},
            "clip": np.column_stack((default_pos - 10.0, default_pos + 10.0)).tolist(),
            "formula": "q_target = alpha*clip(offset + scale*a_delayed) + (1-alpha)*q_target_prev",
        },
        "actuators": {
            "type": "implicit_pd",
            "stiffness": [1.0] * 23,
            "damping": [1.0] * 23,
            "effort_limit": [1.0] * 23,
            "velocity_limit": [1.0] * 23,
            "armature": [0.0] * 23,
        },
        "reference": {
            "fps": 30.0,
            "num_frames": 91,
            "phase_names": ["IDLE", "CROUCH", "TAKEOFF", "FLIGHT", "LAND", "STAND"],
            "phase_frame_ranges": [[0, 6], [6, 19], [19, 26], [26, 43], [43, 60], [60, 91]],
            "preview_offsets_frames": [1, 4, 7],
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
        "tables": {
            "reference_preview": preview_path.name,
            "jump_phase": phase_path.name,
        },
    }
    manifest_path = directory / "deploy_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, preview, phase, default_pos, default_vel, scale


def _trigger(runtime: JumpGoalRuntime, joint_pos: np.ndarray, *, dx: float = 0.4, dy: float = -0.2) -> None:
    runtime.arm(dx, dy, 0.4)
    runtime.trigger(np.asarray((1.0, 2.0, 0.8)), np.asarray((1.0, 0.0, 0.0, 0.0)), joint_pos)


def _step(
    runtime: JumpGoalRuntime,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    *,
    odom_x: float = 1.0,
    advance: bool = True,
) -> np.ndarray:
    half_sqrt = math.sqrt(0.5)
    return runtime.step(
        joint_pos,
        joint_vel,
        np.asarray((0.1, -0.2, 0.3)),
        np.asarray((half_sqrt, half_sqrt, 0.0, 0.0)),
        np.asarray((odom_x, 2.0, 0.8)),
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        advance=advance,
    )


def test_assembled_observation_uses_manifest_offsets_and_fills_first_history(tmp_path: Path):
    manifest_path, preview, phase, default_pos, default_vel, _ = _write_bundle(tmp_path)
    runtime = JumpGoalRuntime(manifest_path)
    _trigger(runtime, default_pos)
    joint_pos_delta = np.linspace(0.01, 0.23, 23, dtype=np.float32)
    joint_vel_delta = np.linspace(-0.46, 0.46, 23, dtype=np.float32)

    observation = _step(runtime, default_pos + joint_pos_delta, default_vel + joint_vel_delta)

    assert observation.shape == (326,)
    assert observation.dtype == np.float32
    expected_samples = {
        "joint_pos": joint_pos_delta,
        "joint_vel": joint_vel_delta,
        "goal_remaining": np.asarray((0.4, -0.2, 0.0), dtype=np.float32),
        "base_ang_vel": np.asarray((0.1, -0.2, 0.3), dtype=np.float32),
        # Inverse rotation of world -Z by a +90 degree WXYZ roll.
        "projected_gravity": np.asarray((0.0, -1.0, 0.0), dtype=np.float32),
        "last_action": np.zeros(23, dtype=np.float32),
        "goal_command": np.asarray((0.4, -0.2, 0.0, 0.0, 0.0, math.sin(0.2), math.cos(0.2)), dtype=np.float32),
        "reference_preview": preview[0],
        "jump_phase": phase[0],
    }
    for term in _TERMS:
        sample = expected_samples[term["name"]]
        expected = np.tile(sample, term["history"])
        actual = observation[term["offset"] : term["offset"] + term["total"]]
        np.testing.assert_allclose(actual, expected, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)


def test_history_is_oldest_first_after_the_first_step(tmp_path: Path):
    manifest_path, _, _, default_pos, default_vel, _ = _write_bundle(tmp_path)
    runtime = JumpGoalRuntime(manifest_path)
    _trigger(runtime, default_pos)
    first_delta = np.full(23, 1.0, dtype=np.float32)
    second_delta = np.full(23, 2.0, dtype=np.float32)
    _step(runtime, default_pos + first_delta, default_vel)

    observation = _step(runtime, default_pos + second_delta, default_vel)

    expected_history = np.concatenate((first_delta, first_delta, first_delta, second_delta))
    np.testing.assert_allclose(observation[0:92], expected_history, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)


def test_observation_term_scales_apply_before_history(tmp_path: Path):
    manifest_path, _, _, default_pos, default_vel, _ = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for term in manifest["observation"]["terms"]:
        if term["name"] == "goal_remaining":
            term["scale"] = 4.0
        elif term["name"] == "goal_command":
            term["scale"] = [2.0] * 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = JumpGoalRuntime(manifest_path)
    _trigger(runtime, default_pos)

    observation = _step(runtime, default_pos, default_vel)

    expected_remaining = np.tile(np.asarray((1.6, -0.8, 0.0), dtype=np.float32), 4)
    np.testing.assert_allclose(observation[184:196], expected_remaining, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)
    expected_command = 2.0 * np.asarray((0.4, -0.2, 0.0, 0.0, 0.0, math.sin(0.2), math.cos(0.2)), dtype=np.float32)
    np.testing.assert_allclose(observation[243:250], expected_command, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)


def test_goal_remaining_freezes_last_preflight_value_and_resumes_at_land(tmp_path: Path):
    manifest_path, _, _, default_pos, default_vel, _ = _write_bundle(tmp_path, flight_range=range(2, 5), land_start=5)
    runtime = JumpGoalRuntime(manifest_path)
    _trigger(runtime, default_pos, dx=0.5, dy=0.0)

    observations = [
        _step(runtime, default_pos, default_vel, odom_x=odom_x) for odom_x in (1.0, 1.1, 1.2, 1.3, 1.4, 1.45)
    ]
    latest_goal_offset = 184 + 3 * 3
    latest_x = [float(observation[latest_goal_offset]) for observation in observations]

    np.testing.assert_allclose(latest_x, (0.5, 0.4, 0.4, 0.4, 0.4, 0.05), rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)


def test_latched_goal_remaining_never_requires_live_odometry(tmp_path: Path):
    manifest_path, _, _, default_pos, default_vel, _ = _write_bundle(tmp_path, flight_range=range(2, 5), land_start=5)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["goal"]["remaining_mode"] = "latched"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = JumpGoalRuntime(manifest_path)
    _trigger(runtime, default_pos, dx=0.5, dy=0.0)

    observations = [
        _step(runtime, default_pos, default_vel, odom_x=odom_x) for odom_x in (1.0, 1.1, 1.2, 1.3, 1.4, 1.45)
    ]
    latest_goal_offset = 184 + 3 * 3
    latest_x = [float(observation[latest_goal_offset]) for observation in observations]

    np.testing.assert_allclose(latest_x, (0.5,) * 6, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)


def test_remaining_goal_orientation_uses_current_imu_attitude(tmp_path: Path):
    manifest_path, _, _, default_pos, default_vel, _ = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["goal"]["orientation_mode"] = "remaining"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = JumpGoalRuntime(manifest_path)
    _trigger(runtime, default_pos)

    observation = _step(runtime, default_pos, default_vel)

    goal_command = observation[243:250]
    # q_remaining = inverse(Rx(+90 deg)) * Rz(+0.4 rad), converted WXYZ -> XYZW.
    half_sqrt = math.sqrt(0.5)
    yaw_cos = math.cos(0.2)
    yaw_sin = math.sin(0.2)
    expected = np.asarray(
        (
            0.4,
            -0.2,
            0.0,
            -half_sqrt * yaw_cos,
            half_sqrt * yaw_sin,
            half_sqrt * yaw_sin,
            half_sqrt * yaw_cos,
        ),
        dtype=np.float32,
    )
    np.testing.assert_allclose(goal_command, expected, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)


def test_schema_v16_marks_only_explicit_retrigger_observations(tmp_path: Path) -> None:
    manifest_path, _, _, default_pos, default_vel, _ = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.6"
    manifest["joints"]["position_limits"] = [[-2.0, 2.0]] * 23
    manifest["goal"]["retrigger_indicator"] = {
        "mode": "goal_command_z",
        "fresh_value": 0.0,
        "retrigger_value": 0.25,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = JumpGoalRuntime(manifest_path)

    runtime.arm(0.4, 0.0, 0.0)
    runtime.trigger(
        np.asarray((1.0, 2.0, 0.8)),
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        default_pos,
        retrigger=False,
    )
    fresh_observation = _step(runtime, default_pos, default_vel)
    runtime.cancel()
    runtime.arm(0.4, 0.0, 0.0)
    runtime.trigger(
        np.asarray((1.0, 2.0, 0.8)),
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        default_pos,
        retrigger=True,
    )
    retrigger_observation = _step(runtime, default_pos, default_vel)

    assert fresh_observation[245] == pytest.approx(0.0)
    assert retrigger_observation[245] == pytest.approx(0.25)
    np.testing.assert_allclose(runtime.goal_position_w, (1.4, 2.0, 0.8))


def test_schema_v16_requires_retrigger_indicator_contract(tmp_path: Path) -> None:
    manifest_path, _, _, _, _, _ = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.6"
    manifest["joints"]["position_limits"] = [[-2.0, 2.0]] * 23
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="retrigger_indicator"):
        JumpGoalRuntime(manifest_path)


def test_schema_v17_affine_retrigger_channel_carries_repeat_goal(tmp_path: Path) -> None:
    manifest_path, _, _, default_pos, default_vel, _ = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.7"
    manifest["joints"]["position_limits"] = [[-2.0, 2.0]] * 23
    manifest["goal"]["ranges"]["pos_x"] = [-0.1, 0.1]
    manifest["goal"]["retrigger_indicator"] = {
        "mode": "goal_command_z_affine_pos_x",
        "fresh_value": 0.0,
        "retrigger_value": 0.25,
        "goal_pos_x_scale": 1.5,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = JumpGoalRuntime(manifest_path)

    runtime.arm(-0.1, 0.0, 0.0)
    runtime.trigger(
        np.asarray((1.0, 2.0, 0.8)),
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        default_pos,
        retrigger=False,
    )
    fresh_observation = _step(runtime, default_pos, default_vel)
    runtime.cancel()
    runtime.arm(-0.1, 0.0, 0.0)
    runtime.trigger(
        np.asarray((1.0, 2.0, 0.8)),
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        default_pos,
        retrigger=True,
    )
    retrigger_observation = _step(runtime, default_pos, default_vel)

    assert fresh_observation[245] == pytest.approx(0.0)
    assert retrigger_observation[245] == pytest.approx(0.1)
    np.testing.assert_allclose(runtime.goal_position_w, (0.9, 2.0, 0.8))


def test_action_transform_matches_delay_affine_and_filter_pipeline(tmp_path: Path):
    manifest_path, _, _, default_pos, default_vel, scale = _write_bundle(tmp_path, delay_steps=1)
    runtime = JumpGoalRuntime(manifest_path)
    measured_pos = default_pos + np.linspace(0.2, 0.4, 23, dtype=np.float32)
    _trigger(runtime, measured_pos)
    alpha = np.linspace(0.2, 1.0, 23, dtype=np.float32)
    first_action = np.linspace(-1.0, 1.0, 23, dtype=np.float32)
    second_action = np.linspace(2.0, 3.0, 23, dtype=np.float32)

    initial_observation = _step(runtime, measured_pos, default_vel)
    np.testing.assert_allclose(
        initial_observation[220:243],
        np.zeros(23, dtype=np.float32),
        rtol=_FLOAT32_RTOL,
        atol=_FLOAT32_ATOL,
    )

    first_target = runtime.transform_action(first_action)
    expected_first = alpha * (default_pos + scale * first_action) + (1.0 - alpha) * measured_pos
    np.testing.assert_allclose(first_target, expected_first, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)

    observation = _step(runtime, measured_pos, default_vel)
    np.testing.assert_allclose(observation[220:243], first_action, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)

    second_target = runtime.transform_action(second_action)
    expected_second = alpha * (default_pos + scale * first_action) + (1.0 - alpha) * expected_first
    np.testing.assert_allclose(second_target, expected_second, rtol=_FLOAT32_RTOL, atol=_FLOAT32_ATOL)


def test_frozen_phase_zero_preparation_preserves_goal_history_and_episode_clock(tmp_path: Path):
    manifest_path, preview, phase, default_pos, default_vel, _ = _write_bundle(tmp_path)
    runtime = JumpGoalRuntime(manifest_path)
    _trigger(runtime, default_pos, dx=0.4, dy=0.0)
    prepared_action = np.linspace(-0.5, 0.5, 23, dtype=np.float32)

    prepared_observations = []
    for preparation_step in range(6):
        observation = _step(
            runtime,
            default_pos + 0.01 * preparation_step,
            default_vel,
            advance=False,
        )
        prepared_observations.append(observation)
        runtime.transform_action(prepared_action)

    for observation in prepared_observations:
        np.testing.assert_array_equal(observation[250:320], preview[0])
        np.testing.assert_array_equal(observation[320:326], phase[0])
        assert observation[243] == pytest.approx(0.4)
    np.testing.assert_allclose(
        prepared_observations[-1][0:92].reshape(4, 23)[:, 0],
        (0.02, 0.03, 0.04, 0.05),
        rtol=_FLOAT32_RTOL,
        atol=_FLOAT32_ATOL,
    )

    runtime.reanchor_goal(
        np.asarray((3.0, 4.0, 0.8)),
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        goal_pos_z_w=0.0,
    )
    first_jump_observation = _step(runtime, default_pos, default_vel)
    second_jump_observation = _step(runtime, default_pos, default_vel)

    np.testing.assert_array_equal(first_jump_observation[250:320], preview[0])
    np.testing.assert_array_equal(first_jump_observation[320:326], phase[0])
    np.testing.assert_array_equal(second_jump_observation[250:320], preview[1])
    np.testing.assert_array_equal(second_jump_observation[320:326], phase[1])
    np.testing.assert_allclose(first_jump_observation[220:243], prepared_action)
    np.testing.assert_allclose(runtime.goal_position_w, (3.4, 4.0, 0.0))


def test_cancelled_preparation_clears_runtime_and_allows_a_new_goal(tmp_path: Path):
    manifest_path, _, _, default_pos, default_vel, _ = _write_bundle(tmp_path)
    runtime = JumpGoalRuntime(manifest_path)
    _trigger(runtime, default_pos)
    _step(runtime, default_pos, default_vel, advance=False)
    runtime.transform_action(np.ones(23, dtype=np.float32))

    runtime.cancel()

    with pytest.raises(RuntimeError, match=r"trigger\(\)"):
        _step(runtime, default_pos, default_vel)
    runtime.arm(-0.1, 0.0, 0.0)
    runtime.trigger(np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0)), default_pos)
    observation = _step(runtime, default_pos, default_vel, advance=False)
    assert observation[243] == pytest.approx(-0.1)
    np.testing.assert_array_equal(observation[220:243], 0.0)


def test_completed_episode_can_hold_last_stand_reference_without_advancing(tmp_path: Path):
    manifest_path, preview, phase, default_pos, default_vel, _ = _write_bundle(tmp_path)
    runtime = JumpGoalRuntime(manifest_path)
    _trigger(runtime, default_pos)
    for _ in range(_EPISODE_STEPS):
        _step(runtime, default_pos, default_vel)
    assert runtime.done
    assert runtime.stand_reference_step == _EPISODE_STEPS - 1

    observations = [
        runtime.step(
            default_pos,
            default_vel,
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.asarray((1.0, 2.0, 0.8)),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            advance=False,
            reference_step=runtime.stand_reference_step,
        )
        for _ in range(3)
    ]

    assert runtime.done
    for observation in observations:
        np.testing.assert_array_equal(observation[250:320], preview[-1])
        np.testing.assert_array_equal(observation[320:326], phase[-1])

    with pytest.raises(ValueError, match="advance=False"):
        runtime.step(
            default_pos,
            default_vel,
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.asarray((1.0, 2.0, 0.8)),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            reference_step=runtime.stand_reference_step,
        )


def test_action_clip_is_applied_before_filter_and_reaches_exact_bound(tmp_path: Path):
    manifest_path, _, _, default_pos, default_vel, scale = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clip = np.column_stack((default_pos - scale, default_pos + scale))
    manifest["action"]["clip"] = clip.tolist()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = JumpGoalRuntime(manifest_path)
    measured_pos = default_pos + 2.0 * scale
    _trigger(runtime, measured_pos)

    target = runtime.transform_action(np.full(23, 10.0, dtype=np.float32))

    alpha = np.linspace(0.2, 1.0, 23, dtype=np.float32).astype(np.float64)
    expected = alpha * clip[:, 1] + (1.0 - alpha) * measured_pos
    np.testing.assert_allclose(target, expected, rtol=0.0, atol=1.0e-12)
    assert target[-1] == clip[-1, 1]


def test_action_clip_wrong_length_is_rejected(tmp_path: Path):
    manifest_path, _, _, _, _, _ = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["action"]["clip"] = manifest["action"]["clip"][:-1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"action\.clip.*23"):
        JumpGoalRuntime(manifest_path)


@pytest.mark.parametrize("schema_version", ("1.0", "1.1"))
def test_legacy_manifest_schema_requires_reexport(tmp_path: Path, schema_version: str):
    manifest_path, _, _, _, _, _ = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"schema 1\.2.*re-export"):
        JumpGoalRuntime(manifest_path)


@pytest.mark.parametrize(
    ("dx", "dy", "dyaw"),
    ((-0.31, 0.0, 0.0), (1.01, 0.0, 0.0), (0.0, -0.61, 0.0), (0.0, 0.61, 0.0), (0.0, 0.0, 1.11)),
)
def test_arm_rejects_out_of_range_goals(tmp_path: Path, dx: float, dy: float, dyaw: float):
    manifest_path, _, _, _, _, _ = _write_bundle(tmp_path)
    runtime = JumpGoalRuntime(manifest_path)

    with pytest.raises(ValueError, match="outside manifest range"):
        runtime.arm(dx, dy, dyaw)


def test_wrong_length_inputs_raise_instead_of_broadcasting(tmp_path: Path):
    manifest_path, _, _, default_pos, default_vel, _ = _write_bundle(tmp_path)
    runtime = JumpGoalRuntime(manifest_path)
    runtime.arm(0.4, 0.0, 0.0)
    with pytest.raises(ValueError, match="shape \\(23,\\)"):
        runtime.trigger(np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0)), np.zeros(22))

    runtime.trigger(np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0)), default_pos)
    with pytest.raises(ValueError, match="shape \\(23,\\)"):
        runtime.step(
            default_pos,
            default_vel[:-1],
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
        )
    with pytest.raises(ValueError, match="shape \\(23,\\)"):
        runtime.transform_action(np.zeros(24))

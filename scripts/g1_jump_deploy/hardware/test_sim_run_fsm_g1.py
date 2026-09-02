# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the real-runner-in-the-loop MuJoCo transport simulator."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.g1_jump_deploy.hardware import run_fsm_g1 as runner
from scripts.g1_jump_deploy.hardware import sim_run_fsm_g1 as simulator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"
_MODEL = _REPO_ROOT / "data_storage" / "g1_23dof_holo_compat.xml"
_OVERLAY = _REPO_ROOT / "scripts" / "g1_jump_deploy" / "mujoco" / "model_overlay.xml"


@pytest.fixture
def simulation() -> simulator.RunnerMujocoSimulation:
    return simulator.RunnerMujocoSimulation(_MANIFEST, _MODEL, _OVERLAY, operator_support_fixture=True)


def test_low_state_slot_mapping_round_trip_and_crc(simulation: simulator.RunnerMujocoSimulation) -> None:
    positions = simulation.manifest.default_position + np.linspace(-0.01, 0.01, simulation.manifest.joint_count)
    simulation.data.qpos[simulation._joint_qpos_addresses] = positions
    simulation.data.qvel[simulation._joint_dof_addresses] = 0.0
    simulator.mujoco.mj_forward(simulation.model, simulation.data)

    state = simulation.emit_low_state()
    for policy_index, slot in enumerate(simulation.manifest.sdk_slots):
        assert state.motor_state[slot].q == pytest.approx(positions[policy_index])

    state_buffer = runner._StateBuffer(simulation.manifest, simulation.crc)
    state_buffer.update(state)
    snapshot = state_buffer.snapshot()
    assert snapshot is not None
    np.testing.assert_allclose(snapshot.joint_positions, positions, rtol=0.0, atol=1.0e-7)
    assert state_buffer.valid_packets == 1
    assert state_buffer.crc_errors == 0

    state.motor_state[simulation.manifest.sdk_slots[0]].q += 0.1
    state_buffer.update(state)
    assert state_buffer.valid_packets == 1
    assert state_buffer.crc_errors == 1


@pytest.mark.parametrize(
    ("button", "byte_index", "mask"),
    (("L1", 2, 0x02), ("R1", 2, 0x01), ("A", 3, 0x01), ("B", 3, 0x02), ("Y", 3, 0x08)),
)
def test_remote_button_encoding(button: str, byte_index: int, mask: int) -> None:
    remote = simulator.encode_remote((button,))
    assert len(remote) == 40
    assert remote[byte_index] == mask
    assert sum(remote) == mask


def test_simulated_sleep_steps_physics_and_feedback(simulation: simulator.RunnerMujocoSimulation) -> None:
    state_buffer = runner._StateBuffer(simulation.manifest, simulation.crc)
    simulation.state_handler = state_buffer.update
    simulation.emit_low_state()
    initial_steps = simulation.physics_steps
    initial_updates = simulation.state_updates

    simulation.sleep(0.010)

    assert simulation.physics_steps - initial_steps == 5
    assert simulation.state_updates - initial_updates == 5
    assert simulation.time_s == pytest.approx(0.010)
    assert state_buffer.valid_packets == 6
    assert state_buffer.snapshot() is not None
    assert state_buffer.snapshot().tick == 10


def test_velocity_limit_emulation_reduces_only_opted_in_torque() -> None:
    without_emulation = simulator.RunnerMujocoSimulation(_MANIFEST, _MODEL, _OVERLAY)
    with_emulation = simulator.RunnerMujocoSimulation(
        _MANIFEST,
        _MODEL,
        _OVERLAY,
        emulate_velocity_limit=True,
    )
    for simulation in (without_emulation, with_emulation):
        simulation.loco_client = SimpleNamespace(fsm_id=1, user_control=True)
        simulation.latest_command_q = simulation.joint_positions + 0.1
        simulation.latest_command_kp.fill(100.0)
        simulation.latest_command_kd.fill(0.0)
        simulation.data.qvel[simulation._joint_dof_addresses] = simulation.manifest.velocity_limit
        simulation._step_physics()

    np.testing.assert_allclose(without_emulation.applied_torque, 10.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(with_emulation.applied_torque, 0.0, rtol=0.0, atol=1.0e-12)


def test_velocity_limit_emulation_cli_defaults_off_and_is_opt_in() -> None:
    default_args, _ = simulator._parse_cli(("--", "sim0"))
    enabled_args, _ = simulator._parse_cli(("--emulate_velocity_limit", "--", "sim0"))

    assert not default_args.emulate_velocity_limit
    assert enabled_args.emulate_velocity_limit


def test_loco_client_state_machine(simulation: simulator.RunnerMujocoSimulation) -> None:
    with simulator._patched_runner(simulation, ("sim0",)):
        client = simulator.FakeLocoClient()
        client.SetTimeout(2.0)
        client.Init()
        assert client.GetFsmId() == (0, 1)
        assert client.SetFsmId(801) == 0
        assert client.GetFsmId() == (0, 801)
        assert client.SwitchToUserCtrl() == 0
        assert client.user_control
        assert client.SetVelocity(0.0, 0.0, 0.0, 1.0) == 0
        assert client.SwitchToInternalCtrl(object()) == 0
        assert not client.user_control


def test_native_stand_fixture_releases_at_user_handover() -> None:
    simulation = simulator.RunnerMujocoSimulation(
        _MANIFEST,
        _MODEL,
        _OVERLAY,
        native_fsm_id=801,
        operator_support_fixture=False,
    )
    assert simulation.fixture_active
    with simulator._patched_runner(simulation, ("sim0",)):
        client = simulator.FakeLocoClient()
        assert simulation.fixture_active
        assert client.SetFsmId(1) == 0
        assert simulation.fixture_active
        assert client.SwitchToUserCtrl() == 0
        assert not simulation.fixture_active
        simulation.sleep(0.002)
        assert not simulation.log_arrays()["fixture_active"][-1]
        assert client.SwitchToInternalCtrl(1) == 0
        assert not simulation.fixture_active
        assert client.SetFsmId(801) == 0
        assert simulation.fixture_active


def test_physics_failure_overrides_success_with_exit_three(simulation: simulator.RunnerMujocoSimulation) -> None:
    verdict = simulation.physics_verdict(None, ground_jump=True, contactless_rehearsal=False)
    assert not verdict.passed
    assert "ground jump completed zero flights" in verdict.details
    assert simulator._physics_exit_code(0, verdict) == 3
    assert simulator._physics_exit_code(2, verdict) == 2


def test_main_prints_failed_verdict_and_returns_three(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(simulator.runner, "main", lambda: 0)
    exit_code = simulator.main(("--", "sim0", "--ground_jump"))
    output = capsys.readouterr().out
    assert exit_code == 3
    assert "SIM START ATTITUDE: level." in output
    assert "SIM VERDICT: FAIL ground jump completed zero flights" in output


def test_drop_feedback_fault_stops_runner(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = simulator.main(("--drop_feedback_ms", "6000", "--drop_feedback_at", "0", "--", "sim0"))
    output = capsys.readouterr().out
    assert exit_code == 2
    assert "no valid G1 feedback arrived within 5.0 seconds" in output


def test_feedback_latency_fault_stops_runner(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = simulator.main(("--feedback_latency_ms", "6000", "--", "sim0"))
    output = capsys.readouterr().out
    assert exit_code == 2
    assert "no valid G1 feedback arrived within 5.0 seconds" in output


def test_crc_corruption_fault_stops_runner(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = simulator.main(("--crc_corrupt_at", "1.0", "--", "sim0"))
    output = capsys.readouterr().out
    assert exit_code == 2
    assert "preflight feedback errors (CRC=1, invalid=0)" in output


def test_rpc_failure_stops_runner(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = simulator.main(("--rpc_fail", "GetFsmId", "--", "sim0", "--query_fsm"))
    output = capsys.readouterr().out
    assert exit_code == 2
    assert "GetFsmId returned code 1" in output


def test_stdin_without_script_blocks_instead_of_returning_eof(simulation: simulator.RunnerMujocoSimulation) -> None:
    result: list[str] = []
    with simulator._patched_runner(simulation, ("sim0",)):
        reader = threading.Thread(target=lambda: result.append(sys.stdin.readline()))
        reader.start()
        reader.join(timeout=0.02)
        assert reader.is_alive()
        assert result == []
    reader.join(timeout=1.0)
    assert result == [""]


def test_stdin_script_releases_lines_at_simulated_times(
    tmp_path: Path, simulation: simulator.RunnerMujocoSimulation
) -> None:
    script_path = tmp_path / "stdin.json"
    script_path.write_text(json.dumps([{"t": 0.02, "line": "q"}]), encoding="utf-8")
    simulation.stdin_lines = simulator._parse_stdin_script(script_path)
    result: list[str] = []
    with simulator._patched_runner(simulation, ("sim0",)):
        reader = threading.Thread(target=lambda: result.append(sys.stdin.readline()))
        reader.start()
        simulation.sleep(0.018)
        reader.join(timeout=0.02)
        assert reader.is_alive()
        simulation.sleep(0.002)
        reader.join(timeout=1.0)
        assert result == ["q\n"]


def test_start_attitude_and_native_fixture_hold() -> None:
    level = simulator.RunnerMujocoSimulation(_MANIFEST, _MODEL, _OVERLAY, native_fsm_id=801, start_attitude="level")
    manifest = simulator.RunnerMujocoSimulation(
        _MANIFEST, _MODEL, _OVERLAY, native_fsm_id=801, start_attitude="manifest"
    )
    np.testing.assert_allclose(level.pelvis_quaternion, (1.0, 0.0, 0.0, 0.0), atol=1.0e-12)
    assert runner._body_tilt(manifest.pelvis_quaternion) == pytest.approx(np.deg2rad(7.49), abs=np.deg2rad(0.02))
    initial_manifest_quaternion = manifest.pelvis_quaternion
    manifest.sleep(0.020)
    np.testing.assert_allclose(manifest.pelvis_quaternion, initial_manifest_quaternion, atol=1.0e-12)


def test_full_ground_session_remote_script() -> None:
    script = simulator.default_remote_script(ground_session=True)
    assert [(pulse["t"], pulse["buttons"]) for pulse in script] == [
        (2.10, ["B"]),
        (2.50, ["L1", "R1"]),
        (11.75, ["A"]),
        (14.00, ["Y"]),
        (24.00, ["A"]),
        (26.50, ["Y"]),
        (36.00, ["B"]),
    ]


def test_short_real_runner_stand(capsys: pytest.CaptureFixture[str]) -> None:
    pulses = tuple(
        simulator.RemotePulse(item["t"], item["hold"], tuple(item["buttons"]))
        for item in simulator.default_remote_script()
    )
    simulation = simulator.RunnerMujocoSimulation(
        _MANIFEST,
        _MODEL,
        _OVERLAY,
        remote_pulses=pulses,
        operator_support_fixture=True,
    )

    exit_code = simulator.run_simulation(
        simulation,
        ("sim0", "--enable_control", "--entry_mode", "passive", "--duration", "1", "--effort_scale", "0.75"),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS: requested stand duration completed" in output
    assert simulation.loco_client is not None
    assert simulation.loco_client.GetFsmId() == (0, 1)
    assert not simulation.loco_client.user_control
    assert np.max(np.asarray(simulation._log["body_tilt"])) < np.deg2rad(10.0)

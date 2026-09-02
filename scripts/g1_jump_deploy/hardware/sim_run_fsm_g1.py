# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the hardware G1 runner against deterministic MuJoCo and fake SDK transport.

The module does not replace the hardware boundary.  It calls the real
``run_fsm_g1.main`` and substitutes only DDS transport, native-controller RPC,
audio RPC, the clock, stdin, and network-interface discovery.  Unitree IDL
messages, command/state CRCs, slot mapping, and every runner safety check stay
active.

Remote script times are absolute simulated seconds.  The runner first spends
two seconds in preflight, then requires a released/pressed/released B test and
an L1+R1 chord held continuously for two seconds, followed by release within
five seconds.  Gantry rehearsal requires 4.5 seconds of stabilization before
``REHEARSAL READY``.  Ground mode instead configures a 1.0-second stand-entry
and attitude-target blend and offers ``READY`` as soon as a queued or
interactive goal is available in STAND.  A must then be tapped and released;
Y is valid only after the FSM reports ARMED and for its 15-second confirmation
window.  A latched post-takeoff ground abort clears the pending goal and locks
the session against any further jump; B or an interactive ``q`` then exits.
Use :func:`make_remote_pulse` to construct standard activation, arm, confirm,
and abort pulses.  :func:`default_remote_script` can produce a complete
two-attempt ground session ending with B at 36 seconds.

Without ``--stdin_script``, stdin is a pipe whose write end remains open, so a
future background read blocks instead of observing EOF.  A stdin script is a
JSON list such as ``[{"t": 25.0, "line": "q"}]``; each complete line becomes
readable at its absolute simulated time, and reads block before the next line.

After the runner returns, a separate physics verdict checks jump flight and
touchdown evidence plus final tilt, joint limits, and motor effort. A failed
verdict changes an otherwise successful process result to exit code 3. The
``--drop_feedback_*``, ``--feedback_latency_ms``, ``--crc_corrupt_at``, and
``--rpc_fail`` options inject deterministic boundary faults for safety tests.

TODO: Add an optional wall-clock/threaded mode for interactive visualization.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.g1_jump_deploy.hardware import run_fsm_g1 as runner  # noqa: E402
from scripts.g1_jump_deploy.mujoco.model_overlay import (  # noqa: E402
    apply_initial_ground_clearance,
    compose_model_xml,
)
from scripts.g1_jump_deploy.runtime import saturate_torque_at_velocity_limit  # noqa: E402

_DEFAULT_MANIFEST = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"
_DEFAULT_MODEL = _REPO_ROOT / "data_storage" / "g1_23dof_holo_compat.xml"
_DEFAULT_OVERLAY = _SCRIPT_DIR.parent / "mujoco" / "model_overlay.xml"
_BUTTON_LAYOUT = {"L1": (2, 0x02), "R1": (2, 0x01), "A": (3, 0x01), "B": (3, 0x02), "Y": (3, 0x08)}
_STANDARD_PULSES = {
    "activation": (2.5, ("L1", "R1")),
    "arm": (0.2, ("A",)),
    "confirm": (0.2, ("Y",)),
    "abort": (0.3, ("B",)),
}
_NATIVE_STAND_FSM_IDS = frozenset((500, 801))
_PASSIVE_FSM_ID = 1
_PASSIVE_DAMPING = 1.5
_CONTACT_UNLOADED_THRESHOLD_N = 1.0
_FLIGHT_CONTACT_THRESHOLD_N = 5.0
_TOUCHDOWN_CONTACT_THRESHOLD_N = 5.0
_FLIGHT_MINIMUM_DURATION_S = 0.040
_FINAL_TILT_LIMIT_RAD = math.radians(10.0)
_JOINT_LIMIT_TOLERANCE_RAD = 0.01
_FEET_CLEARANCE_M = 0.45
_GANTRY_POSITION_STIFFNESS_N_M = 1_000.0
_GANTRY_POSITION_DAMPING_N_S_M = 300.0
_GANTRY_ATTITUDE_STIFFNESS_N_M_RAD = 300.0
_GANTRY_ATTITUDE_DAMPING_N_M_S_RAD = 30.0
_GANTRY_FORCE_LIMIT_N = 750.0
_GANTRY_TORQUE_LIMIT_N_M = 200.0


@dataclass(frozen=True)
class RemotePulse:
    """One interval of pressed remote buttons."""

    time_s: float
    hold_s: float
    buttons: tuple[str, ...]


@dataclass(frozen=True)
class TimedStdinLine:
    """One line made readable at an absolute simulated time."""

    time_s: float
    line: str


@dataclass(frozen=True)
class PhysicsVerdict:
    """Independent MuJoCo acceptance result and diagnostic details."""

    passed: bool
    details: tuple[str, ...]


def make_remote_pulse(name: str, time_s: float) -> dict[str, Any]:
    """Create one standard remote pulse beginning at ``time_s``.

    Args:
        name: One of ``activation``, ``arm``, ``confirm``, or ``abort``.
        time_s: Absolute simulated start time [s].

    Returns:
        JSON-compatible pulse dictionary.
    """
    try:
        hold_s, buttons = _STANDARD_PULSES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown standard remote pulse {name!r}.") from exc
    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError("Remote pulse time must be finite and non-negative.")
    return {"t": float(time_s), "hold": hold_s, "buttons": list(buttons)}


def default_remote_script(*, rehearsal: bool = False, ground_session: bool = False) -> list[dict[str, Any]]:
    """Return a deterministic stand, rehearsal, or full ground-session script."""
    if rehearsal and ground_session:
        raise ValueError("A default remote script cannot be both rehearsal and ground-session mode.")
    pulses = [
        {"t": 2.10, "hold": 0.20, "buttons": ["B"]},
        make_remote_pulse("activation", 2.50),
    ]
    if rehearsal:
        pulses.extend((make_remote_pulse("arm", 11.75), make_remote_pulse("confirm", 14.00)))
    elif ground_session:
        pulses.extend(
            (
                make_remote_pulse("arm", 11.75),
                make_remote_pulse("confirm", 14.00),
                make_remote_pulse("arm", 24.00),
                make_remote_pulse("confirm", 26.50),
                make_remote_pulse("abort", 36.00),
            )
        )
    return pulses


def encode_remote(buttons: Iterable[str]) -> bytes:
    """Encode named buttons into the real 40-byte Unitree remote layout."""
    result = bytearray(40)
    for button in buttons:
        try:
            byte_index, mask = _BUTTON_LAYOUT[button]
        except KeyError as exc:
            raise ValueError(f"Unknown remote button {button!r}; expected {sorted(_BUTTON_LAYOUT)}.") from exc
        result[byte_index] |= mask
    return bytes(result)


def _parse_remote_script(
    path: Path | None, *, rehearsal: bool, ground_session: bool = False
) -> tuple[RemotePulse, ...]:
    if path is None:
        raw: Any = default_remote_script(rehearsal=rehearsal, ground_session=ground_session)
    else:
        try:
            with path.resolve().open(encoding="utf-8") as stream:
                raw = json.load(stream)
        except OSError as exc:
            raise ValueError(f"Cannot read remote script {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("Remote script must be a JSON list.")
    pulses: list[RemotePulse] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"t", "hold", "buttons"}:
            raise ValueError(f"Remote pulse {index} must contain exactly t, hold, and buttons.")
        time_s = item["t"]
        hold_s = item["hold"]
        buttons = item["buttons"]
        if (
            isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not math.isfinite(float(time_s))
            or time_s < 0.0
        ):
            raise ValueError(f"Remote pulse {index} t must be finite and non-negative.")
        if (
            isinstance(hold_s, bool)
            or not isinstance(hold_s, (int, float))
            or not math.isfinite(float(hold_s))
            or hold_s <= 0.0
        ):
            raise ValueError(f"Remote pulse {index} hold must be finite and positive.")
        if not isinstance(buttons, list) or not all(isinstance(button, str) for button in buttons):
            raise ValueError(f"Remote pulse {index} buttons must be a string list.")
        encode_remote(buttons)
        pulses.append(RemotePulse(float(time_s), float(hold_s), tuple(buttons)))
    return tuple(pulses)


def _parse_stdin_script(path: Path | None) -> tuple[TimedStdinLine, ...]:
    if path is None:
        return ()
    try:
        with path.resolve().open(encoding="utf-8") as stream:
            raw: Any = json.load(stream)
    except OSError as exc:
        raise ValueError(f"Cannot read stdin script {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("Stdin script must be a JSON list.")
    entries: list[TimedStdinLine] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"t", "line"}:
            raise ValueError(f"Stdin entry {index} must contain exactly t and line.")
        time_s = item["t"]
        line = item["line"]
        if (
            isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not math.isfinite(float(time_s))
            or time_s < 0.0
        ):
            raise ValueError(f"Stdin entry {index} t must be finite and non-negative.")
        if not isinstance(line, str) or "\n" in line or "\r" in line:
            raise ValueError(f"Stdin entry {index} line must be a string without newline characters.")
        entries.append(TimedStdinLine(float(time_s), line))
    return tuple(sorted(entries, key=lambda entry: entry.time_s))


class _BlockingTimedStdin:
    """Pipe-backed stdin that stays open and releases lines on simulated time."""

    def __init__(self, entries: Sequence[TimedStdinLine]):
        read_fd, self._write_fd = os.pipe()
        self.stream = os.fdopen(read_fd, "r", encoding="utf-8")
        self._entries = tuple(entries)
        self._next_entry = 0
        self._closed = False

    def advance(self, time_s: float) -> None:
        """Make all lines scheduled through ``time_s`` readable."""
        while self._next_entry < len(self._entries) and self._entries[self._next_entry].time_s <= time_s + 1.0e-12:
            entry = self._entries[self._next_entry]
            os.write(self._write_fd, f"{entry.line}\n".encode())
            self._next_entry += 1

    def close(self) -> None:
        """Close both pipe ends, waking any daemon reader during teardown."""
        if self._closed:
            return
        self._closed = True
        os.close(self._write_fd)
        self.stream.close()


class FakeLocoClient:
    """Offline implementation of the runner's native locomotion RPC surface."""

    def __init__(self):
        simulation = _require_active_simulation()
        self._simulation = simulation
        self.fsm_id = simulation.native_fsm_id
        self.user_control = False
        self.timeout_s: float | None = None
        self.initialized = False
        self.user_control_started_at: float | None = None
        self.native_stand_fixture_latched = self.fsm_id in _NATIVE_STAND_FSM_IDS
        simulation.loco_client = self

    def _result(self, method: str) -> int:
        return self._simulation.rpc_result(method)

    def Init(self) -> int:  # noqa: N802
        result = self._result("Init")
        if result:
            return result
        self.initialized = True
        return 0

    def SetTimeout(self, timeout_s: float) -> int:  # noqa: N802
        result = self._result("SetTimeout")
        if result:
            return result
        self.timeout_s = float(timeout_s)
        return 0

    def GetFsmId(self) -> tuple[int, int]:  # noqa: N802
        return self._result("GetFsmId"), self.fsm_id

    def SetFsmId(self, fsm_id: int) -> int:  # noqa: N802
        result = self._result("SetFsmId")
        if result:
            return result
        self.fsm_id = int(fsm_id)
        if self.fsm_id in _NATIVE_STAND_FSM_IDS:
            self.native_stand_fixture_latched = True
        return 0

    def SwitchToUserCtrl(self) -> int:  # noqa: N802
        result = self._result("SwitchToUserCtrl")
        if result:
            return result
        self.user_control = True
        self.native_stand_fixture_latched = False
        self.user_control_started_at = self._simulation.time_s
        return 0

    def SwitchToInternalCtrl(self, mode: Any) -> int:  # noqa: N802
        result = self._result("SwitchToInternalCtrl")
        if result:
            return result
        self.user_control = False
        try:
            mode_value = int(mode.value) if hasattr(mode, "value") else int(mode)
        except (TypeError, ValueError):
            mode_value = 1
        if mode_value == 2:
            self.fsm_id = 801
            self.native_stand_fixture_latched = True
        elif mode_value == 1:
            self.fsm_id = 1
            self.native_stand_fixture_latched = False
        else:
            self.native_stand_fixture_latched = self.fsm_id in _NATIVE_STAND_FSM_IDS
        return 0

    def SetVelocity(self, *_args: Any) -> int:  # noqa: N802
        return self._result("SetVelocity")


class _FakeAudioClient:
    def SetTimeout(self, _timeout_s: float) -> None:  # noqa: N802
        pass

    def Init(self) -> None:  # noqa: N802
        pass

    def PlayStream(self, _app_name: str, _stream_id: str, _chunk: bytes) -> tuple[int, str]:  # noqa: N802
        return 0, ""

    def PlayStop(self, _app_name: str) -> int:  # noqa: N802
        return 0


class _FakeChannelSubscriber:
    def __init__(self, _topic: str, _message_type: type):
        self._simulation = _require_active_simulation()

    def Init(self, handler: Any, _depth: int) -> None:  # noqa: N802
        self._simulation.state_handler = handler
        self._simulation.emit_low_state()

    def Close(self) -> None:  # noqa: N802
        self._simulation.state_handler = None


class _FakeChannelPublisher:
    def __init__(self, _topic: str, _message_type: type):
        self._simulation = _require_active_simulation()

    def Init(self) -> None:  # noqa: N802
        pass

    def Write(self, command: Any) -> bool:  # noqa: N802
        return self._simulation.accept_low_command(command)

    def Close(self) -> None:  # noqa: N802
        pass


def _fake_channel_factory_initialize(_domain_id: int, _interface: str) -> None:
    pass


_ACTIVE_SIMULATION: RunnerMujocoSimulation | None = None


def _require_active_simulation() -> RunnerMujocoSimulation:
    if _ACTIVE_SIMULATION is None:
        raise RuntimeError("No runner-in-the-loop simulation is active.")
    return _ACTIVE_SIMULATION


class RunnerMujocoSimulation:
    """MuJoCo plant, LowState producer, deterministic clock, and truth logger."""

    def __init__(
        self,
        manifest_path: Path,
        model_path: Path,
        overlay_path: Path,
        *,
        remote_pulses: Sequence[RemotePulse] = (),
        native_fsm_id: int = _PASSIVE_FSM_ID,
        gantry_support_fraction: float = 0.0,
        feet_clear: bool = False,
        start_attitude: str = "level",
        stdin_lines: Sequence[TimedStdinLine] = (),
        operator_support_fixture: bool = False,
        emulate_velocity_limit: bool = False,
        drop_feedback_ms: float | None = None,
        drop_feedback_at: float | None = None,
        rpc_fail: str | None = None,
        crc_corrupt_at: float | None = None,
        feedback_latency_ms: float = 0.0,
        metadata_args: dict[str, Any] | None = None,
    ):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.manifest_path = manifest_path.resolve()
        self.model_path = model_path.resolve()
        self.overlay_path = overlay_path.resolve()
        self.manifest = runner._load_hardware_manifest(self.manifest_path)
        with self.manifest_path.open(encoding="utf-8") as stream:
            raw_manifest = json.load(stream)
        armature = np.asarray(raw_manifest["actuators"]["armature"], dtype=np.float64)
        sim_dt = float(raw_manifest["control"]["sim_dt"])
        if not math.isclose(sim_dt, runner._FAST_DT, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"Manifest sim_dt must be {runner._FAST_DT}, got {sim_dt}.")
        if not math.isfinite(gantry_support_fraction) or not 0.0 <= gantry_support_fraction <= 1.0:
            raise ValueError("gantry_support_fraction must be finite and in [0, 1].")
        if feet_clear and gantry_support_fraction <= 0.0:
            raise ValueError("--feet_clear requires nonzero --gantry_support_fraction.")
        if start_attitude not in ("level", "manifest"):
            raise ValueError("start_attitude must be 'level' or 'manifest'.")
        if not isinstance(emulate_velocity_limit, bool):
            raise ValueError("emulate_velocity_limit must be a boolean.")
        if (drop_feedback_ms is None) != (drop_feedback_at is None):
            raise ValueError("--drop_feedback_ms and --drop_feedback_at must be supplied together.")
        if drop_feedback_ms is not None and (not math.isfinite(drop_feedback_ms) or drop_feedback_ms <= 0.0):
            raise ValueError("--drop_feedback_ms must be finite and positive.")
        if drop_feedback_at is not None and (not math.isfinite(drop_feedback_at) or drop_feedback_at < 0.0):
            raise ValueError("--drop_feedback_at must be finite and non-negative.")
        if crc_corrupt_at is not None and (not math.isfinite(crc_corrupt_at) or crc_corrupt_at < 0.0):
            raise ValueError("--crc_corrupt_at must be finite and non-negative.")
        if not math.isfinite(feedback_latency_ms) or feedback_latency_ms < 0.0:
            raise ValueError("--feedback_latency_ms must be finite and non-negative.")

        model_xml, overlay_dt = compose_model_xml(self.model_path, self.overlay_path)
        if not math.isclose(overlay_dt, sim_dt, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Overlay and manifest timesteps differ.")
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = sim_dt
        self._resolve_model()
        self.model.dof_armature[self._joint_dof_addresses] = armature
        self.model.actuator_ctrllimited[self._actuator_ids_policy] = 1
        self.model.actuator_ctrlrange[self._actuator_ids_policy, 0] = -self.manifest.effort_limit
        self.model.actuator_ctrlrange[self._actuator_ids_policy, 1] = self.manifest.effort_limit
        self.model.jnt_actfrclimited[self._joint_ids] = 1
        self.model.jnt_actfrcrange[self._joint_ids, 0] = -self.manifest.effort_limit
        self.model.jnt_actfrcrange[self._joint_ids, 1] = self.manifest.effort_limit
        self.start_attitude = start_attitude
        self._reset(feet_clear, start_attitude)

        self.remote_pulses = tuple(remote_pulses)
        self.stdin_lines = tuple(stdin_lines)
        self._stdin_transport: _BlockingTimedStdin | None = None
        self.native_fsm_id = int(native_fsm_id)
        self.gantry_support_fraction = float(gantry_support_fraction)
        self.operator_support_fixture = bool(operator_support_fixture)
        self.emulate_velocity_limit = emulate_velocity_limit
        self._gantry_reference_position = self.pelvis_position.copy()
        self._gantry_reference_quaternion = self.pelvis_quaternion.copy()
        self._internal_root_qpos = np.asarray(
            self.data.qpos[self._root_qpos_address : self._root_qpos_address + 7], dtype=np.float64
        ).copy()
        self._gantry_support_force_world = (
            -self.gantry_support_fraction
            * float(mujoco.mj_getTotalmass(self.model))
            * np.asarray(self.model.opt.gravity, dtype=np.float64)
        )
        self._state_factory = unitree_hg_msg_dds__LowState_
        self.crc = CRC()
        self.state_handler: Any | None = None
        self.loco_client: FakeLocoClient | None = None
        self.drop_feedback_at = None if drop_feedback_at is None else float(drop_feedback_at)
        self.drop_feedback_duration_s = None if drop_feedback_ms is None else float(drop_feedback_ms) / 1_000.0
        self.rpc_fail = rpc_fail
        self._rpc_failure_injected = False
        self.crc_corrupt_at = None if crc_corrupt_at is None else float(crc_corrupt_at)
        self._crc_corruption_injected = False
        self.feedback_latency_s = float(feedback_latency_ms) / 1_000.0
        self._pending_states: deque[tuple[float, Any]] = deque()
        self.latest_command_q = self.manifest.default_position.copy()
        self.latest_command_kp = np.zeros(self.manifest.joint_count, dtype=np.float64)
        self.latest_command_kd = np.full(self.manifest.joint_count, _PASSIVE_DAMPING, dtype=np.float64)
        self.applied_torque = np.zeros(self.manifest.joint_count, dtype=np.float64)
        self.time_s = 0.0
        self._next_physics_time_s = self.sim_dt
        self.physics_steps = 0
        self.state_updates = 0
        self._initial_pelvis_position = self.pelvis_position.copy()
        self._metadata_args = metadata_args or {}
        self._log: dict[str, list[Any]] = {
            key: []
            for key in (
                "time",
                "qpos",
                "qvel",
                "applied_torque",
                "pelvis_position",
                "pelvis_quaternion_wxyz",
                "pelvis_linear_velocity",
                "pelvis_angular_velocity",
                "foot_contact_normal_forces",
                "body_tilt",
                "lowcmd_q",
                "lowcmd_kp",
                "lowcmd_kd",
                "wireless_remote",
                "fsm_id",
                "user_control",
                "fixture_active",
            )
        }
        self._record_truth()

    def _resolve_model(self) -> None:
        free_joint_ids = [
            joint_id
            for joint_id in range(self.model.njnt)
            if int(self.model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if len(free_joint_ids) != 1:
            raise ValueError(f"Expected one floating-base joint, got {free_joint_ids}.")
        self._root_joint_id = free_joint_ids[0]
        self._root_body_id = int(self.model.jnt_bodyid[self._root_joint_id])
        self._root_qpos_address = int(self.model.jnt_qposadr[self._root_joint_id])
        self._root_dof_address = int(self.model.jnt_dofadr[self._root_joint_id])
        self._joint_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.manifest.joint_names],
            dtype=np.int32,
        )
        if np.any(self._joint_ids < 0) or len(set(self._joint_ids.tolist())) != self.manifest.joint_count:
            raise ValueError("Manifest joints do not map uniquely into MuJoCo.")
        self._joint_qpos_addresses = np.asarray(self.model.jnt_qposadr[self._joint_ids], dtype=np.int32)
        self._joint_dof_addresses = np.asarray(self.model.jnt_dofadr[self._joint_ids], dtype=np.int32)
        actuator_joint_ids = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        actuator_by_joint = {int(joint_id): index for index, joint_id in enumerate(actuator_joint_ids)}
        try:
            self._actuator_ids_policy = np.asarray(
                [actuator_by_joint[int(joint_id)] for joint_id in self._joint_ids], dtype=np.int32
            )
        except KeyError as exc:
            raise ValueError("A manifest joint has no MuJoCo torque actuator.") from exc
        if self.model.nu != self.manifest.joint_count or len(set(self._actuator_ids_policy.tolist())) != self.model.nu:
            raise ValueError("MuJoCo actuators and manifest joints are not a name-mapped bijection.")
        expected_gear = np.zeros_like(self.model.actuator_gear)
        expected_gear[:, 0] = 1.0
        if not np.allclose(self.model.actuator_gear, expected_gear):
            raise ValueError("MuJoCo torque actuators must have unit joint gear.")
        self._ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sim2sim_ground")
        foot_geom_ids: list[frozenset[int]] = []
        for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geom_ids = frozenset(
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) == body_id
                and (int(self.model.geom_contype[geom_id]) or int(self.model.geom_conaffinity[geom_id]))
            )
            if body_id < 0 or not geom_ids:
                raise ValueError(f"MuJoCo model is missing collidable foot body {body_name!r}.")
            foot_geom_ids.append(geom_ids)
        self._foot_geom_ids = (foot_geom_ids[0], foot_geom_ids[1])
        self._sensor_addresses = {}
        for name, dimension in (("imu_quat", 4), ("imu_gyro", 3), ("frame_pos", 3), ("frame_vel", 3)):
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id < 0 or int(self.model.sensor_dim[sensor_id]) != dimension:
                raise ValueError(f"MuJoCo model is missing {dimension}-D sensor {name!r}.")
            self._sensor_addresses[name] = (int(self.model.sensor_adr[sensor_id]), dimension)

    def _reset(self, feet_clear: bool, start_attitude: str) -> None:
        with self.manifest_path.open(encoding="utf-8") as stream:
            raw_manifest = json.load(stream)
        root_frame = raw_manifest["reference"]["root_frame0"]
        root_position = np.asarray(root_frame["pos"], dtype=np.float64)
        root_quaternion_wxyz = np.roll(np.asarray(root_frame["quat_xyzw"], dtype=np.float64), 1)
        if start_attitude == "level":
            # The exported frame-0 attitude includes the training terrain's
            # pitch. Native Unitree stand instead starts with a level pelvis.
            root_quaternion_wxyz = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        mujoco.mj_resetData(self.model, self.data)
        root_qpos = self._root_qpos_address
        self.data.qpos[root_qpos : root_qpos + 3] = root_position
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = root_quaternion_wxyz
        self.data.qpos[self._joint_qpos_addresses] = self.manifest.default_position
        self.data.qvel[:] = 0.0
        apply_initial_ground_clearance(
            self.model,
            self.data,
            root_qpos,
            self._ground_geom_id,
            self._foot_geom_ids[0] | self._foot_geom_ids[1],
        )
        if feet_clear:
            self.data.qpos[root_qpos + 2] += _FEET_CLEARANCE_M
        mujoco.mj_forward(self.model, self.data)

    def _sensor(self, name: str) -> np.ndarray:
        address, dimension = self._sensor_addresses[name]
        return np.asarray(self.data.sensordata[address : address + dimension], dtype=np.float64).copy()

    @property
    def joint_positions(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self._joint_qpos_addresses], dtype=np.float64).copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        return np.asarray(self.data.qvel[self._joint_dof_addresses], dtype=np.float64).copy()

    @property
    def pelvis_position(self) -> np.ndarray:
        return self._sensor("frame_pos")

    @property
    def pelvis_quaternion(self) -> np.ndarray:
        return self._sensor("imu_quat")

    @property
    def pelvis_linear_velocity(self) -> np.ndarray:
        return self._sensor("frame_vel")

    @property
    def pelvis_angular_velocity(self) -> np.ndarray:
        return self._sensor("imu_gyro")

    def remote_bytes(self) -> bytes:
        active: list[str] = []
        for pulse in self.remote_pulses:
            if pulse.time_s <= self.time_s < pulse.time_s + pulse.hold_s:
                active.extend(pulse.buttons)
        return encode_remote(active)

    @property
    def fixture_active(self) -> bool:
        """Whether native or explicit operator support currently holds the base."""
        if self.operator_support_fixture:
            return True
        if self.loco_client is None:
            return self.native_fsm_id in _NATIVE_STAND_FSM_IDS
        return not self.loco_client.user_control and self.loco_client.native_stand_fixture_latched

    def rpc_result(self, method: str) -> int:
        """Return one injected RPC failure, then resume successful calls."""
        if self.rpc_fail == method and not self._rpc_failure_injected:
            self._rpc_failure_injected = True
            return 1
        return 0

    def accept_low_command(self, command: Any) -> bool:
        if self.crc.Crc(command) != command.crc:
            return False
        self.latest_command_q = np.asarray(
            [command.motor_cmd[slot].q for slot in self.manifest.sdk_slots], dtype=np.float64
        )
        self.latest_command_kp = np.asarray(
            [command.motor_cmd[slot].kp for slot in self.manifest.sdk_slots], dtype=np.float64
        )
        self.latest_command_kd = np.asarray(
            [command.motor_cmd[slot].kd for slot in self.manifest.sdk_slots], dtype=np.float64
        )
        return bool(
            np.all(np.isfinite(self.latest_command_q))
            and np.all(np.isfinite(self.latest_command_kp))
            and np.all(np.isfinite(self.latest_command_kd))
        )

    def emit_low_state(self) -> Any:
        self._deliver_pending_states()
        state = self._state_factory()
        positions = self.joint_positions
        velocities = self.joint_velocities
        for policy_index, slot in enumerate(self.manifest.sdk_slots):
            motor = state.motor_state[slot]
            motor.q = float(positions[policy_index])
            motor.dq = float(velocities[policy_index])
            motor.tau_est = float(self.applied_torque[policy_index])
            motor.temperature[:] = [30, 30]
        state.imu_state.quaternion[:] = self.pelvis_quaternion.tolist()
        state.imu_state.gyroscope[:] = self.pelvis_angular_velocity.tolist()
        state.tick = int(round(1_000.0 * self.time_s))
        state.mode_pr = 0
        state.mode_machine = 4
        state.wireless_remote[:] = self.remote_bytes()
        state.crc = self.crc.Crc(state)
        if self.crc_corrupt_at is not None and not self._crc_corruption_injected and self.time_s >= self.crc_corrupt_at:
            state.crc ^= 1
            self._crc_corruption_injected = True
        feedback_dropped = self._feedback_drop_active()
        if not feedback_dropped:
            self._pending_states.append((self.time_s + self.feedback_latency_s, state))
            self._deliver_pending_states()
        return state

    def _feedback_drop_active(self) -> bool:
        return bool(
            self.drop_feedback_at is not None
            and self.drop_feedback_duration_s is not None
            and self.drop_feedback_at <= self.time_s < self.drop_feedback_at + self.drop_feedback_duration_s
        )

    def _deliver_pending_states(self) -> None:
        if self._feedback_drop_active():
            return
        while self._pending_states and self._pending_states[0][0] <= self.time_s + 1.0e-12:
            _, state = self._pending_states.popleft()
            if self.state_handler is not None:
                self.state_handler(state)
                self.state_updates += 1

    def sleep(self, duration_s: float) -> None:
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("sleep duration must be finite and non-negative")
        end_time = self.time_s + float(duration_s)
        epsilon = 1.0e-12
        while self._next_physics_time_s <= end_time + epsilon:
            self.time_s = self._next_physics_time_s
            self._feed_scheduled_stdin()
            self._step_physics()
            self._next_physics_time_s += self.sim_dt
        self.time_s = end_time
        self._feed_scheduled_stdin()

    def monotonic(self) -> float:
        """Return deterministic time with a nanosecond computation-order tick."""
        # The hardware recorder requires separately published commands to have
        # strictly increasing timestamps. Real Python work naturally consumes
        # time between calls; model that ordering without adding a physics step.
        self.time_s += 1.0e-9
        self._feed_scheduled_stdin()
        return self.time_s

    def _feed_scheduled_stdin(self) -> None:
        if self._stdin_transport is not None:
            self._stdin_transport.advance(self.time_s)

    def _step_physics(self) -> None:
        positions = self.joint_positions
        velocities = self.joint_velocities
        active_fsm_id = self.native_fsm_id if self.loco_client is None else self.loco_client.fsm_id
        if self.loco_client is not None and self.loco_client.user_control:
            target = self.latest_command_q
            stiffness = self.latest_command_kp
            damping = self.latest_command_kd
        elif active_fsm_id in _NATIVE_STAND_FSM_IDS:
            target = self.manifest.default_position
            stiffness = self.manifest.stiffness
            damping = self.manifest.damping
        else:
            target = positions
            stiffness = np.zeros(self.manifest.joint_count, dtype=np.float64)
            damping = np.full(self.manifest.joint_count, _PASSIVE_DAMPING, dtype=np.float64)
        torque = stiffness * (target - positions) - damping * velocities
        self.applied_torque = np.clip(torque, -self.manifest.effort_limit, self.manifest.effort_limit)
        if self.emulate_velocity_limit:
            self.applied_torque = saturate_torque_at_velocity_limit(
                self.applied_torque,
                velocities,
                self.manifest.velocity_limit,
            )
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self._actuator_ids_policy] = self.applied_torque
        self._apply_gantry_wrench()
        mujoco.mj_step(self.model, self.data)
        if self.fixture_active:
            # The MJCF has no native whole-body controller. Keep the floating
            # base at its initial stand pose while native stand owns it or an
            # explicit operator-support fixture is active. Joints remain physical.
            root_qpos = self._root_qpos_address
            root_dof = self._root_dof_address
            self.data.qpos[root_qpos : root_qpos + 7] = self._internal_root_qpos
            self.data.qvel[root_dof : root_dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.physics_steps += 1
        self.emit_low_state()
        self._record_truth()

    def _apply_gantry_wrench(self) -> None:
        self.data.xfrc_applied[self._root_body_id, :] = 0.0
        if self.gantry_support_fraction == 0.0:
            return
        root_qpos = self._root_qpos_address
        root_dof = self._root_dof_address
        position = np.asarray(self.data.qpos[root_qpos : root_qpos + 3], dtype=np.float64)
        linear_velocity = np.asarray(self.data.qvel[root_dof : root_dof + 3], dtype=np.float64)
        force = self._gantry_support_force_world.copy()
        force[2] -= _GANTRY_POSITION_STIFFNESS_N_M * (position[2] - self._gantry_reference_position[2])
        force[2] -= _GANTRY_POSITION_DAMPING_N_S_M * linear_velocity[2]
        force = np.clip(force, -_GANTRY_FORCE_LIMIT_N, _GANTRY_FORCE_LIMIT_N)
        quaternion = np.asarray(self.data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float64)
        attitude_error = np.zeros(3, dtype=np.float64)
        mujoco.mju_subQuat(attitude_error, quaternion, self._gantry_reference_quaternion)
        attitude_error[2] = 0.0
        angular_velocity = np.asarray(self.data.qvel[root_dof + 3 : root_dof + 6], dtype=np.float64)
        torque = -_GANTRY_ATTITUDE_STIFFNESS_N_M_RAD * attitude_error
        torque -= _GANTRY_ATTITUDE_DAMPING_N_M_S_RAD * angular_velocity
        torque[2] = 0.0
        torque = np.clip(torque, -_GANTRY_TORQUE_LIMIT_N_M, _GANTRY_TORQUE_LIMIT_N_M)
        self.data.xfrc_applied[self._root_body_id, :3] = force
        self.data.xfrc_applied[self._root_body_id, 3:] = torque

    def foot_contact_forces(self) -> np.ndarray:
        forces = np.zeros((2, 3), dtype=np.float64)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self._ground_geom_id not in (geom1, geom2):
                continue
            wrench_contact = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_id, wrench_contact)
            force_world = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3).T @ wrench_contact[:3]
            for foot_index, geom_ids in enumerate(self._foot_geom_ids):
                if geom2 in geom_ids:
                    forces[foot_index] += force_world
                elif geom1 in geom_ids:
                    forces[foot_index] -= force_world
        return np.maximum(forces[:, 2], 0.0)

    def _record_truth(self) -> None:
        quaternion = self.pelvis_quaternion
        self._log["time"].append(self.time_s)
        self._log["qpos"].append(np.asarray(self.data.qpos, dtype=np.float64).copy())
        self._log["qvel"].append(np.asarray(self.data.qvel, dtype=np.float64).copy())
        self._log["applied_torque"].append(self.applied_torque.copy())
        self._log["pelvis_position"].append(self.pelvis_position)
        self._log["pelvis_quaternion_wxyz"].append(quaternion)
        self._log["pelvis_linear_velocity"].append(self.pelvis_linear_velocity)
        self._log["pelvis_angular_velocity"].append(self.pelvis_angular_velocity)
        self._log["foot_contact_normal_forces"].append(self.foot_contact_forces())
        self._log["body_tilt"].append(runner._body_tilt(quaternion))
        self._log["lowcmd_q"].append(self.latest_command_q.copy())
        self._log["lowcmd_kp"].append(self.latest_command_kp.copy())
        self._log["lowcmd_kd"].append(self.latest_command_kd.copy())
        self._log["wireless_remote"].append(np.frombuffer(self.remote_bytes(), dtype=np.uint8).copy())
        self._log["fsm_id"].append(self.native_fsm_id if self.loco_client is None else self.loco_client.fsm_id)
        self._log["user_control"].append(False if self.loco_client is None else self.loco_client.user_control)
        self._log["fixture_active"].append(self.fixture_active)

    def log_arrays(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self._log.items()}

    def finalize_log(self) -> None:
        """Reflect runner cleanup RPCs in the final 500 Hz truth row."""
        if self.loco_client is not None:
            self._log["fsm_id"][-1] = self.loco_client.fsm_id
            self._log["user_control"][-1] = self.loco_client.user_control
            self._log["fixture_active"][-1] = self.fixture_active

    def write_log(self, path: Path) -> None:
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "args": self._metadata_args,
            "emulate_velocity_limit": self.emulate_velocity_limit,
            "model_path": str(self.model_path),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        }
        metadata_json = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(resolved, **self.log_arrays(), metadata_json=metadata_json)

    def summary(self, exit_code: int) -> str:
        arrays = self.log_arrays()
        maximum_tilt_deg = math.degrees(float(np.max(arrays["body_tilt"])))
        maximum_torque_fraction = float(
            np.max(np.abs(arrays["applied_torque"]) / self.manifest.effort_limit[np.newaxis, :])
        )
        apex_height = float(np.max(arrays["pelvis_position"][:, 2]))
        maximum_foot_contact = float(np.max(arrays["foot_contact_normal_forces"]))
        final_position = arrays["pelvis_position"][-1]
        unloaded = np.all(arrays["foot_contact_normal_forces"] <= _CONTACT_UNLOADED_THRESHOLD_N, axis=1)
        minimum_steps = int(math.ceil(0.040 / self.sim_dt))
        ever_left_ground = bool(
            len(unloaded) >= minimum_steps
            and np.convolve(unloaded.astype(np.int32), np.ones(minimum_steps, dtype=np.int32), mode="valid").max()
            >= minimum_steps
        )
        planar_displacement = float(np.linalg.norm(final_position[:2] - self._initial_pelvis_position[:2]))
        return (
            f"SIM SUMMARY: runner_exit={exit_code}, max_tilt={maximum_tilt_deg:.2f} deg, "
            f"max_torque_limit_fraction={maximum_torque_fraction:.3f}, apex_pelvis_height={apex_height:.3f} m, "
            f"max_foot_contact={maximum_foot_contact:.2f} N, left_ground={ever_left_ground}, "
            f"final_pelvis_height={final_position[2]:.3f} m, "
            f"final_planar_displacement={planar_displacement:.3f} m, fixture_active={self.fixture_active}"
        )

    def physics_verdict(
        self,
        audit_path: Path | None,
        *,
        ground_jump: bool,
        contactless_rehearsal: bool,
    ) -> PhysicsVerdict:
        """Evaluate runner-independent physical constraints and jump outcomes."""
        arrays = self.log_arrays()
        failures: list[str] = []
        facts: list[str] = []
        final_tilt = float(arrays["body_tilt"][-1])
        if final_tilt > _FINAL_TILT_LIMIT_RAD:
            failures.append(f"final tilt {math.degrees(final_tilt):.2f} deg > 10.00 deg")
        joint_positions = arrays["qpos"][:, self._joint_qpos_addresses]
        lower_violation = self.manifest.joint_position_lower - joint_positions
        upper_violation = joint_positions - self.manifest.joint_position_upper
        maximum_joint_violation = float(max(np.max(lower_violation), np.max(upper_violation), 0.0))
        if maximum_joint_violation > _JOINT_LIMIT_TOLERANCE_RAD:
            failures.append(f"joint-limit violation {maximum_joint_violation:.4f} rad > 0.0100 rad")
        maximum_torque_fraction = float(
            np.max(np.abs(arrays["applied_torque"]) / self.manifest.effort_limit[np.newaxis, :])
        )
        if maximum_torque_fraction > 1.0 + 1.0e-9:
            failures.append(f"torque fraction {maximum_torque_fraction:.6f} > 1.0")

        jump_intervals = _audit_jump_intervals(audit_path)
        completed_flights = 0
        for jump_index, (start_s, end_s) in enumerate(jump_intervals):
            interval_end = max(end_s, start_s + self.sim_dt)
            sample_indices = np.flatnonzero((arrays["time"] >= start_s) & (arrays["time"] <= interval_end))
            unloaded = (
                np.all(arrays["foot_contact_normal_forces"][sample_indices] < _FLIGHT_CONTACT_THRESHOLD_N, axis=1)
                if sample_indices.size
                else np.zeros(0, dtype=np.bool_)
            )
            flight_run = _first_true_run(unloaded, int(math.ceil(_FLIGHT_MINIMUM_DURATION_S / self.sim_dt)))
            if flight_run is None:
                failures.append(f"jump {jump_index}: no 40 ms flight")
                continue
            flight_start_index = int(sample_indices[flight_run[0]])
            flight_end_index = int(sample_indices[flight_run[1]])
            apex_height = float(np.max(arrays["pelvis_position"][sample_indices, 2]))
            if contactless_rehearsal:
                completed_flights += 1
                facts.append(f"jump {jump_index}: contactless flight, apex={apex_height:.3f} m")
                continue
            touchdown_candidates = np.flatnonzero(
                np.all(
                    arrays["foot_contact_normal_forces"][flight_end_index + 1 :] >= _TOUCHDOWN_CONTACT_THRESHOLD_N,
                    axis=1,
                )
            )
            if touchdown_candidates.size == 0:
                failures.append(f"jump {jump_index}: no bilateral touchdown after flight")
                continue
            touchdown_index = flight_end_index + 1 + int(touchdown_candidates[0])
            displacement_index = min(
                int(np.searchsorted(arrays["time"], arrays["time"][touchdown_index] + 1.0)), len(arrays["time"]) - 1
            )
            start_index = int(np.searchsorted(arrays["time"], start_s))
            planar_displacement = float(
                np.linalg.norm(
                    arrays["pelvis_position"][displacement_index, :2] - arrays["pelvis_position"][start_index, :2]
                )
            )
            completed_flights += 1
            flight_duration = arrays["time"][flight_end_index] - arrays["time"][flight_start_index]
            facts.append(
                f"jump {jump_index}: flight={flight_duration:.3f} s, "
                f"apex={apex_height:.3f} m, displacement={planar_displacement:.3f} m"
            )
        if ground_jump and completed_flights == 0:
            failures.append("ground jump completed zero flights")
        details = tuple(failures if failures else (facts or ["no jump intervals; static constraints passed"]))
        return PhysicsVerdict(not failures, details)


def _first_true_run(values: np.ndarray, minimum_length: int) -> tuple[int, int] | None:
    run_start: int | None = None
    for index, value in enumerate(values):
        if value and run_start is None:
            run_start = index
        if not value and run_start is not None:
            if index - run_start >= minimum_length:
                return run_start, index - 1
            run_start = None
    if run_start is not None and len(values) - run_start >= minimum_length:
        return run_start, len(values) - 1
    return None


def _audit_jump_intervals(path: Path | None) -> tuple[tuple[float, float], ...]:
    if path is None or not path.exists():
        return ()
    with np.load(path, allow_pickle=False) as audit:
        if "fsm_state" not in audit or "published_at" not in audit:
            return ()
        states = np.asarray(audit["fsm_state"])
        times = np.asarray(audit["published_at"], dtype=np.float64)
    intervals: list[tuple[float, float]] = []
    index = 0
    while index < len(states):
        if states[index] != "JUMP":
            index += 1
            continue
        start = float(times[index])
        while index + 1 < len(states) and states[index + 1] == "JUMP":
            index += 1
        end = float(times[index + 1]) if index + 1 < len(times) else float(times[index])
        intervals.append((start, end))
        index += 1
    return tuple(intervals)


def _physics_exit_code(runner_exit_code: int, verdict: PhysicsVerdict) -> int:
    return 3 if runner_exit_code == 0 and not verdict.passed else runner_exit_code


@contextlib.contextmanager
def _patched_runner(simulation: RunnerMujocoSimulation, runner_argv: Sequence[str]):
    import unitree_sdk2py.core.channel as channel_module
    import unitree_sdk2py.g1.audio.g1_audio_client as audio_module
    import unitree_sdk2py.g1.loco.g1_loco_client as loco_module

    global _ACTIVE_SIMULATION
    if _ACTIVE_SIMULATION is not None:
        raise RuntimeError("Nested runner-in-the-loop simulations are not supported.")
    patches: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, value: Any) -> None:
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    old_argv = sys.argv
    old_stdin = sys.stdin
    stdin_transport = _BlockingTimedStdin(simulation.stdin_lines)
    _ACTIVE_SIMULATION = simulation
    simulation._stdin_transport = stdin_transport
    try:
        patch(channel_module, "ChannelFactoryInitialize", _fake_channel_factory_initialize)
        patch(channel_module, "ChannelSubscriber", _FakeChannelSubscriber)
        patch(channel_module, "ChannelPublisher", _FakeChannelPublisher)
        patch(loco_module, "LocoClient", FakeLocoClient)
        patch(audio_module, "AudioClient", _FakeAudioClient)
        patch(runner.time, "monotonic", simulation.monotonic)
        patch(runner.time, "perf_counter", simulation.monotonic)
        patch(runner.time, "sleep", simulation.sleep)
        patch(runner.socket, "if_nameindex", lambda: [(1, str(runner_argv[0]))])
        sys.argv = [str(runner.__file__), *runner_argv]
        sys.stdin = stdin_transport.stream
        stdin_transport.advance(simulation.time_s)
        yield
    finally:
        sys.argv = old_argv
        sys.stdin = old_stdin
        simulation._stdin_transport = None
        stdin_transport.close()
        for owner, name, value in reversed(patches):
            setattr(owner, name, value)
        _ACTIVE_SIMULATION = None


def run_simulation(
    simulation: RunnerMujocoSimulation,
    runner_argv: Sequence[str],
) -> int:
    """Call the unmodified runner main with deterministic simulation patches."""
    with _patched_runner(simulation, runner_argv):
        return runner.main()


def _runner_requests_rehearsal(runner_args: Sequence[str]) -> bool:
    return "--gantry_policy_rehearsal" in runner_args


def _runner_option(runner_args: Sequence[str], name: str, default: str | None = None) -> str | None:
    for index, argument in enumerate(runner_args):
        if argument == name:
            if index + 1 >= len(runner_args):
                return default
            return runner_args[index + 1]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return default


def _parse_cli(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    try:
        separator = argv.index("--")
    except ValueError:
        separator = -1
    if separator < 0:
        raise ValueError("Separate simulator options from run_fsm_g1 arguments with --.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--overlay", type=Path, default=_DEFAULT_OVERLAY)
    parser.add_argument("--gantry_support_fraction", type=float, default=0.0)
    parser.add_argument("--feet_clear", action="store_true")
    parser.add_argument("--start_attitude", choices=("level", "manifest"), default="level")
    parser.add_argument("--remote_script", type=Path)
    parser.add_argument("--stdin_script", type=Path)
    parser.add_argument("--sim_log", type=Path)
    parser.add_argument("--native_fsm_id", type=int, default=_PASSIVE_FSM_ID)
    parser.add_argument("--drop_feedback_ms", type=float)
    parser.add_argument("--drop_feedback_at", type=float)
    parser.add_argument(
        "--rpc_fail",
        choices=(
            "Init",
            "SetTimeout",
            "GetFsmId",
            "SetFsmId",
            "SwitchToUserCtrl",
            "SwitchToInternalCtrl",
            "SetVelocity",
        ),
    )
    parser.add_argument("--crc_corrupt_at", type=float)
    parser.add_argument("--feedback_latency_ms", type=float, default=0.0)
    parser.add_argument(
        "--emulate_velocity_limit",
        action="store_true",
        help="Emulate manifest actuator velocity limits with torque-speed saturation.",
    )
    args = parser.parse_args(argv[:separator])
    runner_args = list(argv[separator + 1 :])
    if not runner_args:
        raise ValueError("At least the runner network-interface argument is required after --.")
    return args, runner_args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real hardware runner against deterministic MuJoCo."""
    cli_args = list(sys.argv[1:] if argv is None else argv)
    try:
        args, runner_args = _parse_cli(cli_args)
        rehearsal = _runner_requests_rehearsal(runner_args)
        ground_jump = "--ground_jump" in runner_args
        operator_support_fixture = not rehearsal and not ground_jump
        remote_pulses = _parse_remote_script(args.remote_script, rehearsal=rehearsal, ground_session=ground_jump)
        stdin_lines = _parse_stdin_script(args.stdin_script)
        metadata_args = {
            "simulator": cli_args[: cli_args.index("--")],
            "runner": runner_args,
            "operator_support_fixture": operator_support_fixture,
            "emulate_velocity_limit": args.emulate_velocity_limit,
        }
        simulation = RunnerMujocoSimulation(
            args.manifest,
            args.model,
            args.overlay,
            remote_pulses=remote_pulses,
            native_fsm_id=args.native_fsm_id,
            gantry_support_fraction=args.gantry_support_fraction,
            feet_clear=args.feet_clear,
            start_attitude=args.start_attitude,
            stdin_lines=stdin_lines,
            operator_support_fixture=operator_support_fixture,
            emulate_velocity_limit=args.emulate_velocity_limit,
            drop_feedback_ms=args.drop_feedback_ms,
            drop_feedback_at=args.drop_feedback_at,
            rpc_fail=args.rpc_fail,
            crc_corrupt_at=args.crc_corrupt_at,
            feedback_latency_ms=args.feedback_latency_ms,
            metadata_args=metadata_args,
        )
    except (OSError, ValueError) as exc:
        print(f"SIM REFUSED: {exc}")
        return 2
    print(f"SIM START ATTITUDE: {simulation.start_attitude}.")
    if simulation.operator_support_fixture:
        print("SIM OPERATOR FIXTURE: stand-mode base support remains active; joint physics remains active.")
    elif simulation.fixture_active:
        print("SIM NATIVE FIXTURE: native stand holds the base until SwitchToUserCtrl; joint physics remains active.")
    try:
        exit_code = run_simulation(simulation, runner_args)
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"SIM ERROR: {type(exc).__name__}: {exc}")
        exit_code = 1
    simulation.finalize_log()
    audit_value = _runner_option(
        runner_args,
        "--ground_log" if ground_jump else "--rehearsal_log",
    )
    audit_path = None if audit_value is None else Path(audit_value).resolve()
    verdict = simulation.physics_verdict(
        audit_path,
        ground_jump=ground_jump,
        contactless_rehearsal=rehearsal,
    )
    if args.sim_log is not None:
        simulation.write_log(args.sim_log)
    print(simulation.summary(exit_code))
    print(f"SIM VERDICT: {'PASS' if verdict.passed else 'FAIL'} {'; '.join(verdict.details)}")
    return _physics_exit_code(exit_code, verdict)


if __name__ == "__main__":
    raise SystemExit(main())

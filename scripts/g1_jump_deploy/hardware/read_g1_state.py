# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read and validate G1 low-level state without creating a command publisher."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_DEFAULT_MANIFEST = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"
_LOW_STATE_TOPIC = "rt/lowstate"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to G1 LowState and validate feedback. This program cannot send motor commands."
    )
    parser.add_argument("network_interface", help="Ethernet interface connected to G1, for example enp131s0.")
    parser.add_argument("--duration", type=float, default=10.0, help="Acquisition duration in seconds.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Deployment manifest whose 23-DOF SDK2 mapping should be checked.",
    )
    args = parser.parse_args()
    if args.duration <= 0.0 or not math.isfinite(args.duration):
        parser.error("--duration must be a finite positive number")
    return args


def _validate_network_interface(name: str) -> None:
    available = {interface_name for _, interface_name in socket.if_nameindex()}
    if name not in available:
        raise ValueError(f"Network interface {name!r} does not exist; available interfaces: {sorted(available)}")


def _load_manifest_mapping(path: Path) -> tuple[tuple[str, ...], tuple[int, ...]]:
    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    joints = manifest.get("joints")
    if not isinstance(joints, dict):
        raise ValueError(f"Manifest has no joints object: {path}")
    names = tuple(joints.get("names", ()))
    slots = tuple(joints.get("unitree_sdk2_slots", ()))
    if len(names) != 23 or len(slots) != 23:
        raise ValueError(f"Expected a 23-DOF manifest mapping, got {len(names)} names and {len(slots)} slots")
    if len(set(slots)) != len(slots) or any(not isinstance(slot, int) or slot < 0 or slot >= 35 for slot in slots):
        raise ValueError("Manifest SDK2 slots must be 23 unique integers in [0, 35)")
    return names, slots


def _print_first_state(state: object, names: tuple[str, ...], slots: tuple[int, ...]) -> None:
    imu = state.imu_state
    print(f"First valid packet: tick={state.tick}, mode_pr={state.mode_pr}, mode_machine={state.mode_machine}")
    print(f"IMU quaternion(raw)={list(imu.quaternion)}")
    print(f"IMU rpy(rad)={list(imu.rpy)}")
    print(f"IMU gyroscope(rad/s)={list(imu.gyroscope)}")
    print("Manifest-order 23-DOF joint state:")
    for name, slot in zip(names, slots, strict=True):
        motor = state.motor_state[slot]
        print(
            f"  slot={slot:2d} {name:27s} q={motor.q:+.6f} rad  "
            f"dq={motor.dq:+.6f} rad/s  tau_est={motor.tau_est:+.3f} N m"
        )


def _state_is_finite(state: object, slots: tuple[int, ...]) -> bool:
    imu_values = (*state.imu_state.quaternion, *state.imu_state.gyroscope, *state.imu_state.rpy)
    motor_values = tuple(
        value
        for slot in slots
        for value in (state.motor_state[slot].q, state.motor_state[slot].dq, state.motor_state[slot].tau_est)
    )
    return all(math.isfinite(value) for value in (*imu_values, *motor_values))


def _run(args: argparse.Namespace) -> int:
    _validate_network_interface(args.network_interface)
    names, slots = _load_manifest_mapping(args.manifest.resolve())

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as exc:
        raise RuntimeError("Run this probe from the g1_deploy Conda environment.") from exc

    print("READ-ONLY G1 PROBE: no command publisher or control client is created.")
    print(f"Interface={args.network_interface}, topic={_LOW_STATE_TOPIC}, duration={args.duration:.1f}s")
    ChannelFactoryInitialize(0, args.network_interface)
    subscriber = ChannelSubscriber(_LOW_STATE_TOPIC, LowState_)
    subscriber.Init()
    crc = CRC()

    start_time = time.monotonic()
    deadline = start_time + args.duration
    first_valid_time: float | None = None
    previous_valid_time: float | None = None
    valid_packets = 0
    crc_errors = 0
    nonfinite_packets = 0
    maximum_gap_s = 0.0
    last_state = None
    torque_sums = [0.0] * len(slots)
    torque_maximum_abs = [0.0] * len(slots)

    try:
        while time.monotonic() < deadline:
            timeout_s = min(0.5, max(0.001, deadline - time.monotonic()))
            state = subscriber.Read(timeout_s)
            if state is None:
                continue
            if crc.Crc(state) != state.crc:
                crc_errors += 1
                continue
            if len(state.motor_state) != 35 or not _state_is_finite(state, slots):
                nonfinite_packets += 1
                continue

            arrival_time = time.monotonic()
            if first_valid_time is None:
                first_valid_time = arrival_time
                _print_first_state(state, names, slots)
            if previous_valid_time is not None:
                maximum_gap_s = max(maximum_gap_s, arrival_time - previous_valid_time)
            previous_valid_time = arrival_time
            valid_packets += 1
            last_state = state
            for index, slot in enumerate(slots):
                torque = float(state.motor_state[slot].tau_est)
                torque_sums[index] += torque
                torque_maximum_abs[index] = max(torque_maximum_abs[index], abs(torque))
    finally:
        subscriber.Close()

    elapsed_s = time.monotonic() - start_time
    rate_hz = valid_packets / elapsed_s
    print(
        f"Summary: valid={valid_packets}, rate={rate_hz:.1f}Hz, max_gap={maximum_gap_s * 1000.0:.2f}ms, "
        f"crc_errors={crc_errors}, nonfinite={nonfinite_packets}"
    )
    if last_state is not None:
        print(
            f"Last packet: tick={last_state.tick}, mode_pr={last_state.mode_pr}, mode_machine={last_state.mode_machine}"
        )
        print("Manifest-order torque-estimate statistics:")
        for index, (name, slot) in enumerate(zip(names, slots, strict=True)):
            mean_torque = torque_sums[index] / valid_packets
            print(
                f"  slot={slot:2d} {name:27s} mean={mean_torque:+.3f} N m  max_abs={torque_maximum_abs[index]:.3f} N m"
            )

    if valid_packets == 0:
        print(
            "FAIL: no valid LowState packets received. Check robot power, cable, interface address, and DDS interface."
        )
        return 2
    if crc_errors or nonfinite_packets:
        print("FAIL: feedback integrity errors were detected; motor commands must remain disabled.")
        return 3
    print("PASS: read-only G1 feedback is valid and the manifest slot mapping loaded successfully.")
    return 0


def main() -> int:
    """Run the read-only G1 state probe."""
    return _run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

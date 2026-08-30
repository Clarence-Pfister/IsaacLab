# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Speak a short mode announcement through the G1 audio service."""

from __future__ import annotations

import argparse
import socket


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use G1 text-to-speech without creating any motor command channel.")
    parser.add_argument("network_interface", help="Ethernet interface connected to G1, for example enp131s0.")
    parser.add_argument("text", nargs="?", default="Jump mode", help="Short phrase to speak.")
    parser.add_argument("--speaker_id", type=int, default=0, help="Unitree text-to-speech speaker ID.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Audio service timeout in seconds.")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("text must not be empty")
    if args.speaker_id < 0:
        parser.error("--speaker_id must be non-negative")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    """Request one G1 text-to-speech announcement."""
    args = _parse_args()
    available_interfaces = {name for _, name in socket.if_nameindex()}
    if args.network_interface not in available_interfaces:
        raise ValueError(
            f"Network interface {args.network_interface!r} does not exist; "
            f"available interfaces: {sorted(available_interfaces)}"
        )

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    except ImportError as exc:
        raise RuntimeError("Run this command from the g1_deploy Conda environment.") from exc

    print("AUDIO-ONLY G1 CUE: no motor command publisher is created.")
    ChannelFactoryInitialize(0, args.network_interface)
    audio_client = AudioClient()
    audio_client.SetTimeout(args.timeout)
    audio_client.Init()
    return_code = audio_client.TtsMaker(args.text, args.speaker_id)
    if return_code != 0:
        print(f"FAIL: G1 audio service returned code {return_code}.")
        return 2
    print(f"PASS: requested speech {args.text!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

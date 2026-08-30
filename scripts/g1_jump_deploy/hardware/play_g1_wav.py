# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a 16 kHz mono WAV cue through the G1 audio service."""

from __future__ import annotations

import argparse
import socket
import time
import wave
from pathlib import Path

_SAMPLE_RATE = 16_000
_CHANNEL_COUNT = 1
_SAMPLE_WIDTH_BYTES = 2
_CHUNK_DURATION_SECONDS = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a WAV file on G1 without creating any motor command channel.")
    parser.add_argument("network_interface", help="Ethernet interface connected to G1, for example enp131s0.")
    parser.add_argument("wav_path", type=Path, help="16-bit, 16 kHz, mono PCM WAV file to play.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Audio service timeout in seconds.")
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    return args


def _read_pcm(wav_path: Path) -> tuple[bytes, float]:
    """Read and validate a WAV file for the G1 audio stream.

    Args:
        wav_path: Path to the PCM WAV file.

    Returns:
        Raw PCM bytes and audio duration [s].
    """
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("WAV compression is not supported")
            if wav_file.getnchannels() != _CHANNEL_COUNT:
                raise ValueError("WAV must be mono")
            if wav_file.getsampwidth() != _SAMPLE_WIDTH_BYTES:
                raise ValueError("WAV must use 16-bit samples")
            if wav_file.getframerate() != _SAMPLE_RATE:
                raise ValueError("WAV sample rate must be 16000 Hz")
            frame_count = wav_file.getnframes()
            pcm_data = wav_file.readframes(frame_count)
    except (FileNotFoundError, wave.Error) as exc:
        raise ValueError(f"Cannot read WAV file {wav_path}: {exc}") from exc

    if not pcm_data:
        raise ValueError("WAV contains no audio samples")
    return pcm_data, frame_count / _SAMPLE_RATE


def main() -> int:
    """Play one WAV cue using only the G1 audio service."""
    args = _parse_args()
    available_interfaces = {name for _, name in socket.if_nameindex()}
    if args.network_interface not in available_interfaces:
        raise ValueError(
            f"Network interface {args.network_interface!r} does not exist; "
            f"available interfaces: {sorted(available_interfaces)}"
        )

    pcm_data, duration = _read_pcm(args.wav_path)

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

    app_name = "g1_jump_deploy"
    stream_id = str(time.time_ns())
    bytes_per_second = _SAMPLE_RATE * _CHANNEL_COUNT * _SAMPLE_WIDTH_BYTES
    chunk_size = int(bytes_per_second * _CHUNK_DURATION_SECONDS)
    for chunk_index, offset in enumerate(range(0, len(pcm_data), chunk_size)):
        chunk = pcm_data[offset : offset + chunk_size]
        return_code, _ = audio_client.PlayStream(app_name, stream_id, chunk)
        if return_code != 0:
            audio_client.PlayStop(app_name)
            print(f"FAIL: G1 audio stream chunk {chunk_index} returned code {return_code}.")
            return 2
        time.sleep(len(chunk) / bytes_per_second)

    time.sleep(0.25)
    stop_code = audio_client.PlayStop(app_name)
    if stop_code != 0:
        print(f"WARNING: cue played, but PlayStop returned code {stop_code}.")
    print(f"PASS: streamed {args.wav_path} ({duration:.2f} s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

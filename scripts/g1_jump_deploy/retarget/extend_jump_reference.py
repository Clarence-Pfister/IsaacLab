# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extend the G1 jump CSV with default-stance holds at both ends."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = _REPO_ROOT / "data_storage" / "perfect_jump_ground_aligned.csv"
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "data_storage" / "perfect_jump_extended.csv"
DEFAULT_MANIFEST_PATH = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"
ROOT_POSITION_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_QUATERNION_COLUMNS = (
    "root_quaternionW",
    "root_quaternionX",
    "root_quaternionY",
    "root_quaternionZ",
)
FOOT_POSITION_COLUMNS = (
    "left_foot_x",
    "left_foot_y",
    "left_foot_z",
    "right_foot_x",
    "right_foot_y",
    "right_foot_z",
)


def quintic_smoothstep(value: float) -> float:
    """Return a quintic blend with zero first derivative at both endpoints."""
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def quintic_smoothstep_derivative(value: float) -> float:
    """Return the derivative of :func:`quintic_smoothstep`."""
    return 30.0 * value * value * (value - 1.0) * (value - 1.0)


def _slerp_wxyz(start: np.ndarray, end: np.ndarray, blend: float) -> np.ndarray:
    """Spherically interpolate normalized WXYZ quaternions along the short arc."""
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start + blend * (end - start)
    else:
        angle = math.acos(dot)
        result = (math.sin((1.0 - blend) * angle) * start + math.sin(blend * angle) * end) / math.sin(angle)
    return result / np.linalg.norm(result)


def _seconds_to_frames(duration_s: float, fps: float, name: str, *, positive: bool = False) -> int:
    if not math.isfinite(duration_s) or duration_s < 0.0 or (positive and duration_s <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}, got {duration_s}.")
    frame_count = round(duration_s * fps)
    if not math.isclose(duration_s * fps, frame_count, abs_tol=1.0e-9):
        raise ValueError(f"{name} must resolve to a whole number of frames at {fps:g} FPS, got {duration_s}.")
    return frame_count


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV is empty: {path}.") from exc
        rows = [[float(value) for value in row] for row in reader]
    if not rows:
        raise ValueError(f"Input CSV contains no motion frames: {path}.")
    if any(len(row) != len(columns) for row in rows):
        raise ValueError(f"Input CSV contains a row whose width does not match its header: {path}.")
    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Input CSV contains non-finite values: {path}.")
    return columns, values


def _load_stance(manifest_path: Path, columns: list[str], frame_zero: np.ndarray) -> tuple[list[int], np.ndarray]:
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    joints = manifest.get("joints", {})
    joint_names = joints.get("names")
    default_pos = joints.get("default_pos")
    if not isinstance(joint_names, list) or not isinstance(default_pos, list) or len(joint_names) != len(default_pos):
        raise ValueError(f"Manifest joints.names/default_pos are missing or inconsistent: {manifest_path}.")
    try:
        joint_indices = [columns.index(name) for name in joint_names]
    except ValueError as exc:
        raise ValueError(f"Manifest joint is missing from input CSV: {exc}.") from exc
    stance = np.asarray(default_pos, dtype=np.float64)
    if stance.shape != (len(joint_names),) or not np.all(np.isfinite(stance)):
        raise ValueError(f"Manifest joints.default_pos must contain finite scalar values: {manifest_path}.")
    maximum_difference = float(np.max(np.abs(frame_zero[joint_indices] - stance)))
    if maximum_difference >= 1.0e-6:
        raise ValueError(
            "Input frame 0 is not the manifest default stance: "
            f"maximum joint difference is {maximum_difference:.9g} rad (must be < 1e-6 rad)."
        )
    return joint_indices, stance


def extend_reference(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    lead_in_hold_s: float = 1.5,
    lead_out_ramp_s: float = 0.5,
    lead_out_hold_s: float = 1.5,
    fps: float = 30.0,
) -> dict[str, int | float]:
    """Extend a jump reference and write it to disk.

    Args:
        input_path: Source motion CSV path.
        output_path: Destination motion CSV path.
        manifest_path: Deployment manifest providing the exact default joint pose.
        lead_in_hold_s: Duration of the repeated frame-zero hold [s].
        lead_out_ramp_s: Duration of the quintic return-to-stance ramp [s].
        lead_out_hold_s: Duration of the final settled hold [s].
        fps: Reference sampling frequency [Hz].

    Returns:
        Generated frame counts, phase boundaries, and final-hold velocity statistics.
    """
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}.")
    lead_in_frames = _seconds_to_frames(lead_in_hold_s, fps, "lead_in_hold_s")
    ramp_frames = _seconds_to_frames(lead_out_ramp_s, fps, "lead_out_ramp_s", positive=True)
    lead_out_frames = _seconds_to_frames(lead_out_hold_s, fps, "lead_out_hold_s")

    columns, original = _read_csv(input_path)
    required_columns = (*ROOT_POSITION_COLUMNS, *ROOT_QUATERNION_COLUMNS, *FOOT_POSITION_COLUMNS)
    missing_columns = [name for name in required_columns if name not in columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing_columns)}.")

    frame_zero = original[0]
    final_frame = original[-1]
    manifest_joint_indices, manifest_stance = _load_stance(manifest_path, columns, frame_zero)
    root_pos_indices = [columns.index(name) for name in ROOT_POSITION_COLUMNS]
    root_quat_indices = [columns.index(name) for name in ROOT_QUATERNION_COLUMNS]
    foot_indices = [columns.index(name) for name in FOOT_POSITION_COLUMNS]
    joint_indices = [
        index
        for index in range(columns.index(ROOT_QUATERNION_COLUMNS[-1]) + 1, columns.index(FOOT_POSITION_COLUMNS[0]))
    ]

    settled = frame_zero.copy()
    settled[manifest_joint_indices] = manifest_stance
    settled[root_pos_indices[:2]] = final_frame[root_pos_indices[:2]]
    horizontal_offset = final_frame[root_pos_indices[:2]] - frame_zero[root_pos_indices[:2]]
    settled[foot_indices[0:2]] += horizontal_offset
    settled[foot_indices[3:5]] += horizontal_offset

    ramp = np.empty((ramp_frames, len(columns)), dtype=np.float64)
    start_quat = final_frame[root_quat_indices]
    end_quat = frame_zero[root_quat_indices]
    for index in range(ramp_frames):
        blend = quintic_smoothstep((index + 1) / ramp_frames)
        ramp[index] = final_frame + blend * (settled - final_frame)
        ramp[index, root_pos_indices[:2]] = final_frame[root_pos_indices[:2]]
        ramp[index, root_quat_indices] = _slerp_wxyz(start_quat, end_quat, blend)
    ramp[-1] = settled

    lead_in = np.repeat(frame_zero[np.newaxis, :], lead_in_frames, axis=0)
    lead_out = np.repeat(settled[np.newaxis, :], lead_out_frames, axis=0)
    extended = np.concatenate((lead_in, original, ramp, lead_out), axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(extended)

    original_start = lead_in_frames
    ramp_start = original_start + len(original)
    hold_start = ramp_start + ramp_frames
    final_window_frames = round(0.5 * fps)
    final_window = extended[-(final_window_frames + 1) :, joint_indices]
    maximum_final_velocity = float(np.max(np.abs(np.diff(final_window, axis=0) * fps)))
    return {
        "original_frames": len(original),
        "new_frames": len(extended),
        "duration_s": len(extended) / fps,
        "lead_in_start": 0,
        "original_start": original_start,
        "ramp_start": ramp_start,
        "hold_start": hold_start,
        "end": len(extended),
        "max_final_joint_velocity": maximum_final_velocity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--lead-in-hold", type=float, default=1.5)
    parser.add_argument("--lead-out-ramp", type=float, default=0.5)
    parser.add_argument("--lead-out-hold", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    stats = extend_reference(
        args.input,
        args.output,
        args.manifest,
        args.lead_in_hold,
        args.lead_out_ramp,
        args.lead_out_hold,
        args.fps,
    )
    print(f"Frames: {stats['original_frames']} -> {stats['new_frames']}")
    print(f"Duration: {stats['duration_s']:.3f} s at {args.fps:g} FPS")
    print(
        "Boundaries [start, end): "
        f"lead-in [0, {stats['original_start']}), "
        f"original [{stats['original_start']}, {stats['ramp_start']}), "
        f"lead-out ramp [{stats['ramp_start']}, {stats['hold_start']}), "
        f"lead-out hold [{stats['hold_start']}, {stats['end']})"
    )
    print(f"Max joint velocity in final 0.5 s: {stats['max_final_joint_velocity']:.3f} rad/s")


if __name__ == "__main__":
    main()

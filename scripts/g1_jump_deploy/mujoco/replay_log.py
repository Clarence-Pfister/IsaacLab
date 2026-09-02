# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay G1 jump logs in MuJoCo's passive viewer."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from model_overlay import compose_model_xml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_DEFAULT_MODEL = _REPO_ROOT / "data_storage" / "g1_23dof_holo_compat.xml"
_DEFAULT_OVERLAY = _SCRIPT_DIR / "model_overlay.xml"


def detect_log_format(keys: set[str]) -> str:
    """Detect one of the supported G1 jump log formats from its array keys."""
    if {"qpos", "pelvis_position", "pelvis_quaternion_wxyz", "fsm_id", "user_control", "fixture_active"} <= keys:
        return "sim_run_fsm_g1"
    if {"qpos", "pelvis_pose", "fsm_state", "time"} <= keys:
        return "run_fsm_mujoco"
    if {"qpos", "pelvis_pose", "phase", "time"} <= keys:
        return "deploy_mujoco"
    raise ValueError("Log does not match a supported deploy_mujoco, run_fsm_mujoco, or sim_run_fsm_g1 format.")


def _metadata(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in npz:
        return {}
    value = np.asarray(npz["metadata_json"])
    if value.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string.")
    metadata = json.loads(str(value.item()))
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object.")
    return metadata


def _model_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is not None:
            names.append(name)
    return tuple(names)


def _metadata_joint_names(metadata: dict[str, Any], format_name: str) -> tuple[str, ...]:
    field = "mujoco_joint_names" if format_name == "deploy_mujoco" else "joint_names"
    names = metadata.get(field)
    if isinstance(names, list) and all(isinstance(name, str) and name for name in names):
        return tuple(names)
    if format_name != "sim_run_fsm_g1":
        raise ValueError(f"{format_name} metadata is missing {field}.")

    # Compatibility with existing simulator logs written before joint_names was
    # copied into their metadata. Their manifest remains the source of truth.
    manifest_path = metadata.get("manifest_path")
    if not isinstance(manifest_path, str):
        raise ValueError("sim_run_fsm_g1 metadata is missing joint_names and manifest_path.")
    with Path(manifest_path).resolve().open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    names = manifest.get("joints", {}).get("names")
    if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names):
        raise ValueError("Simulator manifest is missing joints.names.")
    return tuple(names)


def _goal(metadata: dict[str, Any]) -> Any:
    if "goal" in metadata:
        return metadata["goal"]
    timeline = metadata.get("operator_timeline")
    if isinstance(timeline, list):
        goals = [entry.get("goal") for entry in timeline if isinstance(entry, dict) and entry.get("goal") is not None]
        if goals:
            return goals[0] if len(goals) == 1 else goals
    args = metadata.get("args")
    runner = args.get("runner") if isinstance(args, dict) else None
    if isinstance(runner, list):
        for index, item in enumerate(runner[:-1]):
            if item == "--goal_sequence":
                return {"pos_x": runner[index + 1]}
    return None


@dataclass
class ReplayLog:
    """Validated replay arrays and their mapping into a compiled model."""

    path: Path
    format_name: str
    metadata: dict[str, Any]
    time_s: np.ndarray
    root_pose: np.ndarray
    joint_position: np.ndarray
    joint_qpos_addresses: np.ndarray
    labels: np.ndarray

    @property
    def frame_count(self) -> int:
        """Number of frames in the log."""
        return len(self.time_s)

    def apply_frame(self, data: mujoco.MjData, frame: int) -> None:
        """Apply one recorded pose and update derived MuJoCo state."""
        data.qpos[:7] = self.root_pose[frame]
        data.qpos[self.joint_qpos_addresses] = self.joint_position[frame]
        mujoco.mj_forward(data.model, data)

    def frame_label(self, frame: int) -> str:
        """Return the phase or FSM description for one frame."""
        if self.format_name == "deploy_mujoco":
            phase = int(self.labels[frame])
            names = self.metadata.get("phase_names", ())
            name = str(names[phase]) if isinstance(names, list) and 0 <= phase < len(names) else str(phase)
            return f"phase={name}"
        if self.format_name == "run_fsm_mujoco":
            return f"fsm_state={self.labels[frame]}"
        fsm_id, user_control, fixture_active = self.labels[frame]
        return f"fsm_id={int(fsm_id)}, user_control={bool(user_control)}, fixture_active={bool(fixture_active)}"


def load_replay_log(path: Path, model: mujoco.MjModel) -> ReplayLog:
    """Load and validate a supported log against a compiled MuJoCo model."""
    path = path.resolve()
    with np.load(path, allow_pickle=False) as npz:
        format_name = detect_log_format(set(npz.files))
        metadata = _metadata(npz)
        time_s = np.asarray(npz["time"], dtype=np.float64).copy()
        recorded_qpos = np.asarray(npz["qpos"], dtype=np.float64).copy()
        if format_name == "sim_run_fsm_g1":
            root_pose = np.concatenate(
                (
                    np.asarray(npz["pelvis_position"], dtype=np.float64),
                    np.asarray(npz["pelvis_quaternion_wxyz"], dtype=np.float64),
                ),
                axis=1,
            )
            labels = np.column_stack((npz["fsm_id"], npz["user_control"], npz["fixture_active"])).copy()
        else:
            root_pose = np.asarray(npz["pelvis_pose"], dtype=np.float64).copy()
            labels = np.asarray(npz["phase" if format_name == "deploy_mujoco" else "fsm_state"]).copy()

    if time_s.ndim != 1 or len(time_s) == 0 or not np.all(np.isfinite(time_s)):
        raise ValueError("time must be a non-empty finite one-dimensional array.")
    if np.any(np.diff(time_s) < 0.0):
        raise ValueError("time must be non-decreasing.")
    if recorded_qpos.ndim != 2 or recorded_qpos.shape[0] != len(time_s):
        raise ValueError("qpos must have one row per time sample.")
    if root_pose.shape != (len(time_s), 7) or not np.all(np.isfinite(root_pose)):
        raise ValueError("The recorded pelvis pose must have finite shape [frame_count, 7].")
    if len(labels) != len(time_s):
        raise ValueError("State labels must have one entry per time sample.")

    free_joint_ids = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if len(free_joint_ids) != 1 or int(model.jnt_qposadr[free_joint_ids[0]]) != 0:
        raise ValueError("Replay model must have one floating-base joint at qpos address zero.")
    source_names = _metadata_joint_names(metadata, format_name)
    if len(source_names) != len(set(source_names)):
        raise ValueError("Recorded joint names must be unique.")
    model_names = _model_joint_names(model)
    if set(source_names) != set(model_names):
        missing = sorted(set(model_names) - set(source_names))
        extra = sorted(set(source_names) - set(model_names))
        raise ValueError(f"Recorded joints do not match the model: missing={missing}, extra={extra}.")

    source_by_name = {name: index for index, name in enumerate(source_names)}
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in model_names], dtype=np.int32
    )
    joint_qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
    if recorded_qpos.shape[1] == len(source_names):
        joint_position = recorded_qpos[:, [source_by_name[name] for name in model_names]]
    elif format_name == "sim_run_fsm_g1" and recorded_qpos.shape[1] == model.nq:
        source_addresses = np.asarray(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in model_names]
        )
        joint_position = recorded_qpos[:, source_addresses]
    else:
        raise ValueError(f"qpos has {recorded_qpos.shape[1]} columns for {len(source_names)} recorded joints.")
    if not np.all(np.isfinite(joint_position)):
        raise ValueError("Recorded joint positions must be finite.")
    return ReplayLog(
        path,
        format_name,
        metadata,
        time_s,
        root_pose,
        joint_position,
        joint_qpos_addresses,
        labels,
    )


def _frame_indices(replay: ReplayLog, start_s: float | None, end_s: float | None) -> np.ndarray:
    start = replay.time_s[0] if start_s is None else start_s
    end = replay.time_s[-1] if end_s is None else end_s
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError("Replay start/end times must be finite and end_s must not precede start_s.")
    indices = np.flatnonzero((replay.time_s >= start) & (replay.time_s <= end))
    if len(indices) == 0:
        raise ValueError(f"No frames fall in the requested interval [{start:.6f}, {end:.6f}] s.")
    return indices


def _print_summary(replay: ReplayLog, indices: np.ndarray) -> None:
    duration = float(replay.time_s[indices[-1]] - replay.time_s[indices[0]])
    print(f"Log format: {replay.format_name}")
    print(f"Goal: {json.dumps(_goal(replay.metadata), sort_keys=True)}")
    print(f"Frames: {len(indices)}; duration: {duration:.6f} s")


def _apply_frames(replay: ReplayLog, data: mujoco.MjData, indices: np.ndarray, viewer: Any | None = None) -> bool:
    previous_label: str | None = None
    wall_start = time.perf_counter()
    replay_start = replay.time_s[indices[0]]
    for frame in indices:
        if viewer is not None:
            remaining = (replay.time_s[frame] - replay_start) / viewer.speed - (time.perf_counter() - wall_start)
            if remaining > 0.0:
                time.sleep(remaining)
            if not viewer.handle.is_running():
                return False
        replay.apply_frame(data, int(frame))
        label = replay.frame_label(int(frame))
        if label != previous_label:
            print(f"t={replay.time_s[frame]:.6f} s: {label}")
            previous_label = label
        if viewer is not None:
            viewer.handle.sync()
    return True


@dataclass
class _ViewerState:
    handle: Any
    speed: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=0.25, help="Playback speed relative to recorded time.")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start_s", type=float, default=None)
    parser.add_argument("--end_s", type=float, default=None)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--overlay", type=Path, default=_DEFAULT_OVERLAY)
    parser.add_argument("--headless_check", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be a positive finite multiplier.")
    return args


def run(args: argparse.Namespace) -> None:
    """Run a headless mapping check or interactive replay."""
    composed_xml, _ = compose_model_xml(args.model, args.overlay)
    model = mujoco.MjModel.from_xml_string(composed_xml)
    data = mujoco.MjData(model)
    replay = load_replay_log(args.log, model)
    indices = _frame_indices(replay, args.start_s, args.end_s)
    _print_summary(replay, indices)
    if args.headless_check:
        _apply_frames(replay, data, indices)
        print("Headless check: PASS")
        return

    from mujoco import viewer as mujoco_viewer

    handle = mujoco_viewer.launch_passive(model, data)
    viewer = _ViewerState(handle, args.speed)
    try:
        while _apply_frames(replay, data, indices, viewer):
            if not args.loop:
                break
    finally:
        handle.close()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()

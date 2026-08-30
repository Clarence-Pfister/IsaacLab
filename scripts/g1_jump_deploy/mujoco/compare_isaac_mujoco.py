# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare deterministic Isaac and MuJoCo G1 open-loop trajectory logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_REQUIRED_ARRAYS = {
    "time",
    "phase",
    "qpos",
    "qvel",
    "action",
    "delayed_action",
    "q_target",
    "applied_tau",
    "pelvis_pose",
    "pelvis_velocity",
    "foot_contact_forces",
    "observation",
    "metadata_json",
}
_OPTIONAL_ARRAYS = {"tendon_length", "tendon_limit_slack", "tendon_limit_force", "tendon_limit_active"}


def _canonical_digest(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_log(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_REQUIRED_ARRAYS - set(archive.files))
        if missing:
            raise ValueError(f"Log {path} is missing arrays: {missing}.")
        arrays = {name: np.asarray(archive[name]) for name in _REQUIRED_ARRAYS if name != "metadata_json"}
        arrays.update({name: np.asarray(archive[name]) for name in _OPTIONAL_ARRAYS if name in archive.files})
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Log {path} has invalid metadata_json.") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Log {path} metadata_json must decode to an object.")
    return arrays, metadata


def _load_manifest(metadata: dict[str, Any], override: Path | None, label: str) -> dict[str, Any]:
    manifest_path = override if override is not None else Path(str(metadata.get("manifest", "")))
    if not manifest_path.is_file():
        raise ValueError(
            f"Cannot read {label} manifest {manifest_path}. Supply --manifest when comparing relocated logs."
        )
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"{label} manifest must contain a JSON object.")
    digest = _canonical_digest(manifest)
    logged_digest = metadata.get("manifest_sha256")
    if logged_digest is not None and logged_digest != digest:
        raise ValueError(f"{label} log names manifest digest {logged_digest}, but {manifest_path} has digest {digest}.")
    return manifest


def _joint_names(manifest: dict[str, Any]) -> tuple[str, ...]:
    try:
        names = tuple(manifest["joints"]["names"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Manifest does not contain joints.names.") from exc
    if len(names) != 23 or len(set(names)) != 23 or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"Manifest joints.names must contain 23 unique strings, got {names}.")
    return names


def _policy_permutation(metadata: dict[str, Any], policy_names: tuple[str, ...], label: str) -> np.ndarray:
    log_names_value = metadata.get("qpos_qvel_order")
    if not isinstance(log_names_value, (list, tuple)):
        raise ValueError(f"{label} metadata lacks qpos_qvel_order.")
    log_names = tuple(log_names_value)
    if len(log_names) != len(set(log_names)) or set(log_names) != set(policy_names):
        raise ValueError(f"{label} log joint names are not a bijection of manifest joints: {log_names}.")
    log_index = {name: index for index, name in enumerate(log_names)}
    permutation = np.asarray([log_index[name] for name in policy_names], dtype=np.int64)
    if not np.array_equal(np.sort(permutation), np.arange(len(policy_names))):
        raise ValueError(f"{label} joint permutation is not bijective: {permutation.tolist()}.")
    return permutation


def _validate_log_shapes(arrays: dict[str, np.ndarray], joint_count: int, label: str) -> int:
    count = len(arrays["time"])
    expected_shapes = {
        "time": (count,),
        "phase": (count,),
        "qpos": (count, joint_count),
        "qvel": (count, joint_count),
        "action": (count, joint_count),
        "delayed_action": (count, joint_count),
        "q_target": (count, joint_count),
        "applied_tau": (count, joint_count),
        "pelvis_pose": (count, 7),
        "pelvis_velocity": (count, 6),
        "foot_contact_forces": (count, 2, 3),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{label} {name} must have shape {expected}, got {arrays[name].shape}.")
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{label} {name} contains non-finite values.")
    if arrays["observation"].shape != (count, 326) or not np.all(np.isfinite(arrays["observation"])):
        raise ValueError(f"{label} observation must be a finite array shaped ({count}, 326).")
    return count


def _orientation_error_deg(isaac_wxyz: np.ndarray, mujoco_wxyz: np.ndarray) -> np.ndarray:
    isaac_length = np.linalg.norm(isaac_wxyz, axis=1, keepdims=True)
    mujoco_length = np.linalg.norm(mujoco_wxyz, axis=1, keepdims=True)
    if np.any(isaac_length <= np.finfo(np.float64).eps) or np.any(mujoco_length <= np.finfo(np.float64).eps):
        raise ValueError("A pelvis orientation quaternion has zero norm.")
    isaac_norm = isaac_wxyz / isaac_length
    mujoco_norm = mujoco_wxyz / mujoco_length
    dot = np.clip(np.abs(np.sum(isaac_norm * mujoco_norm, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _contact_onsets(active: np.ndarray) -> np.ndarray:
    previous = np.concatenate((np.zeros(1, dtype=np.bool_), active[:-1]))
    return np.flatnonzero(active & ~previous)


def _format_first(index: int | None, times: np.ndarray) -> str:
    return "-" if index is None else f"{index} ({times[index]:.3f}s)"


def _compare(args: argparse.Namespace) -> None:  # noqa: C901 - one validation/reporting pipeline
    isaac, isaac_metadata = _load_log(args.isaac_log)
    mujoco, mujoco_metadata = _load_log(args.mujoco_log)
    isaac_manifest = _load_manifest(isaac_metadata, args.manifest, "Isaac")
    mujoco_manifest = _load_manifest(mujoco_metadata, args.manifest, "MuJoCo")
    if isaac_manifest != mujoco_manifest:
        raise ValueError(
            "Refusing to compare logs produced from different deployment manifests: "
            f"Isaac={_canonical_digest(isaac_manifest)}, MuJoCo={_canonical_digest(mujoco_manifest)}."
        )
    isaac_digest = isaac_metadata.get("manifest_sha256")
    mujoco_digest = mujoco_metadata.get("manifest_sha256")
    if isaac_digest is not None and mujoco_digest is not None and isaac_digest != mujoco_digest:
        raise ValueError("Refusing to compare logs whose recorded manifest digests disagree.")
    if isaac_metadata.get("task") != mujoco_metadata.get("task"):
        raise ValueError("Refusing to compare logs whose task metadata disagree.")

    policy_names = _joint_names(isaac_manifest)
    isaac_count = _validate_log_shapes(isaac, len(policy_names), "Isaac")
    mujoco_count = _validate_log_shapes(mujoco, len(policy_names), "MuJoCo")
    if isaac_count != mujoco_count:
        raise ValueError(f"Logs have different sample counts: Isaac={isaac_count}, MuJoCo={mujoco_count}.")
    if not np.allclose(isaac["time"], mujoco["time"], rtol=0.0, atol=1e-10):
        raise ValueError("Logs do not use the same physics-step timestamps.")
    for metadata_name in ("sim_dt", "policy_dt", "decimation", "goal"):
        if isaac_metadata.get(metadata_name) != mujoco_metadata.get(metadata_name):
            raise ValueError(f"Logs disagree on metadata field {metadata_name!r}.")
    for label, metadata in (("Isaac", isaac_metadata), ("MuJoCo", mujoco_metadata)):
        if metadata.get("pelvis_pose_convention") != "position_world_xyz[m], quaternion_world_from_body_wxyz":
            raise ValueError(f"{label} log uses an unsupported pelvis pose convention.")

    isaac_order = _policy_permutation(isaac_metadata, policy_names, "Isaac")
    mujoco_order = _policy_permutation(mujoco_metadata, policy_names, "MuJoCo")
    joint_fields = ("qpos", "qvel", "action", "delayed_action", "q_target", "applied_tau")
    isaac_policy = {name: isaac[name][:, isaac_order] for name in joint_fields}
    mujoco_policy = {name: mujoco[name][:, mujoco_order] for name in joint_fields}

    if not np.isclose(isaac["time"][0], 0.0, rtol=0.0, atol=1e-12) or not np.isclose(
        mujoco["time"][0], 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Both logs must include their true pre-physics initial state at t=0.")
    initial_joint_error = np.abs(isaac_policy["qpos"][0] - mujoco_policy["qpos"][0])
    initial_joint_velocity_error = np.abs(isaac_policy["qvel"][0] - mujoco_policy["qvel"][0])
    initial_pose_error = np.abs(isaac["pelvis_pose"][0] - mujoco["pelvis_pose"][0])
    initial_velocity_error = np.abs(isaac["pelvis_velocity"][0] - mujoco["pelvis_velocity"][0])
    if np.any(initial_joint_error > args.initial_tolerance):
        joint_index = int(np.argmax(initial_joint_error))
        raise ValueError(
            "Refusing to compare different initial joint states: "
            f"{policy_names[joint_index]} differs by {initial_joint_error[joint_index]:.3e} rad "
            f"(tolerance {args.initial_tolerance:.3e})."
        )
    if np.any(initial_joint_velocity_error > args.initial_tolerance):
        joint_index = int(np.argmax(initial_joint_velocity_error))
        raise ValueError(
            "Refusing to compare different initial joint velocities: "
            f"{policy_names[joint_index]} differs by {initial_joint_velocity_error[joint_index]:.3e} rad/s."
        )
    if np.any(initial_pose_error > args.initial_tolerance):
        component = int(np.argmax(initial_pose_error))
        raise ValueError(
            "Refusing to compare different initial pelvis poses: "
            f"component {component} differs by {initial_pose_error[component]:.3e} "
            f"(tolerance {args.initial_tolerance:.3e})."
        )
    if np.any(initial_velocity_error > args.initial_tolerance):
        component = int(np.argmax(initial_velocity_error))
        raise ValueError(
            "Refusing to compare different initial pelvis velocities: "
            f"component {component} differs by {initial_velocity_error[component]:.3e}."
        )
    if not np.allclose(isaac_policy["action"], mujoco_policy["action"], rtol=0.0, atol=1e-7):
        maximum = float(np.max(np.abs(isaac_policy["action"] - mujoco_policy["action"])))
        raise ValueError(f"Logs were not driven by identical raw actions (max difference {maximum:.3e}).")

    delayed_difference = float(np.max(np.abs(isaac_policy["delayed_action"] - mujoco_policy["delayed_action"])))
    target_difference = float(np.max(np.abs(isaac_policy["q_target"] - mujoco_policy["q_target"])))

    difference = mujoco_policy["qpos"] - isaac_policy["qpos"]
    absolute = np.abs(difference)
    rmse = np.sqrt(np.mean(np.square(difference), axis=0))
    maximum = np.max(absolute, axis=0)
    first_by_joint: list[int | None] = []
    for joint_index in range(len(policy_names)):
        indices = np.flatnonzero(absolute[:, joint_index] > args.joint_threshold)
        first_by_joint.append(int(indices[0]) if len(indices) else None)

    print(f"Manifest: {_canonical_digest(isaac_manifest)}")
    print(f"Samples: {isaac_count} at dt={isaac_metadata['sim_dt']} s; joint threshold={args.joint_threshold:g} rad")
    print(f"Action pipeline max difference: delayed={delayed_difference:.3e}, q_target={target_difference:.3e} rad")
    print("\nJoint qpos divergence (policy order)")
    print(f"{'joint':30s} {'RMSE(rad)':>11s} {'max(rad)':>11s} {'first > threshold':>20s}")
    for name, joint_rmse, joint_maximum, first_index in zip(policy_names, rmse, maximum, first_by_joint):
        print(f"{name:30s} {joint_rmse:11.6f} {joint_maximum:11.6f} {_format_first(first_index, isaac['time']):>20s}")

    position_error = np.linalg.norm(mujoco["pelvis_pose"][:, :3] - isaac["pelvis_pose"][:, :3], axis=1)
    orientation_error = _orientation_error_deg(isaac["pelvis_pose"][:, 3:], mujoco["pelvis_pose"][:, 3:])
    print("\nPelvis divergence")
    print(f"{'metric':24s} {'RMSE':>12s} {'max':>12s} {'worst step':>12s}")
    position_worst = int(np.argmax(position_error))
    orientation_worst = int(np.argmax(orientation_error))
    print(
        f"{'position norm (m)':24s} {math.sqrt(float(np.mean(position_error**2))):12.6f} "
        f"{position_error[position_worst]:12.6f} {position_worst:12d}"
    )
    print(
        f"{'orientation (deg)':24s} {math.sqrt(float(np.mean(orientation_error**2))):12.6f} "
        f"{orientation_error[orientation_worst]:12.6f} {orientation_worst:12d}"
    )

    isaac_contact = np.linalg.norm(isaac["foot_contact_forces"], axis=2) > args.contact_force_threshold
    mujoco_contact = np.linalg.norm(mujoco["foot_contact_forces"], axis=2) > args.contact_force_threshold
    print("\nContact timing")
    print(f"{'foot':8s} {'Isaac onsets':>13s} {'MuJoCo onsets':>14s} {'first delta':>13s} {'mismatch steps':>15s}")
    for foot_index, foot_name in enumerate(("left", "right")):
        isaac_onsets = _contact_onsets(isaac_contact[:, foot_index])
        mujoco_onsets = _contact_onsets(mujoco_contact[:, foot_index])
        first_delta = "-"
        if len(isaac_onsets) and len(mujoco_onsets):
            step_delta = int(mujoco_onsets[0]) - int(isaac_onsets[0])
            first_delta = f"{step_delta:+d} ({step_delta * isaac_metadata['sim_dt']:+.3f}s)"
        mismatch = int(np.count_nonzero(isaac_contact[:, foot_index] != mujoco_contact[:, foot_index]))
        print(
            f"{foot_name:8s} {str(isaac_onsets.tolist()):>13s} {str(mujoco_onsets.tolist()):>14s} "
            f"{first_delta:>13s} {mismatch:15d}"
        )

    all_over = np.argwhere(absolute > args.joint_threshold)
    if len(all_over):
        first_step = int(np.min(all_over[:, 0]))
        first_joint_candidates = all_over[all_over[:, 0] == first_step, 1]
        first_joint = int(first_joint_candidates[np.argmax(absolute[first_step, first_joint_candidates])])
        first_summary = (
            f"step {first_step} ({isaac['time'][first_step]:.3f}s), {policy_names[first_joint]}, "
            f"error={absolute[first_step, first_joint]:.6f} rad"
        )
    else:
        first_summary = "none"
    worst_step, worst_joint = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    print(f"\nFirst joint threshold crossing: {first_summary}")
    print(
        f"Worst offender: {policy_names[worst_joint]} at step {worst_step} "
        f"({isaac['time'][worst_step]:.3f}s), error={absolute[worst_step, worst_joint]:.6f} rad"
    )

    tendon_fields = ("tendon_length", "tendon_limit_slack", "tendon_limit_force", "tendon_limit_active")
    present_tendon_fields = [name for name in tendon_fields if name in mujoco]
    if present_tendon_fields and len(present_tendon_fields) != len(tendon_fields):
        raise ValueError(f"MuJoCo log has an incomplete tendon diagnostic set: {present_tendon_fields}.")
    if len(present_tendon_fields) == len(tendon_fields):
        tendon_names = tuple(mujoco_metadata.get("limited_tendon_names", ()))
        expected_shape = (mujoco_count, len(tendon_names))
        for name in tendon_fields:
            if mujoco[name].shape != expected_shape:
                raise ValueError(f"MuJoCo {name} must have shape {expected_shape}, got {mujoco[name].shape}.")
        if (
            not np.all(np.isfinite(mujoco["tendon_length"]))
            or not np.all(np.isfinite(mujoco["tendon_limit_slack"]))
            or not np.all(np.isfinite(mujoco["tendon_limit_force"]))
        ):
            raise ValueError("MuJoCo tendon diagnostics contain non-finite values.")

        tendon_active = np.asarray(mujoco["tendon_limit_active"], dtype=np.bool_)
        active_any = np.any(tendon_active, axis=1)
        ankle_indices = [index for index, name in enumerate(policy_names) if "ankle" in name]
        ankle_error = np.sqrt(np.mean(np.square(difference[:, ankle_indices]), axis=1))
        print("\nMuJoCo tendon-limit diagnostic")
        print(f"Active at any sample: {bool(np.any(active_any))}; active samples: {int(np.count_nonzero(active_any))}")
        for tendon_index, tendon_name in enumerate(tendon_names):
            count = int(np.count_nonzero(tendon_active[:, tendon_index]))
            minimum_slack = float(np.min(mujoco["tendon_limit_slack"][:, tendon_index]))
            maximum_force = float(np.max(mujoco["tendon_limit_force"][:, tendon_index]))
            print(f"{tendon_name:20s} active={count:4d} min_slack={minimum_slack:+.6e} max_force={maximum_force:.6e}")
        if np.any(active_any) and np.any(~active_any):
            correlation = float(np.corrcoef(active_any.astype(np.float64), ankle_error)[0, 1])
            print(
                "Ankle-error correlation with any active tendon: "
                f"r={correlation:+.6f}, mean_active={np.mean(ankle_error[active_any]):.6e} rad, "
                f"mean_inactive={np.mean(ankle_error[~active_any]):.6e} rad"
            )
        else:
            print("Ankle-error correlation with tendon activity: unavailable (activity state did not vary).")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isaac_log", type=Path)
    parser.add_argument("mujoco_log", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=None, help="Manifest location to use when logs were relocated."
    )
    parser.add_argument("--joint_threshold", type=float, default=0.05, help="Joint divergence threshold [rad].")
    parser.add_argument("--contact_force_threshold", type=float, default=1.0, help="Contact activation threshold [N].")
    parser.add_argument(
        "--initial_tolerance", type=float, default=1e-6, help="Required initial state agreement tolerance."
    )
    args = parser.parse_args()
    if args.joint_threshold < 0.0 or args.contact_force_threshold < 0.0 or args.initial_tolerance < 0.0:
        parser.error("Thresholds must be non-negative.")
    return args


if __name__ == "__main__":
    _compare(_parse_args())

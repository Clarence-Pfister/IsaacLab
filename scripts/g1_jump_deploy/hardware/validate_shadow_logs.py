# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate live G1 full-policy shadow logs and issue read-only admission evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.g1_jump_deploy.hardware.run_fsm_g1 import (  # noqa: E402
    _DEFAULT_MANIFEST,
    _DEFAULT_VALIDATION_RECORD,
    _MAX_BASE_ANGULAR_SPEED_RAD_S,
    _MAX_MOTOR_TEMPERATURE_C,
    _MAX_TILT_RAD,
    _OBSERVATION_DIM,
    _SHADOW_MAX_JOINT_SPEED_RAD_S,
    FeedbackSnapshot,
    HardwareManifest,
    SafetyFault,
    _body_tilt,
    _load_hardware_manifest,
    _project_shadow_target,
    _sha256,
    _verify_validated_bundle,
)
from scripts.g1_jump_deploy.runtime import JumpGoalRuntime, OnnxPolicy  # noqa: E402

_EXPECTED_STEPS = 152
_EXPECTED_GOALS_X = np.asarray((-0.1, 0.0, 0.1), dtype=np.float64)
_FLOAT_ATOL = 1.0e-7


@dataclass(frozen=True)
class ShadowLogSummary:
    """Validated metrics from one read-only hardware shadow log.

    Attributes:
        path: Validated NPZ log path.
        sha256: SHA-256 digest of the NPZ log.
        goal_pos_x_m: Requested longitudinal displacement [m].
        steps: Number of policy steps.
        duration_s: Time from the first to final policy sample [s].
        inference_p99_ms: 99th-percentile inference latency [ms].
        inference_maximum_ms: Maximum inference latency [ms].
        feedback_age_maximum_ms: Maximum feedback age [ms].
        body_tilt_maximum_deg: Maximum measured body tilt [deg].
        joint_speed_maximum_rad_s: Maximum measured joint speed [rad/s].
        measured_torque_maximum_fraction: Maximum measured torque estimate as
            a fraction of the manifest effort limit.
        projected_torque_maximum_fraction: Maximum counterfactual projected PD
            torque as a fraction of the manifest effort limit.
        unique_feedback_ticks: Number of distinct feedback ticks sampled.
    """

    path: str
    sha256: str
    goal_pos_x_m: float
    steps: int
    duration_s: float
    inference_p99_ms: float
    inference_maximum_ms: float
    feedback_age_maximum_ms: float
    body_tilt_maximum_deg: float
    joint_speed_maximum_rad_s: float
    measured_torque_maximum_fraction: float
    projected_torque_maximum_fraction: float
    unique_feedback_ticks: int


def _load_log(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files if name != "metadata_json"}
            if "metadata_json" not in archive.files:
                raise ValueError("missing metadata_json")
            raw_metadata = archive["metadata_json"].item()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read shadow log {path}: {exc}") from exc
    if not isinstance(raw_metadata, str):
        raise ValueError(f"Shadow log {path} metadata_json must contain a string")
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Shadow log {path} has invalid metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Shadow log {path} metadata must be an object")
    return arrays, metadata


def _require_array(
    arrays: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    *,
    finite: bool = True,
) -> np.ndarray:
    value = arrays.get(name)
    if value is None or value.shape != shape:
        actual = None if value is None else value.shape
        raise ValueError(f"Shadow log field {name} must have shape {shape}, got {actual}")
    if finite and not np.all(np.isfinite(value)):
        raise ValueError(f"Shadow log field {name} contains non-finite values")
    return value


def _require_close(actual: np.ndarray, expected: np.ndarray, name: str, atol: float = _FLOAT_ATOL) -> None:
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=atol):
        maximum_error = math.inf if actual.shape != expected.shape else float(np.max(np.abs(actual - expected)))
        raise ValueError(f"Shadow log {name} replay mismatch; maximum error={maximum_error:.9g}")


def validate_shadow_log(  # noqa: C901
    path: Path,
    manifest_path: Path,
    policy_path: Path,
    manifest: HardwareManifest,
    policy: OnnxPolicy,
    *,
    expected_steps: int = _EXPECTED_STEPS,
) -> ShadowLogSummary:
    """Validate one log structurally and replay its complete policy pipeline.

    Args:
        path: Full-policy shadow NPZ path.
        manifest_path: Accepted deployment manifest path.
        policy_path: Accepted ONNX policy path.
        manifest: Validated hardware manifest.
        policy: Loaded accepted ONNX policy.
        expected_steps: Required number of 50 Hz samples.

    Returns:
        Validated log metrics.

    Raises:
        ValueError: If metadata, timing, feedback, or replay checks fail.
        SafetyFault: If replayed command projection violates its envelope.
    """
    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    resolved_path = path.resolve()
    arrays, metadata = _load_log(resolved_path)
    joint_count = manifest.joint_count
    vector_shape = (expected_steps,)
    joint_shape = (expected_steps, joint_count)

    sample_time = _require_array(arrays, "time", vector_shape)
    feedback_age_ms = _require_array(arrays, "feedback_age_ms", vector_shape)
    ticks = _require_array(arrays, "tick", vector_shape)
    mode_pr = _require_array(arrays, "mode_pr", vector_shape)
    mode_machine = _require_array(arrays, "mode_machine", vector_shape)
    temperatures = _require_array(arrays, "maximum_temperature_c", vector_shape)
    joint_position = _require_array(arrays, "joint_position", joint_shape)
    joint_velocity = _require_array(arrays, "joint_velocity", joint_shape)
    joint_torque_estimate = _require_array(arrays, "joint_torque_estimate", joint_shape)
    imu_quaternion = _require_array(arrays, "imu_quaternion_wxyz", (expected_steps, 4))
    imu_gyroscope = _require_array(arrays, "imu_gyroscope", (expected_steps, 3))
    observation = _require_array(arrays, "observation", (expected_steps, _OBSERVATION_DIM))
    raw_action = _require_array(arrays, "raw_action", joint_shape)
    delayed_action = _require_array(arrays, "delayed_action", joint_shape)
    requested_target = _require_array(arrays, "requested_target", joint_shape)
    projected_target = _require_array(arrays, "projected_target", joint_shape)
    unprojected_torque = _require_array(arrays, "unprojected_torque", joint_shape)
    projected_torque = _require_array(arrays, "projected_torque", joint_shape)
    effort_ratio = _require_array(arrays, "effort_ratio", joint_shape)
    inference_latency_ms = _require_array(arrays, "inference_latency_ms", vector_shape)
    logged_tilt_deg = _require_array(arrays, "body_tilt_deg", vector_shape)
    logged_maximum_speed = _require_array(arrays, "maximum_joint_speed_rad_s", vector_shape)

    if metadata.get("schema_version") != "1.0":
        raise ValueError("Shadow log metadata must use schema 1.0")
    if metadata.get("read_only") is not True or metadata.get("command_publisher_created") is not False:
        raise ValueError("Shadow log does not attest to read-only execution")
    if metadata.get("feedback_mode") != "stationary_live_lowstate_counterfactual_targets":
        raise ValueError("Shadow log feedback mode is unsupported")
    if metadata.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Shadow log manifest digest differs from the accepted manifest")
    if metadata.get("policy_sha256") != _sha256(policy_path):
        raise ValueError("Shadow log policy digest differs from the accepted policy")
    if metadata.get("joint_names") != list(manifest.joint_names):
        raise ValueError("Shadow log joint order differs from the accepted manifest")
    if not math.isclose(float(metadata.get("policy_dt_s", math.nan)), manifest.policy_dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Shadow log policy period differs from the accepted manifest")
    feedback_counters = metadata.get("feedback_counters")
    if not isinstance(feedback_counters, dict):
        raise ValueError("Shadow log has no feedback integrity counters")
    if feedback_counters.get("crc_errors") != 0 or feedback_counters.get("invalid_packets") != 0:
        raise ValueError("Shadow log reports feedback integrity errors")
    valid_packets = feedback_counters.get("valid_packets")
    if isinstance(valid_packets, bool) or not isinstance(valid_packets, int) or valid_packets < expected_steps:
        raise ValueError("Shadow log reports too few valid feedback packets")

    goal = metadata.get("goal")
    if not isinstance(goal, dict):
        raise ValueError("Shadow log has no goal metadata")
    try:
        goal_values = {name: float(goal.get(name, math.nan)) for name in ("pos_x", "pos_y", "yaw", "roll", "pitch")}
    except (TypeError, ValueError) as exc:
        raise ValueError("Shadow log goal must contain numeric values") from exc
    if not all(math.isfinite(value) for value in goal_values.values()):
        raise ValueError("Shadow log goal contains non-finite values")
    if any(abs(goal_values[name]) > 1.0e-12 for name in ("pos_y", "yaw", "roll", "pitch")):
        raise ValueError("Hardware admission shadows require zero lateral and orientation commands")
    effort_scale = float(metadata.get("effort_scale", math.nan))
    if not math.isfinite(effort_scale) or not 0.0 < effort_scale <= 1.0:
        raise ValueError("Shadow log effort_scale is invalid")

    if abs(float(sample_time[0])) > 0.05:
        raise ValueError("Shadow log first policy timestamp is too far from zero")
    intervals = np.diff(sample_time)
    if np.any(intervals <= 0.0) or np.max(intervals) > 2.0 * manifest.policy_dt + 1.0e-6:
        raise ValueError("Shadow log policy timestamps are non-monotonic or miss the 50 Hz schedule")
    expected_duration = (expected_steps - 1) * manifest.policy_dt
    if abs(float(sample_time[-1] - sample_time[0]) - expected_duration) > 0.1:
        raise ValueError("Shadow log duration differs from the complete policy timeline")
    if np.any(feedback_age_ms < 0.0) or np.max(feedback_age_ms) > 20.0:
        raise ValueError("Shadow log feedback age left the 20 ms hardware envelope")
    if np.any(inference_latency_ms < 0.0):
        raise ValueError("Shadow log inference latency is negative")
    inference_p99_ms = float(np.percentile(inference_latency_ms, 99.0))
    inference_maximum_ms = float(np.max(inference_latency_ms))
    if inference_p99_ms > 0.5 * manifest.policy_dt * 1000.0 or inference_maximum_ms > manifest.policy_dt * 1000.0:
        raise ValueError("Shadow log inference timing is unsafe")
    if not np.all(mode_pr == mode_pr[0]) or not np.all(mode_machine == mode_machine[0]):
        raise ValueError("Shadow log native mode changed during the policy timeline")
    if np.max(temperatures) > _MAX_MOTOR_TEMPERATURE_C:
        raise ValueError("Shadow log motor temperature exceeded the hardware limit")
    if np.any(joint_position < manifest.joint_position_lower[None, :]) or np.any(
        joint_position > manifest.joint_position_upper[None, :]
    ):
        raise ValueError("Shadow log measured joint position exceeded physical limits")
    maximum_joint_speed = float(np.max(np.abs(joint_velocity)))
    if maximum_joint_speed > _SHADOW_MAX_JOINT_SPEED_RAD_S:
        raise ValueError("Shadow log robot was not stationary")
    if np.max(np.abs(imu_gyroscope)) > _MAX_BASE_ANGULAR_SPEED_RAD_S:
        raise ValueError("Shadow log base angular speed exceeded the hardware limit")
    calculated_tilt_deg = np.asarray([math.degrees(_body_tilt(value)) for value in imu_quaternion])
    _require_close(logged_tilt_deg, calculated_tilt_deg, "body_tilt_deg")
    if np.max(calculated_tilt_deg) > math.degrees(_MAX_TILT_RAD):
        raise ValueError("Shadow log body tilt exceeded the hardware limit")
    _require_close(logged_maximum_speed, np.max(np.abs(joint_velocity), axis=1), "maximum_joint_speed_rad_s")
    if np.any(requested_target < manifest.target_position_lower[None, :]) or np.any(
        requested_target > manifest.target_position_upper[None, :]
    ):
        raise ValueError("Shadow log requested target exceeded target limits")
    if np.any(projected_target < manifest.target_position_lower[None, :]) or np.any(
        projected_target > manifest.target_position_upper[None, :]
    ):
        raise ValueError("Shadow log projected target exceeded target limits")
    expected_effort_ratio = np.full(joint_count, effort_scale)
    if manifest.effort_limit_ratio is not None:
        expected_effort_ratio = np.minimum(expected_effort_ratio, manifest.effort_limit_ratio)
    _require_close(effort_ratio, np.broadcast_to(expected_effort_ratio, joint_shape), "effort_ratio")
    recomputed_unprojected_torque = manifest.stiffness * (requested_target - joint_position)
    recomputed_unprojected_torque -= manifest.damping * joint_velocity
    recomputed_projected_torque = manifest.stiffness * (projected_target - joint_position)
    recomputed_projected_torque -= manifest.damping * joint_velocity
    _require_close(unprojected_torque, recomputed_unprojected_torque, "unprojected_torque", atol=1.0e-6)
    _require_close(projected_torque, recomputed_projected_torque, "projected_torque", atol=1.0e-6)
    projected_fraction = np.abs(projected_torque) / manifest.effort_limit[None, :]
    if np.any(projected_fraction > effort_ratio + 1.0e-6):
        raise ValueError("Shadow log projected torque exceeded its effort envelope")

    runtime = JumpGoalRuntime(manifest_path, freeze_during_flight=True)
    runtime.arm(
        goal_values["pos_x"],
        goal_values["pos_y"],
        goal_values["yaw"],
        roll=goal_values["roll"],
        pitch=goal_values["pitch"],
    )
    root_position = np.asarray((0.0, 0.0, manifest.initial_root_height), dtype=np.float64)
    runtime.trigger(root_position, imu_quaternion[0], joint_position[0], goal_pos_z_w=0.0)
    for index in range(expected_steps):
        replayed_observation = runtime.step(
            joint_position[index],
            joint_velocity[index],
            imu_gyroscope[index],
            imu_quaternion[index],
            root_position,
            imu_quaternion[index],
        )
        _require_close(observation[index], replayed_observation, f"observation[{index}]")
        replayed_action = policy(replayed_observation)
        _require_close(raw_action[index], replayed_action, f"raw_action[{index}]", atol=1.0e-6)
        snapshot = FeedbackSnapshot(
            received_at=0.0,
            tick=int(ticks[index]),
            mode_pr=int(mode_pr[index]),
            mode_machine=int(mode_machine[index]),
            joint_positions=joint_position[index],
            joint_velocities=joint_velocity[index],
            joint_torque_estimates=joint_torque_estimate[index],
            imu_quaternion=imu_quaternion[index],
            imu_gyroscope=imu_gyroscope[index],
            wireless_remote=bytes(40),
            maximum_temperature_c=int(temperatures[index]),
        )
        replayed_requested, replayed_projected, replayed_unprojected_torque, replayed_projected_torque, _ = (
            _project_shadow_target(runtime, replayed_action, manifest, snapshot, effort_scale)
        )
        _require_close(delayed_action[index], runtime.delayed_action, f"delayed_action[{index}]")
        _require_close(requested_target[index], replayed_requested, f"requested_target[{index}]")
        _require_close(projected_target[index], replayed_projected, f"projected_target[{index}]")
        _require_close(
            unprojected_torque[index],
            replayed_unprojected_torque,
            f"unprojected_torque[{index}]",
            atol=1.0e-6,
        )
        _require_close(
            projected_torque[index],
            replayed_projected_torque,
            f"projected_torque[{index}]",
            atol=1.0e-6,
        )
    if not runtime.done:
        raise ValueError("Shadow log replay did not complete the policy timeline")

    measured_torque_fraction = np.abs(joint_torque_estimate) / manifest.effort_limit[None, :]
    return ShadowLogSummary(
        path=str(resolved_path),
        sha256=_sha256(resolved_path),
        goal_pos_x_m=goal_values["pos_x"],
        steps=expected_steps,
        duration_s=float(sample_time[-1] - sample_time[0]),
        inference_p99_ms=inference_p99_ms,
        inference_maximum_ms=inference_maximum_ms,
        feedback_age_maximum_ms=float(np.max(feedback_age_ms)),
        body_tilt_maximum_deg=float(np.max(calculated_tilt_deg)),
        joint_speed_maximum_rad_s=maximum_joint_speed,
        measured_torque_maximum_fraction=float(np.max(measured_torque_fraction)),
        projected_torque_maximum_fraction=float(np.max(projected_fraction)),
        unique_feedback_ticks=int(np.unique(ticks).size),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate three read-only G1 full-policy shadow logs.")
    parser.add_argument("logs", nargs=3, type=Path, help="Shadow NPZ logs for goals -0.1, 0.0, and 0.1 m.")
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST, help="Accepted deployment manifest.")
    parser.add_argument(
        "--validation_record",
        type=Path,
        default=_DEFAULT_VALIDATION_RECORD,
        help="Accepted artifact SHA-256 record.",
    )
    parser.add_argument("--policy", type=Path, default=None, help="Defaults to policy.onnx beside the manifest.")
    parser.add_argument(
        "--admission_output",
        type=Path,
        default=None,
        help="Optional new JSON evidence file; existing files are refused.",
    )
    args = parser.parse_args()
    resolved_logs = [path.resolve() for path in args.logs]
    if len(set(resolved_logs)) != 3:
        parser.error("The three shadow log paths must be distinct")
    return args


def _write_admission(path: Path, manifest_path: Path, policy_path: Path, summaries: list[ShadowLogSummary]) -> None:
    resolved_path = path.resolve()
    evidence = {
        "schema_version": "1.0",
        "issued_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "read_only_shadow_admission": True,
        "authorizes_motor_control": False,
        "manifest_sha256": _sha256(manifest_path),
        "policy_sha256": _sha256(policy_path),
        "logs": [asdict(summary) for summary in summaries],
    }
    try:
        with resolved_path.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError as exc:
        raise ValueError(f"Cannot create admission evidence {resolved_path}: {exc}") from exc


def main() -> int:
    """Validate the shadow matrix and optionally issue read-only evidence."""
    args = _parse_args()
    manifest_path = args.manifest.resolve()
    policy_path = args.policy.resolve() if args.policy is not None else manifest_path.with_name("policy.onnx")
    try:
        _verify_validated_bundle(manifest_path, args.validation_record.resolve(), policy_path)
        manifest = _load_hardware_manifest(manifest_path)
        policy = OnnxPolicy(policy_path, _OBSERVATION_DIM, manifest.joint_count)
        policy.warm_up()
        summaries = [validate_shadow_log(path, manifest_path, policy_path, manifest, policy) for path in args.logs]
        summaries.sort(key=lambda summary: summary.goal_pos_x_m)
        actual_goals = np.asarray([summary.goal_pos_x_m for summary in summaries])
        if not np.allclose(actual_goals, _EXPECTED_GOALS_X, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Shadow goals must be -0.1, 0.0, and 0.1 m, got {actual_goals.tolist()}")
        for summary in summaries:
            print(
                f"PASS goal={summary.goal_pos_x_m:+.3f} m: steps={summary.steps}, "
                f"p99/max={summary.inference_p99_ms:.3f}/{summary.inference_maximum_ms:.3f} ms, "
                f"feedback_age_max={summary.feedback_age_maximum_ms:.3f} ms, "
                f"tilt_max={summary.body_tilt_maximum_deg:.2f} deg, "
                f"projected_torque_max={100.0 * summary.projected_torque_maximum_fraction:.1f}%."
            )
        if args.admission_output is not None:
            _write_admission(args.admission_output, manifest_path, policy_path, summaries)
            print(f"Wrote read-only shadow admission evidence: {args.admission_output.resolve()}")
        print("PASS: all three shadow logs replay exactly against the accepted policy pipeline.")
        print("This evidence does not authorize motor control or a real jump.")
        return 0
    except (SafetyFault, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

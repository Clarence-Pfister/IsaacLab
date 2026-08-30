# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project G1 jump ankle poses into the MJCF tendon-feasible region.

For each frame and each ankle, this tool computes the exact Euclidean projection
onto the convex polygon formed by the MJCF fixed-tendon upper limits and joint
box limits. It then applies one light, feasibility-preserving temporal smoothing
pass to the four ankle columns. All other CSV fields are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_ANKLE_JOINTS = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
_SIDES = ("left", "right")
_EXPECTED_TENDON_COUNT = 8
_TENDONS_PER_SIDE = 4
_SMOOTHING_STRENGTH = 0.1
_FEASIBILITY_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class _HalfPlane:
    """A two-dimensional half-plane in ankle pitch-roll coordinates."""

    pitch_coef: float
    roll_coef: float
    limit: float
    label: str

    def value(self, point: tuple[float, float]) -> float:
        """Evaluate the left-hand side at an ankle pose [rad]."""

        pitch, roll = point
        return self.pitch_coef * pitch + self.roll_coef * roll


@dataclass(frozen=True)
class _TendonConstraint:
    """A fixed-tendon upper-limit constraint parsed from the MJCF."""

    side: str
    side_index: int
    plane: _HalfPlane


@dataclass(frozen=True)
class _ModelLimits:
    """Ankle tendon and box limits parsed from the MJCF."""

    tendons: tuple[_TendonConstraint, ...]
    joint_ranges: dict[str, tuple[float, float]]


@dataclass
class _MotionCsv:
    """Raw CSV storage plus parsed ankle positions [rad]."""

    header_line: bytes
    header: tuple[str, ...]
    rows: list[tuple[list[bytes], bytes]]
    ankle_values: dict[str, list[float]]


def _parse_float_list(text: str, expected_count: int, description: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split())
    if len(values) != expected_count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{description} must contain {expected_count} finite values: {text!r}")
    return values


def _parse_model_limits(mjcf_path: Path) -> _ModelLimits:
    root = ET.parse(mjcf_path).getroot()

    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle", "degree").lower() != "radian":
        raise ValueError("the MJCF compiler must specify angle='radian'")

    tendon_element = root.find("tendon")
    fixed_tendons = [] if tendon_element is None else tendon_element.findall("fixed")
    if len(fixed_tendons) != _EXPECTED_TENDON_COUNT:
        raise ValueError(
            f"expected exactly {_EXPECTED_TENDON_COUNT} fixed tendon constraints, found {len(fixed_tendons)}"
        )

    tendons: list[_TendonConstraint] = []
    side_counts = dict.fromkeys(_SIDES, 0)
    for tendon_index, fixed in enumerate(fixed_tendons, start=1):
        if fixed.get("limited", "false").lower() != "true":
            raise ValueError(f"fixed tendon {tendon_index} is not limited")
        if "range" not in fixed.attrib:
            raise ValueError(f"fixed tendon {tendon_index} has no range")
        _, upper_limit = _parse_float_list(fixed.attrib["range"], 2, f"fixed tendon {tendon_index} range")

        tendon_joints = fixed.findall("joint")
        if len(tendon_joints) != 2:
            raise ValueError(f"fixed tendon {tendon_index} must reference exactly two joints")
        coefficients: dict[str, float] = {}
        for joint in tendon_joints:
            joint_name = joint.get("joint")
            coefficient_text = joint.get("coef")
            if joint_name is None or coefficient_text is None:
                raise ValueError(f"fixed tendon {tendon_index} has an incomplete joint reference")
            coefficient = float(coefficient_text)
            if not math.isfinite(coefficient):
                raise ValueError(f"fixed tendon {tendon_index} has a non-finite coefficient")
            if joint_name in coefficients:
                raise ValueError(f"fixed tendon {tendon_index} references {joint_name!r} more than once")
            coefficients[joint_name] = coefficient

        matching_sides = []
        for side in _SIDES:
            expected_joints = {f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint"}
            if set(coefficients) == expected_joints:
                matching_sides.append(side)
        if len(matching_sides) != 1:
            raise ValueError(f"fixed tendon {tendon_index} does not couple one pitch-roll ankle pair")

        side = matching_sides[0]
        side_counts[side] += 1
        pitch_joint = f"{side}_ankle_pitch_joint"
        roll_joint = f"{side}_ankle_roll_joint"
        plane = _HalfPlane(
            pitch_coef=coefficients[pitch_joint],
            roll_coef=coefficients[roll_joint],
            limit=upper_limit,
            label=f"{side} constraint {side_counts[side]}",
        )
        tendons.append(_TendonConstraint(side=side, side_index=side_counts[side], plane=plane))

    for side, count in side_counts.items():
        if count != _TENDONS_PER_SIDE:
            raise ValueError(f"expected {_TENDONS_PER_SIDE} fixed tendons for {side} ankle, found {count}")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("the MJCF has no worldbody")
    joint_ranges: dict[str, tuple[float, float]] = {}
    for joint_name in _ANKLE_JOINTS:
        matching_joints = [joint for joint in worldbody.iter("joint") if joint.get("name") == joint_name]
        if len(matching_joints) != 1:
            raise ValueError(f"expected exactly one body joint named {joint_name!r}, found {len(matching_joints)}")
        range_text = matching_joints[0].get("range")
        if range_text is None:
            raise ValueError(f"joint {joint_name!r} has no range")
        lower, upper = _parse_float_list(range_text, 2, f"joint {joint_name!r} range")
        if lower >= upper:
            raise ValueError(f"joint {joint_name!r} has an invalid range")
        joint_ranges[joint_name] = (lower, upper)

    return _ModelLimits(tendons=tuple(tendons), joint_ranges=joint_ranges)


def _split_line_ending(line: bytes) -> tuple[bytes, bytes]:
    if line.endswith(b"\r\n"):
        return line[:-2], b"\r\n"
    if line.endswith(b"\n") or line.endswith(b"\r"):
        return line[:-1], line[-1:]
    return line, b""


def _read_motion_csv(csv_path: Path) -> _MotionCsv:
    lines = csv_path.read_bytes().splitlines(keepends=True)
    if len(lines) < 2:
        raise ValueError(f"motion CSV {csv_path} must contain a header and at least one frame")

    header_body, _ = _split_line_ending(lines[0])
    try:
        header = tuple(field.decode("ascii") for field in header_body.split(b","))
    except UnicodeDecodeError as error:
        raise ValueError(f"motion CSV {csv_path} header is not ASCII") from error
    if len(set(header)) != len(header):
        raise ValueError(f"motion CSV {csv_path} contains duplicate column names")
    missing_columns = set(_ANKLE_JOINTS).difference(header)
    if missing_columns:
        raise ValueError(f"motion CSV {csv_path} is missing ankle columns: {sorted(missing_columns)}")

    ankle_indices = {joint_name: header.index(joint_name) for joint_name in _ANKLE_JOINTS}
    ankle_values = {joint_name: [] for joint_name in _ANKLE_JOINTS}
    rows: list[tuple[list[bytes], bytes]] = []
    for frame, line in enumerate(lines[1:]):
        body, line_ending = _split_line_ending(line)
        fields = body.split(b",")
        if len(fields) != len(header):
            raise ValueError(f"motion CSV {csv_path} frame {frame} has {len(fields)} fields; expected {len(header)}")
        for joint_name, column_index in ankle_indices.items():
            try:
                value = float(fields[column_index])
            except ValueError as error:
                raise ValueError(
                    f"motion CSV {csv_path} frame {frame} has a non-numeric value for {joint_name}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"motion CSV {csv_path} frame {frame} has a non-finite value for {joint_name}")
            ankle_values[joint_name].append(value)
        rows.append((fields, line_ending))

    return _MotionCsv(header_line=lines[0], header=header, rows=rows, ankle_values=ankle_values)


def _constraints_for_side(model: _ModelLimits, side: str) -> tuple[_HalfPlane, ...]:
    pitch_joint = f"{side}_ankle_pitch_joint"
    roll_joint = f"{side}_ankle_roll_joint"
    pitch_lower, pitch_upper = model.joint_ranges[pitch_joint]
    roll_lower, roll_upper = model.joint_ranges[roll_joint]
    tendon_planes = tuple(tendon.plane for tendon in model.tendons if tendon.side == side)
    box_planes = (
        _HalfPlane(1.0, 0.0, pitch_upper, f"{pitch_joint} upper limit"),
        _HalfPlane(-1.0, 0.0, -pitch_lower, f"{pitch_joint} lower limit"),
        _HalfPlane(0.0, 1.0, roll_upper, f"{roll_joint} upper limit"),
        _HalfPlane(0.0, -1.0, -roll_lower, f"{roll_joint} lower limit"),
    )
    return tendon_planes + box_planes


def _is_feasible(
    point: tuple[float, float], constraints: tuple[_HalfPlane, ...], tolerance: float = _FEASIBILITY_TOLERANCE
) -> bool:
    return all(constraint.value(point) <= constraint.limit + tolerance for constraint in constraints)


def _project_to_polygon(point: tuple[float, float], constraints: tuple[_HalfPlane, ...]) -> tuple[float, float]:
    """Return the exact Euclidean projection onto a convex half-plane polygon."""

    if _is_feasible(point, constraints, tolerance=0.0):
        return point

    pitch, roll = point
    candidates: list[tuple[float, float]] = []

    # A projection in the relative interior of an edge is perpendicular to that
    # edge, so test the orthogonal projection onto every boundary line.
    for constraint in constraints:
        norm_squared = constraint.pitch_coef**2 + constraint.roll_coef**2
        if norm_squared == 0.0:
            raise ValueError(f"constraint {constraint.label!r} has zero coefficients")
        scale = (constraint.value(point) - constraint.limit) / norm_squared
        candidate = (pitch - scale * constraint.pitch_coef, roll - scale * constraint.roll_coef)
        if _is_feasible(candidate, constraints):
            candidates.append(candidate)

    # Otherwise the projection is a polygon vertex. Enumerate every pairwise
    # boundary intersection and retain only feasible intersections.
    for first_index, first in enumerate(constraints):
        for second in constraints[first_index + 1 :]:
            determinant = first.pitch_coef * second.roll_coef - first.roll_coef * second.pitch_coef
            if abs(determinant) <= 1.0e-15:
                continue
            candidate_pitch = (first.limit * second.roll_coef - first.roll_coef * second.limit) / determinant
            candidate_roll = (first.pitch_coef * second.limit - first.limit * second.pitch_coef) / determinant
            candidate = (candidate_pitch, candidate_roll)
            if _is_feasible(candidate, constraints):
                candidates.append(candidate)

    if not candidates:
        raise ValueError("ankle constraints define an empty feasible polygon")
    return min(candidates, key=lambda candidate: (candidate[0] - pitch) ** 2 + (candidate[1] - roll) ** 2)


def _project_motion(motion: _MotionCsv, model: _ModelLimits) -> dict[str, list[float]]:
    projected = {joint_name: list(values) for joint_name, values in motion.ankle_values.items()}
    for side in _SIDES:
        pitch_joint = f"{side}_ankle_pitch_joint"
        roll_joint = f"{side}_ankle_roll_joint"
        constraints = _constraints_for_side(model, side)
        for frame, point in enumerate(zip(motion.ankle_values[pitch_joint], motion.ankle_values[roll_joint])):
            projected_pitch, projected_roll = _project_to_polygon(point, constraints)
            projected[pitch_joint][frame] = projected_pitch
            projected[roll_joint][frame] = projected_roll
    return projected


def _smooth_series(values: list[float], strength: float) -> list[float]:
    if len(values) == 1:
        return list(values)
    smoothed = []
    for frame, value in enumerate(values):
        if frame == 0:
            neighbor_average = values[1]
        elif frame == len(values) - 1:
            neighbor_average = values[-2]
        else:
            neighbor_average = 0.5 * (values[frame - 1] + values[frame + 1])
        smoothed.append((1.0 - strength) * value + strength * neighbor_average)
    return smoothed


def _motion_is_feasible(values: dict[str, list[float]], model: _ModelLimits, tolerance: float = 0.0) -> bool:
    for side in _SIDES:
        pitch_joint = f"{side}_ankle_pitch_joint"
        roll_joint = f"{side}_ankle_roll_joint"
        constraints = _constraints_for_side(model, side)
        points = zip(values[pitch_joint], values[roll_joint])
        if not all(_is_feasible(point, constraints, tolerance=tolerance) for point in points):
            return False
    return True


def _smooth_motion(projected: dict[str, list[float]], model: _ModelLimits) -> tuple[dict[str, list[float]], float]:
    # A convex average of feasible poses remains feasible. Recheck anyway so a
    # future model or floating-point edge case cannot silently corrupt output.
    for backoff_step in range(11):
        strength = round(_SMOOTHING_STRENGTH - 0.01 * backoff_step, 10)
        smoothed = {joint_name: _smooth_series(values, strength) for joint_name, values in projected.items()}
        if _motion_is_feasible(smoothed, model):
            return smoothed, strength
    raise ValueError("projected motion is not feasible after smoothing backoff")


def _format_float(value: float) -> bytes:
    return format(value, ".17g").encode("ascii")


def _serialize_motion(motion: _MotionCsv, ankle_values: dict[str, list[float]]) -> bytes:
    ankle_indices = {joint_name: motion.header.index(joint_name) for joint_name in _ANKLE_JOINTS}
    output = bytearray(motion.header_line)
    for frame, (input_fields, line_ending) in enumerate(motion.rows):
        output_fields = list(input_fields)
        for joint_name, column_index in ankle_indices.items():
            output_value = ankle_values[joint_name][frame]
            if output_value != motion.ankle_values[joint_name][frame]:
                output_fields[column_index] = _format_float(output_value)
        output.extend(b",".join(output_fields))
        output.extend(line_ending)
    return bytes(output)


def _write_atomic(output_path: Path, payload: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
        ) as temporary_file:
            temporary_file.write(payload)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _report_movement(input_motion: _MotionCsv, output_motion: _MotionCsv, smoothing_strength: float) -> None:
    print(f"Temporal smoothing strength: {smoothing_strength:.10g}")
    print("Per-frame ankle delta (output - input) [rad]:")
    print("frame," + ",".join(_ANKLE_JOINTS))
    changed_frames = []
    max_deltas = dict.fromkeys(_ANKLE_JOINTS, 0.0)
    max_delta_frames = dict.fromkeys(_ANKLE_JOINTS, 0)
    for frame in range(len(input_motion.rows)):
        deltas = []
        for joint_name in _ANKLE_JOINTS:
            delta = output_motion.ankle_values[joint_name][frame] - input_motion.ankle_values[joint_name][frame]
            deltas.append(delta)
            if abs(delta) > max_deltas[joint_name]:
                max_deltas[joint_name] = abs(delta)
                max_delta_frames[joint_name] = frame
        if any(delta != 0.0 for delta in deltas):
            changed_frames.append(frame)
        print(f"{frame}," + ",".join(f"{delta:+.17e}" for delta in deltas))

    print("Maximum absolute ankle delta [rad]:")
    for joint_name in _ANKLE_JOINTS:
        print(f"  {joint_name}: {max_deltas[joint_name]:.17e} at frame {max_delta_frames[joint_name]}")
    if changed_frames:
        print(f"Changed frames ({len(changed_frames)}): " + ",".join(str(frame) for frame in changed_frames))
    else:
        print("Changed frames (0): none")


def _report_constraints(label: str, motion: _MotionCsv, model: _ModelLimits) -> None:
    print(f"{label} tendon constraints:")
    for tendon in model.tendons:
        pitch_values = motion.ankle_values[f"{tendon.side}_ankle_pitch_joint"]
        roll_values = motion.ankle_values[f"{tendon.side}_ankle_roll_joint"]
        values = [tendon.plane.value(point) for point in zip(pitch_values, roll_values)]
        maximum = max(values)
        maximum_frame = values.index(maximum)
        violation_count = sum(value > tendon.plane.limit for value in values)
        status = "OK" if violation_count == 0 else "VIOLATION"
        print(
            f"  {tendon.plane.label}: max {maximum:+.9f} at frame {maximum_frame}; "
            f"limit {tendon.plane.limit:+.9f}; violations {violation_count} [{status}]"
        )


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="source motion CSV (never modified)")
    parser.add_argument("output_csv", type=Path, help="path for the tendon-feasible motion CSV")
    parser.add_argument(
        "--mjcf_path",
        type=Path,
        default=repository_root / "data_storage" / "g1_23dof_holo_compat.xml",
        help="G1 MJCF containing ankle tendon and joint limits",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="report all tendon maxima and violation counts for the input and output",
    )
    return parser.parse_args()


def main() -> int:
    """Retarget an ankle motion and write its feasibility and movement reports."""

    args = _parse_args()
    input_path = args.input_csv.resolve()
    output_path = args.output_csv.resolve()
    if input_path == output_path:
        print("error: input and output paths must differ; the source motion will not be overwritten", file=sys.stderr)
        return 2

    try:
        model = _parse_model_limits(args.mjcf_path)
        input_motion = _read_motion_csv(input_path)
        projected = _project_motion(input_motion, model)
        smoothed, smoothing_strength = _smooth_motion(projected, model)
        payload = _serialize_motion(input_motion, smoothed)
        _write_atomic(output_path, payload)
        output_motion = _read_motion_csv(output_path)
        if not _motion_is_feasible(output_motion.ankle_values, model):
            raise ValueError("serialized output is not tendon and joint-limit feasible")
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(output_motion.rows)} frames to {output_path}")
    _report_movement(input_motion, output_motion, smoothing_strength)
    if args.verify:
        _report_constraints("Input", input_motion, model)
        _report_constraints("Output", output_motion, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

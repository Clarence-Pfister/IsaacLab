# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.g1_jump_deploy.retarget.extend_jump_reference import (
    extend_reference,
    quintic_smoothstep,
    quintic_smoothstep_derivative,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INPUT_PATH = _REPO_ROOT / "data_storage" / "perfect_jump_ground_aligned.csv"
_MANIFEST_PATH = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated" / "deploy_manifest.json"


def _read(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        return columns, np.asarray([[float(value) for value in row] for row in reader])


@pytest.fixture
def extended_reference(tmp_path: Path) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    output_path = tmp_path / "extended.csv"
    extend_reference(_INPUT_PATH, output_path, _MANIFEST_PATH)
    input_columns, original = _read(_INPUT_PATH)
    output_columns, extended = _read(output_path)
    return input_columns, original, output_columns, extended


def test_default_extension_frame_counts_and_columns(
    extended_reference: tuple[list[str], np.ndarray, list[str], np.ndarray],
) -> None:
    input_columns, original, output_columns, extended = extended_reference

    assert len(original) == 91
    assert len(extended) == 45 + 91 + 15 + 45 == 196
    assert output_columns == input_columns
    np.testing.assert_array_equal(extended[45:136], original)


def test_extension_holds_exact_stance_and_zero_velocity(
    extended_reference: tuple[list[str], np.ndarray, list[str], np.ndarray],
) -> None:
    columns, original, _, extended = extended_reference
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))["joints"]
    manifest_indices = [columns.index(name) for name in manifest["names"]]
    all_joint_indices = list(range(7, columns.index("left_foot_x")))

    np.testing.assert_array_equal(extended[:45], np.repeat(original[[0]], 45, axis=0))
    np.testing.assert_allclose(extended[0, manifest_indices], manifest["default_pos"], rtol=0.0, atol=1.0e-6)
    np.testing.assert_array_equal(extended[151:], np.repeat(extended[[151]], 45, axis=0))
    np.testing.assert_array_equal(extended[-1, manifest_indices], manifest["default_pos"])
    np.testing.assert_array_equal(np.diff(extended[:45, all_joint_indices], axis=0), 0.0)
    np.testing.assert_array_equal(np.diff(extended[151:, all_joint_indices], axis=0), 0.0)


def test_ramp_has_smooth_joins_and_preserves_horizontal_displacement(
    extended_reference: tuple[list[str], np.ndarray, list[str], np.ndarray],
) -> None:
    columns, original, _, extended = extended_reference
    joint_indices = list(range(7, columns.index("left_foot_x")))
    ramp_with_start = extended[135:151, joint_indices]
    ramp_steps = np.linalg.norm(np.diff(ramp_with_start, axis=0), axis=1)

    assert quintic_smoothstep(0.0) == 0.0
    assert quintic_smoothstep(1.0) == 1.0
    assert quintic_smoothstep_derivative(0.0) == 0.0
    assert quintic_smoothstep_derivative(1.0) == 0.0
    assert ramp_steps[0] < ramp_steps[1]
    assert ramp_steps[-1] < ramp_steps[-2]
    np.testing.assert_array_equal(extended[150, joint_indices], extended[151, joint_indices])

    root_x = columns.index("root_translateX")
    root_y = columns.index("root_translateY")
    expected_horizontal_position = np.repeat(original[-1:, [root_x, root_y]], 60, axis=0)
    np.testing.assert_array_equal(extended[136:, [root_x, root_y]], expected_horizontal_position)


def test_rejects_frame_zero_that_is_not_manifest_stance(tmp_path: Path) -> None:
    columns, original = _read(_INPUT_PATH)
    original[0, columns.index("left_knee_joint")] += 1.0e-3
    bad_input = tmp_path / "bad.csv"
    with bad_input.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(original)

    with pytest.raises(ValueError, match="not the manifest default stance"):
        extend_reference(bad_input, tmp_path / "unused.csv", _MANIFEST_PATH)

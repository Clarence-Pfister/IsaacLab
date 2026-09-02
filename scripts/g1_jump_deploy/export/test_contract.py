# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deployment-export contract validation."""

import pytest

from scripts.g1_jump_deploy.export.contract import validate_deployment_table_shapes, validate_joint_name_contract


def test_non_152_episode_table_shapes_are_valid() -> None:
    validate_deployment_table_shapes((327, 70), (327, 6), episode_steps=327, preview_dim=70, phase_count=6)


@pytest.mark.parametrize(
    ("reference_shape", "phase_shape", "message"),
    [
        ((326, 70), (327, 6), "Reference preview must have shape \\(327, 70\\), got \\(326, 70\\)"),
        ((327, 70), (327, 5), "Jump phase must have shape \\(327, 6\\), got \\(327, 5\\)"),
    ],
)
def test_invalid_deployment_table_shape_reports_expected_and_actual(
    reference_shape: tuple[int, int], phase_shape: tuple[int, int], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_deployment_table_shapes(
            reference_shape,
            phase_shape,
            episode_steps=327,
            preview_dim=70,
            phase_count=6,
        )


def test_reordered_joint_contract_is_valid() -> None:
    assert not validate_joint_name_contract(("joint_b", "joint_a"), ("joint_a", "joint_b"))


def test_missing_or_extra_joint_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"Missing=\['joint_b'\], extra=\['joint_c'\]"):
        validate_joint_name_contract(("joint_a", "joint_c"), ("joint_a", "joint_b"))

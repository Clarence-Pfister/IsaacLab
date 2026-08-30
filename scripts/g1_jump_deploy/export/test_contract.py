# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deployment-export contract validation."""

import pytest

from scripts.g1_jump_deploy.export.contract import validate_joint_name_contract


def test_reordered_joint_contract_is_valid() -> None:
    assert not validate_joint_name_contract(("joint_b", "joint_a"), ("joint_a", "joint_b"))


def test_missing_or_extra_joint_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"Missing=\['joint_b'\], extra=\['joint_c'\]"):
        validate_joint_name_contract(("joint_a", "joint_c"), ("joint_a", "joint_b"))

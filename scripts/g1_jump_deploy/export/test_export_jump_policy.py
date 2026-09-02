# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for jump-policy manifest export."""

from scripts.g1_jump_deploy.export.contract import deployment_table_filenames, goal_command_contract


def test_deployment_table_filenames_are_derived_from_shapes() -> None:
    assert deployment_table_filenames((152, 70), (152, 6)) == (
        "reference_preview_152x70.npy",
        "jump_phase_152x6.npy",
    )
    assert deployment_table_filenames((327, 70), (327, 6)) == (
        "reference_preview_327x70.npy",
        "jump_phase_327x6.npy",
    )


def test_retrigger_aware_goal_command_exports_explicit_schema_contract() -> None:
    orientation_mode, retrigger_indicator = goal_command_contract("obs_goal_command_remaining_orientation_retrigger")

    assert orientation_mode == "remaining"
    assert retrigger_indicator == {
        "mode": "goal_command_z",
        "fresh_value": 0.0,
        "retrigger_value": 0.25,
    }


def test_legacy_goal_command_does_not_claim_retrigger_input() -> None:
    orientation_mode, retrigger_indicator = goal_command_contract("obs_goal_command_remaining_orientation")

    assert orientation_mode == "remaining"
    assert retrigger_indicator is None


def test_retrigger_goal_aware_command_exports_affine_schema_contract() -> None:
    orientation_mode, retrigger_indicator = goal_command_contract(
        "obs_goal_command_remaining_orientation_retrigger_goal",
        {"retrigger_value": 0.25, "retrigger_goal_pos_x_scale": 1.5},
    )

    assert orientation_mode == "remaining"
    assert retrigger_indicator == {
        "mode": "goal_command_z_affine_pos_x",
        "fresh_value": 0.0,
        "retrigger_value": 0.25,
        "goal_pos_x_scale": 1.5,
    }

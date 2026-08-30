# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure validation helpers for G1 deployment exports."""

from collections.abc import Mapping, Sequence


def goal_command_contract(
    function_name: str,
    params: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, str | float] | None]:
    """Resolve goal-orientation and repeated-jump observation semantics.

    Args:
        function_name: Resolved goal-command observation function name.
        params: Resolved observation function parameters.

    Returns:
        Orientation mode and an optional repeated-jump indicator contract.

    Raises:
        RuntimeError: If the observation function is not deployable.
    """
    resolved_params = params or {}
    if function_name == "obs_goal_command":
        return "trigger_relative", None
    if function_name == "obs_goal_command_remaining_orientation":
        return "remaining", None
    if function_name == "obs_goal_command_remaining_orientation_retrigger":
        return (
            "remaining",
            {
                "mode": "goal_command_z",
                "fresh_value": 0.0,
                "retrigger_value": float(resolved_params.get("retrigger_value", 0.25)),
            },
        )
    if function_name == "obs_goal_command_remaining_orientation_retrigger_goal":
        return (
            "remaining",
            {
                "mode": "goal_command_z_affine_pos_x",
                "fresh_value": 0.0,
                "retrigger_value": float(resolved_params.get("retrigger_value", 0.25)),
                "goal_pos_x_scale": float(resolved_params.get("retrigger_goal_pos_x_scale", 1.0)),
            },
        )
    raise RuntimeError(f"Unsupported goal_command observation function: {function_name!r}.")


def validate_joint_name_contract(runtime_names: Sequence[str], contract_names: Sequence[str]) -> bool:
    """Validate that runtime joints are a permutation of the policy contract.

    Args:
        runtime_names: Joint names in resolved articulation order.
        contract_names: Joint names in the task's declared order.

    Returns:
        Whether the two valid name sequences also have identical ordering.

    Raises:
        ValueError: If either sequence contains duplicates or the name sets differ.
    """
    runtime = tuple(runtime_names)
    contract = tuple(contract_names)
    if len(set(runtime)) != len(runtime):
        raise ValueError("Runtime joint names must be unique.")
    if len(set(contract)) != len(contract):
        raise ValueError("Contract joint names must be unique.")
    if set(runtime) != set(contract):
        missing = sorted(set(contract) - set(runtime))
        extra = sorted(set(runtime) - set(contract))
        raise ValueError(f"Runtime joints differ from the G1 jump contract. Missing={missing}, extra={extra}.")
    return runtime == contract

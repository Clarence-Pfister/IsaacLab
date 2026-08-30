# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrain PPO actor updates to one observation input column."""

from __future__ import annotations

from typing import Any

import torch


def configure_actor_input_adapter(runner: Any, observation_index: int) -> torch.nn.Parameter:
    """Allow training only through one first-layer actor input.

    The critic remains fully trainable. All actor parameters other than the
    selected first-layer column are frozen, and resumed optimizer momentum for
    that matrix is cleared. Therefore an input value of zero at the selected
    index produces exactly the pre-adaptation actor function.

    Args:
        runner: RSL-RL runner containing an actor and optimizer.
        observation_index: Actor observation column to adapt.

    Returns:
        The first-layer actor weight receiving constrained gradients.

    Raises:
        TypeError: If the actor does not expose a linear first MLP layer.
        ValueError: If the index is invalid or the optimizer uses weight decay.
    """
    actor = runner.alg.actor
    optimizer = runner.alg.optimizer
    first_layer = actor.mlp[0]
    if not isinstance(first_layer, torch.nn.Linear):
        raise TypeError("The actor's first MLP layer must be torch.nn.Linear.")
    if not 0 <= observation_index < first_layer.in_features:
        raise ValueError(f"observation_index must be in [0, {first_layer.in_features}), got {observation_index}.")
    if any(float(group.get("weight_decay", 0.0)) != 0.0 for group in optimizer.param_groups):
        raise ValueError("Actor input adaptation requires zero optimizer weight decay.")

    for parameter in actor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    weight = first_layer.weight
    weight.requires_grad_(True)
    optimizer.state.pop(weight, None)

    def _marker_column_gradient(gradient: torch.Tensor) -> torch.Tensor:
        constrained = torch.zeros_like(gradient)
        constrained[:, observation_index] = gradient[:, observation_index]
        return constrained

    weight.register_hook(_marker_column_gradient)
    return weight

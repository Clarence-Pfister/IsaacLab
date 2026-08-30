# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for retrigger-only actor adaptation."""

from itertools import chain
from types import SimpleNamespace

import pytest
import torch

from scripts.g1_jump_deploy.train.retrigger_adapter import configure_actor_input_adapter


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, 5),
            torch.nn.ELU(),
            torch.nn.Linear(5, 2),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)


def _runner(weight_decay: float = 0.0) -> SimpleNamespace:
    actor = _Actor()
    critic = torch.nn.Linear(4, 1)
    optimizer = torch.optim.Adam(
        chain(actor.parameters(), critic.parameters()),
        lr=1.0e-2,
        weight_decay=weight_decay,
    )
    # Populate Adam state to reproduce a resumed PPO optimizer.
    loss = actor(torch.ones(2, 4)).sum() + critic(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic, optimizer=optimizer))


def test_adapter_changes_only_marker_column_and_preserves_fresh_policy() -> None:
    runner = _runner()
    actor = runner.alg.actor
    first_weight = actor.mlp[0].weight
    actor_before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    fresh_observation = torch.randn(8, 4)
    fresh_observation[:, 2] = 0.0
    fresh_output_before = actor(fresh_observation).detach().clone()

    configure_actor_input_adapter(runner, observation_index=2)

    assert first_weight.requires_grad
    assert first_weight not in runner.alg.optimizer.state
    assert all(parameter.requires_grad for parameter in runner.alg.critic.parameters())
    assert all(not parameter.requires_grad for parameter in actor.parameters() if parameter is not first_weight)

    retrigger_observation = torch.randn(8, 4)
    retrigger_observation[:, 2] = 1.0
    loss = actor(retrigger_observation).square().mean() + runner.alg.critic(retrigger_observation).square().mean()
    loss.backward()
    runner.alg.optimizer.step()

    assert not torch.equal(first_weight[:, 2], actor_before["mlp.0.weight"][:, 2])
    torch.testing.assert_close(first_weight[:, :2], actor_before["mlp.0.weight"][:, :2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_weight[:, 3:], actor_before["mlp.0.weight"][:, 3:], rtol=0.0, atol=0.0)
    for name, parameter in actor.named_parameters():
        if name != "mlp.0.weight":
            torch.testing.assert_close(parameter, actor_before[name], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actor(fresh_observation), fresh_output_before, rtol=0.0, atol=0.0)


def test_adapter_rejects_optimizer_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight decay"):
        configure_actor_input_adapter(_runner(weight_decay=1.0e-3), observation_index=2)


def test_adapter_rejects_out_of_range_observation_index() -> None:
    with pytest.raises(ValueError, match="observation_index"):
        configure_actor_input_adapter(_runner(), observation_index=4)

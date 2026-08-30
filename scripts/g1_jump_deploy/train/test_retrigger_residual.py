# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the repeat-only residual actor."""

from __future__ import annotations

from itertools import chain
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.distributions import (
    TanhGaussianDistribution,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.retrigger_residual import (
    RetriggerResidualMLPModel,
    configure_retrigger_residual_actor,
)

from scripts.g1_jump_deploy.train import train_retrigger_residual
from scripts.g1_jump_deploy.train.train_retrigger_residual import load_retrigger_actor_state


def _model() -> RetriggerResidualMLPModel:
    observations = TensorDict({"policy": torch.zeros(4, 6)}, batch_size=[4])
    return RetriggerResidualMLPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        output_dim=2,
        hidden_dims=[8, 8],
        residual_hidden_dims=[5],
        retrigger_observation_index=2,
        distribution_cfg=None,
    )


def test_residual_is_zero_initialized_and_fresh_path_remains_exact() -> None:
    model = _model()
    observation = TensorDict({"policy": torch.randn(4, 6)}, batch_size=[4])
    observation["policy"][:, 2] = 0.0
    latent = model.get_latent(observation)

    torch.testing.assert_close(model(observation), model.mlp(latent), rtol=0.0, atol=0.0)

    with torch.no_grad():
        model.retrigger_residual[-1].bias.fill_(0.5)
    torch.testing.assert_close(model(observation), model.mlp(latent), rtol=0.0, atol=0.0)

    observation["policy"][:, 2] = 0.25
    expected = model.mlp(model.get_latent(observation)) + 0.5
    torch.testing.assert_close(model(observation), expected)


def test_residual_export_matches_tensor_dict_inference() -> None:
    model = _model()
    with torch.no_grad():
        model.retrigger_residual[-1].bias.fill_(0.25)
    concatenated = torch.randn(4, 6)
    concatenated[:, 2] = torch.tensor((0.0, 0.1, 0.2, 0.3))
    observation = TensorDict({"policy": concatenated}, batch_size=[4])

    torch.testing.assert_close(model.as_onnx(verbose=False)(concatenated), model(observation))
    torch.testing.assert_close(model.as_jit()(concatenated), model(observation))


def test_stochastic_exploration_is_applied_only_to_retrigger_rows() -> None:
    model = _model()
    model.distribution = TanhGaussianDistribution(output_dim=2, init_std=0.15)
    concatenated = torch.randn(4, 6)
    concatenated[:, 2] = torch.tensor((0.0, 0.25, 0.0, 0.15))
    observation = TensorDict({"policy": concatenated}, batch_size=[4])

    deterministic = model(observation)
    torch.manual_seed(7)
    stochastic = model(observation, stochastic_output=True)

    torch.testing.assert_close(stochastic[[0, 2]], deterministic[[0, 2]], rtol=0.0, atol=0.0)
    assert not torch.equal(stochastic[[1, 3]], deterministic[[1, 3]])


def test_configure_residual_freezes_base_actor_and_keeps_critic_trainable() -> None:
    actor = _model()
    critic = torch.nn.Linear(6, 1)
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=1.0e-3)
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic, optimizer=optimizer))

    configure_retrigger_residual_actor(runner)

    assert all(parameter.requires_grad for parameter in actor.retrigger_residual.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in actor.named_parameters()
        if not name.startswith("retrigger_residual.")
    )
    assert all(parameter.requires_grad for parameter in critic.parameters())
    assert not optimizer.state


def test_configure_residual_resets_and_freezes_exploration_std() -> None:
    actor = _model()
    actor.distribution = TanhGaussianDistribution(output_dim=2, init_std=0.5)
    critic = torch.nn.Linear(6, 1)
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=1.0e-3)
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic, optimizer=optimizer))

    configure_retrigger_residual_actor(runner, exploration_std=0.15)
    actor.distribution.update(torch.zeros(1, 2))

    torch.testing.assert_close(actor.distribution.std, torch.full((1, 2), 0.15))
    assert actor.distribution.std_param.requires_grad is False


def test_load_retrigger_actor_state_accepts_base_and_residual_checkpoints() -> None:
    source = _model()
    full_state = source.state_dict()
    base_state = {name: value for name, value in full_state.items() if not name.startswith("retrigger_residual.")}

    assert load_retrigger_actor_state(_model(), base_state) == "initialized"
    assert load_retrigger_actor_state(_model(), full_state) == "resumed"


def test_retrigger_training_disables_random_initial_episode_lengths(monkeypatch) -> None:
    received = {}

    def standard_learn(_runner, num_learning_iterations, init_at_random_ep_len=False):
        received["iterations"] = num_learning_iterations
        received["randomized"] = init_at_random_ep_len
        return "complete"

    monkeypatch.setattr(train_retrigger_residual, "_STANDARD_ON_POLICY_LEARN", standard_learn)

    result = train_retrigger_residual.learn_retrigger_from_full_episode(
        object(),
        num_learning_iterations=7,
        init_at_random_ep_len=True,
    )

    assert result == "complete"
    assert received == {"iterations": 7, "randomized": False}

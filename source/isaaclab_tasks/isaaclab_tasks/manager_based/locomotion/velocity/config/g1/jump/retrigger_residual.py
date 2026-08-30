# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Repeat-only residual policy model for the G1 jump task."""

from __future__ import annotations

import copy
from typing import Any

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict

from .distributions import TanhGaussianDistribution


class RetriggerResidualMLPModel(MLPModel):
    """Add a gated residual controller without changing fresh-jump outputs."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        retrigger_observation_index: int = 245,
        residual_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
    ) -> None:
        """Initialize the base policy and repeat-only residual branch.

        Args:
            obs: Observation dictionary used to resolve the actor input.
            obs_groups: Mapping from model sets to environment observation groups.
            obs_set: Observation set consumed by this model.
            output_dim: Number of policy outputs.
            hidden_dims: Base-policy hidden layer sizes.
            activation: Neural-network activation name.
            obs_normalization: Whether to normalize observations online.
            distribution_cfg: Optional output distribution configuration.
            retrigger_observation_index: Index of the repeat-only gate in the
                concatenated actor observation.
            residual_hidden_dims: Hidden layer sizes of the residual branch.

        Raises:
            ValueError: If the gate index or residual dimensions are invalid,
                or observation normalization could make a zero gate nonzero.
        """
        if obs_normalization:
            raise ValueError("RetriggerResidualMLPModel requires obs_normalization=False.")
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        if not 0 <= retrigger_observation_index < self.obs_dim:
            raise ValueError(
                f"retrigger_observation_index must be in [0, {self.obs_dim}), got {retrigger_observation_index}."
            )
        if not residual_hidden_dims or any(value <= 0 for value in residual_hidden_dims):
            raise ValueError("residual_hidden_dims must contain positive layer sizes.")
        residual_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.retrigger_observation_index = retrigger_observation_index
        self.retrigger_residual = MLP(
            self.obs_dim,
            residual_output_dim,
            residual_hidden_dims,
            activation,
        )
        output_layer = self.retrigger_residual[-1]
        if not isinstance(output_layer, torch.nn.Linear):
            raise TypeError("The repeat residual must end in a linear layer.")
        torch.nn.init.zeros_(output_layer.weight)
        torch.nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Evaluate the base policy plus the repeat-gated residual."""
        latent = self.get_latent(obs, masks, hidden_state)
        gate = latent[..., self.retrigger_observation_index : self.retrigger_observation_index + 1].ne(0)
        base_output = self.mlp(latent)
        residual = gate.to(dtype=latent.dtype) * self.retrigger_residual(latent)
        model_output = base_output + residual
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(model_output)
                sampled_output = self.distribution.sample()
                deterministic_output = self.distribution.deterministic_output(model_output)
                return torch.where(gate, sampled_output, deterministic_output)
            return self.distribution.deterministic_output(model_output)
        return model_output

    def as_jit(self) -> torch.nn.Module:
        """Return an exportable deterministic residual policy."""
        return _RetriggerResidualExport(self)

    def as_onnx(self, verbose: bool) -> torch.nn.Module:
        """Return an ONNX-compatible deterministic residual policy."""
        del verbose
        return _RetriggerResidualExport(self)


class _RetriggerResidualExport(torch.nn.Module):
    """Tensor-input export form of :class:`RetriggerResidualMLPModel`."""

    is_recurrent: bool = False

    def __init__(self, model: RetriggerResidualMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.retrigger_residual = copy.deepcopy(model.retrigger_residual)
        self.retrigger_observation_index = model.retrigger_observation_index
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else torch.nn.Identity()
        )
        self.input_size = model.obs_dim

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Evaluate deterministic actions from a concatenated observation."""
        latent = self.obs_normalizer(observation)
        gate = latent[..., self.retrigger_observation_index : self.retrigger_observation_index + 1].ne(0)
        output = self.mlp(latent) + gate.to(dtype=latent.dtype) * self.retrigger_residual(latent)
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent state; this feed-forward model has none."""

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return a representative ONNX input."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output names."""
        return ["actions"]


def configure_retrigger_residual_actor(
    runner: Any,
    exploration_std: float | None = None,
) -> None:
    """Freeze the base actor and train only its repeat residual branch.

    The critic remains trainable. Optimizer state is cleared so a resumed base
    checkpoint cannot update newly frozen parameters through stale momentum.

    Args:
        runner: RSL-RL runner containing the actor, critic, and optimizer.
        exploration_std: Optional fixed latent standard deviation for residual
            training. The parameter is frozen together with the base actor.

    Raises:
        TypeError: If the configured actor is not repeat-residual capable.
    """
    actor = runner.alg.actor
    if not isinstance(actor, RetriggerResidualMLPModel):
        raise TypeError("The actor must be RetriggerResidualMLPModel.")
    if exploration_std is not None:
        if not isinstance(actor.distribution, TanhGaussianDistribution):
            raise TypeError("Exploration reset requires TanhGaussianDistribution.")
        actor.distribution.set_std(exploration_std)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in actor.retrigger_residual.parameters():
        parameter.requires_grad_(True)
    for parameter in runner.alg.critic.parameters():
        parameter.requires_grad_(True)
    runner.alg.optimizer.state.clear()

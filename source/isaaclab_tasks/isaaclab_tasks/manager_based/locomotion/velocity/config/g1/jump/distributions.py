# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded policy distributions for the G1 jump task."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from rsl_rl.modules import Distribution
from torch import nn
from torch.distributions import Normal


class TanhGaussianDistribution(Distribution):
    """Diagonal Gaussian transformed to the closed interval ``[-1, 1]`` by ``tanh``.

    PPO operates on the transformed action and uses the change-of-variables correction in
    :meth:`log_prob`. The latent Gaussian parameters remain available through :attr:`params`,
    so its analytic KL divergence can still drive the adaptive learning-rate schedule.
    """

    LATENT_MEAN_LIMIT = 3.0
    """Maximum magnitude of the latent Gaussian mean."""

    MIN_STD = 0.005
    """Minimum latent Gaussian standard deviation."""

    MAX_STD = 1.0
    """Maximum latent Gaussian standard deviation."""

    class _DeterministicOutput(nn.Module):
        """Export-friendly deterministic transform for the policy MLP output."""

        def __init__(self, latent_mean_limit: float) -> None:
            super().__init__()
            self.latent_mean_limit = latent_mean_limit

        def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
            """Bound the latent mean, then squash it to the action interval."""
            latent_mean = self.latent_mean_limit * torch.tanh(mlp_output / self.latent_mean_limit)
            return torch.tanh(latent_mean)

    def __init__(self, output_dim: int, init_std: float = 1.0, std_type: str = "scalar") -> None:
        """Initialize the transformed Gaussian.

        Args:
            output_dim: Number of policy outputs.
            init_std: Initial latent Gaussian standard deviation.
            std_type: Standard-deviation parameterization, either ``"scalar"`` or ``"log"``.

        Raises:
            ValueError: If :paramref:`std_type` is unsupported or :paramref:`init_std` is not positive.
        """
        super().__init__(output_dim)
        if not self.MIN_STD <= init_std <= self.MAX_STD:
            raise ValueError(f"Initial standard deviation must be in [{self.MIN_STD}, {self.MAX_STD}], got {init_std}.")
        self.std_type = std_type
        if std_type == "scalar":
            # Store an unconstrained logit while retaining the conventional state-dict key.
            # This prevents a PPO update from making the sampled standard deviation negative.
            normalized_std = (init_std - self.MIN_STD) / (self.MAX_STD - self.MIN_STD)
            epsilon = torch.finfo(torch.float32).eps
            normalized_std = min(max(normalized_std, epsilon), 1.0 - epsilon)
            initial_parameter = math.log(normalized_std / (1.0 - normalized_std))
            self.std_param = nn.Parameter(initial_parameter * torch.ones(output_dim))
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)))
        else:
            raise ValueError(f"Unknown standard deviation type {std_type!r}; expected 'scalar' or 'log'.")
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the latent Gaussian from an MLP output."""
        latent_mean = self.LATENT_MEAN_LIMIT * torch.tanh(mlp_output / self.LATENT_MEAN_LIMIT)
        if self.std_type == "scalar":
            std = self.MIN_STD + (self.MAX_STD - self.MIN_STD) * torch.sigmoid(self.std_param)
            std = std.expand_as(latent_mean)
        else:
            std = torch.exp(self.log_std_param).clamp(min=self.MIN_STD, max=self.MAX_STD).expand_as(latent_mean)
        self._distribution = Normal(latent_mean, std)

    def set_std(self, std: float) -> None:
        """Set a fixed latent exploration standard deviation.

        Args:
            std: Standard deviation to encode in the learned distribution
                parameter.

        Raises:
            ValueError: If :paramref:`std` is non-finite or outside the
                supported interval.
        """
        if not math.isfinite(std) or not self.MIN_STD <= std <= self.MAX_STD:
            raise ValueError(f"Exploration standard deviation must be in [{self.MIN_STD}, {self.MAX_STD}], got {std}.")
        with torch.no_grad():
            if self.std_type == "scalar":
                normalized_std = (std - self.MIN_STD) / (self.MAX_STD - self.MIN_STD)
                epsilon = torch.finfo(self.std_param.dtype).eps
                normalized_std = min(max(normalized_std, epsilon), 1.0 - epsilon)
                self.std_param.fill_(math.log(normalized_std / (1.0 - normalized_std)))
            else:
                self.log_std_param.fill_(math.log(std))
        self._distribution = None

    def sample(self) -> torch.Tensor:
        """Sample and squash an action from the current latent Gaussian."""
        return torch.tanh(self._require_distribution().sample())

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Squash the latent mean for deterministic inference."""
        return self._DeterministicOutput(self.LATENT_MEAN_LIMIT)(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return the export-friendly deterministic squashing module."""
        return self._DeterministicOutput(self.LATENT_MEAN_LIMIT)

    @property
    def input_dim(self) -> int:
        """Number of latent values required from the MLP."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Latent Gaussian mean used for KL computation."""
        return self._require_distribution().mean

    @property
    def std(self) -> torch.Tensor:
        """Latent Gaussian standard deviation used for KL computation."""
        return self._require_distribution().stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Latent Gaussian entropy summed over action dimensions.

        The transformed distribution has no closed-form entropy. The jump task disables the
        PPO entropy bonus, so this stable analytic value is exposed only for diagnostics.
        """
        return self._require_distribution().entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Latent mean and standard deviation used to evaluate analytic KL divergence."""
        return self.mean, self.std

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the transformed action density with its Jacobian correction."""
        epsilon = torch.finfo(outputs.dtype).eps
        bounded_outputs = torch.clamp(outputs, min=-1.0 + epsilon, max=1.0 - epsilon)
        latent = torch.atanh(bounded_outputs)
        # Numerically stable log(1 - tanh(latent)^2), as used by squashed-Gaussian policies.
        log_jacobian = 2.0 * (math.log(2.0) - latent - functional.softplus(-2.0 * latent))
        return (self._require_distribution().log_prob(latent) - log_jacobian).sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Compute ``KL(old || new)`` in latent space.

        The ``tanh`` transform is a shared bijection away from its measure-zero endpoints, so
        it preserves KL divergence.
        """
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_distribution = Normal(old_mean, old_std)
        new_distribution = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_distribution, new_distribution).sum(dim=-1)

    def _require_distribution(self) -> Normal:
        """Return the current latent distribution or reject use before :meth:`update`."""
        if self._distribution is None:
            raise RuntimeError("update() must be called before using the stochastic distribution.")
        return self._distribution

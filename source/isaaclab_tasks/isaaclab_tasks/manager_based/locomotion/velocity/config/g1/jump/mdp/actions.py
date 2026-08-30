# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Filtered joint position action class for the G1 jump task.

Resolved from its config's ``class_type`` string after the app starts; see actions_cfg.py."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils.buffers import DelayBuffer

from .torque_projection import project_pd_position_target, project_position_target_to_lower_limit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import LowPassJointPositionActionCfg


class LowPassJointPositionAction(JointPositionAction):
    """Joint position action with an exponential low-pass filter on position targets."""

    cfg: LowPassJointPositionActionCfg

    def __init__(self, cfg: LowPassJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        if cfg.min_delay_steps < 0:
            raise ValueError(f"Minimum action delay must be non-negative. Got {cfg.min_delay_steps}.")
        if cfg.min_delay_steps > cfg.max_delay_steps:
            raise ValueError(
                "Minimum action delay must not exceed maximum action delay. "
                f"Got {cfg.min_delay_steps} and {cfg.max_delay_steps}."
            )
        self._action_delay_buffer = DelayBuffer(cfg.max_delay_steps, self.num_envs, device=self.device)

        self._alpha = torch.ones((1, self.action_dim), device=self.device)
        if isinstance(cfg.alpha, (float, int)):
            if not 0.0 < cfg.alpha <= 1.0:
                raise ValueError(f"Filter alpha must be in the range (0, 1]. Got {cfg.alpha}.")
            self._alpha.fill_(float(cfg.alpha))
        elif isinstance(cfg.alpha, dict):
            joint_ids, joint_names, alpha_values = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            for joint_name, alpha in zip(joint_names, alpha_values):
                if not 0.0 < alpha <= 1.0:
                    raise ValueError(f"Filter alpha must be in the range (0, 1]. Got {alpha} for {joint_name}.")
            self._alpha[:, joint_ids] = torch.tensor(alpha_values, device=self.device)
        else:
            raise TypeError(f"Unsupported filter alpha type: {type(cfg.alpha)}.")

        self._previous_targets = torch.zeros_like(self.processed_actions)

        self._effort_limit_ratio: torch.Tensor | None = None
        if isinstance(cfg.effort_limit_ratio, (float, int)):
            if not 0.0 < cfg.effort_limit_ratio <= 1.0:
                raise ValueError(f"Effort-limit ratio must be in the range (0, 1]. Got {cfg.effort_limit_ratio}.")
            self._effort_limit_ratio = torch.full(
                (1, self.action_dim), float(cfg.effort_limit_ratio), device=self.device
            )
        elif isinstance(cfg.effort_limit_ratio, dict):
            effort_limit_ratio = torch.ones((1, self.action_dim), device=self.device)
            joint_ids, joint_names, ratio_values = string_utils.resolve_matching_names_values(
                cfg.effort_limit_ratio, self._joint_names
            )
            for joint_name, ratio in zip(joint_names, ratio_values):
                if not 0.0 < ratio <= 1.0:
                    raise ValueError(f"Effort-limit ratio must be in the range (0, 1]. Got {ratio} for {joint_name}.")
            effort_limit_ratio[:, joint_ids] = torch.tensor(ratio_values, device=self.device)
            self._effort_limit_ratio = effort_limit_ratio
        elif cfg.effort_limit_ratio is not None:
            raise TypeError(f"Unsupported effort-limit ratio type: {type(cfg.effort_limit_ratio)}.")

        self._lower_limit_velocity_lookahead: torch.Tensor | None = None
        if cfg.lower_limit_velocity_lookahead is not None:
            if cfg.clip is None or not hasattr(self, "_clip"):
                raise ValueError("Lower-limit velocity lookahead requires finite action clip bounds.")
            velocity_lookahead = torch.zeros((1, self.action_dim), device=self.device)
            joint_ids, joint_names, lookahead_values = string_utils.resolve_matching_names_values(
                cfg.lower_limit_velocity_lookahead, self._joint_names
            )
            for joint_name, lookahead in zip(joint_names, lookahead_values):
                if lookahead < 0.0:
                    raise ValueError(
                        f"Lower-limit velocity lookahead must be non-negative. Got {lookahead} s for {joint_name}."
                    )
            velocity_lookahead[:, joint_ids] = torch.tensor(lookahead_values, device=self.device)
            active = velocity_lookahead > 0.0
            finite_bounds = torch.isfinite(self._clip[:, :, 0]) & torch.isfinite(self._clip[:, :, 1])
            if not bool(torch.all(finite_bounds | ~active)):
                raise ValueError("Lower-limit velocity lookahead requires finite clip bounds for active joints.")
            self._lower_limit_velocity_lookahead = velocity_lookahead

    def process_actions(self, actions: torch.Tensor) -> None:
        delayed_actions = self._action_delay_buffer.compute(actions)
        super().process_actions(delayed_actions)
        self._processed_actions[:] = (
            self._alpha * self._processed_actions + (1.0 - self._alpha) * self._previous_targets
        )
        self._previous_targets[:] = self._processed_actions

    def apply_actions(self) -> None:
        if self._effort_limit_ratio is None and self._lower_limit_velocity_lookahead is None:
            super().apply_actions()
            return

        data = self._asset.data
        joint_pos = data.joint_pos.torch[:, self._joint_ids]
        joint_vel = data.joint_vel.torch[:, self._joint_ids]
        requested_target = self.processed_actions
        if self._lower_limit_velocity_lookahead is not None:
            requested_target = project_position_target_to_lower_limit(
                requested_target,
                joint_vel,
                self._clip[:, :, 0],
                self._clip[:, :, 1],
                self._lower_limit_velocity_lookahead,
            )
        if self._effort_limit_ratio is None:
            self._asset.set_joint_position_target_index(target=requested_target, joint_ids=self._joint_ids)
            return

        stiffness = data.joint_stiffness.torch[:, self._joint_ids]
        damping = data.joint_damping.torch[:, self._joint_ids]
        effort_limit = data.joint_effort_limits.torch[:, self._joint_ids]
        projected_target = project_pd_position_target(
            requested_target,
            joint_pos,
            joint_vel,
            stiffness,
            damping,
            effort_limit,
            self._effort_limit_ratio,
        )
        self._asset.set_joint_position_target_index(target=projected_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)

        if self.cfg.min_delay_steps == self.cfg.max_delay_steps:
            time_lags = self.cfg.min_delay_steps
        else:
            num_envs = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
            time_lags = torch.randint(
                low=self.cfg.min_delay_steps,
                high=self.cfg.max_delay_steps + 1,
                size=(num_envs,),
                dtype=torch.int,
                device=self.device,
            )
        self._action_delay_buffer.set_time_lag(time_lags, env_ids)
        self._action_delay_buffer.reset(env_ids)

        self._previous_targets[env_ids] = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]

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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import LowPassJointPositionActionCfg


class LowPassJointPositionAction(JointPositionAction):
    """Joint position action with an exponential low-pass filter on position targets."""

    cfg: LowPassJointPositionActionCfg

    def __init__(self, cfg: LowPassJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

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

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions[:] = (
            self._alpha * self._processed_actions + (1.0 - self._alpha) * self._previous_targets
        )
        self._previous_targets[:] = self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._previous_targets[env_ids] = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]

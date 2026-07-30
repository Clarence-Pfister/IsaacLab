# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the G1 jump task."""

from __future__ import annotations

import torch

from ..constants import JUMP_PHASES, REFERENCE_MOTION_FPS
from .motion import get_env_time, get_jump_phase, get_loader


def obs_future_reference_preview(env) -> torch.Tensor:
    """Return the reference preview at offsets of 1, 4, and 7 reference frames.

    The offsets are 0.0333, 0.1333, and 0.2333 seconds at 30 FPS. The preview is
    ``[qz^r(t), qm^r(t+1), qm^r(t+4), qm^r(t+7)]`` and has 70 elements.
    """
    loader = get_loader(env)
    current_time = get_env_time(env)
    reference_dt = 1.0 / REFERENCE_MOTION_FPS

    # Define future time offsets for preview
    t_0 = current_time
    t_1 = current_time + (1 * reference_dt)
    t_4 = current_time + (4 * reference_dt)
    t_7 = current_time + (7 * reference_dt)

    # Fetch reference states at respective times
    _, _, ref_root_0, _, _, _, _ = loader.get_state(t_0)
    ref_pos_1, _, _, _, _, _, _ = loader.get_state(t_1)
    ref_pos_4, _, _, _, _, _, _ = loader.get_state(t_4)
    ref_pos_7, _, _, _, _, _, _ = loader.get_state(t_7)

    # qz^r(t) is the root z position
    qz_t = ref_root_0[:, 2:3]

    # Concatenate [qz^r(t), qm^r(t+1), qm^r(t+4), qm^r(t+7)]
    preview = torch.cat((qz_t, ref_pos_1, ref_pos_4, ref_pos_7), dim=-1)
    return preview


def obs_jump_phase(env) -> torch.Tensor:
    """Returns the current jump phase as a one-hot policy observation."""
    phase = get_jump_phase(env)
    phase_obs = torch.zeros((env.num_envs, len(JUMP_PHASES)), device=env.device)
    phase_obs.scatter_(1, phase.unsqueeze(-1), 1.0)
    return phase_obs

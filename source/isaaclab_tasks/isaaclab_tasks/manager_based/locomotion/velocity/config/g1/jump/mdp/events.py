# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the G1 jump task, including reference-state initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from ..constants import REFERENCE_MOTION_FPS
from .motion import get_loader, warp_to_torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def reference_state_initialization(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    init_start_prob: float = 0.2,
):
    asset: Articulation = env.scene[asset_cfg.name]
    loader = get_loader(env)

    do_rsi = torch.rand(len(env_ids), device=env.device) < init_start_prob
    rsi_env_ids = env_ids[do_rsi]
    std_env_ids = env_ids[~do_rsi]

    if len(rsi_env_ids) > 0:
        random_frame_ids = torch.randint(0, loader.length, (len(rsi_env_ids),), device=env.device)
        start_times = random_frame_ids / REFERENCE_MOTION_FPS
        if not hasattr(env, "start_times"):
            env.start_times = torch.zeros(env.num_envs, device=env.device)
        env.start_times[rsi_env_ids] = start_times

        ref_joint_pos = loader.ref_joint_pos[random_frame_ids]
        ref_joint_vel = loader.ref_joint_vel[random_frame_ids]
        ref_root_pos = loader.ref_root_pos[random_frame_ids]
        ref_root_vel = loader.ref_root_vel[random_frame_ids]
        ref_root_ang_vel = loader.ref_root_ang_vel[random_frame_ids]
        ref_root_quat = loader.ref_root_quat[random_frame_ids]

        if loader.joint_ids is None:
            joint_ids, _ = asset.find_joints(loader.joint_names, preserve_order=True)
            loader.joint_ids = torch.tensor(joint_ids, device=env.device)

        init_joint_pos = warp_to_torch(asset.data.default_joint_pos)[rsi_env_ids].clone()
        init_joint_pos[:, loader.joint_ids] = ref_joint_pos
        init_joint_vel = torch.zeros_like(init_joint_pos)
        init_joint_vel[:, loader.joint_ids] = ref_joint_vel
        init_root_pos = torch.zeros((len(rsi_env_ids), 3), device=env.device)
        env_origins = env.scene.env_origins[rsi_env_ids]
        init_root_pos[:, :2] = env_origins[:, :2]  # Need to change in Stage 2&3
        init_root_pos[:, 2] = ref_root_pos[:, 2]
        init_root_pose = torch.cat([init_root_pos, ref_root_quat], dim=-1)
        init_root_vel = torch.zeros((len(rsi_env_ids), 6), device=env.device)
        init_root_vel[:, :3] = ref_root_vel
        init_root_vel[:, 3:] = ref_root_ang_vel

        asset.write_joint_position_to_sim_index(position=init_joint_pos, env_ids=rsi_env_ids)
        asset.write_joint_velocity_to_sim_index(velocity=init_joint_vel, env_ids=rsi_env_ids)
        asset.write_root_pose_to_sim_index(root_pose=init_root_pose, env_ids=rsi_env_ids)
        # Reference velocities are link-derived, so use the matching link-frame writer.
        asset.write_root_link_velocity_to_sim_index(root_velocity=init_root_vel, env_ids=rsi_env_ids)

    if len(std_env_ids) > 0:
        if not hasattr(env, "start_times"):
            env.start_times = torch.zeros(env.num_envs, device=env.device)
        env.start_times[std_env_ids] = 0.0

        default_joint_pos = warp_to_torch(asset.data.default_joint_pos)[std_env_ids].clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        default_root_pose = warp_to_torch(asset.data.default_root_pose)[std_env_ids].clone()
        default_root_pose[:, :3] += env.scene.env_origins[std_env_ids]
        default_root_vel = torch.zeros((len(std_env_ids), 6), device=env.device)

        asset.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=std_env_ids)
        asset.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=std_env_ids)
        asset.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=std_env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=std_env_ids)

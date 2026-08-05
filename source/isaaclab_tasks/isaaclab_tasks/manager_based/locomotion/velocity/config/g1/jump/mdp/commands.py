# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal command term for the G1 jump task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    quat_unique,
)

from .motion import warp_to_torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class JumpGoalCommand(CommandTerm):
    """Generates random target jump pose relative to robot, fixed in world frame."""

    cfg: JumpGoalCommandCfg

    def __init__(self, cfg: JumpGoalCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 6] = 1.0
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 6] = 1.0
        self.target_displacement_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.target_yaw_displacement_w = torch.zeros(self.num_envs, device=self.device)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        r = self.cfg.ranges
        num_resampling = len(env_ids)

        # Generate relative targets
        dx = torch.zeros(num_resampling, device=self.device).uniform_(*r.pos_x)
        dy = torch.zeros(num_resampling, device=self.device).uniform_(*r.pos_y)
        self.pose_command_b[env_ids, 0] = dx  # cx
        self.pose_command_b[env_ids, 1] = dy  # cy
        self.pose_command_b[env_ids, 2] = 0.0  # cz
        euler_angles = torch.zeros((num_resampling, 3), device=self.device)
        euler_angles[:, 0].uniform_(*r.roll)  # c_psi
        euler_angles[:, 1].uniform_(*r.pitch)  # c_theta
        euler_angles[:, 2].uniform_(*r.yaw)  # c_phi
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

        # Fetch current roots
        if (
            self.robot.is_initialized
            and hasattr(self.robot, "data")
            and self.robot.data.root_pos_w.shape[0] == self.num_envs
            and self.robot.data.root_quat_w.shape[0] == self.num_envs
        ):
            current_root_pos = warp_to_torch(self.robot.data.root_pos_w).to(self.device)[env_ids]
            current_root_quat = warp_to_torch(self.robot.data.root_quat_w).to(self.device)[env_ids]
        else:
            # Fall back only for pre-initialization resampling; prefer real state when available.
            current_root_pos = torch.zeros((num_resampling, 3), device=self.device)
            if hasattr(self._env.scene, "env_origins"):
                env_origins = warp_to_torch(self._env.scene.env_origins).to(self.device)
                current_root_pos[:, :2] = env_origins[env_ids, :2]
            initial_rot = torch.tensor(self.robot.cfg.init_state.rot, device=self.device)
            current_root_quat = initial_rot.unsqueeze(0).expand(num_resampling, -1)

        # Convert to world frame
        pos_w, quat_w = combine_frame_transforms(
            current_root_pos,
            current_root_quat,
            self.pose_command_b[env_ids, :3],
            self.pose_command_b[env_ids, 3:],
        )
        self.pose_command_w[env_ids, :3] = pos_w
        self.pose_command_w[env_ids, 3:] = quat_w
        self.target_displacement_w[env_ids] = pos_w[:, :2] - current_root_pos[:, :2]
        _, _, goal_yaw = euler_xyz_from_quat(quat_w)
        _, _, root_yaw = euler_xyz_from_quat(current_root_quat)
        delta = goal_yaw - root_yaw
        self.target_yaw_displacement_w[env_ids] = torch.atan2(torch.sin(delta), torch.cos(delta))

        # Query terrain height at the target (x, y) position and set cz accordingly
        self.pose_command_w[env_ids, 2] = self._query_terrain_height(
            self.pose_command_w[env_ids, 0], self.pose_command_w[env_ids, 1]
        )

    def _update_command(self):
        pass  # Command is static during episode, so no update needed

    def _update_metrics(self):
        if (
            self.robot.is_initialized
            and hasattr(self.robot, "data")
            and self.robot.data.root_pos_w.shape[0] == self.num_envs
        ):
            current_pos_w = warp_to_torch(self.robot.data.root_pos_w)[:, :3]
        else:
            if hasattr(self._env.scene, "env_origins"):
                current_pos_w = warp_to_torch(self._env.scene.env_origins).to(self.device)
            else:
                current_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.metrics["position_error"] = torch.linalg.norm(self.pose_command_w[:, :2] - current_pos_w[:, :2], dim=-1)

    def _query_terrain_height(self, x_targets: torch.Tensor, y_targets: torch.Tensor) -> torch.Tensor:
        """Return terrain heights at target positions.

        Args:
            x_targets: Target world-frame x-coordinates [m].
            y_targets: Target world-frame y-coordinates [m].

        Returns:
            Terrain heights [m], with the same shape and dtype as ``x_targets``.

        Raises:
            NotImplementedError: If the configured terrain is not a plane.
        """
        terrain_type = self._env.scene.cfg.terrain.terrain_type
        if terrain_type == "plane":
            return torch.zeros_like(x_targets)

        raise NotImplementedError(
            "Per-target terrain height lookup is not implemented for terrain type "
            f"{terrain_type!r}. A height query (for example, an "
            "isaaclab.sensors.RayCaster) must be wired up before using non-flat "
            "terrain; silently returning 0.0 would place goals below or above the "
            "real ground."
        )

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_w

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # The command is resampled on reset, before the articulation exists on the first frame.
        if not self.robot.is_initialized:
            return
        # Lift the marker clear of the ground plane so the triad is not half-buried in it.
        marker_pos = self.pose_command_w[:, :3].clone()
        marker_pos[:, 2] += self.cfg.goal_marker_height
        self.goal_pose_visualizer.visualize(marker_pos, self.pose_command_w[:, 3:])


@configclass
class JumpGoalCommandCfg(CommandTermCfg):
    class_type: type = JumpGoalCommand
    asset_name: str = "robot"
    make_quat_unique: bool = True
    resampling_time_range: tuple[float, float] = (
        30.0,
        30.0,
    )  # Resample every 30 seconds

    @configclass
    class Ranges:
        pos_x = (0.0, 0.0)  # cx
        pos_y = (0.0, 0.0)  # cy
        roll = (0.0, 0.0)  # c_psi
        pitch = (0.0, 0.0)  # c_theta
        yaw = (0.0, 0.0)  # c_phi

    ranges: Ranges = Ranges()

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/jump_goal")
    """Marker drawn at the commanded landing pose when ``debug_vis`` is set.

    A frame triad rather than a sphere: the goal carries a turn as well as a position, and only
    an oriented marker shows whether the robot finished facing where it was told to.
    """

    goal_marker_height: float = 0.05
    """Height above the goal at which the marker is drawn [m]."""

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.25, 0.25, 0.25)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal command term for the G1 jump task."""

from __future__ import annotations

import math
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


def _sample_planar_displacement(
    sample_count: int,
    pos_x_range: tuple[float, float],
    pos_y_range: tuple[float, float],
    zero_goal_probability: float,
    boundary_goal_probability: float,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a mixture of uniform, zero, and boundary planar goals.

    Args:
        sample_count: Number of displacement samples.
        pos_x_range: Minimum and maximum forward displacement [m].
        pos_y_range: Minimum and maximum lateral displacement [m].
        zero_goal_probability: Probability of sampling exactly zero displacement.
        boundary_goal_probability: Probability of sampling each coordinate at one of
            its configured range boundaries.
        device: Torch device on which to create the samples.

    Returns:
        Forward and lateral displacement samples [m], each with shape ``(sample_count,)``.

    Raises:
        ValueError: If either probability is outside ``[0, 1]`` or their sum exceeds one.
    """
    probabilities = (zero_goal_probability, boundary_goal_probability)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError(f"Goal-mixture probabilities must be finite and in [0, 1], got {probabilities}.")
    if sum(probabilities) > 1.0:
        raise ValueError(f"Goal-mixture probabilities must sum to at most 1, got {probabilities}.")

    dx = torch.empty(sample_count, device=device).uniform_(*pos_x_range)
    dy = torch.empty(sample_count, device=device).uniform_(*pos_y_range)
    category = torch.rand(sample_count, device=device)
    zero_goal = category < zero_goal_probability
    boundary_goal = (category >= zero_goal_probability) & (category < zero_goal_probability + boundary_goal_probability)
    if torch.any(zero_goal):
        dx[zero_goal] = 0.0
        dy[zero_goal] = 0.0
    if torch.any(boundary_goal):
        boundary_count = int(torch.count_nonzero(boundary_goal).item())
        choose_upper = torch.rand((boundary_count, 2), device=device) >= 0.5
        dx[boundary_goal] = torch.where(
            choose_upper[:, 0],
            torch.as_tensor(pos_x_range[1], device=device),
            torch.as_tensor(pos_x_range[0], device=device),
        )
        dy[boundary_goal] = torch.where(
            choose_upper[:, 1],
            torch.as_tensor(pos_y_range[1], device=device),
            torch.as_tensor(pos_y_range[0], device=device),
        )
    return dx, dy


def _next_longitudinal_cycle_goal(
    previous_goal: torch.Tensor,
    pos_x_range: tuple[float, float],
    reverse_cycle: torch.Tensor,
) -> torch.Tensor:
    """Advance longitudinal goals through one of two three-command cycles.

    The forward cycle is ``lower -> zero -> upper -> lower`` and the reverse
    cycle traverses the same anchors in the opposite order. Assigning both
    cycles across environments balances every directed transition between
    the lower boundary, zero, and the upper boundary.

    Args:
        previous_goal: Previous longitudinal displacement command [m].
        pos_x_range: Lower and upper longitudinal command boundaries [m].
        reverse_cycle: Whether each sample follows the reverse cycle.

    Returns:
        Next longitudinal displacement commands [m], with the same shape and
        dtype as :paramref:`previous_goal`.

    Raises:
        ValueError: If tensor shapes differ, a previous goal is not finite, or
            the configured range does not strictly span zero.
    """
    if previous_goal.shape != reverse_cycle.shape:
        raise ValueError(
            "previous_goal and reverse_cycle must have the same shape, "
            f"got {previous_goal.shape} and {reverse_cycle.shape}."
        )
    lower, upper = pos_x_range
    if not all(math.isfinite(value) for value in pos_x_range) or not lower < 0.0 < upper:
        raise ValueError(
            f"Longitudinal cycle goals require a finite range that strictly spans zero, got {pos_x_range}."
        )
    if not bool(torch.all(torch.isfinite(previous_goal))):
        raise ValueError("previous_goal must contain only finite values.")

    anchors = previous_goal.new_tensor((lower, 0.0, upper))
    closest_anchor = torch.argmin(
        torch.abs(previous_goal.unsqueeze(-1) - anchors),
        dim=-1,
    )
    direction = torch.where(
        reverse_cycle.to(device=previous_goal.device, dtype=torch.bool),
        -torch.ones_like(closest_anchor),
        torch.ones_like(closest_anchor),
    )
    return anchors[torch.remainder(closest_anchor + direction, len(anchors))]


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
        self.metrics["yaw_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["retrigger_reset"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        r = self.cfg.ranges
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        num_resampling = len(env_ids)

        # Generate relative targets
        dx, dy = _sample_planar_displacement(
            num_resampling,
            r.pos_x,
            r.pos_y,
            self.cfg.zero_goal_probability,
            self.cfg.boundary_goal_probability,
            self.device,
        )
        cycle_probability = self.cfg.retrigger_cycle_goal_probability
        if not math.isfinite(cycle_probability) or not 0.0 <= cycle_probability <= 1.0:
            raise ValueError(
                f"retrigger_cycle_goal_probability must be finite and lie in [0, 1], got {cycle_probability}."
            )
        if cycle_probability > 0.0:
            retrigger_mask = getattr(self._env, "retrigger_reset_mask", None)
            if retrigger_mask is not None:
                use_cycle = retrigger_mask[env_ids].bool()
                if cycle_probability < 1.0:
                    use_cycle &= torch.rand(num_resampling, device=self.device) < cycle_probability
                if torch.any(use_cycle):
                    previous_goal = self.pose_command_b[env_ids, 0].clone()
                    reverse_cycle = torch.remainder(env_ids, 2).bool()
                    dx[use_cycle] = _next_longitudinal_cycle_goal(
                        previous_goal[use_cycle],
                        r.pos_x,
                        reverse_cycle[use_cycle],
                    )
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
            and self.robot.data.root_quat_w.shape[0] == self.num_envs
        ):
            current_pos_w = warp_to_torch(self.robot.data.root_pos_w)[:, :3]
            current_quat_w = warp_to_torch(self.robot.data.root_quat_w)
        else:
            if hasattr(self._env.scene, "env_origins"):
                current_pos_w = warp_to_torch(self._env.scene.env_origins).to(self.device)
            else:
                current_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
            current_quat_w = torch.tensor(self.robot.cfg.init_state.rot, device=self.device).expand(self.num_envs, -1)
        self.metrics["position_error"] = torch.linalg.norm(self.pose_command_w[:, :2] - current_pos_w[:, :2], dim=-1)
        _, _, current_yaw = euler_xyz_from_quat(current_quat_w)
        _, _, target_yaw = euler_xyz_from_quat(self.pose_command_w[:, 3:])
        yaw_delta = current_yaw - target_yaw
        self.metrics["yaw_error"] = torch.abs(torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)))
        retrigger_reset_mask = getattr(self._env, "retrigger_reset_mask", None)
        if retrigger_reset_mask is None:
            self.metrics["retrigger_reset"] = torch.zeros(len(current_pos_w), device=current_pos_w.device)
        else:
            self.metrics["retrigger_reset"] = retrigger_reset_mask.to(
                device=current_pos_w.device,
                dtype=torch.float32,
            )

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

    zero_goal_probability: float = 0.0
    """Probability of sampling an exact in-place landing command."""

    boundary_goal_probability: float = 0.0
    """Probability of sampling both planar coordinates at their range boundaries."""

    retrigger_cycle_goal_probability: float = 0.0
    """Probability of cycling a carried longitudinal goal through both boundaries and zero.

    This affects only episodes carried from a preceding safe landing. Even and
    odd environment indices use opposite cycle directions so training covers
    every directed transition without changing fresh-start command sampling.
    """

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/jump_goal")
    """Marker drawn at the commanded landing pose when ``debug_vis`` is set.

    A frame triad rather than a sphere: the goal carries a turn as well as a position, and only
    an oriented marker shows whether the robot finished facing where it was told to.
    """

    goal_marker_height: float = 0.05
    """Height above the goal at which the marker is drawn [m]."""

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.25, 0.25, 0.25)

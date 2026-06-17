# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
import pandas as pd
from typing import TYPE_CHECKING
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    convert_quat,
    quat_unique,
    combine_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.assets import Articulation

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

# =============================================================================
# CONSTANTS
# =============================================================================

DATA_STORAGE_DIR = Path(__file__).resolve().parents[8] / "data_storage"
CSV_MOTION_PATH = str(DATA_STORAGE_DIR / "perfect_jump_processed.csv")
REFERENCE_NUM_FRAMES = 91
REFERENCE_MOTION_FPS = 30.0
REFERENCE_DURATION_S = REFERENCE_NUM_FRAMES / REFERENCE_MOTION_FPS
NUMBER_OF_JOINTS = 23

JUMP_PHASE_STAND_PREP = 0
JUMP_PHASE_CROUCH = 1
JUMP_PHASE_TAKEOFF = 2
JUMP_PHASE_FLIGHT_PEAK = 3
JUMP_PHASE_LANDING = 4
JUMP_PHASE_FINAL_STAND = 5
JUMP_PHASE_COUNT = 6
JUMP_PHASE_FRAME_ENDS = (6, 19, 26, 43, 60)
JUMP_PHASE_TIME_ENDS_S = tuple(
    frame / REFERENCE_MOTION_FPS for frame in JUMP_PHASE_FRAME_ENDS
)
JUMP_STAND_PREP_END_S = JUMP_PHASE_TIME_ENDS_S[0]
JUMP_CROUCH_END_S = JUMP_PHASE_TIME_ENDS_S[1]
JUMP_TAKEOFF_END_S = JUMP_PHASE_TIME_ENDS_S[2]
JUMP_FLIGHT_PEAK_END_S = JUMP_PHASE_TIME_ENDS_S[3]
JUMP_LANDING_END_S = JUMP_PHASE_TIME_ENDS_S[4]
JUMP_EPISODE_LENGTH_S = REFERENCE_DURATION_S

G1_USD_PATH = str(
    DATA_STORAGE_DIR / "g1_23dof_holo_compat" / "g1_23dof_holo_compat.usda"
)
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]
G1_23DOF_HOLO_COMPAT_ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            "waist_yaw_joint",
        ],
        effort_limit_sim=300,
        stiffness={
            ".*_hip_yaw_joint": 150.0,
            ".*_hip_roll_joint": 150.0,
            ".*_hip_pitch_joint": 200.0,
            ".*_knee_joint": 200.0,
            "waist_yaw_joint": 200.0,
        },
        damping={
            ".*_hip_yaw_joint": 5.0,
            ".*_hip_roll_joint": 5.0,
            ".*_hip_pitch_joint": 5.0,
            ".*_knee_joint": 5.0,
            "waist_yaw_joint": 5.0,
        },
        armature={
            ".*_hip_.*": 0.01,
            ".*_knee_joint": 0.01,
            "waist_yaw_joint": 0.01,
        },
    ),
    "feet": ImplicitActuatorCfg(
        effort_limit_sim=20,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=20.0,
        damping=2.0,
        armature=0.01,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
        ],
        effort_limit_sim=300,
        stiffness=40.0,
        damping=10.0,
        armature={
            ".*_shoulder_.*": 0.01,
            ".*_elbow_joint": 0.01,
            ".*_wrist_roll_joint": 0.01,
        },
    ),
}
G1_23DOF_HOLO_COMPAT_CFG = G1_MINIMAL_CFG.copy()
G1_23DOF_HOLO_COMPAT_CFG.spawn.usd_path = G1_USD_PATH
G1_23DOF_HOLO_COMPAT_CFG.actuators = G1_23DOF_HOLO_COMPAT_ACTUATORS

# =============================================================================
# UTILITY FUNCTIONS & MOTION LOADER
# =============================================================================


class MotionLoader:
    """Handles loading and interpolation of joint and root motion data from a CSV file."""

    def __init__(self, csv_path: str, device: str):
        self.device = device
        self.csv_path = os.path.abspath(csv_path)

        self.joint_names = JOINT_NAMES
        self.joint_ids = None
        self.foot_ids = None

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Reference motion CSV not found: {self.csv_path}.")

        df = pd.read_csv(self.csv_path)
        motion_dt = 1 / REFERENCE_MOTION_FPS

        # Joint Positions & Velocities
        self.ref_joint_pos = torch.tensor(
            df[JOINT_NAMES].values, device=device, dtype=torch.float32
        )
        self.ref_joint_vel = torch.zeros_like(self.ref_joint_pos)
        self.ref_joint_vel[1:] = (
            self.ref_joint_pos[1:] - self.ref_joint_pos[:-1]
        ) / motion_dt
        # Root Translation & Linear Velocity
        root_pos_cols = ["root_translateX", "root_translateY", "root_translateZ"]
        self.ref_root_pos = torch.tensor(
            df[root_pos_cols].values, device=device, dtype=torch.float32
        )
        self.ref_root_vel = torch.zeros_like(self.ref_root_pos)
        self.ref_root_vel[1:] = (
            self.ref_root_pos[1:] - self.ref_root_pos[:-1]
        ) / motion_dt
        # Root Quaternions
        root_quat_cols = [
            "root_quaternionW",
            "root_quaternionX",
            "root_quaternionY",
            "root_quaternionZ",
        ]
        self.ref_root_quat = torch.tensor(
            df[root_quat_cols].values, device=device, dtype=torch.float32
        )
        # Foot Heights
        foot_cols = ["left_foot_height", "right_foot_height"]
        self.ref_foot_height = torch.tensor(
            df[foot_cols].values, device=device, dtype=torch.float32
        )

        self.length = self.ref_joint_pos.shape[0]
        self.num_joints = self.ref_joint_pos.shape[1]

        if self.length != REFERENCE_NUM_FRAMES:
            raise ValueError(
                f"Expected {REFERENCE_NUM_FRAMES} frames in reference motion, but found {self.length}."
            )
        if self.num_joints != NUMBER_OF_JOINTS:
            raise ValueError(
                f"Expected {NUMBER_OF_JOINTS} joints in reference motion, but found {self.num_joints}."
            )
        print(f"Loaded reference motion from {self.csv_path}")
        print(f"Motion length: {self.length}, Number of joints: {self.num_joints}")

    def get_state(self, current_time: torch.Tensor):
        """Returns linearly interpolated reference states."""
        frame_idx_float = current_time * REFERENCE_MOTION_FPS
        idx_low = torch.floor(frame_idx_float).to(torch.long)
        idx_high = torch.ceil(frame_idx_float).to(torch.long)

        alpha = (frame_idx_float - idx_low).unsqueeze(-1)
        idx_low = torch.clamp(idx_low, max=self.length - 1)
        idx_high = torch.clamp(idx_high, max=self.length - 1)
        is_clamp = (idx_low >= self.length - 1).unsqueeze(-1)

        # Interpolate
        interp_joint_pos = (1.0 - alpha) * self.ref_joint_pos[
            idx_low
        ] + alpha * self.ref_joint_pos[idx_high]
        interp_root_pos = (1.0 - alpha) * self.ref_root_pos[
            idx_low
        ] + alpha * self.ref_root_pos[idx_high]
        interp_foot_height = (1.0 - alpha) * self.ref_foot_height[
            idx_low
        ] + alpha * self.ref_foot_height[idx_high]

        interp_joint_vel = (1.0 - alpha) * self.ref_joint_vel[
            idx_low
        ] + alpha * self.ref_joint_vel[idx_high]
        interp_root_vel = (1.0 - alpha) * self.ref_root_vel[
            idx_low
        ] + alpha * self.ref_root_vel[idx_high]

        # Zero velocities if animation finished
        interp_joint_vel = torch.where(
            is_clamp, torch.zeros_like(interp_joint_vel), interp_joint_vel
        )
        interp_root_vel = torch.where(
            is_clamp[..., :3], torch.zeros_like(interp_root_vel), interp_root_vel
        )

        return (
            interp_joint_pos,
            interp_joint_vel,
            interp_root_pos,
            interp_root_vel,
            interp_foot_height,
        )


def get_loader(env) -> MotionLoader:
    """Singleton accessor for MotionLoader."""
    if not hasattr(env, "motion_loader"):
        env.motion_loader = MotionLoader(CSV_MOTION_PATH, env.device)
    return env.motion_loader


def get_reward(u: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    """Computes bounded reward using exponential of negative squared error."""
    return torch.exp(-alpha * torch.sum(torch.square(u - v), dim=-1))


def get_jump_phase(env) -> torch.Tensor:
    """Returns the current reference jump phase for each environment."""
    current_time = get_env_time(env)
    phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = torch.where(
        current_time >= JUMP_STAND_PREP_END_S,
        torch.full_like(phase, JUMP_PHASE_CROUCH),
        phase,
    )
    phase = torch.where(
        current_time >= JUMP_CROUCH_END_S,
        torch.full_like(phase, JUMP_PHASE_TAKEOFF),
        phase,
    )
    phase = torch.where(
        current_time >= JUMP_TAKEOFF_END_S,
        torch.full_like(phase, JUMP_PHASE_FLIGHT_PEAK),
        phase,
    )
    phase = torch.where(
        current_time >= JUMP_FLIGHT_PEAK_END_S,
        torch.full_like(phase, JUMP_PHASE_LANDING),
        phase,
    )
    phase = torch.where(
        current_time >= JUMP_LANDING_END_S,
        torch.full_like(phase, JUMP_PHASE_FINAL_STAND),
        phase,
    )
    return phase


def get_phase_weight(env, phase_weights: Sequence[float]) -> torch.Tensor:
    """Returns a per-environment scalar weight selected by jump phase."""
    if len(phase_weights) != JUMP_PHASE_COUNT:
        raise ValueError(
            f"Expected {JUMP_PHASE_COUNT} phase weights, got {len(phase_weights)}."
        )
    weights = torch.tensor(phase_weights, device=env.device, dtype=torch.float32)
    return weights[get_jump_phase(env)]


def get_weight(
    env,
    Tj: float,
    w_before: float,
    w_after: float,
    phase_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Returns dynamic weight based on jump phase or elapsed episode time."""
    if phase_weights is not None:
        return get_phase_weight(env, phase_weights)

    current_time = get_env_time(env)
    return torch.where(current_time <= Tj, w_before, w_after)


def get_env_time(env) -> torch.Tensor:
    if not hasattr(env, "start_times"):
        env.start_times = torch.zeros(env.num_envs, device=env.device)
    return env.start_times + env.episode_length_buf * env.step_dt


def warp_to_torch(warp_array):
    """Utility to convert a Warp array to a PyTorch tensor with zero-copy if possible."""
    if isinstance(warp_array, torch.Tensor):
        return warp_array
    else:
        import warp

        return warp.to_torch(warp_array)


# =============================================================================
# COMMAND TERM
# =============================================================================


class JumpGoalCommand(CommandTerm):
    """Generates random target jump pose relative to robot, fixed in world frame."""

    cfg: JumpGoalCommandCfg

    def __init__(self, cfg: JumpGoalCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 6] = 1.0
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 6] = 1.0

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        # flag to indicate first resampling to handle initial root fetch
        self._is_first_resample = True

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
        quat = quat_from_euler_xyz(
            euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        )
        self.pose_command_b[env_ids, 3:] = (
            quat_unique(quat) if self.cfg.make_quat_unique else quat
        )

        # Fetch current roots
        if self._is_first_resample:
            current_root_pos = torch.zeros((num_resampling, 3), device=self.device)
            if hasattr(self._env.scene, "env_origins"):
                env_origins = warp_to_torch(self._env.scene.env_origins).to(self.device)
                current_root_pos[:, :2] = env_origins[env_ids, :2]
            current_root_quat = torch.zeros((num_resampling, 4), device=self.device)
            current_root_quat[:, 3] = 1.0
            self._is_first_resample = False
        else:
            current_root_pos = warp_to_torch(self.robot.data.root_pos_w).to(
                self.device
            )[env_ids]
            current_root_quat = warp_to_torch(self.robot.data.root_quat_w).to(
                self.device
            )[env_ids]

        # Convert to world frame
        pos_w, quat_w = combine_frame_transforms(
            current_root_pos,
            current_root_quat,
            self.pose_command_b[env_ids, :3],
            self.pose_command_b[env_ids, 3:],
        )
        self.pose_command_w[env_ids, :3] = pos_w
        self.pose_command_w[env_ids, 3:] = quat_w

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
                current_pos_w = warp_to_torch(self._env.scene.env_origins).to(
                    self.device
                )
            else:
                current_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.metrics["position_error"] = torch.linalg.norm(
            self.pose_command_w[:, :3] - current_pos_w, dim=-1
        )

    def _query_terrain_height(
        self, x_targets: torch.Tensor, y_targets: torch.Tensor
    ) -> torch.Tensor:
        """Helper function to query terrain height at specified (x, y) target positions using raycasting."""
        import omni.physx
        import carb

        physx_query = omni.physx.get_physx_scene_query_interface()
        heights = torch.zeros_like(x_targets, device=x_targets.device)

        for i in range(len(x_targets)):
            x_val = float(x_targets[i].item())
            y_val = float(y_targets[i].item())

            ray_origin = carb.Float3(x_val, y_val, 50.0)
            ray_direction = carb.Float3(0.0, 0.0, -1.0)
            ray_distance = 100.0

            hit = physx_query.raycast_closest(ray_origin, ray_direction, ray_distance)
            if hit["hit"]:
                heights[i] = hit["position"][2]
            else:
                heights[i] = 0.0

        return heights

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_w


# =============================================================================
# OBSERVATION TERM
# =============================================================================


def obs_future_reference_preview(env) -> torch.Tensor:
    """
    Implements the reference trajectory preview [qz^r(t), qm^r(t+1), qm^r(t+4), qm^r(t+7)].
    """
    loader = get_loader(env)
    current_time = get_env_time(env)
    step_dt = env.step_dt

    # Define future time offsets for preview
    t_0 = current_time
    t_1 = current_time + (1 * step_dt)
    t_4 = current_time + (4 * step_dt)
    t_7 = current_time + (7 * step_dt)

    # Fetch reference states at respective times
    _, _, ref_root_0, _, _ = loader.get_state(t_0)
    ref_pos_1, _, _, _, _ = loader.get_state(t_1)
    ref_pos_4, _, _, _, _ = loader.get_state(t_4)
    ref_pos_7, _, _, _, _ = loader.get_state(t_7)

    # qz^r(t) is the root z position
    qz_t = ref_root_0[:, 2:3]

    # Concatenate [qz^r(t), qm^r(t+1), qm^r(t+4), qm^r(t+7)]
    preview = torch.cat((qz_t, ref_pos_1, ref_pos_4, ref_pos_7), dim=-1)
    return preview


def obs_jump_phase(env) -> torch.Tensor:
    """Returns the current jump phase as a one-hot policy observation."""
    phase = get_jump_phase(env)
    phase_obs = torch.zeros((env.num_envs, JUMP_PHASE_COUNT), device=env.device)
    phase_obs.scatter_(1, phase.unsqueeze(-1), 1.0)
    return phase_obs


# =============================================================================
# EVENT TERM
# =============================================================================


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
        random_frame_ids = torch.randint(
            0, loader.length, (len(rsi_env_ids),), device=env.device
        )
        start_times = random_frame_ids / REFERENCE_MOTION_FPS
        if not hasattr(env, "start_times"):
            env.start_times = torch.zeros(env.num_envs, device=env.device)
        env.start_times[rsi_env_ids] = start_times

        ref_joint_pos = loader.ref_joint_pos[random_frame_ids]
        ref_joint_vel = loader.ref_joint_vel[random_frame_ids]
        ref_root_pos = loader.ref_root_pos[random_frame_ids]
        ref_root_vel = loader.ref_root_vel[random_frame_ids]
        ref_root_quat = loader.ref_root_quat[random_frame_ids]

        if loader.joint_ids is None:
            joint_ids, _ = asset.find_joints(loader.joint_names, preserve_order=True)
            loader.joint_ids = torch.tensor(joint_ids, device=env.device)

        init_joint_pos = warp_to_torch(asset.data.default_joint_pos)[
            rsi_env_ids
        ].clone()
        init_joint_pos[:, loader.joint_ids] = ref_joint_pos
        init_joint_vel = torch.zeros_like(init_joint_pos)
        init_joint_vel[:, loader.joint_ids] = ref_joint_vel
        init_root_pos = torch.zeros((len(rsi_env_ids), 3), device=env.device)
        env_origins = env.scene.env_origins[rsi_env_ids]
        init_root_pos[:, :2] = env_origins[:, :2]  # Need to fix this line later
        init_root_pos[:, 2] = ref_root_pos[:, 2]
        init_root_quat = convert_quat(ref_root_quat, to="xyzw")
        init_root_pose = torch.cat([init_root_pos, init_root_quat], dim=-1)
        init_root_vel = torch.zeros((len(rsi_env_ids), 6), device=env.device)
        init_root_vel[:, :3] = ref_root_vel  # Put 0 for angular rate

        asset.write_joint_position_to_sim_index(
            position=init_joint_pos, env_ids=rsi_env_ids
        )
        asset.write_joint_velocity_to_sim_index(
            velocity=init_joint_vel, env_ids=rsi_env_ids
        )
        asset.write_root_pose_to_sim_index(
            root_pose=init_root_pose, env_ids=rsi_env_ids
        )
        asset.write_root_velocity_to_sim_index(
            root_velocity=init_root_vel, env_ids=rsi_env_ids
        )

    if len(std_env_ids) > 0:
        if not hasattr(env, "start_times"):
            env.start_times = torch.zeros(env.num_envs, device=env.device)
        env.start_times[std_env_ids] = 0.0

        default_joint_pos = warp_to_torch(asset.data.default_joint_pos)[
            std_env_ids
        ].clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        default_root_pose = warp_to_torch(asset.data.default_root_pose)[
            std_env_ids
        ].clone()
        default_root_pose[:, :3] += env.scene.env_origins[std_env_ids]
        default_root_vel = torch.zeros((len(std_env_ids), 6), device=env.device)

        asset.write_joint_position_to_sim_index(
            position=default_joint_pos, env_ids=std_env_ids
        )
        asset.write_joint_velocity_to_sim_index(
            velocity=default_joint_vel, env_ids=std_env_ids
        )
        asset.write_root_pose_to_sim_index(
            root_pose=default_root_pose, env_ids=std_env_ids
        )
        asset.write_root_velocity_to_sim_index(
            root_velocity=default_root_vel, env_ids=std_env_ids
        )


# =============================================================================
# REWARD TERMS
# =============================================================================


def track_joint_pos(env, alpha, Tj, w_before, w_after, phase_weights=None):
    loader = get_loader(env)
    robot = env.scene["robot"]
    current_time = get_env_time(env)
    ref_joint_pos, _, _, _, _ = loader.get_state(current_time)

    if loader.joint_ids is None:
        joint_ids, _ = robot.find_joints(loader.joint_names, preserve_order=True)
        loader.joint_ids = torch.tensor(joint_ids, device=env.device)

    joint_pos = warp_to_torch(robot.data.joint_pos)
    current_joint_pos = joint_pos[:, loader.joint_ids]

    r = get_reward(current_joint_pos, ref_joint_pos, alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def track_pelvis_height(env, alpha, Tj, w_before, w_after, phase_weights=None):
    current_time = get_env_time(env)
    _, _, ref_root_pos, _, _ = get_loader(env).get_state(current_time)
    cz = env.command_manager.get_command("jump_goal")[:, 2:3]
    target_z = ref_root_pos[:, 2:3] + cz
    current_z = warp_to_torch(env.scene["robot"].data.root_pos_w)[:, 2:3]

    r = get_reward(current_z, target_z, alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def track_foot_height(env, alpha, Tj, w_before, w_after, phase_weights=None):
    loader = get_loader(env)
    robot = env.scene["robot"]
    current_time = get_env_time(env)
    _, _, _, _, ref_foot_height = loader.get_state(current_time)

    if loader.foot_ids is None:
        foot_ids, _ = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        loader.foot_ids = torch.tensor(foot_ids, device=env.device)

    body_pos_w = warp_to_torch(robot.data.body_pos_w)
    left_foot_z = body_pos_w[:, loader.foot_ids[0], 2].unsqueeze(1)
    right_foot_z = body_pos_w[:, loader.foot_ids[1], 2].unsqueeze(1)
    current_foot_height = torch.cat((left_foot_z, right_foot_z), dim=1)

    r = get_reward(current_foot_height, ref_foot_height, alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def target_pelvis_position(env, alpha, Tj, w_before, w_after, phase_weights=None):
    current_xy = warp_to_torch(env.scene["robot"].data.root_pos_w)[:, :2]
    target_xy = env.command_manager.get_command("jump_goal")[:, :2]

    r = get_reward(current_xy, target_xy, alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def target_pelvis_velocity(env, alpha, Tj, w_before, w_after, phase_weights=None):
    current_time = get_env_time(env)
    current_vel_xy = warp_to_torch(env.scene["robot"].data.root_lin_vel_w)[:, :2]
    target_vel_xy = torch.where(
        (current_time <= Tj).unsqueeze(-1),
        env.command_manager.get_command("jump_goal")[:, :2] / Tj,
        torch.zeros_like(current_vel_xy),
    )

    r = get_reward(current_vel_xy, target_vel_xy, alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def target_orientation(env, alpha, Tj, w_before, w_after, phase_weights=None):
    current_quat = warp_to_torch(env.scene["robot"].data.root_quat_w)
    target_quat = env.command_manager.get_command("jump_goal")[:, 3:7]
    dot_product = torch.sum(current_quat * target_quat, dim=-1, keepdim=True)
    quat_error = 1.0 - torch.square(dot_product)

    r = torch.exp(-alpha * quat_error).squeeze(-1)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def target_angular_rate(env, alpha, Tj, w_before, w_after, phase_weights=None):
    current_time = get_env_time(env)
    current_ang_vel = warp_to_torch(env.scene["robot"].data.root_ang_vel_w)
    target_quat = env.command_manager.get_command("jump_goal")[:, 3:7]
    _, _, target_yaw = euler_xyz_from_quat(target_quat)
    target_ang_vel = torch.zeros(env.num_envs, 3, device=env.device)
    target_ang_vel[:, 2] = torch.where(
        current_time <= Tj, target_yaw / Tj, torch.zeros_like(target_yaw)
    )

    r = get_reward(current_ang_vel, target_ang_vel, alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def penalize_ground_impact(env, alpha, Tj, w_before, w_after, phase_weights=None):
    net_forces_w = warp_to_torch(env.scene["contact_forces"].data.net_forces_w)
    net_contact_forces_z = net_forces_w[:, :, 2]
    fz_total = torch.sum(torch.abs(net_contact_forces_z), dim=1, keepdim=True)
    r = get_reward(fz_total, torch.zeros_like(fz_total), alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def penalize_torque_consumption(env, alpha, Tj, w_before, w_after, phase_weights=None):
    torques = warp_to_torch(env.scene["robot"].data.applied_torque)
    r = get_reward(torques, torch.zeros_like(torques), alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def penalize_motor_velocity(env, alpha, Tj, w_before, w_after, phase_weights=None):
    joint_vel = warp_to_torch(env.scene["robot"].data.joint_vel)
    r = get_reward(joint_vel, torch.zeros_like(joint_vel), alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


def penalize_joint_acceleration(env, alpha, Tj, w_before, w_after, phase_weights=None):
    accel = warp_to_torch(env.scene["robot"].data.joint_acc)
    r = get_reward(accel, torch.zeros_like(accel), alpha)
    return r * get_weight(env, Tj, w_before, w_after, phase_weights)


# =============================================================================
# TERMINATION TERMS
# =============================================================================


def foot_tracking_error(
    env,
    threshold: float,
    active_phases: Sequence[int] | None = None,
) -> torch.Tensor:
    """Terminates if foot height tracking error exceeds the threshold."""
    loader = get_loader(env)
    robot = env.scene["robot"]
    current_time = get_env_time(env)
    _, _, _, _, ref_foot_height = loader.get_state(current_time)

    if loader.foot_ids is None:
        foot_ids, _ = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        loader.foot_ids = torch.tensor(foot_ids, device=env.device)

    body_pos_w = warp_to_torch(robot.data.body_pos_w)
    left_foot_z = body_pos_w[:, loader.foot_ids[0], 2].unsqueeze(1)
    right_foot_z = body_pos_w[:, loader.foot_ids[1], 2].unsqueeze(1)
    current_foot_height = torch.cat((left_foot_z, right_foot_z), dim=1)

    error = torch.abs(current_foot_height - ref_foot_height)
    terminated = torch.any(error > threshold, dim=-1)

    if active_phases is not None:
        current_phase = get_jump_phase(env)
        active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for phase in active_phases:
            active |= current_phase == phase
        terminated &= active

    return terminated


def task_completion_error(
    env,
    pos_threshold: float,
    yaw_threshold: float,
    start_phase: int = JUMP_PHASE_FINAL_STAND,
) -> torch.Tensor:
    """Terminates if task error exceeds bounds once recovery should begin."""
    current_phase = get_jump_phase(env)

    current_xy = warp_to_torch(env.scene["robot"].data.root_pos_w)[:, :2]
    target_xy = env.command_manager.get_command("jump_goal")[:, :2]
    pos_error = torch.linalg.norm(current_xy - target_xy, dim=-1)

    current_quat = warp_to_torch(env.scene["robot"].data.root_quat_w)
    target_quat = env.command_manager.get_command("jump_goal")[:, 3:7]
    _, _, current_yaw = euler_xyz_from_quat(current_quat)
    _, _, target_yaw = euler_xyz_from_quat(target_quat)

    yaw_error = torch.abs(
        torch.atan2(
            torch.sin(current_yaw - target_yaw), torch.cos(current_yaw - target_yaw)
        )
    )

    error_exceeded = (pos_error > pos_threshold) | (yaw_error > yaw_threshold)
    phase_exceeded = current_phase >= start_phase

    return phase_exceeded & error_exceeded


# =============================================================================
# CONFIGURATIONS
# =============================================================================


@configclass
class G1JumpSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )
    robot: ArticulationCfg = G1_23DOF_HOLO_COMPAT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.78),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={r".*_joint": 0.0},
        ),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*",
        history_length=3,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=None,  # To be set in __post_init__ to avoid circular imports
    )

    def __post_init__(self):
        self.sky_light.spawn = sim_utils.DomeLightCfg(intensity=750.0)
        self.robot.spawn.articulation_props.enabled_self_collisions = True


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


@configclass
class G1JumpCommandCfg:
    jump_goal: JumpGoalCommandCfg = JumpGoalCommandCfg()


@configclass
class G1JumpActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class G1JumpObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
        )
        root_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        root_vel = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos"},
        )
        reference_preview = ObsTerm(
            func=obs_future_reference_preview,
        )
        jump_phase = ObsTerm(
            func=obs_jump_phase,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1JumpEventCfg:
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )
    reset_to_reference = EventTerm(
        func=reference_state_initialization,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "init_start_prob": 0.2,  # 20% of RSI
        },
    )


@configclass
class G1JumpRewardsCfg:
    # Reference Motion Tracking
    track_joint_pos = RewTerm(
        func=track_joint_pos,
        weight=2.0,
        params={
            "alpha": 0.1745,
            "Tj": REFERENCE_DURATION_S,
            "w_before": 15.0,
            "w_after": 15.0,
            "phase_weights": (15.0, 15.0, 15.0, 10.0, 12.0, 15.0),
        },
    )
    track_pelvis_height = RewTerm(
        func=track_pelvis_height,
        weight=1.0,
        params={
            "alpha": 65.85,
            "Tj": REFERENCE_DURATION_S,
            "w_before": 5.0,
            "w_after": 5.0,
            "phase_weights": (5.0, 8.0, 8.0, 8.0, 5.0, 5.0),
        },
    )
    track_foot_height = RewTerm(
        func=track_foot_height,
        weight=1.0,
        params={
            "alpha": 131.7,
            "Tj": REFERENCE_DURATION_S,
            "w_before": 10.0,
            "w_after": 10.0,
            "phase_weights": (10.0, 10.0, 10.0, 12.0, 10.0, 10.0),
        },
    )
    # Task Completion
    target_pelvis_position = RewTerm(
        func=target_pelvis_position,
        weight=1.0,
        params={
            "alpha": 21.07,
            "Tj": REFERENCE_DURATION_S,
            "w_before": 12.5,
            "w_after": 12.5,
            "phase_weights": (3.0, 5.0, 8.0, 12.5, 12.5, 12.5),
        },
    )
    target_pelvis_velocity = RewTerm(
        func=target_pelvis_velocity,
        weight=1.0,
        params={
            "alpha": 1.317,
            "Tj": JUMP_LANDING_END_S,
            "w_before": 0.0,
            "w_after": 3.0,
            "phase_weights": (0.0, 0.0, 3.0, 3.0, 1.0, 0.0),
        },
    )
    target_orientation = RewTerm(
        func=target_orientation,
        weight=1.0,
        params={
            "alpha": 2.634,
            "Tj": REFERENCE_DURATION_S,
            "w_before": 0.0,
            "w_after": 12.5,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 6.0, 12.5),
        },
    )
    target_angular_rate = RewTerm(
        func=target_angular_rate,
        weight=1.0,
        params={
            "alpha": 0.14,
            "Tj": JUMP_LANDING_END_S,
            "w_before": 0.0,
            "w_after": 3.0,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 3.0, 6.0),
        },
    )
    # Smoothing
    penalize_ground_impact = RewTerm(
        func=penalize_ground_impact,
        weight=1.0,
        params={
            "alpha": 4.7e-8,
            "Tj": REFERENCE_DURATION_S,
            "w_before": 5.0,
            "w_after": 0.0,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 5.0, 0.0),
        },
    )
    penalize_torque_consumption = RewTerm(
        func=penalize_torque_consumption,
        weight=1.0,
        params={
            "alpha": 6.9e-7,
            "Tj": REFERENCE_DURATION_S,
            "w_before": 3.0,
            "w_after": 3.0,
            "phase_weights": (1.0, 1.0, 0.5, 1.0, 3.0, 5.0),
        },
    )
    penalize_motor_velocity = RewTerm(
        func=penalize_motor_velocity,
        weight=1.0,
        params={
            "alpha": 1.1e-4,
            "Tj": REFERENCE_DURATION_S,
            "w_before": 0.0,
            "w_after": 15.0,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 5.0, 15.0),
        },
    )
    penalize_joint_acceleration = RewTerm(
        func=penalize_joint_acceleration,
        weight=1.0,
        params={
            "alpha": 2.8e-7,
            "Tj": REFERENCE_DURATION_S,
            "w_before": 3.0,
            "w_after": 10.0,
            "phase_weights": (1.0, 1.0, 0.0, 0.0, 5.0, 10.0),
        },
    )
    # Termination
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1JumpTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=r"^(?!.*_ankle_roll_link$).*",
            ),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": 1.8,
        },
    )
    foot_tracking_error = DoneTerm(
        func=foot_tracking_error,
        params={
            "threshold": 0.22,
            "active_phases": (
                JUMP_PHASE_STAND_PREP,
                JUMP_PHASE_CROUCH,
                JUMP_PHASE_TAKEOFF,
                JUMP_PHASE_FLIGHT_PEAK,
                JUMP_PHASE_LANDING,
            ),
        },
    )
    task_completion_error = DoneTerm(
        func=task_completion_error,
        params={
            "pos_threshold": 1.0,
            "yaw_threshold": 45.0 * (torch.pi / 180.0),
            "start_phase": JUMP_PHASE_FINAL_STAND,
        },
    )


@configclass
class G1JumpEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1JumpSceneCfg = G1JumpSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: G1JumpCommandCfg = G1JumpCommandCfg()
    actions: G1JumpActionsCfg = G1JumpActionsCfg()
    observations: G1JumpObservationsCfg = G1JumpObservationsCfg()
    events: G1JumpEventCfg = G1JumpEventCfg()
    rewards: G1JumpRewardsCfg = G1JumpRewardsCfg()
    terminations: G1JumpTerminationsCfg = G1JumpTerminationsCfg()

    def __post_init__(self):
        self.decimation = 10
        self.sim.dt = 0.002
        self.episode_length_s = JUMP_EPISODE_LENGTH_S
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1JumpEnvCfg_PLAY(G1JumpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = JUMP_EPISODE_LENGTH_S

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.scene.terrain.max_init_terrain_level = None
        self.commands.jump_goal.ranges.pos_x = (0.0, 0.0)
        self.commands.jump_goal.ranges.pos_y = (0.0, 0.0)
        self.commands.jump_goal.ranges.yaw = (0.0, 0.0)
        self.observations.policy.enable_corruption = True

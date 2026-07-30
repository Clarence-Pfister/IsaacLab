# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the Unitree G1 reference-motion jump task."""

from __future__ import annotations

import torch
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from .constants import (
    CONTACT_SENSOR_NAMES,
    CONTACT_SENSOR_PRIM_PATHS,
    G1_23DOF_HOLO_COMPAT_CFG,
    JOINT_ACTION_SCALES,
    JOINT_NAMES,
    NON_FOOT_CONTACT_SENSOR_NAMES,
    PLAY_JOINT_ACTION_FILTER_ALPHA,
    REFERENCE_DURATION_S,
)
from .mdp import (
    JumpGoalCommandCfg,
    LowPassJointPositionActionCfg,
    foot_tracking_error,
    ground_contact,
    obs_future_reference_preview,
    obs_jump_phase,
    penalize_ground_impact,
    penalize_joint_acc,
    penalize_joint_vel,
    penalize_torque_consumption,
    reference_motion_complete,
    reference_state_initialization,
    set_contact_sensor,
    target_angular_rate,
    target_orientation,
    target_position,
    target_velocity,
    task_completion_error,
    track_foot_xy,
    track_foot_z,
    track_joint_pos,
    track_joint_vel,
    track_root_angular_rate,
    track_root_orientation,
    track_root_pos_z,
    track_root_vel_z,
)
from .mdp.motion import get_reference_initial_state


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
            # Populated from the reference motion in G1JumpEnvCfg.__post_init__.
            joint_pos={},
        ),
    )
    contact_forces_left_foot = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["left_foot"])
    contact_forces_right_foot = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["right_foot"])
    contact_forces_pelvis = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["pelvis"])
    contact_forces_left_thigh = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["left_thigh"])
    contact_forces_left_shin = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["left_shin"])
    contact_forces_right_thigh = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["right_thigh"])
    contact_forces_right_shin = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["right_shin"])
    contact_forces_torso = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["torso"])
    contact_forces_left_upper_arm = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["left_upper_arm"])
    contact_forces_left_lower_arm = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["left_lower_arm"])
    contact_forces_left_hand = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["left_hand"])
    contact_forces_right_upper_arm = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["right_upper_arm"])
    contact_forces_right_lower_arm = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["right_lower_arm"])
    contact_forces_right_hand = set_contact_sensor(CONTACT_SENSOR_PRIM_PATHS["right_hand"])
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=None,  # To be set in __post_init__ to avoid circular imports
    )

    def __post_init__(self):
        self.sky_light.spawn = sim_utils.DomeLightCfg(intensity=750.0)
        self.robot.spawn.articulation_props.enabled_self_collisions = False


@configclass
class G1JumpCommandCfg:
    jump_goal: JumpGoalCommandCfg = JumpGoalCommandCfg()


@configclass
class G1JumpActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=JOINT_ACTION_SCALES,
        use_default_offset=True,
    )


@configclass
class G1JumpPlayActionsCfg:
    joint_pos = LowPassJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=JOINT_ACTION_SCALES,
        use_default_offset=True,
        alpha=PLAY_JOINT_ACTION_FILTER_ALPHA,
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
        root_quat_w = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        root_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
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
        weight=1.0,
        params={
            "gradient": 0.15,
            "phase_weights": (12.0, 16.0, 18.0, 12.0, 14.0, 16.0),
        },
    )
    track_joint_vel = RewTerm(
        func=track_joint_vel,
        weight=1.0,
        params={
            "gradient": 0.0046,
            "phase_weights": (0.0, 1.0, 2.0, 2.0, 1.0, 0.0),
        },
    )
    track_root_pos_z = RewTerm(
        func=track_root_pos_z,
        weight=1.0,
        params={
            "gradient": 65.85,
            "phase_weights": (4.0, 8.0, 12.0, 14.0, 10.0, 6.0),
        },
    )
    track_root_vel_z = RewTerm(
        func=track_root_vel_z,
        weight=1.0,
        params={
            "gradient": 2.634,
            "phase_weights": (0.0, 6.0, 14.0, 10.0, 6.0, 0.0),
        },
    )
    track_root_orientation = RewTerm(
        func=track_root_orientation,
        weight=1.0,
        params={
            "gradient": 6.18,
            "phase_weights": (4.0, 6.0, 4.0, 2.0, 8.0, 10.0),
        },
    )
    track_root_angular_rate = RewTerm(
        func=track_root_angular_rate,
        weight=1.0,
        params={
            "gradient": 0.14,
            "phase_weights": (0.0, 2.0, 4.0, 2.0, 4.0, 2.0),
        },
    )
    track_foot_z = RewTerm(
        func=track_foot_z,
        weight=1.0,
        params={
            "gradient": 58.53,
            "phase_weights": (10.0, 12.0, 14.0, 16.0, 14.0, 10.0),
        },
    )
    track_foot_xy = RewTerm(
        func=track_foot_xy,
        weight=1.0,
        params={
            "gradient": 30.0,
            "phase_weights": (8.0, 12.0, 14.0, 6.0, 14.0, 12.0),
        },
    )
    # Task Completion
    target_position = RewTerm(
        func=target_position,
        weight=1.0,
        params={
            "gradient": 21.07,
            "phase_weights": (0.0, 1.0, 2.0, 4.0, 8.0, 12.0),
        },
    )
    target_velocity = RewTerm(
        func=target_velocity,
        weight=1.0,
        params={
            "gradient": 1.317,
            "phase_weights": (0.0, 0.0, 3.0, 3.0, 1.0, 0.0),
        },
    )
    target_orientation = RewTerm(
        func=target_orientation,
        weight=1.0,
        params={
            "gradient": 13.87,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
    )
    target_angular_rate = RewTerm(
        func=target_angular_rate,
        weight=1.0,
        params={
            "gradient": 0.14,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
    )
    # Smoothing
    # The smoothness gradients below are calibrated so that each kernel sits near 0.55 at
    # the error observed WITHIN the phases where that term carries non-zero weight. The
    # previous values left every one of them pinned near 1.0 in its own active phase
    # (0.986, 0.932, 0.988, 0.858 measured), where exp(-k*e) has slope -k*exp(-k*e) and so
    # supplies almost no gradient: the agent collected a near-full bonus regardless of how
    # hard it landed or how fast it moved.
    penalize_ground_impact = RewTerm(
        func=penalize_ground_impact,
        weight=1.0,
        params={
            "gradient": 2.0e-6,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 8.0, 2.0),
        },
    )
    penalize_torque_consumption = RewTerm(
        func=penalize_torque_consumption,
        weight=1.0,
        params={
            "gradient": 9.3e-6,
            "phase_weights": (1.0, 1.0, 0.25, 0.5, 2.0, 4.0),
        },
    )
    penalize_joint_vel = RewTerm(
        func=penalize_joint_vel,
        weight=1.0,
        params={
            "gradient": 6.3e-3,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 4.0, 12.0),
        },
    )
    penalize_joint_acc = RewTerm(
        func=penalize_joint_acc,
        weight=1.0,
        params={
            "gradient": 1.1e-6,
            "phase_weights": (0.5, 0.5, 0.0, 0.0, 4.0, 8.0),
        },
    )
    # Termination
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)


@configclass
class G1JumpTerminationsCfg:
    time_out = DoneTerm(func=reference_motion_complete, time_out=True)
    base_contact = DoneTerm(
        func=ground_contact,
        params={
            "threshold": 1.0,
            "sensor_names": NON_FOOT_CONTACT_SENSOR_NAMES,
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
            "threshold": 0.50,
            "active_phases": ("IDLE", "CROUCH", "TAKEOFF", "FLIGHT", "LAND"),
        },
    )
    task_completion_error = DoneTerm(
        func=task_completion_error,
        params={
            "pos_threshold": 1.0,
            "yaw_threshold": 45.0 * (torch.pi / 180.0),
            "start_phase": "STAND",
        },
    )


@configclass
class G1JumpEnvCfg(ManagerBasedRLEnvCfg):
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        physics=PhysxCfg(enable_external_forces_every_iteration=True)
    )
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
        self.episode_length_s = REFERENCE_DURATION_S
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        for sensor_name in CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        # Keep default_joint_pos, non-RSI resets, and the default action offset aligned
        # with reference frame 0.
        init_joint_pos, init_root_z = get_reference_initial_state()
        self.scene.robot.init_state.joint_pos = init_joint_pos
        self.scene.robot.init_state.pos = (0.0, 0.0, init_root_z)


@configclass
class G1JumpEnvCfg_PLAY(G1JumpEnvCfg):
    actions: G1JumpPlayActionsCfg = G1JumpPlayActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = REFERENCE_DURATION_S

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.scene.terrain.max_init_terrain_level = None
        self.commands.jump_goal.ranges.pos_x = (0.0, 0.0)
        self.commands.jump_goal.ranges.pos_y = (0.0, 0.0)
        self.commands.jump_goal.ranges.yaw = (0.0, 0.0)
        self.observations.policy.enable_corruption = True

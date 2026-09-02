# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the Unitree G1 reference-motion jump task."""

from __future__ import annotations

import math

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
from isaaclab.utils.noise import UniformNoiseCfg

from .constants import (
    CONTACT_SENSOR_NAMES,
    CONTACT_SENSOR_PRIM_PATHS,
    CSV_MOTION_PATH,
    CSV_MOTION_PATH_EXTENDED,
    G1_23DOF_HOLO_COMPAT_CFG,
    JOINT_ACTION_SCALES,
    JOINT_NAMES,
    JOINT_POSITION_LIMITS,
    JUMP_PHASES,
    JUMP_PHASES_EXTENDED,
    NON_FOOT_CONTACT_SENSOR_NAMES,
    PLAY_JOINT_ACTION_FILTER_ALPHA,
    REFERENCE_DURATION_S,
    REFERENCE_DURATION_S_EXTENDED,
    REFERENCE_MOTION_FPS,
    REFERENCE_NUM_FRAMES,
    REFERENCE_NUM_FRAMES_EXTENDED,
)
from .mdp import (
    JumpGoalCommandCfg,
    LowPassJointPositionActionCfg,
    foot_tracking_error,
    ground_contact,
    joint_position_limit_margin,
    joint_target_lower_limit,
    joint_torque_demand_limit,
    obs_future_reference_preview,
    obs_goal_command,
    obs_goal_command_remaining_orientation,
    obs_goal_command_remaining_orientation_retrigger,
    obs_goal_command_remaining_orientation_retrigger_goal,
    obs_goal_remaining,
    obs_goal_remaining_latched,
    obs_jump_phase,
    obs_projected_gravity,
    penalize_ground_impact,
    penalize_joint_acc,
    penalize_joint_vel,
    penalize_torque_consumption,
    perturb_trigger_state,
    randomize_contact_compliance,
    reference_joint_target_deviation,
    reference_motion_complete,
    reference_or_terminal_state_initialization,
    reference_state_initialization,
    set_contact_sensor,
    target_angular_rate,
    target_heading,
    target_orientation,
    target_position,
    target_position_error,
    target_velocity,
    target_velocity_error,
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
from .mdp.motion import get_reference_initial_pose

# Training and deployment use the same physical target envelope. A target beyond a joint
# stop does not add reachable motion; it only keeps the implicit PD controller saturated
# after the joint reaches the stop, which was the main non-transferable behavior in the
# previous policy.
_JOINT_ACTION_CLIP = JOINT_POSITION_LIMITS


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
        clip=_JOINT_ACTION_CLIP,
    )


@configclass
class G1JumpPlayActionsCfg:
    joint_pos = LowPassJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=JOINT_ACTION_SCALES,
        use_default_offset=True,
        clip=_JOINT_ACTION_CLIP,
        alpha=PLAY_JOINT_ACTION_FILTER_ALPHA,
    )


@configclass
class G1JumpDeployActionsCfg:
    """Action pipeline used for the sim-to-real stage and intended for the robot.

    Training and deployment have to drive the same pipeline, or the policy is tuned against
    an actuation path it will never see. Stage 3 therefore trains through the same low-pass
    filter and command delay that the deployed controller runs, rather than the unfiltered,
    zero-latency path the earlier stages use. This is deliberately separate from
    :class:`G1JumpPlayActionsCfg`, whose filter exists to make playback legible: retuning a
    visualization setting must not silently change what Stage 3 trains against.
    """

    joint_pos = LowPassJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=JOINT_ACTION_SCALES,
        use_default_offset=True,
        clip=_JOINT_ACTION_CLIP,
        alpha=PLAY_JOINT_ACTION_FILTER_ALPHA,
    )


@configclass
class G1JumpObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
            history_length=4,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
            history_length=4,
        )
        # Every term in this group has to be assemblable on a real G1, which rules out both a
        # world frame and base linear velocity. The environment-frame position, world-frame
        # velocities and absolute yaw that the root terms replace are ground truth the robot
        # cannot measure: IMU yaw drifts with no absolute reference, and the only base-velocity
        # source is a contact-based leg-odometry estimate that stops updating the moment both
        # feet leave the ground. This task spends roughly a third of a second airborne, so that
        # estimate is unavailable exactly when the landing is being committed. Unitree's own
        # deployment stack, NVIDIA's ProtoMotions and LeCAR-Lab's ASAP all reach the same
        # conclusion and keep base linear velocity out of the deployed actor, so it lives in the
        # critic group below, where being sim-only costs nothing.
        #
        # The history on the terms that carry it is what pays for that removal: differencing
        # goal_remaining across four steps recovers velocity relative to the goal, which is the
        # quantity base linear velocity was really supplying. Noise is applied per step before
        # it enters the history buffer, so a corrupted history stays a realistic one.
        goal_remaining = ObsTerm(
            func=obs_goal_remaining,
            history_length=4,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            history_length=4,
        )
        projected_gravity = ObsTerm(
            func=obs_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            history_length=4,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos"},
        )
        goal_command = ObsTerm(
            func=obs_goal_command,
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
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    critic: CriticCfg = CriticCfg()


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
    target_position_error = RewTerm(
        func=target_position_error,
        weight=0.0,
        params={
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "retrigger_only": False,
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
    target_velocity_error = RewTerm(
        func=target_velocity_error,
        weight=0.0,
        params={"phase_weights": (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)},
    )
    target_orientation = RewTerm(
        func=target_orientation,
        weight=1.0,
        params={
            "gradient": 13.87,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
    )
    target_heading = RewTerm(
        func=target_heading,
        weight=1.0,
        params={
            "gradient": 30.0,
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
    joint_torque_demand_limit = RewTerm(
        func=joint_torque_demand_limit,
        weight=-2.0,
        params={
            "soft_ratio": 0.9,
            "maximum_excess": 2.0,
            # Takeoff and flight may briefly need the full rated effort. Persistent
            # saturation while standing or absorbing landing is never acceptable.
            "phase_weights": (1.0, 1.0, 0.25, 0.25, 1.0, 1.0),
        },
    )
    knee_target_lower_limit = RewTerm(
        func=joint_target_lower_limit,
        weight=0.0,
        params={
            "lower_limit": 0.1,
            "normalization": 0.1,
            "phase_weights": (0.0, 0.0, 0.0, 0.0, 1.0, 2.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
        },
    )
    ankle_roll_position_limit_margin = RewTerm(
        func=joint_position_limit_margin,
        weight=0.0,
        params={
            "margin": 0.01,
            "phase_weights": (1.0, 2.0, 4.0, 4.0, 8.0, 10.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"]),
        },
    )
    reference_joint_target_deviation = RewTerm(
        func=reference_joint_target_deviation,
        weight=-10.0,
        params={
            # The start and settled phases should command targets near the reference.
            # Crouch and landing retain room for command-dependent balance corrections,
            # while takeoff and flight can use larger residuals to reach the goal.
            "phase_weights": (4.0, 1.0, 0.25, 0.25, 1.0, 2.0),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    penalize_joint_vel = RewTerm(
        func=penalize_joint_vel,
        weight=1.0,
        params={
            # Eased from 6.3e-3: at that value the term lost 81% of its return over the
            # first training run, penalising the joint speed the policy still needs to
            # reach the landing pose before it has learned to land at all.
            "gradient": 3.0e-3,
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
    reference_motion_path: str = CSV_MOTION_PATH
    reference_num_frames: int = REFERENCE_NUM_FRAMES
    reference_motion_fps: float = REFERENCE_MOTION_FPS
    reference_duration_s: float = REFERENCE_DURATION_S
    jump_phases: dict[str, tuple[int, int]] = JUMP_PHASES
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
        self.episode_length_s = self.reference_duration_s
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        for sensor_name in CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        # Keep default_joint_pos, non-RSI resets, and the default action offset aligned
        # with reference frame 0.
        init_joint_pos, init_root_pos, init_root_quat = get_reference_initial_pose(self.reference_motion_path)
        self.scene.robot.init_state.joint_pos = init_joint_pos
        self.scene.robot.init_state.pos = (0.0, 0.0, init_root_pos[2])
        self.scene.robot.init_state.rot = init_root_quat


@configclass
class G1JumpStage1DeployEnvCfg(G1JumpEnvCfg):
    """In-place jump imitation through the deployment action filter.

    This stage keeps the fixed Stage 1 goal while introducing the target filter used by the
    robot. Its strong reference-target prior prevents the actor from replacing the recorded
    motion with targets that merely drive the implicit PD controller into effort clipping.
    """

    actions: G1JumpDeployActionsCfg = G1JumpDeployActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.rewards.reference_joint_target_deviation.weight = -50.0
        self.rewards.action_rate.weight = -0.5


@configclass
class G1JumpStage2EnvCfg(G1JumpEnvCfg):
    """Multi-goal jump training.

    Stage 1 fixes the goal at the origin with no turn, so the policy only has to imitate the
    reference. Here the goal is resampled per episode and the policy has to reach it, which
    changes what the reference is good for: it was recorded as an in-place jump, so the terms
    that track its heading and its foot ground track now describe a motion the robot is being
    asked *not* to perform. Those are dropped in favour of the task terms, following the
    weight shift in Table II of Li et al. (2023).

    Dynamics randomization belongs to stage 3 and is deliberately absent here.
    """

    def __post_init__(self):
        super().__post_init__()

        # Flat-ground policy: position and turning vary, elevation does not. The paper trains
        # elevation as a separate policy, and _query_terrain_height only supports flat ground.
        # These are about a third of the paper's ranges (it uses +/-1.5 m and +/-100 deg on a
        # robot it had already trained through this stage); widen once the policy holds up.
        self.commands.jump_goal.ranges.pos_x = (-0.4, 0.4)
        self.commands.jump_goal.ranges.pos_y = (-0.3, 0.3)
        self.commands.jump_goal.ranges.yaw = (-30.0 * torch.pi / 180.0, 30.0 * torch.pi / 180.0)

        # Introduce the deployable observation contract while the goal envelope is
        # still narrow. The physical G1 supplies no root position in LowState, so
        # the actor receives the trigger-time displacement throughout the jump;
        # the asymmetric critic retains true remaining displacement in simulation.
        self.observations.policy.goal_remaining.func = obs_goal_remaining_latched
        self.observations.policy.goal_remaining.params = {}
        self.observations.critic.goal_remaining.func = obs_goal_remaining
        self.observations.critic.goal_remaining.params = {}

        # The reference motion cannot describe where the robot was told to go, so the terms
        # that would hold it on the reference's heading and ground track are switched off.
        # Their task-space counterparts below take over.
        self.rewards.track_root_orientation.params["phase_weights"] = (0.0,) * 6
        self.rewards.track_root_angular_rate.params["phase_weights"] = (0.0,) * 6
        self.rewards.track_foot_xy.params["phase_weights"] = (0.0,) * 6

        # Halve joint-position tracking before landing so the policy can deviate from the
        # in-place posture to travel, while keeping it after landing where the reference
        # still describes the pose we want (Table II: 15 -> 7.5 before, 15 after).
        self.rewards.track_joint_pos.params["phase_weights"] = (6.0, 8.0, 9.0, 6.0, 14.0, 16.0)

        # Heading is now a task, not an imitation target. Orientation is weighted towards
        # landing and standing where the commanded turn must actually hold; angular rate is
        # weighted towards take-off and flight, which is when the turn is executed.
        #
        # The base-task kernels were calibrated against the much larger angular rates in the
        # reference motion. At this stage, ignoring the largest commanded turn still retained
        # 39.5% of the orientation score and 98.8% of the angular-rate score. Recalibrate both
        # so turning supplies a meaningful gradient instead of an almost constant bonus.
        self.rewards.target_orientation.params["gradient"] = 30.0
        self.rewards.target_orientation.params["phase_weights"] = (0.0, 1.0, 2.0, 3.0, 6.0, 8.0)
        self.rewards.target_angular_rate.params["gradient"] = 7.0
        self.rewards.target_angular_rate.params["phase_weights"] = (0.0, 2.0, 4.0, 3.0, 2.0, 0.0)


@configclass
class G1JumpStage2DeployEnvCfg(G1JumpStage2EnvCfg):
    """Narrow command training through the deployment action filter.

    The reference-target prior is relaxed from Stage 1 so the policy can create the joint
    asymmetries needed for translation and turning while retaining smooth, physical targets.
    """

    actions: G1JumpDeployActionsCfg = G1JumpDeployActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.rewards.reference_joint_target_deviation.weight = -20.0
        self.rewards.action_rate.weight = -0.1


@configclass
class G1JumpStage2DeployTranslationEnvCfg(G1JumpStage2DeployEnvCfg):
    """Translation-first deployment curriculum with relative attitude feedback.

    This stage narrows translation and holds the requested turn at zero while the policy
    learns to remove the reference motion's repeatable heading bias. The actor receives the
    remaining target orientation from the simulated IMU using the same calculation available
    from G1 ``LowState`` during deployment. Turning is introduced only after this stage can
    land accurately without accumulating uncommanded yaw.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.ranges.pos_x = (-0.2, 0.2)
        self.commands.jump_goal.ranges.pos_y = (-0.15, 0.15)
        self.commands.jump_goal.ranges.yaw = (0.0, 0.0)
        self.observations.policy.goal_command.func = obs_goal_command_remaining_orientation
        self.observations.critic.goal_command.func = obs_goal_command_remaining_orientation
        self.rewards.target_heading.params["gradient"] = 30.0
        self.rewards.target_heading.params["phase_weights"] = (0.0, 0.0, 0.0, 0.0, 8.0, 12.0)
        # Keep instantaneous implicit-PD demand below the standard G1's advertised
        # 90 N.m knee maximum: 60% of the model's 139 N.m envelope is 83.4 N.m.
        self.actions.joint_pos.effort_limit_ratio = 0.6


@configclass
class G1JumpStage2DeployLongitudinalEnvCfg(G1JumpStage2DeployTranslationEnvCfg):
    """Signed forward/backward deployment curriculum used by the first commandable policy.

    Lateral displacement and heading are deliberately disabled until their command-response
    gains pass the same deployment evaluation as the longitudinal axis. The actor and critic
    goal features use the scale baked into the selected checkpoint's training contract.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.ranges.pos_y = (0.0, 0.0)
        self.commands.jump_goal.zero_goal_probability = 0.25
        self.commands.jump_goal.boundary_goal_probability = 0.5
        self.observations.policy.goal_remaining.scale = 4.0
        self.observations.critic.goal_remaining.scale = 4.0
        self.observations.policy.goal_command.scale = 4.0
        self.observations.critic.goal_command.scale = 4.0

        self.rewards.target_position.weight = 8.0
        self.rewards.target_velocity.weight = 0.0
        self.rewards.target_velocity_error.weight = -50.0
        self.rewards.target_heading.weight = 3.0
        self.rewards.reference_joint_target_deviation.weight = -5.0


@configclass
class G1JumpStage2DeployLongitudinalUniformEnvCfg(G1JumpStage2DeployLongitudinalEnvCfg):
    """Uniform-dominant curriculum for a smooth longitudinal command response.

    The endpoint-heavy curriculum is useful for establishing signed authority, but it
    permits the actor to learn disconnected endpoint behaviors. This stage retains a
    small number of exact zero and boundary commands while drawing most goals uniformly
    from the interior of the trained range.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.zero_goal_probability = 0.1
        self.commands.jump_goal.boundary_goal_probability = 0.1


@configclass
class G1JumpStage2DeployLongitudinalSmoothEnvCfg(G1JumpStage2DeployLongitudinalUniformEnvCfg):
    """Latched-feedback curriculum with bandwidth-limited leg position targets.

    This retains the trigger-latched horizontal command available from G1
    :class:`LowState` while limiting how quickly policy-state differences can become
    distinct leg contact modes.
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.alpha = {
            ".*_hip_.*": 0.3,
            ".*_knee_joint": 0.3,
            ".*_ankle_.*": 0.3,
        }


@configclass
class G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg(G1JumpStage2DeployLongitudinalSmoothEnvCfg):
    """Deployment task restricted to the validated longitudinal command range.

    The selected policy remains upright over its wider training range, but MuJoCo
    validation identifies nonlinear forward overshoot outside this narrower envelope.
    Exporting through this task records and enforces only the validated range.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.ranges.pos_x = (-0.1, 0.1)
        action_clip = dict(self.actions.joint_pos.clip)
        for joint_name in ("left_knee_joint", "right_knee_joint"):
            _, upper = action_clip[joint_name]
            action_clip[joint_name] = (0.1, upper)
        self.actions.joint_pos.clip = action_clip
        self.actions.joint_pos.lower_limit_velocity_lookahead = {".*_knee_joint": 0.028}


@configclass
class G1JumpStage2DeployLongitudinalSmoothNarrowExtendedEnvCfg(G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg):
    """Narrow deployment task fine-tuned on the stance-extended jump reference."""

    reference_motion_path: str = CSV_MOTION_PATH_EXTENDED
    reference_num_frames: int = REFERENCE_NUM_FRAMES_EXTENDED
    reference_duration_s: float = REFERENCE_DURATION_S_EXTENDED
    jump_phases: dict[str, tuple[int, int]] = JUMP_PHASES_EXTENDED


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeEnvCfg(G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg):
    """Parameterized longitudinal goal-range widening stage.

    Re-scaling the Narrow position kernel from 21.07 to 5.2675 for Range020 made a
    0.2 m miss score 0.81 instead of 0.43 and allowed command response to collapse.
    The settled-displacement x gain changed from 0.70 at model 825 (offset +0.011
    [m], correlation 0.97) to 0.944, 0.935, 0.695, 0.558, 0.324, and 0.340 at
    models 900, 950, 1000, 1050, 1100, and 1124 respectively; model 1124's
    correlation fell to 0.78. Every range therefore retains the Narrow kernel.
    Longer stages still adopt the WideLand touchdown emphasis once their range
    reaches 0.6 m.
    """

    goal_pos_x_range: tuple[float, float] = (-0.1, 0.1)
    """Longitudinal command range [m]."""

    def __post_init__(self):
        super().__post_init__()

        narrow_pos_threshold = self.terminations.task_completion_error.params["pos_threshold"]
        range_min, range_max = self.goal_pos_x_range
        stage_half_range = max(abs(range_min), abs(range_max))

        self.commands.jump_goal.ranges.pos_x = self.goal_pos_x_range
        stage_pos_threshold = min(max(0.35 * stage_half_range / 0.65, 0.20), 0.35)
        self.terminations.task_completion_error.params["pos_threshold"] = min(narrow_pos_threshold, stage_pos_threshold)

        if stage_half_range >= 0.6:
            self.rewards.target_position.params["phase_weights"] = (0.0, 1.0, 2.0, 4.0, 12.0, 10.0)
            self.rewards.target_velocity.params["phase_weights"] = (0.0, 0.0, 3.0, 3.0, 6.0, 2.0)
            self.rewards.track_root_pos_z.params["phase_weights"] = (4.0, 8.0, 12.0, 8.0, 6.0, 6.0)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRange020EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeEnvCfg):
    """Deployment curriculum with uniformly sampled longitudinal goals [m] in [-0.2, 0.2]."""

    goal_pos_x_range: tuple[float, float] = (-0.2, 0.2)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRange040EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeEnvCfg):
    """Deployment curriculum with uniformly sampled longitudinal goals [m] in [-0.4, 0.4]."""

    goal_pos_x_range: tuple[float, float] = (-0.4, 0.4)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRange060EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeEnvCfg):
    """Deployment curriculum with uniformly sampled longitudinal goals [m] in [-0.6, 0.6]."""

    goal_pos_x_range: tuple[float, float] = (-0.6, 0.6)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRange080EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeEnvCfg):
    """Deployment curriculum with uniformly sampled longitudinal goals [m] in [-0.8, 0.8]."""

    goal_pos_x_range: tuple[float, float] = (-0.8, 0.8)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRange100EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeEnvCfg):
    """Deployment curriculum with uniformly sampled longitudinal goals [m] in [-1.0, 1.0]."""

    goal_pos_x_range: tuple[float, float] = (-1.0, 1.0)


@configclass
class G1JumpStage2DeployLongitudinalSmoothNarrowRepeatEnvCfg(G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg):
    """Latched deployment contract for policies trained to repeat narrow jumps."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.lower_limit_velocity_lookahead = {".*_knee_joint": 0.032}


@configclass
class G1JumpStage2DeployLongitudinalLatchedSmoothNarrowDampedEnvCfg(
    G1JumpStage2DeployLongitudinalSmoothNarrowRepeatEnvCfg
):
    """Deployment contract with additional ankle-roll landing damping."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.actuators["feet"].damping = {
            ".*_ankle_pitch_joint": 2.0,
            ".*_ankle_roll_joint": 4.0,
        }


@configclass
class G1JumpStage2DeployLongitudinalSmoothNarrowHandoffEnvCfg(G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg):
    """Fine-tuning task for a gravity-loaded stand-to-jump handoff.

    The normal jump reset starts at the exact first reference frame and applies the
    policy immediately. A deployment FSM instead holds the robot under gravity before
    confirmation, so the trigger state contains small joint deflections and residual
    velocities. This task retains the narrow signed command and deployment action
    contracts while perturbing the physical reset state around that handoff envelope.
    """

    @configclass
    class EventCfg(G1JumpEventCfg):
        reset_to_reference = EventTerm(
            func=reference_state_initialization,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "init_start_prob": 0.0,
                "roll_range": (-0.035, 0.035),
                "pitch_range": (-0.035, 0.035),
                "lin_vel_range": (-0.05, 0.05),
            },
        )
        reset_handoff_leg_state = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_hip_.*", ".*_knee_joint"],
                ),
                "position_range": (-0.03, 0.03),
                "velocity_range": (-0.15, 0.15),
            },
        )
        reset_handoff_ankle_state = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*"]),
                "position_range": (-0.08, 0.08),
                "velocity_range": (-0.2, 0.2),
            },
        )
        reset_handoff_upper_state = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["waist_.*", ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
                ),
                "position_range": (-0.02, 0.02),
                "velocity_range": (-0.1, 0.1),
            },
        )

    events: EventCfg = EventCfg()


@configclass
class G1JumpStage2DeployLongitudinalOdometrySmoothNarrowRepeatEnvCfg(
    G1JumpStage2DeployLongitudinalSmoothNarrowRepeatEnvCfg
):
    """Fine-tuning task that chains safe policy landings into new jump commands.

    Eligible time-limit resets retain the robot's terminal pose, translate it back to
    its environment origin, reset its velocity and motion phase to zero, and sample a
    new goal. Failed, excessively tilted, or joint-limit-violating states use the normal
    frame-zero reset instead. Fine-tuning retains the selected checkpoint's live
    remaining-displacement observation; export and deployment still substitute the
    previously validated latched signal with the same shape and scale.
    """

    @configclass
    class EventCfg(G1JumpEventCfg):
        reset_to_reference = EventTerm(
            func=reference_or_terminal_state_initialization,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "retrigger_probability": 1.0,
                "init_start_prob": 0.0,
                "root_height_range": (0.65, 0.9),
                "max_tilt_rad": 0.15,
                "max_root_linear_speed": 0.5,
                "max_root_angular_speed": 2.0,
                "max_joint_speed": 5.0,
                "joint_limit_margin": 0.0,
                "joint_limit_tolerance": 0.001,
                "zero_retrigger_velocity": True,
            },
        )

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.goal_remaining.func = obs_goal_remaining
        self.observations.policy.goal_remaining.params = {}
        self.rewards.ankle_roll_position_limit_margin.weight = -20.0


@configclass
class G1JumpStage2DeployLongitudinalLatchedSmoothNarrowRepeatEnvCfg(
    G1JumpStage2DeployLongitudinalOdometrySmoothNarrowRepeatEnvCfg
):
    """Repeat curriculum matching the latched, stand-to-phase-zero FSM handoff.

    Each complete episode holds the policy's final STAND reference for one second.
    Eligible terminal poses then start a new relative goal at rest and spend 0.26 s
    at phase zero, matching the 14 phase-zero evaluations made by the 50 Hz FSM's
    0.25 s preparation plus its first jump tick. Exact zero and boundary commands
    remain frequent while interior goals preserve a continuous command response.
    """

    def __post_init__(self):
        super().__post_init__()

        terminal_hold_duration_s = 1.0
        retrigger_prepare_duration_s = 0.26
        self.events.reset_to_reference.params["terminal_hold_duration_s"] = terminal_hold_duration_s
        self.events.reset_to_reference.params["retrigger_prepare_duration_s"] = retrigger_prepare_duration_s
        self.events.reset_to_reference.params["fresh_prepare_duration_s"] = retrigger_prepare_duration_s
        # Preserve the first-jump command map while exposing enough policy-native landings
        # to learn the repeated handoff. Both reset paths now receive the same phase-zero
        # preparation, so the actor can converge them to a common cyclic start target.
        self.events.reset_to_reference.params["retrigger_probability"] = 0.25
        self.terminations.time_out.params = {"hold_duration_s": terminal_hold_duration_s}
        self.episode_length_s = REFERENCE_DURATION_S + terminal_hold_duration_s + retrigger_prepare_duration_s

        self.observations.policy.goal_remaining.func = obs_goal_remaining_latched
        self.observations.policy.goal_remaining.params = {}
        self.commands.jump_goal.zero_goal_probability = 0.25
        self.commands.jump_goal.boundary_goal_probability = 0.5
        self.rewards.target_position.weight = 16.0
        self.rewards.target_velocity_error.weight = -75.0
        self.rewards.reference_joint_target_deviation.weight = -20.0
        self.rewards.reference_joint_target_deviation.params["phase_weights"] = (
            8.0,
            2.0,
            0.25,
            0.25,
            2.0,
            10.0,
        )


@configclass
class G1JumpStage2DeployLongitudinalLatchedSmoothNarrowDirectRepeatEnvCfg(
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowRepeatEnvCfg
):
    """Fine-tune direct policy-stand retriggers with an ankle safety margin.

    Every safe full-length landing is reused at rest with a newly sampled
    trigger-latched command. The new episode starts at phase zero immediately,
    matching an FSM that holds the previous final STAND controller until the
    operator confirms the next jump. A wider ankle-roll limit margin penalizes
    landing impulses before they reach the modeled joint stop.
    """

    def __post_init__(self):
        super().__post_init__()

        terminal_hold_duration_s = 1.0
        self.events.reset_to_reference.params["terminal_hold_duration_s"] = terminal_hold_duration_s
        self.events.reset_to_reference.params["retrigger_prepare_duration_s"] = 0.0
        self.events.reset_to_reference.params["fresh_prepare_duration_s"] = 0.0
        self.events.reset_to_reference.params["retrigger_probability"] = 1.0
        self.events.reset_to_reference.params["retrigger_after_retrigger_probability"] = 1.0
        self.terminations.time_out.params["hold_duration_s"] = terminal_hold_duration_s
        self.episode_length_s = REFERENCE_DURATION_S + terminal_hold_duration_s
        self.rewards.ankle_roll_position_limit_margin.weight = -100.0
        self.rewards.ankle_roll_position_limit_margin.params["margin"] = 0.04


@configclass
class G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatEnvCfg(
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowDirectRepeatEnvCfg
):
    """Fine-tune settled repeated jumps without sacrificing fresh commands.

    Fresh reference starts remain half of eligible resets. The other half carry
    a four-second policy-native stand into a newly latched command, matching the
    repeated FSM after it has converged. An unsaturated terminal position cost is
    applied only to those carried episodes so the actor learns their systematic
    handoff bias while retaining the original start-state command map.
    """

    def __post_init__(self):
        super().__post_init__()

        terminal_hold_duration_s = 4.0
        self.events.reset_to_reference.params["terminal_hold_duration_s"] = terminal_hold_duration_s
        self.events.reset_to_reference.params["retrigger_probability"] = 0.5
        self.terminations.time_out.params["hold_duration_s"] = terminal_hold_duration_s
        self.episode_length_s = REFERENCE_DURATION_S + terminal_hold_duration_s
        self.rewards.target_position_error.weight = -200.0
        self.rewards.target_position_error.params["phase_weights"] = (0.0, 0.0, 0.0, 0.0, 4.0, 12.0)
        self.rewards.target_position_error.params["retrigger_only"] = True
        self.rewards.ankle_roll_position_limit_margin.weight = -50.0
        self.rewards.ankle_roll_position_limit_margin.params["margin"] = 0.03
        self.scene.robot.actuators["feet"].damping = {
            ".*_ankle_pitch_joint": 2.0,
            ".*_ankle_roll_joint": 4.0,
        }


@configclass
class G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatStrongEnvCfg(
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatEnvCfg
):
    """Emphasize signed command response from settled repeated-jump states.

    Three quarters of eligible resets carry a safe policy-native landing into
    the next relative command. The unsaturated landing error dominates the
    bounded proximity reward, while stronger takeoff-velocity and ankle-margin
    costs retain a usable command gradient without accepting joint-stop contact.
    Fresh reference starts remain in the batch to constrain first-jump behavior.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_to_reference.params["retrigger_probability"] = 0.75
        self.commands.jump_goal.zero_goal_probability = 0.2
        self.commands.jump_goal.boundary_goal_probability = 0.6
        self.rewards.target_position.weight = 0.0
        self.rewards.target_position_error.weight = -2000.0
        self.rewards.target_velocity_error.weight = -150.0
        self.rewards.ankle_roll_position_limit_margin.weight = -200.0
        self.rewards.ankle_roll_position_limit_margin.params["margin"] = 0.04
        self.rewards.reference_joint_target_deviation.params["phase_weights"] = (
            8.0,
            2.0,
            0.05,
            0.05,
            2.0,
            10.0,
        )


@configclass
class G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatDenseEnvCfg(
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatStrongEnvCfg
):
    """Give command tracking dense credit before and through touchdown.

    The carried-state landing error otherwise arrives mostly in the four-second
    terminal hold, after the takeoff actions that caused it have left a short
    PPO rollout. This variant applies unsaturated position error from takeoff
    onward and raises the signed planar-velocity cost during takeoff and flight.
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards.target_position_error.weight = -500.0
        self.rewards.target_position_error.params["phase_weights"] = (0.0, 0.0, 2.0, 4.0, 12.0, 2.0)
        self.rewards.target_position_error.params["retrigger_only"] = False
        self.rewards.target_velocity_error.weight = -1000.0


@configclass
class G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerAwareEnvCfg(
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatDenseEnvCfg
):
    """Condition the policy explicitly on fresh versus carried jump state.

    Joint and IMU feedback alone leave the policy to infer whether phase zero
    began at the canonical reference pose or at its preceding policy-native
    landing. The FSM already knows that state. This task exposes it through the
    otherwise-unused vertical goal-command component while retaining the
    checkpoint's 326-element observation shape.
    """

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.goal_command.func = obs_goal_command_remaining_orientation_retrigger
        self.observations.critic.goal_command.func = obs_goal_command_remaining_orientation_retrigger

        # Preserve the bounded goal reward that shaped reliable fresh jumps,
        # while applying the signed position correction only to carried state.
        self.rewards.target_position.weight = 16.0
        self.rewards.target_position_error.weight = -300.0
        self.rewards.target_position_error.params["retrigger_only"] = True
        self.rewards.target_velocity_error.weight = -150.0

        # A repeat is usable only if the preceding policy-native stand leaves
        # enough ankle-roll authority for the next takeoff.
        self.rewards.ankle_roll_position_limit_margin.weight = -1000.0
        self.rewards.ankle_roll_position_limit_margin.params["margin"] = 0.05


@configclass
class G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerGoalEnvCfg(
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerAwareEnvCfg
):
    """Expose the longitudinal goal through a repeat-only actor channel.

    The vertical command component remains exactly zero throughout every
    fresh episode. Carried episodes receive ``0.25 + goal_pos_x`` before the
    inherited observation scale, allowing the retrigger adapter to learn a
    signed command correction without changing the validated first jump.
    """

    def __post_init__(self):
        super().__post_init__()

        params = {
            "retrigger_value": 0.25,
            "retrigger_goal_pos_x_scale": 1.0,
        }
        self.observations.policy.goal_command.func = obs_goal_command_remaining_orientation_retrigger_goal
        self.observations.policy.goal_command.params = dict(params)
        self.observations.critic.goal_command.func = obs_goal_command_remaining_orientation_retrigger_goal
        self.observations.critic.goal_command.params = dict(params)
        self.events.reset_to_reference.params["use_soft_joint_limits"] = False
        self.events.reset_to_reference.params["joint_limit_margin"] = 0.01


@configclass
class G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerChainEnvCfg(
    G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerGoalEnvCfg
):
    """Train the repeat residual on uninterrupted signed command chains.

    Every safe landing is carried into another relative command so later jumps
    are represented as often as the first retrigger. Exact zero and both
    longitudinal boundaries span the deployable command set. Unsaturated
    position and velocity costs supply signed credit through landing, while a
    stronger termination cost rejects policies that gain distance by falling.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_to_reference.params["retrigger_probability"] = 1.0
        self.commands.jump_goal.retrigger_cycle_goal_probability = 1.0
        self.commands.jump_goal.zero_goal_probability = 0.2
        self.commands.jump_goal.boundary_goal_probability = 0.8
        self.rewards.target_position.weight = 0.0
        self.rewards.target_position_error.weight = -2000.0
        self.rewards.target_position_error.params["phase_weights"] = (
            0.0,
            0.0,
            2.0,
            4.0,
            16.0,
            4.0,
        )
        self.rewards.target_position_error.params["retrigger_only"] = True
        self.rewards.target_velocity_error.weight = -1000.0
        self.rewards.termination_penalty.weight = -500.0
        self.rewards.ankle_roll_position_limit_margin.params["retrigger_only"] = True
        self.rewards.ankle_roll_position_limit_margin.params["use_soft_joint_limits"] = False


@configclass
class G1JumpStage2DeployLongitudinalOdometryEnvCfg(G1JumpStage2DeployLongitudinalUniformEnvCfg):
    """Longitudinal curriculum with live remaining-displacement feedback.

    This diagnostic stage keeps the deployable observation dimensions and action contract
    unchanged while replacing the actor's trigger-latched displacement with live odometry.
    It is suitable for deployment only when the runtime supplies validated horizontal
    odometry; G1 ``LowState`` does not provide that measurement directly.
    """

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.goal_remaining.func = obs_goal_remaining
        self.observations.policy.goal_remaining.params = {}


@configclass
class G1JumpStage2DeployLongitudinalOdometrySmoothEnvCfg(G1JumpStage2DeployLongitudinalSmoothEnvCfg):
    """Odometry curriculum with bandwidth-limited leg position targets.

    The faster deployment filter lets small policy-state differences become distinct
    contact modes before the target interpolation can attenuate them. This stage lowers
    the leg filter bandwidth while preserving the checkpoint's observations, target
    scaling, torque projection, and zero-delay contract.
    """

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.goal_remaining.func = obs_goal_remaining
        self.observations.policy.goal_remaining.params = {}


@configclass
class G1JumpStage2DeployLongitudinalOdometrySmoothNarrowEnvCfg(G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg):
    """Safe-target narrow curriculum with live remaining-displacement feedback."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.goal_remaining.func = obs_goal_remaining
        self.observations.policy.goal_remaining.params = {}


@configclass
class G1JumpStage2DeployLongitudinalOdometrySmoothTargetSafeEnvCfg(G1JumpStage2DeployLongitudinalOdometrySmoothEnvCfg):
    """Narrow live-feedback curriculum that teaches knee-target stop avoidance."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.ranges.pos_x = (-0.1, 0.1)
        self.rewards.knee_target_lower_limit.weight = -2.0


@configclass
class G1JumpStage2DeployLongitudinalContactEnvCfg(G1JumpStage2DeployLongitudinalEnvCfg):
    """Contact-robust bridge between longitudinal command training and full Stage 3.

    This curriculum varies only contact compliance. It lets the policy adapt to the
    dominant Isaac-to-MuJoCo mismatch before introducing mass, center-of-mass, gain,
    sensing, push, and latency randomization together.
    """

    @configclass
    class EventCfg(G1JumpEventCfg):
        contact_compliance = EventTerm(
            func=randomize_contact_compliance,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_range": (1.0e5, 1.0e6),
                "damping_ratio_range": (0.8, 1.2),
                "rigid_probability": 0.25,
            },
        )

    events: EventCfg = EventCfg()


@configclass
class G1JumpStage2DeployLongitudinalOdometryRobustEnvCfg(G1JumpStage2DeployLongitudinalContactEnvCfg):
    """Odometry curriculum with sensing noise and contact variation.

    This bridge targets excessive closed-loop policy gain without introducing mass,
    center-of-mass, actuator-gain, push, or action-delay randomization. It retains the
    live horizontal odometry requirement of
    :class:`G1JumpStage2DeployLongitudinalOdometryEnvCfg`.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.zero_goal_probability = 0.1
        self.commands.jump_goal.boundary_goal_probability = 0.1
        self.observations.policy.enable_corruption = True
        self.observations.policy.joint_pos.noise = UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
        self.observations.policy.joint_vel.noise = UniformNoiseCfg(n_min=-1.0, n_max=1.0, operation="add")
        self.observations.policy.base_ang_vel.noise = UniformNoiseCfg(n_min=-0.3, n_max=0.3, operation="add")
        self.observations.policy.projected_gravity.noise = UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add")
        self.observations.policy.goal_remaining.func = obs_goal_remaining
        self.observations.policy.goal_remaining.params = {}
        self.observations.policy.goal_remaining.noise = UniformNoiseCfg(n_min=-0.02, n_max=0.02, operation="add")


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeContactEnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeEnvCfg):
    """Longitudinal range widening with randomized contact and actor sensing."""

    @configclass
    class EventCfg(G1JumpEventCfg):
        contact_compliance = EventTerm(
            func=randomize_contact_compliance,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_range": (1.0e5, 1.0e6),
                "damping_ratio_range": (0.8, 1.2),
                "rigid_probability": 0.25,
            },
        )

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.observations.policy.joint_vel.noise = UniformNoiseCfg(n_min=-1.0, n_max=1.0, operation="add")
        self.observations.policy.base_ang_vel.noise = UniformNoiseCfg(n_min=-0.3, n_max=0.3, operation="add")
        self.observations.policy.projected_gravity.noise = UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add")


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeContactTriggerEnvCfg(
    G1JumpStage2DeployLongitudinalSmoothRangeContactEnvCfg
):
    """Contact-robust range curriculum with randomized deployment trigger states."""

    @configclass
    class EventCfg(G1JumpStage2DeployLongitudinalSmoothRangeContactEnvCfg.EventCfg):
        perturb_trigger_state = EventTerm(
            func=perturb_trigger_state,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "leg_joint_pos_noise_rad": 0.05,
                "ankle_pitch_offset_range_rad": (-0.15, 0.15),
                "ankle_roll_noise_rad": 0.03,
                "root_pitch_noise_rad": math.radians(3.0),
                "root_roll_noise_rad": math.radians(1.5),
                "root_height_offset_range_m": (0.0, 0.01),
                "joint_vel_noise_rad_s": 0.1,
            },
        )

    events: EventCfg = EventCfg()


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeContactTrigger020EnvCfg(
    G1JumpStage2DeployLongitudinalSmoothRangeContactTriggerEnvCfg
):
    """Trigger-randomized contact curriculum with longitudinal goals [m] in [-0.2, 0.2]."""

    goal_pos_x_range: tuple[float, float] = (-0.2, 0.2)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeContactTrigger040EnvCfg(
    G1JumpStage2DeployLongitudinalSmoothRangeContactTriggerEnvCfg
):
    """Trigger-randomized contact curriculum with longitudinal goals [m] in [-0.4, 0.4]."""

    goal_pos_x_range: tuple[float, float] = (-0.4, 0.4)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeContact020EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeContactEnvCfg):
    """Contact-randomized deployment curriculum with longitudinal goals [m] in [-0.2, 0.2]."""

    goal_pos_x_range: tuple[float, float] = (-0.2, 0.2)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeContact040EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeContactEnvCfg):
    """Contact-randomized deployment curriculum with longitudinal goals [m] in [-0.4, 0.4]."""

    goal_pos_x_range: tuple[float, float] = (-0.4, 0.4)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeContact060EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeContactEnvCfg):
    """Contact-randomized deployment curriculum with longitudinal goals [m] in [-0.6, 0.6]."""

    goal_pos_x_range: tuple[float, float] = (-0.6, 0.6)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeContact080EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeContactEnvCfg):
    """Contact-randomized deployment curriculum with longitudinal goals [m] in [-0.8, 0.8]."""

    goal_pos_x_range: tuple[float, float] = (-0.8, 0.8)


@configclass
class G1JumpStage2DeployLongitudinalSmoothRangeContact100EnvCfg(G1JumpStage2DeployLongitudinalSmoothRangeContactEnvCfg):
    """Contact-randomized deployment curriculum with longitudinal goals [m] in [-1.0, 1.0]."""

    goal_pos_x_range: tuple[float, float] = (-1.0, 1.0)


@configclass
class G1JumpStage2WideEnvCfg(G1JumpStage2EnvCfg):
    """Stage 2 at roughly two thirds of the paper's goal ranges.

    Kept as its own task rather than widening stage 2 in place so the narrower policy stays
    reproducible. The forward bias in ``pos_x`` follows the paper, which samples U(-0.5, 1.5):
    a jump forward is the case the task is really about, and a backward jump of the same size
    is harder for a robot whose reference motion travels nowhere.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.ranges.pos_x = (-0.3, 1.0)
        self.commands.jump_goal.ranges.pos_y = (-0.6, 0.6)
        self.commands.jump_goal.ranges.yaw = (-60.0 * torch.pi / 180.0, 60.0 * torch.pi / 180.0)

        # The task reward kernels are exp(-k * squared error), so k is only meaningful relative
        # to the errors actually seen. Both were calibrated for the narrow ranges and go flat
        # over the wider ones: at the 0.62 m rms distance of an unmoved robot, position scores
        # 3e-4 and supplies essentially no gradient. Re-scale both so the kernel at that
        # distance matches what the narrow stage saw at its own rms distance (0.24).
        self.rewards.target_position.params["gradient"] = 3.72
        self.rewards.target_orientation.params["gradient"] = 6.0

        # With goals up to 1.17 m away the stage 1 bound of 1.0 m was unreachable in the wrong
        # direction and vacuous in every other; tighten to the paper's stage 2 values so a
        # landing that misses is actually terminated.
        self.terminations.task_completion_error.params["pos_threshold"] = 0.35
        self.terminations.task_completion_error.params["yaw_threshold"] = 35.0 * (torch.pi / 180.0)


@configclass
class G1JumpStage2WideLandEnvCfg(G1JumpStage2WideEnvCfg):
    """Wide stage 2, rewarded for arriving at touchdown rather than after settling.

    The wide policy lands short and walks the rest in. Measured over 600 episodes with
    observation corruption off, it covers 0.874 of the goal distance by touchdown but 0.986
    of it once settled, and the shortfall grows with distance: goals past 0.75 m reach only
    0.851 at touchdown against 0.891 for goals under 0.45 m.

    Kept as its own task so the wide stage stays reproducible.
    """

    def __post_init__(self):
        super().__post_init__()

        # Inherited weights put 12 on standing against 8 on landing, so the policy scores
        # better arriving after it has settled than at touchdown, and a short landing followed
        # by a corrective hop is worth more than a landing on the mark. Weight landing above
        # standing so the position error is judged where we want it to be small.
        #
        # A first pass used 14 against 8, which overcorrected: median reach at touchdown rose
        # from 0.959 to 0.997 and far goals improved from 0.851 to 0.873, but the robot began
        # arriving with momentum it could not shed, and settled overshoot went from 32% of
        # episodes to 48% while settled error rose from 0.071 m to 0.077 m. 12 against 10 keeps
        # landing dominant with less of a push past the goal.
        self.rewards.target_position.params["phase_weights"] = (0.0, 1.0, 2.0, 4.0, 12.0, 10.0)

        # Landing carried a weight of 1, so nothing asked the robot to be stopped at touchdown
        # and the position term alone drove it long. Weight arriving at rest, which is what
        # bounds the overshoot the position change above introduces.
        self.rewards.target_velocity.params["phase_weights"] = (0.0, 0.0, 3.0, 3.0, 6.0, 2.0)

        # The reference is a single fixed jump, so its pelvis arc only describes the distance
        # it was recorded at. Holding the robot to that arc through flight is what caps range,
        # which is why the shortfall appears only on far goals. Crouch and take-off keep their
        # weights, so leaving the ground is still required; flight and landing are relaxed so
        # the arc can stretch. This reduces the term rather than removing it: with foot height
        # it is the inductive bias that keeps the behaviour a jump instead of a walk
        # (Li et al., section IV-D, remark 2), and the paper keeps it through every stage.
        self.rewards.track_root_pos_z.params["phase_weights"] = (4.0, 8.0, 12.0, 8.0, 6.0, 6.0)


@configclass
class G1JumpStage3EnvCfg(G1JumpStage2WideLandEnvCfg):
    """Sim-to-real stage with randomized dynamics, sensing, and action latency.

    The actor is already restricted to observations available on the physical G1, while the
    simulation-only critic retains privileged, uncorrupted observations.
    """

    @configclass
    class EventCfg(G1JumpEventCfg):
        # The reference's 7.49 deg pitch is correct against feet-flat kinematics, but an
        # identical attitude and velocity on every reset gives the policy unrealistic zero
        # variance. Mild floor slope and standing residuals fit within +/-3 deg; at the 0.78 m
        # pelvis height that changes foot height by only about 1.1 mm, below the existing
        # roughly 5 mm ground penetration, so height compensation is unnecessary.
        reset_to_reference = EventTerm(
            func=reference_state_initialization,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "init_start_prob": 0.2,
                "roll_range": (-0.052, 0.052),
                "pitch_range": (-0.052, 0.052),
                "lin_vel_range": (-0.05, 0.05),
            },
        )
        # These contact ranges cover low-grip surfaces through high-grip rubber contact and
        # allow imperfectly inelastic contacts without introducing highly elastic impacts.
        physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.5, 1.25),
                "dynamic_friction_range": (0.4, 1.0),
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        )
        # Scaling every link by ±20% covers aggregate mass and link-level model error.
        robot_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )
        # A 5 cm pelvis offset covers uncertainty in torso equipment and payload placement.
        pelvis_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "com_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05),
                },
            },
        )
        # ±25% gain scaling covers actuator calibration and unmodelled drivetrain response.
        actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.75, 1.25),
                "damping_distribution_params": (0.75, 1.25),
                "operation": "scale",
            },
        )
        # Frequent 0.5 m/s lateral velocity changes exercise recovery during every jump.
        push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.5, 3.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

    actions: G1JumpDeployActionsCfg = G1JumpDeployActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        # Zero to two control steps represents 0-40 ms latency at the 50 Hz policy rate.
        self.actions.joint_pos.min_delay_steps = 0
        self.actions.joint_pos.max_delay_steps = 2
        self.rewards.action_rate.weight = -0.1

        self.observations.policy.enable_corruption = True
        self.observations.policy.joint_pos.noise = UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
        self.observations.policy.joint_vel.noise = UniformNoiseCfg(n_min=-1.0, n_max=1.0, operation="add")
        self.observations.policy.base_ang_vel.noise = UniformNoiseCfg(n_min=-0.3, n_max=0.3, operation="add")
        self.observations.policy.projected_gravity.noise = UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add")
        # G1 LowState has no root position once the native controller is released.
        # Keep the actor conditioned on the trigger-time relative goal so real inference
        # needs only joint and IMU feedback. The privileged critic remains closed-loop.
        self.observations.policy.goal_remaining.func = obs_goal_remaining_latched
        self.observations.policy.goal_remaining.params = {}
        self.observations.policy.goal_remaining.noise = UniformNoiseCfg(n_min=-0.02, n_max=0.02, operation="add")

        # The critic is simulation-only and must not inherit actor corruption or stale odometry.
        self.observations.critic.enable_corruption = False
        self.observations.critic.goal_remaining.func = obs_goal_remaining
        self.observations.critic.goal_remaining.params = {}


@configclass
class G1JumpStage3DeployTranslationEnvCfg(G1JumpStage3EnvCfg):
    """Sim-to-real robustness stage for the narrow translation-only policy.

    This stage preserves the deployable remaining-attitude observation introduced by
    :class:`G1JumpStage2DeployTranslationEnvCfg` while adding Stage 3 dynamics, sensing,
    and latency randomization. Contact compliance spans soft shoe/floor contact through
    rigid PhysX contact so the policy cannot specialize to a single solver response.
    """

    @configclass
    class EventCfg(G1JumpStage3EnvCfg.EventCfg):
        contact_compliance = EventTerm(
            func=randomize_contact_compliance,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_range": (1.0e4, 1.0e6),
                "damping_ratio_range": (0.7, 1.4),
                "rigid_probability": 0.25,
            },
        )

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.ranges.pos_x = (-0.2, 0.2)
        self.commands.jump_goal.ranges.pos_y = (-0.15, 0.15)
        self.commands.jump_goal.ranges.yaw = (0.0, 0.0)
        self.observations.policy.goal_command.func = obs_goal_command_remaining_orientation
        self.observations.critic.goal_command.func = obs_goal_command_remaining_orientation

        self.rewards.target_position.params["gradient"] = 21.07
        self.rewards.target_orientation.params["gradient"] = 30.0
        self.rewards.target_heading.params["gradient"] = 30.0
        self.rewards.target_heading.params["phase_weights"] = (0.0, 0.0, 0.0, 0.0, 8.0, 12.0)

        # Unitree advertises a 90 N.m maximum knee torque for the standard G1. Sixty
        # percent of the 139 N.m model envelope is 83.4 N.m, leaving margin below that
        # maximum while applying the same conservative fraction to every actuator.
        self.rewards.joint_torque_demand_limit.weight = -50.0
        self.rewards.joint_torque_demand_limit.params["soft_ratio"] = 0.6
        self.actions.joint_pos.effort_limit_ratio = 0.6


@configclass
class G1JumpStage3DeployLongitudinalEnvCfg(G1JumpStage3DeployTranslationEnvCfg):
    """Robustness curriculum for the signed forward/backward deployment policy."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.ranges.pos_y = (0.0, 0.0)
        self.commands.jump_goal.zero_goal_probability = 0.25
        self.commands.jump_goal.boundary_goal_probability = 0.5
        self.observations.policy.goal_remaining.scale = 4.0
        self.observations.critic.goal_remaining.scale = 4.0
        self.observations.policy.goal_command.scale = 4.0
        self.observations.critic.goal_command.scale = 4.0

        self.rewards.target_position.weight = 8.0
        self.rewards.target_velocity.weight = 0.0
        self.rewards.target_velocity_error.weight = -50.0
        self.rewards.target_heading.weight = 3.0
        self.rewards.reference_joint_target_deviation.weight = -5.0


@configclass
class G1JumpStage3NarrowEnvCfg(G1JumpStage3EnvCfg):
    """Sim-to-real stage that retains the narrow Stage 2 command envelope.

    This task introduces the deployment action filter, latency, observation noise, and
    dynamics randomization without simultaneously increasing the commanded displacement.
    Once this stage meets the deployment acceptance thresholds, training can continue with
    :class:`G1JumpStage3EnvCfg` to widen the command envelope independently.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.jump_goal.ranges.pos_x = (-0.4, 0.4)
        self.commands.jump_goal.ranges.pos_y = (-0.3, 0.3)
        self.commands.jump_goal.ranges.yaw = (-30.0 * torch.pi / 180.0, 30.0 * torch.pi / 180.0)

        # Restore the kernels calibrated for the narrow goal distribution. The wide-stage
        # kernels deliberately trade precision for gradient at metre-scale errors.
        self.rewards.target_position.params["gradient"] = 21.07
        self.rewards.target_orientation.params["gradient"] = 30.0


@configclass
class G1JumpStage3NarrowEnvCfg_PLAY(G1JumpStage3NarrowEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = True
        self.commands.jump_goal.debug_vis = True


@configclass
class G1JumpStage3EnvCfg_PLAY(G1JumpStage3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = True
        self.commands.jump_goal.debug_vis = True


@configclass
class G1JumpStage2WideLandEnvCfg_PLAY(G1JumpStage2WideLandEnvCfg):
    actions: G1JumpPlayActionsCfg = G1JumpPlayActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = True
        self.commands.jump_goal.debug_vis = True


@configclass
class G1JumpStage2WideEnvCfg_PLAY(G1JumpStage2WideEnvCfg):
    actions: G1JumpPlayActionsCfg = G1JumpPlayActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = True
        self.commands.jump_goal.debug_vis = True


@configclass
class G1JumpStage2EnvCfg_PLAY(G1JumpStage2EnvCfg):
    actions: G1JumpPlayActionsCfg = G1JumpPlayActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = True
        # Each episode draws a different target, so without a marker there is no way to tell a
        # missed landing from a goal that moved. Only enabled for play: the marker costs draw
        # calls and training is headless.
        self.commands.jump_goal.debug_vis = True


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

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static definitions for the G1 jump task: asset paths, joint layout and motion phases."""

from __future__ import annotations

from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

DATA_STORAGE_DIR = Path(__file__).resolve().parents[9] / "data_storage"
# The ankle-feasible retarget of the captured motion, not the raw capture. The G1 drives each
# ankle through a parallel linkage that couples pitch and roll, and the raw capture leaves that
# coupled workspace by up to 47% during takeoff and flight. PhysX models no such coupling, so
# training against the raw file rewards ankle poses the robot physically cannot reach. Only the
# four ankle columns differ, and only around the infeasible frames: the median change is under
# 5e-4 rad and frame 0 -- which supplies the default joint positions, and through them the
# action offsets -- moves by 2e-4 rad.
#
# The root height is also shifted up by 7.521074 mm, the exact signed distance from the foot
# collision geoms to the ground at frame 0. The captured height put the feet BELOW the floor,
# so PhysX spent the first ~16 ms depenetrating the robot: measured, the pelvis rose from
# 0.780409 to 0.786171 at +0.119 m/s and the feet took a 1974 N spike. Every training episode
# began with that kick, which no real robot receives and which MuJoCo cannot even reproduce
# (it goes non-finite from the same initial penetration). The collision geometry itself is
# correct -- the USD capsules match the MJCF fromto midpoints exactly -- so the fix belongs in
# the motion, not the asset.
CSV_MOTION_PATH = str(DATA_STORAGE_DIR / "perfect_jump_ground_aligned.csv")
CSV_MOTION_PATH_EXTENDED = str(DATA_STORAGE_DIR / "perfect_jump_extended.csv")
REFERENCE_NUM_FRAMES = 91
REFERENCE_NUM_FRAMES_EXTENDED = 196
REFERENCE_MOTION_FPS = 30.0
REFERENCE_DURATION_S = REFERENCE_NUM_FRAMES / REFERENCE_MOTION_FPS
REFERENCE_DURATION_S_EXTENDED = REFERENCE_NUM_FRAMES_EXTENDED / REFERENCE_MOTION_FPS
NUMBER_OF_JOINTS = 23
JUMP_PHASES = {
    "IDLE": (0, 6),
    "CROUCH": (6, 19),
    "TAKEOFF": (19, 26),
    "FLIGHT": (26, 43),
    "LAND": (43, 60),
    "STAND": (60, 91),
}
JUMP_PHASES_EXTENDED = {
    "IDLE": (0, 51),
    "CROUCH": (51, 64),
    "TAKEOFF": (64, 71),
    "FLIGHT": (71, 88),
    "LAND": (88, 105),
    "STAND": (105, 196),
}

G1_USD_PATH = str(DATA_STORAGE_DIR / "g1_23dof_holo_compat" / "g1_23dof_holo_compat" / "g1_23dof_holo_compat.usda")
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
# Each normalized action can cover 115% of that joint's largest reference excursion
# from frame zero. The previous scales covered as little as 37% of the reference and
# therefore required raw actor outputs of 2.7 before feedback corrections; once those
# outputs became unbounded, the policy learned a bang-bang controller through joint stops.
# These scales pair with the tanh-squashed actor and mechanical target clip below.
JOINT_ACTION_SCALES = {
    "left_hip_pitch_joint": 2.579558,
    "left_hip_roll_joint": 0.252798,
    "left_hip_yaw_joint": 0.295907,
    "left_knee_joint": 2.935078,
    "left_ankle_pitch_joint": 0.761558,
    "left_ankle_roll_joint": 0.156033,
    "right_hip_pitch_joint": 2.397138,
    "right_hip_roll_joint": 0.223851,
    "right_hip_yaw_joint": 0.264369,
    "right_knee_joint": 3.099232,
    "right_ankle_pitch_joint": 0.719913,
    "right_ankle_roll_joint": 0.249177,
    "waist_yaw_joint": 0.151461,
    "left_shoulder_pitch_joint": 0.715553,
    "left_shoulder_roll_joint": 1.186100,
    "left_shoulder_yaw_joint": 0.621555,
    "left_elbow_joint": 0.888665,
    "left_wrist_roll_joint": 0.339550,
    "right_shoulder_pitch_joint": 0.495058,
    "right_shoulder_roll_joint": 0.679958,
    "right_shoulder_yaw_joint": 0.511980,
    "right_elbow_joint": 0.736179,
    "right_wrist_roll_joint": 0.242058,
}
PLAY_JOINT_ACTION_FILTER_ALPHA = {
    ".*_hip_.*": 0.70,
    ".*_knee_joint": 0.70,
    ".*_ankle_.*": 0.65,
}
# These limits come from Unitree's official 23-DOF URDF. Matching the model's actuator
# envelope matters for transfer, but the 139 N·m knee limit is a model limit rather than a
# verified safe continuous limit; Unitree advertises 90 N·m for the non-EDU G1.
#
# Gains are reduced only where the larger normalized action range would otherwise multiply
# one unit of action into a much larger PD demand. This keeps the normalized action-to-torque
# authority close to the original controller while allowing the actor to represent the whole
# reference motion without commanding past the target envelope.
G1_23DOF_HOLO_COMPAT_ACTUATORS = {
    "hip_pitch_yaw_waist": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_yaw_joint", ".*_hip_pitch_joint", "waist_yaw_joint"],
        effort_limit_sim=88,
        velocity_limit_sim=32,
        stiffness={".*_hip_yaw_joint": 150.0, ".*_hip_pitch_joint": 54.5, "waist_yaw_joint": 200.0},
        damping={".*_hip_yaw_joint": 5.0, ".*_hip_pitch_joint": 2.61, "waist_yaw_joint": 5.0},
        armature={".*_hip_.*": 0.01, "waist_yaw_joint": 0.01},
    ),
    "hip_roll_knee": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
        effort_limit_sim=139,
        velocity_limit_sim=20,
        stiffness={".*_hip_roll_joint": 150.0, ".*_knee_joint": 71.7},
        damping={".*_hip_roll_joint": 5.0, ".*_knee_joint": 2.99},
        armature={".*_hip_roll_joint": 0.01, ".*_knee_joint": 0.01},
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=35,
        velocity_limit_sim=30,
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
        effort_limit_sim=25,
        velocity_limit_sim=37,
        stiffness={
            ".*_shoulder_roll_joint": 33.7,
            ".*_(shoulder_pitch|shoulder_yaw|elbow|wrist_roll)_joint": 40.0,
        },
        damping={
            ".*_shoulder_roll_joint": 9.18,
            ".*_(shoulder_pitch|shoulder_yaw|elbow|wrist_roll)_joint": 10.0,
        },
        armature={".*_shoulder_.*": 0.01, ".*_elbow_joint": 0.01, ".*_wrist_roll_joint": 0.01},
    ),
}

# Mechanical travel of each joint [rad], from the G1 MJCF and cross-checked against the USD.
# The action clip below uses these rather than default +/- scale: a position target beyond a
# mechanical stop is never meaningful, and the physics engine clamps there regardless, so
# bounding the command changes nothing dynamically while keeping the deployed target honest.
JOINT_POSITION_LIMITS = {
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
}
G1_23DOF_HOLO_COMPAT_CFG = G1_MINIMAL_CFG.copy()
G1_23DOF_HOLO_COMPAT_CFG.spawn.usd_path = G1_USD_PATH
G1_23DOF_HOLO_COMPAT_CFG.spawn.activate_contact_sensors = True
G1_23DOF_HOLO_COMPAT_CFG.actuators = G1_23DOF_HOLO_COMPAT_ACTUATORS

# The G1 asset nests every link prim under its parent, so a contact sensor path is the whole
# kinematic chain down from the pelvis. Compose each path from the one above it so the nesting
# stays readable instead of repeating the shared prefixes.
_GEOMETRY_ROOT = "{ENV_REGEX_NS}/Robot/Geometry"
_PELVIS_PATH = f"{_GEOMETRY_ROOT}/pelvis"
_TORSO_PATH = f"{_PELVIS_PATH}/torso_link"
_LEFT_THIGH_PATH = f"{_PELVIS_PATH}/left_hip_pitch_link/left_hip_roll_link"
_LEFT_SHIN_PATH = f"{_LEFT_THIGH_PATH}/left_hip_yaw_link/left_knee_link"
_LEFT_FOOT_PATH = f"{_LEFT_SHIN_PATH}/left_ankle_pitch_link/left_ankle_roll_link"
_RIGHT_THIGH_PATH = f"{_PELVIS_PATH}/right_hip_pitch_link/right_hip_roll_link"
_RIGHT_SHIN_PATH = f"{_RIGHT_THIGH_PATH}/right_hip_yaw_link/right_knee_link"
_RIGHT_FOOT_PATH = f"{_RIGHT_SHIN_PATH}/right_ankle_pitch_link/right_ankle_roll_link"
_LEFT_UPPER_ARM_PATH = f"{_TORSO_PATH}/left_shoulder_pitch_link/left_shoulder_roll_link/left_shoulder_yaw_link"
_LEFT_LOWER_ARM_PATH = f"{_LEFT_UPPER_ARM_PATH}/left_elbow_link"
_LEFT_HAND_PATH = f"{_LEFT_LOWER_ARM_PATH}/left_wrist_roll_rubber_hand"
_RIGHT_UPPER_ARM_PATH = f"{_TORSO_PATH}/right_shoulder_pitch_link/right_shoulder_roll_link/right_shoulder_yaw_link"
_RIGHT_LOWER_ARM_PATH = f"{_RIGHT_UPPER_ARM_PATH}/right_elbow_link"
_RIGHT_HAND_PATH = f"{_RIGHT_LOWER_ARM_PATH}/right_wrist_roll_rubber_hand"

# The two feet must stay first: FOOT_CONTACT_SENSOR_NAMES slices them off the front.
CONTACT_SENSOR_PRIM_PATHS = {
    "left_foot": _LEFT_FOOT_PATH,
    "right_foot": _RIGHT_FOOT_PATH,
    "pelvis": _PELVIS_PATH,
    "left_thigh": _LEFT_THIGH_PATH,
    "left_shin": _LEFT_SHIN_PATH,
    "right_thigh": _RIGHT_THIGH_PATH,
    "right_shin": _RIGHT_SHIN_PATH,
    "torso": _TORSO_PATH,
    "left_upper_arm": _LEFT_UPPER_ARM_PATH,
    "left_lower_arm": _LEFT_LOWER_ARM_PATH,
    "left_hand": _LEFT_HAND_PATH,
    "right_upper_arm": _RIGHT_UPPER_ARM_PATH,
    "right_lower_arm": _RIGHT_LOWER_ARM_PATH,
    "right_hand": _RIGHT_HAND_PATH,
}
CONTACT_SENSOR_NAMES = tuple(f"contact_forces_{name}" for name in CONTACT_SENSOR_PRIM_PATHS)
FOOT_CONTACT_SENSOR_NAMES = CONTACT_SENSOR_NAMES[:2]
NON_FOOT_CONTACT_SENSOR_NAMES = CONTACT_SENSOR_NAMES[2:]

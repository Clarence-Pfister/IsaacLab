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
CSV_MOTION_PATH = str(DATA_STORAGE_DIR / "perfect_jump_processed.csv")
REFERENCE_NUM_FRAMES = 91
REFERENCE_MOTION_FPS = 30.0
REFERENCE_DURATION_S = REFERENCE_NUM_FRAMES / REFERENCE_MOTION_FPS
NUMBER_OF_JOINTS = 23
JUMP_PHASES = {
    "IDLE": (0, 6),
    "CROUCH": (6, 19),
    "TAKEOFF": (19, 26),
    "FLIGHT": (26, 43),
    "LAND": (43, 60),
    "STAND": (60, 91),
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
JOINT_ACTION_SCALES = {
    "left_hip_pitch_joint": 0.85,
    "left_hip_roll_joint": 0.18,
    "left_hip_yaw_joint": 0.18,
    "left_knee_joint": 1.00,
    "left_ankle_pitch_joint": 0.35,
    "left_ankle_roll_joint": 0.12,
    "right_hip_pitch_joint": 0.85,
    "right_hip_roll_joint": 0.18,
    "right_hip_yaw_joint": 0.20,
    "right_knee_joint": 1.00,
    "right_ankle_pitch_joint": 0.35,
    "right_ankle_roll_joint": 0.12,
    # The reference motion rotates the waist by up to 7.6 deg from its start pose, so a
    # zero scale would leave that error uncorrectable by the policy.
    "waist_yaw_joint": 0.20,
    "left_shoulder_pitch_joint": 0.30,
    "left_shoulder_roll_joint": 0.45,
    "left_shoulder_yaw_joint": 0.30,
    "left_elbow_joint": 0.45,
    "left_wrist_roll_joint": 0.12,
    "right_shoulder_pitch_joint": 0.30,
    "right_shoulder_roll_joint": 0.35,
    "right_shoulder_yaw_joint": 0.30,
    "right_elbow_joint": 0.45,
    "right_wrist_roll_joint": 0.12,
}
PLAY_JOINT_ACTION_FILTER_ALPHA = {
    ".*_hip_.*": 0.70,
    ".*_knee_joint": 0.70,
    ".*_ankle_.*": 0.65,
}
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

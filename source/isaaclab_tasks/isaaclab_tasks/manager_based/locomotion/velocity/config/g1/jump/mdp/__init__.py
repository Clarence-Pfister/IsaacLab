# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms specific to the G1 jump task."""

from .actions_cfg import LowPassJointPositionActionCfg
from .commands import JumpGoalCommand, JumpGoalCommandCfg
from .events import (
    perturb_trigger_state,
    randomize_contact_compliance,
    reference_or_terminal_state_initialization,
    reference_state_initialization,
)
from .motion import (
    MotionLoader,
    get_current_foot_pos_w,
    get_env_time,
    get_jump_phase,
    get_loader,
    get_phase_id,
    get_phase_weight,
    get_reference_initial_state,
    get_reward,
    get_root_relative_foot_pos,
    set_contact_sensor,
    slerp_quat,
    warp_to_torch,
)
from .observations import (
    obs_future_reference_preview,
    obs_goal_command,
    obs_goal_command_remaining_orientation,
    obs_goal_command_remaining_orientation_retrigger,
    obs_goal_command_remaining_orientation_retrigger_goal,
    obs_goal_remaining,
    obs_goal_remaining_latched,
    obs_goal_remaining_stale,
    obs_jump_phase,
    obs_projected_gravity,
)
from .rewards import (
    joint_position_limit_margin,
    joint_target_lower_limit,
    joint_torque_demand_limit,
    penalize_ground_impact,
    penalize_joint_acc,
    penalize_joint_vel,
    penalize_torque_consumption,
    reference_joint_target_deviation,
    target_angular_rate,
    target_heading,
    target_orientation,
    target_position,
    target_position_error,
    target_velocity,
    target_velocity_error,
    track_foot_xy,
    track_foot_z,
    track_joint_pos,
    track_joint_vel,
    track_root_angular_rate,
    track_root_orientation,
    track_root_pos_z,
    track_root_vel_z,
)
from .terminations import foot_tracking_error, ground_contact, reference_motion_complete, task_completion_error

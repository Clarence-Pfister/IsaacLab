# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure G1 jump-policy success rates over the configured goal envelope in Isaac."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tensordict import TensorDict

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.constants import FOOT_CONTACT_SENSOR_NAMES
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp import (
    obs_goal_remaining,
    obs_goal_remaining_latched,
)
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


_AGENT_CFG_ENTRY_POINT = "rsl_rl_cfg_entry_point"
_DEFAULT_EPISODES = 4096
_EXPECTED_EPISODE_STEPS = 152
_FAILURE_TERMS = ("base_contact", "bad_orientation", "foot_tracking_error", "task_completion_error")
_GOAL_RANGE_NAMES = ("pos_x", "pos_y", "roll", "pitch", "yaw")
_GOAL_TOLERANCES_M = (0.10, 0.15, 0.20, 0.30)
_SWEEP_RANGE_SCALES = (1.0, 0.75, 0.50, 0.25)
_REFERENCE_ROLL_RAD = math.radians(0.43)
_REFERENCE_PITCH_RAD = math.radians(7.49)
_CONFIDENCE_Z = 1.959963984540054
_SATURATION_RELATIVE_TOLERANCE = 1.0e-5
_AIRBORNE_CONTACT_THRESHOLD_N = 5.0
_AXIS_SCALES: dict[str, float] = {}


@dataclass(frozen=True)
class CommandResponseFit:
    """Linear command-to-settled-displacement response measurements."""

    response_matrix: np.ndarray
    offset_xy: np.ndarray
    axis_correlation: np.ndarray
    mean_absolute_tracking_error_xy: np.ndarray
    tracking_error_norm_percentiles: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    """Episode-level measurements for one goal-range scale."""

    range_scale: float
    scaled_ranges: dict[str, tuple[float, float]]
    goal_displacement_xy: np.ndarray
    final_displacement_xy: np.ndarray
    goal_distance: np.ndarray
    goal_yaw_magnitude: np.ndarray
    final_height: np.ndarray
    final_tilt_error: np.ndarray
    final_goal_error: np.ndarray
    final_yaw_error: np.ndarray
    peak_height: np.ndarray
    maximum_airborne_time: np.ndarray
    termination_failures: dict[str, np.ndarray]
    peak_torque_fraction: np.ndarray
    torque_saturation_fraction: np.ndarray
    torque_saturation_streak: np.ndarray
    peak_torque_demand_fraction: np.ndarray
    torque_demand_exceedance_fraction: np.ndarray
    torque_demand_exceedance_streak: np.ndarray
    joint_names: tuple[str, ...]
    effort_limits: np.ndarray
    physics_dt: float

    @property
    def sample_count(self) -> int:
        """Number of evaluated episodes."""
        return len(self.goal_distance)


def _fit_command_response(commanded_xy: np.ndarray, landed_xy: np.ndarray) -> CommandResponseFit:
    """Fit settled displacement as an affine function of commanded displacement.

    Args:
        commanded_xy: Commanded planar displacement [m], shape ``(sample_count, 2)``.
        landed_xy: Settled planar displacement [m], shape ``(sample_count, 2)``.

    Returns:
        Affine response and direct tracking-error measurements.

    Raises:
        ValueError: If either input is empty or does not have shape ``(sample_count, 2)``.
    """
    commanded_xy = np.asarray(commanded_xy, dtype=np.float64)
    landed_xy = np.asarray(landed_xy, dtype=np.float64)
    if commanded_xy.ndim != 2 or commanded_xy.shape[1:] != (2,) or len(commanded_xy) == 0:
        raise ValueError(f"commanded_xy must have non-empty shape (sample_count, 2), got {commanded_xy.shape}.")
    if landed_xy.shape != commanded_xy.shape:
        raise ValueError(f"landed_xy must have shape {commanded_xy.shape}, got {landed_xy.shape}.")

    commanded_centered = commanded_xy - np.mean(commanded_xy, axis=0)
    axis_norm = np.linalg.norm(commanded_centered, axis=0)
    independence_tolerance = max(np.max(axis_norm) * 1.0e-3, np.finfo(np.float64).eps)
    identifiable_axes: list[int] = []
    for axis in np.argsort(-axis_norm):
        candidate = commanded_centered[:, axis]
        if axis_norm[axis] <= independence_tolerance:
            continue
        if identifiable_axes:
            basis = commanded_centered[:, identifiable_axes]
            projection, _, _, _ = np.linalg.lstsq(basis, candidate, rcond=None)
            candidate = candidate - basis @ projection
        if np.linalg.norm(candidate) > independence_tolerance:
            identifiable_axes.append(int(axis))

    design = np.column_stack((commanded_xy[:, identifiable_axes], np.ones(len(commanded_xy))))
    coefficients, _, _, _ = np.linalg.lstsq(design, landed_xy, rcond=None)
    response_matrix = np.full((2, 2), np.nan, dtype=np.float64)
    response_matrix[:, identifiable_axes] = coefficients[:-1].T
    offset_xy = coefficients[-1]

    axis_correlation = np.full(2, np.nan, dtype=np.float64)
    for axis in identifiable_axes:
        commanded_axis_centered = commanded_centered[:, axis]
        landed_centered = landed_xy[:, axis] - np.mean(landed_xy[:, axis])
        denominator = np.linalg.norm(commanded_axis_centered) * np.linalg.norm(landed_centered)
        if denominator > np.finfo(np.float64).eps:
            axis_correlation[axis] = np.dot(commanded_axis_centered, landed_centered) / denominator

    tracking_error = landed_xy - commanded_xy
    return CommandResponseFit(
        response_matrix=response_matrix,
        offset_xy=offset_xy,
        axis_correlation=axis_correlation,
        mean_absolute_tracking_error_xy=np.mean(np.abs(tracking_error), axis=0),
        tracking_error_norm_percentiles=np.percentile(np.linalg.norm(tracking_error, axis=1), (50.0, 90.0, 95.0)),
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "By default Stage 3 randomization and observation corruption remain enabled. "
            "--no_randomization removes them from the resolved evaluation copy."
        ),
    )
    parser.add_argument("--task", required=True, help="Registered G1 jump task ID.")
    parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint to evaluate.")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel Isaac environments.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=_DEFAULT_EPISODES,
        help=f"Total episodes per range scale (default: {_DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--range_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to every resolved jump-goal range (default: 1.0).",
    )
    parser.add_argument(
        "--axis_scale",
        default=None,
        help=(
            "Per-axis goal-range multipliers as name=value pairs, e.g. 'yaw=0,pos_y=0.5'. "
            "Applied on top of --range_scale. Narrowing one axis at a time is how the "
            "commandable envelope is traded against success rate."
        ),
    )
    parser.add_argument(
        "--sweep_range_scale",
        action="store_true",
        help="Evaluate range scales 1.0, 0.75, 0.50, and 0.25 in one run.",
    )
    parser.add_argument(
        "--no_randomization",
        action="store_true",
        help="Disable Stage 3 dynamics, sensing, reset, push, and action-delay randomization.",
    )
    parser.add_argument(
        "--goal_feedback",
        choices=("task", "live", "latched"),
        default="task",
        help=(
            "Goal-position feedback supplied to the actor. 'task' preserves the resolved task; "
            "'latched' is the real-G1 contract; 'live' is an oracle diagnostic (default: task)."
        ),
    )
    parser.add_argument(
        "--height_floor",
        type=float,
        default=0.6,
        help="Strict lower bound on final pelvis height [m] (default: 0.6).",
    )
    parser.add_argument(
        "--tilt_limit_deg",
        type=float,
        default=30.0,
        help="Final roll-pitch error limit relative to the reference attitude [deg] (default: 30).",
    )
    parser.add_argument(
        "--yaw_tolerance_deg",
        type=float,
        default=15.0,
        help="Maximum absolute final commanded-yaw error [deg] (default: 15).",
    )
    parser.add_argument(
        "--minimum_airborne_time_s",
        type=float,
        default=0.1,
        help="Minimum continuous both-feet-airborne time required for success [s] (default: 0.1).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (default: 0).")
    add_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {args.num_envs}.")
    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}.")
    if not math.isfinite(args.range_scale) or args.range_scale < 0.0:
        raise ValueError(f"--range_scale must be finite and non-negative, got {args.range_scale}.")
    if args.sweep_range_scale and not math.isclose(args.range_scale, 1.0):
        raise ValueError("--range_scale cannot be combined with --sweep_range_scale.")
    if not math.isfinite(args.height_floor):
        raise ValueError(f"--height_floor must be finite, got {args.height_floor}.")
    if not math.isfinite(args.tilt_limit_deg) or not 0.0 <= args.tilt_limit_deg <= 180.0:
        raise ValueError(f"--tilt_limit_deg must be within [0, 180], got {args.tilt_limit_deg}.")
    if not math.isfinite(args.yaw_tolerance_deg) or not 0.0 <= args.yaw_tolerance_deg <= 180.0:
        raise ValueError(f"--yaw_tolerance_deg must be within [0, 180], got {args.yaw_tolerance_deg}.")
    if not math.isfinite(args.minimum_airborne_time_s) or args.minimum_airborne_time_s < 0.0:
        raise ValueError(
            f"--minimum_airborne_time_s must be finite and non-negative, got {args.minimum_airborne_time_s}."
        )


def _range_pair(value: Any, name: str) -> tuple[float, float]:
    pair = tuple(float(item) for item in value)
    if len(pair) != 2 or not all(math.isfinite(item) for item in pair) or pair[0] > pair[1]:
        raise ValueError(f"Resolved task goal range {name!r} is invalid: {value}.")
    return pair


def _read_goal_ranges(env_cfg: Any) -> dict[str, tuple[float, float]]:
    ranges = env_cfg.commands.jump_goal.ranges
    return {name: _range_pair(getattr(ranges, name), name) for name in _GOAL_RANGE_NAMES}


def _parse_axis_scale(spec: str | None) -> dict[str, float]:
    """Parse 'yaw=0,pos_y=0.5' into per-axis multipliers."""
    if not spec:
        return {}
    axis_scales: dict[str, float] = {}
    for item in spec.split(","):
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in _GOAL_RANGE_NAMES:
            raise ValueError(f"Unknown goal axis {name!r}; expected one of {_GOAL_RANGE_NAMES}.")
        axis_scales[name] = float(value)
    return axis_scales


def _scaled_goal_ranges(
    base_ranges: dict[str, tuple[float, float]], scale: float, axis_scales: dict[str, float] | None = None
) -> dict[str, tuple[float, float]]:
    axis_scales = axis_scales or {}
    return {
        name: (lower * scale * axis_scales.get(name, 1.0), upper * scale * axis_scales.get(name, 1.0))
        for name, (lower, upper) in base_ranges.items()
    }


def _write_goal_ranges(ranges: Any, scaled_ranges: dict[str, tuple[float, float]]) -> None:
    for name, bounds in scaled_ranges.items():
        setattr(ranges, name, bounds)


def _configure_evaluation(env_cfg: Any, args: argparse.Namespace) -> tuple[str, ...]:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.commands.jump_goal.debug_vis = False
    if args.device is not None:
        env_cfg.sim.device = args.device

    # Reference-state initialization is a training device, not part of the commanded goal
    # distribution. Starting every rollout at reference frame zero is also required for all
    # episodes to cover the same complete 152-step motion horizon.
    reset_params = env_cfg.events.reset_to_reference.params
    reset_params["init_start_prob"] = 0.0

    if not args.no_randomization:
        return ()

    startup_event_names = (
        "physics_material",
        "robot_mass",
        "pelvis_com",
        "actuator_gains",
        "contact_compliance",
    )
    for event_name in startup_event_names:
        setattr(env_cfg.events, event_name, None)
    env_cfg.events.push_robot = None

    for name in ("roll_range", "pitch_range", "lin_vel_range"):
        reset_params[name] = (0.0, 0.0)

    action_cfg = env_cfg.actions.joint_pos
    action_cfg.min_delay_steps = 0
    action_cfg.max_delay_steps = 0

    policy_observations = env_cfg.observations.policy
    policy_observations.enable_corruption = False
    for term_name in ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "goal_remaining"):
        getattr(policy_observations, term_name).noise = None

    return (
        *(f"startup event events.{name}" for name in startup_event_names),
        "interval event events.push_robot",
        "policy observation corruption and configured observation noise",
        "action delay (min_delay_steps=max_delay_steps=0)",
        "reset attitude/velocity randomization (roll_range, pitch_range, lin_vel_range=0)",
    )


def _configure_goal_feedback(env_cfg: Any, mode: str) -> str:
    """Select actor goal feedback and return its resolved mode name."""
    goal_remaining = env_cfg.observations.policy.goal_remaining
    if mode == "live":
        goal_remaining.func = obs_goal_remaining
        goal_remaining.params = {}
    elif mode == "latched":
        goal_remaining.func = obs_goal_remaining_latched
        goal_remaining.params = {}
    elif mode != "task":
        raise ValueError(f"Unsupported goal feedback mode: {mode!r}.")

    function_name = getattr(goal_remaining.func, "__name__", type(goal_remaining.func).__name__)
    resolved_modes = {
        "obs_goal_remaining": "live",
        "obs_goal_remaining_stale": "flight_frozen",
        "obs_goal_remaining_latched": "latched",
    }
    return resolved_modes.get(function_name, function_name)


def _as_torch(value: Any) -> torch.Tensor:
    tensor = value.torch if hasattr(value, "torch") else value
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch-backed runtime value, got {type(value).__name__}.")
    return tensor


def _pd_torque_demand(
    joint_pos_target: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
) -> torch.Tensor:
    """Compute implicit-PD torque demand before actuator effort clipping.

    Args:
        joint_pos_target: Commanded joint positions [rad].
        joint_pos: Measured joint positions [rad].
        joint_vel: Measured joint velocities [rad/s].
        stiffness: Joint position gains [N·m/rad].
        damping: Joint velocity gains [N·m·s/rad].

    Returns:
        Unclipped joint torque demand [N·m].
    """
    return stiffness * (joint_pos_target - joint_pos) - damping * joint_vel


def _failure_term_functions(env: Any) -> dict[str, tuple[Callable, dict[str, Any]]]:
    missing = sorted(set(_FAILURE_TERMS) - set(env.termination_manager.active_terms))
    if missing:
        raise RuntimeError(f"Resolved task is missing required failure terminations: {missing}.")
    terms = {}
    for name in _FAILURE_TERMS:
        cfg = env.termination_manager.get_term_cfg(name)
        if not callable(cfg.func):
            raise TypeError(f"Resolved termination {name!r} is not callable: {cfg.func}.")
        terms[name] = (cfg.func, cfg.params)
    return terms


def _validate_runtime(env: Any) -> tuple[Any, Any, torch.Tensor, tuple[str, ...]]:
    if env.max_episode_length != _EXPECTED_EPISODE_STEPS:
        raise RuntimeError(
            f"The evaluator requires {_EXPECTED_EPISODE_STEPS} control steps, but the resolved task has "
            f"{env.max_episode_length}."
        )
    if env._physics_handles_decimation:
        raise RuntimeError("Per-substep torque measurement requires a physics backend with exposed decimation steps.")
    if env.recorder_manager.active_terms:
        raise RuntimeError("The evaluator does not support active recorder terms in the resolved task.")

    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    joint_names = tuple(robot.joint_names)
    action_joint_names = tuple(action_term._joint_names)
    if action_joint_names != joint_names:
        raise RuntimeError(
            "Action and articulation joint order differ, so per-joint torque results would be ambiguous: "
            f"action={action_joint_names}, articulation={joint_names}."
        )

    effort_limits = _as_torch(robot.data.joint_effort_limits)
    if effort_limits.shape != (env.num_envs, len(joint_names)):
        raise RuntimeError(
            f"Unexpected effort-limit shape {tuple(effort_limits.shape)}; expected {(env.num_envs, len(joint_names))}."
        )
    if not bool(torch.all(torch.isfinite(effort_limits) & (effort_limits > 0.0))):
        raise RuntimeError("Every resolved joint effort limit must be finite and positive.")
    if not bool(torch.allclose(effort_limits, effort_limits[0].expand_as(effort_limits))):
        raise RuntimeError("Per-environment effort limits differ; a single manifest-limit table would be invalid.")
    return robot, action_term, effort_limits, joint_names


def _policy_observation(observation_dict: dict[str, torch.Tensor], num_envs: int) -> TensorDict:
    return TensorDict(observation_dict, batch_size=[num_envs])


def _step_policy(
    env: Any,
    robot: Any,
    action: torch.Tensor,
    effort_limits: torch.Tensor,
    termination_functions: dict[str, tuple[Callable, dict[str, Any]]],
    termination_failures: dict[str, torch.Tensor],
    active_episodes: torch.Tensor,
    peak_torque_fraction: torch.Tensor,
    torque_sample_count: torch.Tensor,
    torque_saturation_count: torch.Tensor,
    torque_saturation_streak: torch.Tensor,
    torque_saturation_streak_max: torch.Tensor,
    peak_torque_demand_fraction: torch.Tensor,
    torque_demand_exceedance_count: torch.Tensor,
    torque_demand_exceedance_streak: torch.Tensor,
    torque_demand_exceedance_streak_max: torch.Tensor,
    peak_height: torch.Tensor,
    airborne_streak: torch.Tensor,
    airborne_streak_max: torch.Tensor,
    *,
    final_step: bool,
) -> dict[str, torch.Tensor] | None:
    env.action_manager.process_action(action.to(env.device))

    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        data = robot.data
        applied_torque = torch.abs(_as_torch(robot.data.applied_torque))
        current_fraction = applied_torque / effort_limits
        active_joints = active_episodes.unsqueeze(-1)
        peak_torque_fraction[:] = torch.where(
            active_joints,
            torch.maximum(peak_torque_fraction, current_fraction),
            peak_torque_fraction,
        )
        saturated = active_joints & (current_fraction >= 1.0 - _SATURATION_RELATIVE_TOLERANCE)
        torque_sample_count += active_episodes.to(dtype=torque_sample_count.dtype)
        torque_saturation_count += saturated.to(dtype=torque_saturation_count.dtype)
        torque_saturation_streak[:] = torch.where(
            saturated,
            torque_saturation_streak + 1,
            torch.zeros_like(torque_saturation_streak),
        )
        torque_saturation_streak_max[:] = torch.maximum(torque_saturation_streak_max, torque_saturation_streak)

        torque_demand = torch.abs(
            _pd_torque_demand(
                _as_torch(data.joint_pos_target),
                _as_torch(data.joint_pos),
                _as_torch(data.joint_vel),
                _as_torch(data.joint_stiffness),
                _as_torch(data.joint_damping),
            )
        )
        demand_fraction = torque_demand / effort_limits
        peak_torque_demand_fraction[:] = torch.where(
            active_joints,
            torch.maximum(peak_torque_demand_fraction, demand_fraction),
            peak_torque_demand_fraction,
        )
        demand_exceeded = active_joints & (demand_fraction >= 1.0 - _SATURATION_RELATIVE_TOLERANCE)
        torque_demand_exceedance_count += demand_exceeded.to(dtype=torque_demand_exceedance_count.dtype)
        torque_demand_exceedance_streak[:] = torch.where(
            demand_exceeded,
            torque_demand_exceedance_streak + 1,
            torch.zeros_like(torque_demand_exceedance_streak),
        )
        torque_demand_exceedance_streak_max[:] = torch.maximum(
            torque_demand_exceedance_streak_max, torque_demand_exceedance_streak
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        root_height = _as_torch(data.root_pos_w)[:, 2]
        peak_height[:] = torch.where(active_episodes, torch.maximum(peak_height, root_height), peak_height)
        airborne = active_episodes.clone()
        for sensor_name in FOOT_CONTACT_SENSOR_NAMES:
            forces = _as_torch(env.scene.sensors[sensor_name].data.net_forces_w)
            contact = torch.any(
                torch.linalg.vector_norm(forces, dim=-1).reshape(env.num_envs, -1) > _AIRBORNE_CONTACT_THRESHOLD_N,
                dim=1,
            )
            airborne &= ~contact
        airborne_streak[:] = torch.where(
            airborne,
            airborne_streak + 1,
            torch.zeros_like(airborne_streak),
        )
        airborne_streak_max[:] = torch.maximum(airborne_streak_max, airborne_streak)

    env.episode_length_buf += 1
    env.common_step_counter += 1
    step_failure = torch.zeros_like(active_episodes)
    for name, (term_function, params) in termination_functions.items():
        failure = term_function(env, **params)
        if failure.shape != (env.num_envs,) or failure.dtype != torch.bool:
            raise RuntimeError(
                f"Termination {name!r} returned shape={tuple(failure.shape)}, dtype={failure.dtype}; "
                f"expected {(env.num_envs,)}, torch.bool."
            )
        # Match the real environment: retain all terms that fire together on the first
        # failing step, but do not attribute counterfactual terms after an episode ended.
        failure &= active_episodes
        termination_failures[name] |= failure
        step_failure |= failure
    active_episodes &= ~step_failure

    if final_step:
        return None

    env.command_manager.compute(dt=env.step_dt)
    if "interval" in env.event_manager.available_modes:
        env.event_manager.apply(mode="interval", dt=env.step_dt)
    return env.observation_manager.compute(update_history=True)


def _final_tilt_error(root_quaternion: torch.Tensor) -> torch.Tensor:
    roll, pitch, _ = euler_xyz_from_quat(root_quaternion)
    roll_error = torch.atan2(torch.sin(roll - _REFERENCE_ROLL_RAD), torch.cos(roll - _REFERENCE_ROLL_RAD))
    pitch_error = torch.atan2(torch.sin(pitch - _REFERENCE_PITCH_RAD), torch.cos(pitch - _REFERENCE_PITCH_RAD))
    return torch.sqrt(torch.square(roll_error) + torch.square(pitch_error))


def _final_yaw_error(root_quaternion: torch.Tensor, target_quaternion: torch.Tensor) -> torch.Tensor:
    _, _, root_yaw = euler_xyz_from_quat(root_quaternion)
    _, _, target_yaw = euler_xyz_from_quat(target_quaternion)
    return torch.abs(torch.atan2(torch.sin(root_yaw - target_yaw), torch.cos(root_yaw - target_yaw)))


def _collect_batch(
    env: Any,
    policy: Callable,
    policy_reset: Callable[[torch.Tensor], None],
    clip_actions: float | None,
    robot: Any,
    effort_limits: torch.Tensor,
    termination_functions: dict[str, tuple[Callable, dict[str, Any]]],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    observation_dict, _ = env.reset(seed=seed)
    reset_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    policy_reset(reset_mask)
    observation = _policy_observation(observation_dict, env.num_envs)

    peak_torque_fraction = torch.zeros_like(effort_limits)
    torque_sample_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    torque_saturation_count = torch.zeros_like(effort_limits, dtype=torch.int64)
    torque_saturation_streak = torch.zeros_like(effort_limits, dtype=torch.int64)
    torque_saturation_streak_max = torch.zeros_like(effort_limits, dtype=torch.int64)
    peak_torque_demand_fraction = torch.zeros_like(effort_limits)
    torque_demand_exceedance_count = torch.zeros_like(effort_limits, dtype=torch.int64)
    torque_demand_exceedance_streak = torch.zeros_like(effort_limits, dtype=torch.int64)
    torque_demand_exceedance_streak_max = torch.zeros_like(effort_limits, dtype=torch.int64)
    peak_height = _as_torch(robot.data.root_pos_w)[:, 2].clone()
    airborne_streak = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
    airborne_streak_max = torch.zeros_like(airborne_streak)
    active_episodes = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    termination_failures = {
        name: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool) for name in _FAILURE_TERMS
    }

    # no_grad rather than inference_mode: this runs once per batch, and inference_mode marks
    # every tensor it creates -- including the observation manager's history buffers -- as an
    # inference tensor, which the NEXT batch's env.reset() then cannot write to in place.
    with torch.no_grad():
        for step in range(_EXPECTED_EPISODE_STEPS):
            action = policy(observation)
            expected_shape = (env.num_envs, len(robot.joint_names))
            if not isinstance(action, torch.Tensor) or action.shape != expected_shape:
                raise RuntimeError(
                    f"Policy returned {type(action).__name__} with shape {getattr(action, 'shape', None)}; "
                    f"expected {expected_shape}."
                )
            if not bool(torch.all(torch.isfinite(action))):
                raise RuntimeError("Policy returned a non-finite action.")
            if clip_actions is not None:
                action = torch.clamp(action, -clip_actions, clip_actions)

            next_observation = _step_policy(
                env,
                robot,
                action,
                effort_limits,
                termination_functions,
                termination_failures,
                active_episodes,
                peak_torque_fraction,
                torque_sample_count,
                torque_saturation_count,
                torque_saturation_streak,
                torque_saturation_streak_max,
                peak_torque_demand_fraction,
                torque_demand_exceedance_count,
                torque_demand_exceedance_streak,
                torque_demand_exceedance_streak_max,
                peak_height,
                airborne_streak,
                airborne_streak_max,
                final_step=step + 1 == _EXPECTED_EPISODE_STEPS,
            )
            if next_observation is not None:
                observation = _policy_observation(next_observation, env.num_envs)

    command_term = env.command_manager.get_term("jump_goal")
    root_position = _as_torch(robot.data.root_pos_w)
    root_quaternion = _as_torch(robot.data.root_quat_w)
    target_position = command_term.pose_command_w[:, :3]
    target_quaternion = command_term.pose_command_w[:, 3:7]
    command_origin_xy = target_position[:, :2] - command_term.target_displacement_w
    selection = slice(0, sample_count)

    def cpu(value: torch.Tensor) -> np.ndarray:
        return value[selection].detach().cpu().numpy()

    return {
        "goal_displacement_xy": cpu(command_term.target_displacement_w),
        "final_displacement_xy": cpu(root_position[:, :2] - command_origin_xy),
        "goal_distance": cpu(torch.linalg.vector_norm(command_term.target_displacement_w, dim=-1)),
        "goal_yaw_magnitude": cpu(torch.abs(command_term.target_yaw_displacement_w)),
        "final_height": cpu(root_position[:, 2]),
        "final_tilt_error": cpu(_final_tilt_error(root_quaternion)),
        "final_goal_error": cpu(torch.linalg.vector_norm(root_position[:, :2] - target_position[:, :2], dim=-1)),
        "final_yaw_error": cpu(_final_yaw_error(root_quaternion, target_quaternion)),
        "peak_height": cpu(peak_height),
        "maximum_airborne_time": cpu(airborne_streak_max) * env.physics_dt,
        "termination_failures": {name: cpu(values) for name, values in termination_failures.items()},
        "peak_torque_fraction": cpu(peak_torque_fraction),
        "torque_saturation_fraction": cpu(
            torque_saturation_count / torch.clamp(torque_sample_count.unsqueeze(-1), min=1)
        ),
        "torque_saturation_streak": cpu(torque_saturation_streak_max),
        "peak_torque_demand_fraction": cpu(peak_torque_demand_fraction),
        "torque_demand_exceedance_fraction": cpu(
            torque_demand_exceedance_count / torch.clamp(torque_sample_count.unsqueeze(-1), min=1)
        ),
        "torque_demand_exceedance_streak": cpu(torque_demand_exceedance_streak_max),
    }


def _evaluate_scale(
    env: Any,
    policy: Callable,
    policy_reset: Callable[[torch.Tensor], None],
    clip_actions: float | None,
    robot: Any,
    effort_limits: torch.Tensor,
    joint_names: tuple[str, ...],
    termination_functions: dict[str, tuple[Callable, dict[str, Any]]],
    base_ranges: dict[str, tuple[float, float]],
    range_scale: float,
    episodes: int,
    seed: int,
) -> EvaluationResult:
    scaled_ranges = _scaled_goal_ranges(base_ranges, range_scale, _AXIS_SCALES)
    command_term = env.command_manager.get_term("jump_goal")
    _write_goal_ranges(env.cfg.commands.jump_goal.ranges, scaled_ranges)
    if command_term.cfg.ranges is not env.cfg.commands.jump_goal.ranges:
        _write_goal_ranges(command_term.cfg.ranges, scaled_ranges)

    batches: list[dict[str, Any]] = []
    remaining = episodes
    batch_index = 0
    while remaining > 0:
        count = min(remaining, env.num_envs)
        batches.append(
            _collect_batch(
                env,
                policy,
                policy_reset,
                clip_actions,
                robot,
                effort_limits,
                termination_functions,
                count,
                seed + batch_index,
            )
        )
        remaining -= count
        batch_index += 1

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([batch[name] for batch in batches], axis=0)

    termination_failures = {
        name: np.concatenate([batch["termination_failures"][name] for batch in batches], axis=0)
        for name in _FAILURE_TERMS
    }
    return EvaluationResult(
        range_scale=range_scale,
        scaled_ranges=scaled_ranges,
        goal_displacement_xy=concatenate("goal_displacement_xy"),
        final_displacement_xy=concatenate("final_displacement_xy"),
        goal_distance=concatenate("goal_distance"),
        goal_yaw_magnitude=concatenate("goal_yaw_magnitude"),
        final_height=concatenate("final_height"),
        final_tilt_error=concatenate("final_tilt_error"),
        final_goal_error=concatenate("final_goal_error"),
        final_yaw_error=concatenate("final_yaw_error"),
        peak_height=concatenate("peak_height"),
        maximum_airborne_time=concatenate("maximum_airborne_time"),
        termination_failures=termination_failures,
        peak_torque_fraction=concatenate("peak_torque_fraction"),
        torque_saturation_fraction=concatenate("torque_saturation_fraction"),
        torque_saturation_streak=concatenate("torque_saturation_streak"),
        peak_torque_demand_fraction=concatenate("peak_torque_demand_fraction"),
        torque_demand_exceedance_fraction=concatenate("torque_demand_exceedance_fraction"),
        torque_demand_exceedance_streak=concatenate("torque_demand_exceedance_streak"),
        joint_names=joint_names,
        effort_limits=effort_limits[0].detach().cpu().numpy(),
        physics_dt=env.physics_dt,
    )


def _success_masks(
    result: EvaluationResult,
    height_floor: float,
    tilt_limit_rad: float,
    yaw_limit_rad: float,
    minimum_airborne_time_s: float = 0.0,
) -> dict[float, np.ndarray]:
    no_termination = ~np.logical_or.reduce(tuple(result.termination_failures.values()))
    airborne = (
        np.ones_like(result.final_height, dtype=np.bool_)
        if minimum_airborne_time_s <= 0.0
        else result.maximum_airborne_time >= minimum_airborne_time_s
    )
    common_success = (
        no_termination
        & (result.final_height > height_floor)
        & (result.final_tilt_error <= tilt_limit_rad)
        & (result.final_yaw_error <= yaw_limit_rad)
        & airborne
    )
    return {tolerance: common_success & (result.final_goal_error < tolerance) for tolerance in _GOAL_TOLERANCES_M}


def _wilson_interval(success_count: int, sample_count: int) -> tuple[float, float]:
    proportion = success_count / sample_count
    z_squared = _CONFIDENCE_Z**2
    denominator = 1.0 + z_squared / sample_count
    center = (proportion + z_squared / (2.0 * sample_count)) / denominator
    radius = (
        _CONFIDENCE_Z
        * math.sqrt(proportion * (1.0 - proportion) / sample_count + z_squared / (4.0 * sample_count**2))
        / denominator
    )
    return center - radius, center + radius


def _bin_edges(maximum: float, bin_count: int) -> np.ndarray:
    if maximum <= np.finfo(np.float64).eps:
        return np.linspace(0.0, 1.0e-9, bin_count + 1)
    return np.linspace(0.0, maximum, bin_count + 1)


def _bin_rates(
    values: np.ndarray, success_masks: dict[float, np.ndarray], edges: np.ndarray
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    bin_ids = np.digitize(values, edges[1:-1], right=False)
    counts = np.bincount(bin_ids, minlength=len(edges) - 1)
    rates = {}
    for tolerance, success in success_masks.items():
        success_counts = np.bincount(bin_ids, weights=success.astype(np.float64), minlength=len(edges) - 1)
        rates[tolerance] = np.divide(
            success_counts,
            counts,
            out=np.full(len(counts), np.nan, dtype=np.float64),
            where=counts > 0,
        )
    return counts, rates


def _crossing_magnitude(edges: np.ndarray, counts: np.ndarray, rates: np.ndarray, threshold: float) -> str:
    centers = 0.5 * (edges[:-1] + edges[1:])
    populated = counts > 0
    centers = centers[populated]
    rates = rates[populated]
    if len(centers) == 0:
        return "n/a"
    if rates[0] < threshold:
        return f"<{centers[0]:.3f}"
    for index in range(1, len(centers)):
        previous_rate = rates[index - 1]
        current_rate = rates[index]
        if previous_rate >= threshold > current_rate:
            rate_span = previous_rate - current_rate
            blend = 0.0 if rate_span <= 0.0 else (previous_rate - threshold) / rate_span
            crossing = centers[index - 1] + blend * (centers[index] - centers[index - 1])
            return f"{crossing:.3f}"
    return f">{centers[-1]:.3f}"


def _print_ranges(base_ranges: dict[str, tuple[float, float]], scaled_ranges: dict[str, tuple[float, float]]) -> None:
    print("Resolved goal ranges (base -> evaluated):")
    for name in _GOAL_RANGE_NAMES:
        base = base_ranges[name]
        scaled = scaled_ranges[name]
        unit = "rad" if name in ("roll", "pitch", "yaw") else "m"
        print(f"  {name:7s} [{base[0]: .4f}, {base[1]: .4f}] -> [{scaled[0]: .4f}, {scaled[1]: .4f}] {unit}")


def _print_success_rates(success_masks: dict[float, np.ndarray]) -> None:
    print("\nOverall success (95% Wilson binomial confidence interval):")
    print(f"{'goal tol [m]':>12s} {'successes':>10s} {'samples':>9s} {'rate':>9s} {'95% CI':>19s}")
    for tolerance, success in success_masks.items():
        success_count = int(np.count_nonzero(success))
        lower, upper = _wilson_interval(success_count, len(success))
        print(
            f"{tolerance:12.2f} {success_count:10d} {len(success):9d} "
            f"{100.0 * success_count / len(success):8.2f}% "
            f"[{100.0 * lower:6.2f}%, {100.0 * upper:6.2f}%]"
        )


def _print_command_response(result: EvaluationResult) -> None:
    fit = _fit_command_response(result.goal_displacement_xy, result.final_displacement_xy)
    response = np.where(np.isfinite(fit.response_matrix), np.char.mod("%+.3f", fit.response_matrix), "n/a")
    correlation = tuple("n/a" if not math.isfinite(value) else f"{value:+.3f}" for value in fit.axis_correlation)

    print("\nSettled displacement response to the commanded displacement:")
    print("  [landed_x]   [gain_xx gain_xy] [command_x]   [bias_x]")
    print("  [landed_y] = [gain_yx gain_yy] [command_y] + [bias_y]")
    print(f"  response matrix = [[{response[0, 0]}, {response[0, 1]}], [{response[1, 0]}, {response[1, 1]}]]")
    if np.any(~np.isfinite(fit.response_matrix)):
        print("  n/a denotes a command axis that was not independently excited")
    print(f"  offset [m] = [{fit.offset_xy[0]:+.4f}, {fit.offset_xy[1]:+.4f}]")
    print(f"  same-axis Pearson correlation = [x {correlation[0]}, y {correlation[1]}]")
    print(
        f"  mean absolute tracking error [m] = [x {fit.mean_absolute_tracking_error_xy[0]:.4f}, "
        f"y {fit.mean_absolute_tracking_error_xy[1]:.4f}]"
    )
    print(
        "  planar tracking-error norm: "
        f"p50={fit.tracking_error_norm_percentiles[0]:.4f} m, "
        f"p90={fit.tracking_error_norm_percentiles[1]:.4f} m, "
        f"p95={fit.tracking_error_norm_percentiles[2]:.4f} m"
    )


def _print_stability_summary(
    result: EvaluationResult,
    height_floor: float,
    tilt_limit_rad: float,
    minimum_airborne_time_s: float,
) -> None:
    """Report standing-up separately from landing on target.

    The two questions come apart for this task. ``task_completion_error`` and
    ``foot_tracking_error`` are ACCURACY terminations -- the first fires during STAND when
    position or yaw error exceeds a threshold -- so counting them as failures conflates
    missing the goal with falling over. Every episode here runs all 152 steps without a
    reset, so the final pose is genuinely observed even for episodes that tripped a
    termination, and the two can be separated.
    """
    upright = (result.final_height > height_floor) & (result.final_tilt_error <= tilt_limit_rad)
    fell = result.termination_failures["base_contact"] | result.termination_failures["bad_orientation"]
    total = result.sample_count

    print("\nStability, independent of goal accuracy:")
    print(
        f"  upright at episode end (height > {height_floor:.2f} m, tilt <= "
        f"{math.degrees(tilt_limit_rad):.0f} deg): {upright.sum()}/{total} "
        f"({100.0 * upright.mean():.2f}%)"
    )
    print(f"  hard fall (base_contact or bad_orientation): {int(fell.sum())}/{total} ({100.0 * fell.mean():.2f}%)")
    completed_jump = result.maximum_airborne_time >= minimum_airborne_time_s
    print(
        f"  airborne for >= {minimum_airborne_time_s:.3f} s: {int(completed_jump.sum())}/{total} "
        f"({100.0 * completed_jump.mean():.2f}%)"
    )
    print(
        "  maximum airborne time: "
        f"p50={np.percentile(result.maximum_airborne_time, 50.0):.3f} s, "
        f"p05={np.percentile(result.maximum_airborne_time, 5.0):.3f} s; "
        f"peak pelvis height p50={np.percentile(result.peak_height, 50.0):.3f} m"
    )

    print("\n  Of the episodes that FAIL each goal tolerance, how many still end upright:")
    print(f"  {'goal tol [m]':>12} {'failures':>9} {'upright':>9} {'upright %':>11}")
    for tolerance in _GOAL_TOLERANCES_M:
        missed = result.final_goal_error >= tolerance
        if not missed.any():
            continue
        still_up = upright & missed
        print(
            f"  {tolerance:>12.2f} {int(missed.sum()):>9} {int(still_up.sum()):>9} "
            f"{100.0 * still_up.sum() / missed.sum():>10.2f}%"
        )


def _print_failure_histogram(
    result: EvaluationResult,
    height_floor: float,
    tilt_limit_rad: float,
    yaw_limit_rad: float,
    minimum_airborne_time_s: float,
) -> None:
    print("\nFailure-mode histogram (non-exclusive episode counts):")
    entries = [(name, values) for name, values in result.termination_failures.items()]
    entries.extend(
        (
            (f"final_height <= {height_floor:.3f} m", result.final_height <= height_floor),
            (
                f"final_tilt_error > {math.degrees(tilt_limit_rad):.1f} deg",
                result.final_tilt_error > tilt_limit_rad,
            ),
            (
                f"final_yaw_error > {math.degrees(yaw_limit_rad):.1f} deg",
                result.final_yaw_error > yaw_limit_rad,
            ),
            (
                f"maximum_airborne_time < {minimum_airborne_time_s:.3f} s",
                result.maximum_airborne_time < minimum_airborne_time_s,
            ),
        )
    )
    entries.extend(
        (f"final_goal_error >= {tolerance:.2f} m", result.final_goal_error >= tolerance)
        for tolerance in _GOAL_TOLERANCES_M
    )
    for label, failures in entries:
        count = int(np.count_nonzero(failures))
        print(f"  {label:35s} {count:7d}  ({100.0 * count / result.sample_count:6.2f}%)")


def _print_binned_success(
    label: str,
    unit: str,
    values: np.ndarray,
    edges: np.ndarray,
    success_masks: dict[float, np.ndarray],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    counts, rates = _bin_rates(values, success_masks, edges)
    tolerance_headers = " ".join(f"tol={tolerance:.2f}" for tolerance in _GOAL_TOLERANCES_M)
    print(f"\nSuccess binned by {label} [{unit}]:")
    print(f"{'bin':>17s} {'samples':>8s} {tolerance_headers}")
    for index, count in enumerate(counts):
        interval = f"[{edges[index]:.3f}, {edges[index + 1]:.3f}{']' if index + 1 == len(counts) else ')'}"
        rate_values = " ".join(
            f"{100.0 * rates[tolerance][index]:8.2f}%" if math.isfinite(rates[tolerance][index]) else f"{'n/a':>9s}"
            for tolerance in _GOAL_TOLERANCES_M
        )
        print(f"{interval:>17s} {count:8d} {rate_values}")
    return counts, rates


def _print_crossings(
    distance_edges: np.ndarray,
    distance_counts: np.ndarray,
    distance_rates: dict[float, np.ndarray],
    yaw_edges_deg: np.ndarray,
    yaw_counts: np.ndarray,
    yaw_rates: dict[float, np.ndarray],
) -> None:
    print("\nFirst downward success-rate crossing (linear interpolation of populated bin centers):")
    print(f"{'goal tol':>9s} {'target':>8s} {'distance [m]':>14s} {'|dyaw| [deg]':>15s}")
    for tolerance in _GOAL_TOLERANCES_M:
        for threshold in (0.95, 0.99):
            distance = _crossing_magnitude(distance_edges, distance_counts, distance_rates[tolerance], threshold)
            yaw = _crossing_magnitude(yaw_edges_deg, yaw_counts, yaw_rates[tolerance], threshold)
            print(f"{tolerance:9.2f} {100.0 * threshold:7.0f}% {distance:>14s} {yaw:>15s}")


def _print_torque_distribution(result: EvaluationResult) -> None:
    percentiles = np.percentile(result.peak_torque_fraction, (50.0, 90.0, 95.0, 99.0), axis=0)
    maxima = np.max(result.peak_torque_fraction, axis=0)
    saturated = result.peak_torque_fraction >= 1.0 - _SATURATION_RELATIVE_TOLERANCE
    saturation_rates = np.mean(saturated, axis=0)
    episode_saturation = np.any(saturated, axis=1)

    print("\nPeak applied torque / resolved actuator effort limit by episode:")
    print(
        f"{'joint':30s} {'limit N.m':>10s} {'p50':>8s} {'p90':>8s} {'p95':>8s} "
        f"{'p99':>8s} {'max':>8s} {'episodes sat.':>14s}"
    )
    for index, name in enumerate(result.joint_names):
        print(
            f"{name:30s} {result.effort_limits[index]:10.2f} "
            f"{100.0 * percentiles[0, index]:7.1f}% {100.0 * percentiles[1, index]:7.1f}% "
            f"{100.0 * percentiles[2, index]:7.1f}% {100.0 * percentiles[3, index]:7.1f}% "
            f"{100.0 * maxima[index]:7.1f}% {100.0 * saturation_rates[index]:13.2f}%"
        )
    saturated_count = int(np.count_nonzero(episode_saturation))
    lower, upper = _wilson_interval(saturated_count, result.sample_count)
    print(
        f"Episodes with any saturated joint: {saturated_count}/{result.sample_count} "
        f"({100.0 * saturated_count / result.sample_count:.2f}%, "
        f"95% CI [{100.0 * lower:.2f}%, {100.0 * upper:.2f}%])"
    )

    duty_percentiles = np.percentile(result.torque_saturation_fraction, (50.0, 95.0), axis=0)
    duty_maxima = np.max(result.torque_saturation_fraction, axis=0)
    streak_ms = 1000.0 * result.physics_dt * result.torque_saturation_streak
    streak_percentiles = np.percentile(streak_ms, (50.0, 95.0), axis=0)
    streak_maxima = np.max(streak_ms, axis=0)
    print("\nApplied-torque saturation duration by episode:")
    print(
        f"{'joint':30s} {'duty p50':>9s} {'duty p95':>9s} {'duty max':>9s} "
        f"{'streak p50':>12s} {'streak p95':>12s} {'streak max':>12s}"
    )
    for index, name in enumerate(result.joint_names):
        print(
            f"{name:30s} {100.0 * duty_percentiles[0, index]:8.2f}% "
            f"{100.0 * duty_percentiles[1, index]:8.2f}% {100.0 * duty_maxima[index]:8.2f}% "
            f"{streak_percentiles[0, index]:10.1f} ms {streak_percentiles[1, index]:10.1f} ms "
            f"{streak_maxima[index]:10.1f} ms"
        )


def _print_torque_demand_distribution(result: EvaluationResult) -> None:
    percentiles = np.percentile(result.peak_torque_demand_fraction, (50.0, 90.0, 95.0, 99.0), axis=0)
    maxima = np.max(result.peak_torque_demand_fraction, axis=0)
    exceeded = result.peak_torque_demand_fraction >= 1.0 - _SATURATION_RELATIVE_TOLERANCE
    exceedance_rates = np.mean(exceeded, axis=0)
    episode_exceedance = np.any(exceeded, axis=1)

    print("\nPeak unclipped PD torque demand / resolved actuator effort limit by episode:")
    print(
        f"{'joint':30s} {'limit N.m':>10s} {'p50':>8s} {'p90':>8s} {'p95':>8s} "
        f"{'p99':>8s} {'max':>8s} {'episodes over':>14s}"
    )
    for index, name in enumerate(result.joint_names):
        print(
            f"{name:30s} {result.effort_limits[index]:10.2f} "
            f"{100.0 * percentiles[0, index]:7.1f}% {100.0 * percentiles[1, index]:7.1f}% "
            f"{100.0 * percentiles[2, index]:7.1f}% {100.0 * percentiles[3, index]:7.1f}% "
            f"{100.0 * maxima[index]:7.1f}% {100.0 * exceedance_rates[index]:13.2f}%"
        )
    exceeded_count = int(np.count_nonzero(episode_exceedance))
    lower, upper = _wilson_interval(exceeded_count, result.sample_count)
    print(
        f"Episodes with any over-limit torque demand: {exceeded_count}/{result.sample_count} "
        f"({100.0 * exceeded_count / result.sample_count:.2f}%, "
        f"95% CI [{100.0 * lower:.2f}%, {100.0 * upper:.2f}%])"
    )

    duty_percentiles = np.percentile(result.torque_demand_exceedance_fraction, (50.0, 95.0), axis=0)
    duty_maxima = np.max(result.torque_demand_exceedance_fraction, axis=0)
    streak_ms = 1000.0 * result.physics_dt * result.torque_demand_exceedance_streak
    streak_percentiles = np.percentile(streak_ms, (50.0, 95.0), axis=0)
    streak_maxima = np.max(streak_ms, axis=0)
    print("\nOver-limit torque-demand duration by episode:")
    print(
        f"{'joint':30s} {'duty p50':>9s} {'duty p95':>9s} {'duty max':>9s} "
        f"{'streak p50':>12s} {'streak p95':>12s} {'streak max':>12s}"
    )
    for index, name in enumerate(result.joint_names):
        print(
            f"{name:30s} {100.0 * duty_percentiles[0, index]:8.2f}% "
            f"{100.0 * duty_percentiles[1, index]:8.2f}% {100.0 * duty_maxima[index]:8.2f}% "
            f"{streak_percentiles[0, index]:10.1f} ms {streak_percentiles[1, index]:10.1f} ms "
            f"{streak_maxima[index]:10.1f} ms"
        )


def _print_result(
    result: EvaluationResult,
    base_ranges: dict[str, tuple[float, float]],
    height_floor: float,
    tilt_limit_rad: float,
    yaw_limit_rad: float,
    minimum_airborne_time_s: float,
    condition: str,
) -> dict[float, np.ndarray]:
    print("\n" + "=" * 104)
    print(
        f"Condition: {condition}; range scale {result.range_scale:.2f}; "
        f"{result.sample_count} full {_EXPECTED_EPISODE_STEPS}-step episodes"
    )
    _print_ranges(base_ranges, result.scaled_ranges)
    success_masks = _success_masks(
        result,
        height_floor,
        tilt_limit_rad,
        yaw_limit_rad,
        minimum_airborne_time_s,
    )
    _print_success_rates(success_masks)
    _print_command_response(result)
    _print_stability_summary(result, height_floor, tilt_limit_rad, minimum_airborne_time_s)
    yaw_percentiles = np.percentile(np.degrees(result.final_yaw_error), (50.0, 90.0, 95.0, 99.0))
    print(
        "\nFinal commanded-yaw absolute error: "
        f"p50={yaw_percentiles[0]:.2f} deg, p90={yaw_percentiles[1]:.2f} deg, "
        f"p95={yaw_percentiles[2]:.2f} deg, p99={yaw_percentiles[3]:.2f} deg, "
        f"max={math.degrees(float(np.max(result.final_yaw_error))):.2f} deg"
    )
    _print_failure_histogram(
        result,
        height_floor,
        tilt_limit_rad,
        yaw_limit_rad,
        minimum_airborne_time_s,
    )

    max_distance = math.hypot(
        max(abs(value) for value in result.scaled_ranges["pos_x"]),
        max(abs(value) for value in result.scaled_ranges["pos_y"]),
    )
    distance_edges = _bin_edges(max_distance, 5)
    distance_counts, distance_rates = _print_binned_success(
        "commanded distance", "m", result.goal_distance, distance_edges, success_masks
    )

    max_yaw_deg = math.degrees(max(abs(value) for value in result.scaled_ranges["yaw"]))
    yaw_edges_deg = _bin_edges(max_yaw_deg, 4)
    yaw_counts, yaw_rates = _print_binned_success(
        "absolute commanded yaw", "deg", np.degrees(result.goal_yaw_magnitude), yaw_edges_deg, success_masks
    )
    _print_crossings(
        distance_edges,
        distance_counts,
        distance_rates,
        yaw_edges_deg,
        yaw_counts,
        yaw_rates,
    )
    _print_torque_distribution(result)
    _print_torque_demand_distribution(result)
    return success_masks


def _print_sweep_summary(
    results: list[EvaluationResult], success_by_scale: list[dict[float, np.ndarray]], condition: str
) -> None:
    print("\n" + "=" * 104)
    print(f"RANGE-SCALE SWEEP SUMMARY — {condition}")
    headers = " ".join(f"tol={tolerance:.2f}" for tolerance in _GOAL_TOLERANCES_M)
    print(f"{'scale':>7s} {'samples':>8s} {headers} {'any torque sat.':>16s}")
    for result, success_masks in zip(results, success_by_scale):
        success_rates = " ".join(
            f"{100.0 * np.mean(success_masks[tolerance]):8.2f}%" for tolerance in _GOAL_TOLERANCES_M
        )
        any_saturation = np.any(result.peak_torque_fraction >= 1.0 - _SATURATION_RELATIVE_TOLERANCE, axis=1)
        print(
            f"{result.range_scale:7.2f} {result.sample_count:8d} {success_rates} "
            f"{100.0 * np.mean(any_saturation):15.2f}%"
        )


def _make_policy_reset(runner: Any, policy: Any, installed_rsl_rl_version: str) -> Callable[[torch.Tensor], None]:
    if version.parse(installed_rsl_rl_version) >= version.parse("4.0.0"):
        return policy.reset
    policy_network = (
        runner.alg.policy
        if version.parse(installed_rsl_rl_version) >= version.parse("2.3.0")
        else runner.alg.actor_critic
    )
    return policy_network.reset


def main() -> None:
    parser = _create_parser()
    args, hydra_args = setup_preset_cli(parser)
    _validate_args(args)
    global _AXIS_SCALES
    _AXIS_SCALES = _parse_axis_scale(args.axis_scale)
    sys.argv = [sys.argv[0]] + hydra_args

    env_cfg, agent_cfg = resolve_task_config(args.task, _AGENT_CFG_ENTRY_POINT)
    base_ranges = _read_goal_ranges(env_cfg)
    deterministic_overrides = _configure_evaluation(env_cfg, args)
    goal_feedback = _configure_goal_feedback(env_cfg, args.goal_feedback)
    installed_rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    agent_cfg.seed = args.seed
    if args.device is not None:
        agent_cfg.device = args.device
    checkpoint_path = Path(retrieve_file_path(args.checkpoint)).resolve()
    env_cfg.log_dir = str(checkpoint_path.parent)
    range_scales = _SWEEP_RANGE_SCALES if args.sweep_range_scale else (args.range_scale,)
    _write_goal_ranges(
        env_cfg.commands.jump_goal.ranges,
        _scaled_goal_ranges(base_ranges, range_scales[0], _AXIS_SCALES),
    )

    condition = (
        "deterministic (configured randomization disabled)"
        if args.no_randomization
        else "configured task randomization enabled"
    )
    print(f"Evaluation condition: {condition}")
    print(f"Actor goal feedback: {goal_feedback}")
    print("Shared evaluation override: reset_to_reference.init_start_prob=0.0")
    if deterministic_overrides:
        print("Disabled on the resolved environment copy:")
        for override in deterministic_overrides:
            print(f"  - {override}")
    print(f"Task: {args.task}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Success requires no {_FAILURE_TERMS}, final pelvis height > {args.height_floor:.3f} m, "
        f"tilt error <= {args.tilt_limit_deg:.1f} deg relative to "
        f"roll={math.degrees(_REFERENCE_ROLL_RAD):.2f} deg/pitch={math.degrees(_REFERENCE_PITCH_RAD):.2f} deg, "
        f"absolute yaw error <= {args.yaw_tolerance_deg:.1f} deg, and final horizontal goal error below "
        f"the reported tolerance, after at least {args.minimum_airborne_time_s:.3f} s continuously airborne."
    )

    with launch_simulation(env_cfg, args):
        gym_env = gym.make(args.task, cfg=env_cfg)
        rl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        try:
            print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(rl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            if args.no_randomization:
                configure_seed(args.seed, True)
            runner.load(str(checkpoint_path))
            policy = runner.get_inference_policy(device=rl_env.unwrapped.device)
            policy_reset = _make_policy_reset(runner, policy, installed_rsl_rl_version)

            env = rl_env.unwrapped
            robot, _, effort_limits, joint_names = _validate_runtime(env)
            termination_functions = _failure_term_functions(env)
            results = []
            success_by_scale = []
            for range_scale in range_scales:
                result = _evaluate_scale(
                    env,
                    policy,
                    policy_reset,
                    agent_cfg.clip_actions,
                    robot,
                    effort_limits,
                    joint_names,
                    termination_functions,
                    base_ranges,
                    range_scale,
                    args.episodes,
                    args.seed,
                )
                results.append(result)
                success_by_scale.append(
                    _print_result(
                        result,
                        base_ranges,
                        args.height_floor,
                        math.radians(args.tilt_limit_deg),
                        math.radians(args.yaw_tolerance_deg),
                        args.minimum_airborne_time_s,
                        condition,
                    )
                )
            if args.sweep_range_scale:
                _print_sweep_summary(results, success_by_scale, condition)
        finally:
            rl_env.close()


if __name__ == "__main__":
    main()

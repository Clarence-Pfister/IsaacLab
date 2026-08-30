# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the G1 jump task, including reference-state initialization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import normalize, quat_from_euler_xyz, quat_mul

from ..constants import REFERENCE_DURATION_S, REFERENCE_MOTION_FPS
from .motion import get_loader, warp_to_torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def reference_state_initialization(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    init_start_prob: float = 0.2,
    roll_range: tuple[float, float] = (0.0, 0.0),
    pitch_range: tuple[float, float] = (0.0, 0.0),
    lin_vel_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reset the robot from the reference motion or its initial frame.

    Args:
        env: Environment in which to reset the robot.
        env_ids: Environment indices to reset.
        asset_cfg: Robot asset configuration.
        init_start_prob: Probability of initializing from a random reference frame.
        roll_range: Roll perturbation range [rad] added to the reference attitude.
        pitch_range: Pitch perturbation range [rad] added to the reference attitude.
        lin_vel_range: Per-axis root linear velocity perturbation range [m/s].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    loader = get_loader(env)

    def randomize_root_state(root_pose: torch.Tensor, root_velocity: torch.Tensor) -> None:
        if roll_range != (0.0, 0.0) or pitch_range != (0.0, 0.0):
            roll = root_pose.new_empty(len(root_pose)).uniform_(*roll_range)
            pitch = root_pose.new_empty(len(root_pose)).uniform_(*pitch_range)
            attitude_offset = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
            root_pose[:, 3:7] = normalize(quat_mul(root_pose[:, 3:7], attitude_offset))
        if lin_vel_range != (0.0, 0.0):
            root_velocity[:, :3] += root_velocity.new_empty((len(root_velocity), 3)).uniform_(*lin_vel_range)

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
        randomize_root_state(init_root_pose, init_root_vel)

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
        randomize_root_state(default_root_pose, default_root_vel)

        asset.write_joint_position_to_sim_index(position=default_joint_pos, env_ids=std_env_ids)
        asset.write_joint_velocity_to_sim_index(velocity=default_joint_vel, env_ids=std_env_ids)
        asset.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=std_env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=std_env_ids)


def _terminal_state_is_retriggerable(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_velocity: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_pos_limits: torch.Tensor,
    *,
    root_height_range: tuple[float, float],
    max_tilt_rad: float,
    max_root_linear_speed: float,
    max_root_angular_speed: float,
    max_joint_speed: float,
    joint_limit_margin: float,
    joint_limit_tolerance: float,
) -> torch.Tensor:
    """Return which terminal states are safe candidates for another simulated jump."""
    sample_count = root_pos.shape[0]
    expected_shapes = {
        "root_pos": (sample_count, 3),
        "root_quat": (sample_count, 4),
        "root_velocity": (sample_count, 6),
        "joint_velocity": joint_pos.shape,
        "joint_pos_limits": (*joint_pos.shape, 2),
    }
    actual_shapes = {
        "root_pos": tuple(root_pos.shape),
        "root_quat": tuple(root_quat.shape),
        "root_velocity": tuple(root_velocity.shape),
        "joint_velocity": tuple(joint_velocity.shape),
        "joint_pos_limits": tuple(joint_pos_limits.shape),
    }
    for name, expected_shape in expected_shapes.items():
        if actual_shapes[name] != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {actual_shapes[name]}.")
    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be two-dimensional, got shape {tuple(joint_pos.shape)}.")

    quat_norm = torch.linalg.vector_norm(root_quat, dim=-1)
    normalized_quat = root_quat / quat_norm.clamp_min(torch.finfo(root_quat.dtype).eps).unsqueeze(-1)
    # Root quaternions use XYZW order. The world-Z component of body Z is
    # 1 - 2 * (qx**2 + qy**2), independent of yaw.
    body_z_world_z = 1.0 - 2.0 * torch.sum(torch.square(normalized_quat[:, :2]), dim=-1)
    tilt = torch.acos(torch.clamp(body_z_world_z, -1.0, 1.0))

    finite = (
        torch.isfinite(root_pos).all(dim=-1)
        & torch.isfinite(root_quat).all(dim=-1)
        & torch.isfinite(root_velocity).all(dim=-1)
        & torch.isfinite(joint_pos).all(dim=-1)
        & torch.isfinite(joint_velocity).all(dim=-1)
        & torch.isfinite(joint_pos_limits).all(dim=(-1, -2))
        & (quat_norm > torch.finfo(root_quat.dtype).eps)
    )
    height_safe = (root_pos[:, 2] >= root_height_range[0]) & (root_pos[:, 2] <= root_height_range[1])
    tilt_safe = tilt <= max_tilt_rad
    root_linear_speed_safe = torch.linalg.vector_norm(root_velocity[:, :3], dim=-1) <= max_root_linear_speed
    root_angular_speed_safe = torch.linalg.vector_norm(root_velocity[:, 3:], dim=-1) <= max_root_angular_speed
    joint_speed_safe = torch.max(torch.abs(joint_velocity), dim=-1).values <= max_joint_speed
    joint_position_safe = torch.all(
        (joint_pos >= joint_pos_limits[..., 0] + joint_limit_margin - joint_limit_tolerance)
        & (joint_pos <= joint_pos_limits[..., 1] - joint_limit_margin + joint_limit_tolerance),
        dim=-1,
    )
    return (
        finite
        & height_safe
        & tilt_safe
        & root_linear_speed_safe
        & root_angular_speed_safe
        & joint_speed_safe
        & joint_position_safe
    )


def _full_episode_mask(
    elapsed_steps: torch.Tensor,
    previous_retrigger: torch.Tensor,
    step_dt: float,
    terminal_hold_duration_s: float,
    retrigger_prepare_duration_s: float,
    fresh_prepare_duration_s: float,
) -> torch.Tensor:
    """Return episodes that ran their complete configured reference clock.

    Elapsed global steps are measured from the last real reset, rather than the
    environment episode counter. RSL-RL randomizes that counter at the start of
    training, which must not make a shortened initial episode retriggerable.

    Args:
        elapsed_steps: Policy steps elapsed from each environment's last reset.
        previous_retrigger: Whether each episode began from a carried landing.
        step_dt: Policy control period [s].
        terminal_hold_duration_s: Final-reference hold duration [s].
        retrigger_prepare_duration_s: Phase-zero preparation duration for
            carried episodes [s].
        fresh_prepare_duration_s: Phase-zero preparation duration for fresh
            episodes [s].

    Returns:
        Boolean completion mask with the same shape as
        :paramref:`elapsed_steps`.

    Raises:
        ValueError: If tensor shapes differ or :paramref:`step_dt` is not
            finite and positive.
    """
    if elapsed_steps.shape != previous_retrigger.shape:
        raise ValueError(
            "elapsed_steps and previous_retrigger must have the same shape, "
            f"got {elapsed_steps.shape} and {previous_retrigger.shape}."
        )
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError(f"step_dt must be finite and positive, got {step_dt}.")
    minimum_normal_steps = math.ceil(
        (REFERENCE_DURATION_S + terminal_hold_duration_s + fresh_prepare_duration_s) / step_dt
    )
    minimum_retrigger_steps = math.ceil(
        (REFERENCE_DURATION_S + terminal_hold_duration_s + retrigger_prepare_duration_s) / step_dt
    )
    minimum_episode_steps = torch.where(
        previous_retrigger,
        minimum_retrigger_steps,
        minimum_normal_steps,
    )
    return elapsed_steps >= minimum_episode_steps


def _sample_retrigger_mask(
    previous_retrigger: torch.Tensor,
    retrigger_probability: float,
    retrigger_after_retrigger_probability: float | None = None,
) -> torch.Tensor:
    """Sample which eligible terminal states should be carried forward.

    Args:
        previous_retrigger: Whether each completed episode began from a
            carried landing.
        retrigger_probability: Carry probability after a fresh episode.
        retrigger_after_retrigger_probability: Optional carry probability
            after an episode that was itself carried. ``None`` uses
            :paramref:`retrigger_probability` for both cases.

    Returns:
        Boolean sample mask with the same shape as
        :paramref:`previous_retrigger`.

    Raises:
        ValueError: If either probability is outside ``[0, 1]``.
    """
    after_retrigger_probability = (
        retrigger_probability
        if retrigger_after_retrigger_probability is None
        else retrigger_after_retrigger_probability
    )
    probabilities = (retrigger_probability, after_retrigger_probability)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError(f"Retrigger probabilities must be finite and lie in [0, 1], got {probabilities}.")
    probability = torch.full(
        previous_retrigger.shape,
        retrigger_probability,
        device=previous_retrigger.device,
        dtype=torch.float32,
    )
    probability = torch.where(
        previous_retrigger.bool(),
        probability.new_full((), after_retrigger_probability),
        probability,
    )
    return torch.rand_like(probability) < probability


def reference_or_terminal_state_initialization(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    retrigger_probability: float = 0.5,
    retrigger_after_retrigger_probability: float | None = None,
    init_start_prob: float = 0.0,
    root_height_range: tuple[float, float] = (0.65, 0.9),
    max_tilt_rad: float = 0.15,
    max_root_linear_speed: float = 0.3,
    max_root_angular_speed: float = 0.5,
    max_joint_speed: float = 2.0,
    joint_limit_margin: float = 0.0,
    joint_limit_tolerance: float = 0.001,
    use_soft_joint_limits: bool = True,
    zero_retrigger_velocity: bool = False,
    terminal_hold_duration_s: float = 0.0,
    retrigger_prepare_duration_s: float = 0.0,
    fresh_prepare_duration_s: float = 0.0,
) -> None:
    """Reset normally or continue from a safe terminal state with phase reset to zero.

    Only environments that ran for a complete reference duration and ended by timeout,
    without another termination, can carry their physical state into the next episode.
    This excludes the trainer's randomized initial episode lengths and creates real
    command sequences: the command manager samples a new body-relative goal after this
    event, while the stateless policy restarts at phase zero from its own prior landing.

    Args:
        env: Environment in which to reset the robot.
        env_ids: Environment indices to reset.
        asset_cfg: Robot asset configuration.
        retrigger_probability: Probability of carrying an eligible timeout
            state after a fresh episode.
        retrigger_after_retrigger_probability: Optional probability of carrying
            an eligible timeout state after an episode that was itself carried.
            ``None`` uses :paramref:`retrigger_probability`.
        init_start_prob: Probability of random reference-state initialization for states
            that are not carried into the next episode.
        root_height_range: Permitted root-height interval [m].
        max_tilt_rad: Maximum root tilt from world vertical [rad].
        max_root_linear_speed: Maximum root linear speed [m/s].
        max_root_angular_speed: Maximum root angular speed [rad/s].
        max_joint_speed: Maximum absolute joint speed [rad/s].
        joint_limit_margin: Required distance from every joint limit [rad].
        joint_limit_tolerance: Numerical tolerance beyond the modeled joint limits [rad].
        use_soft_joint_limits: Whether to evaluate carried states against the
            training soft limits instead of the physical articulation limits.
        zero_retrigger_velocity: Whether to restart eligible terminal poses at rest,
            matching a settled stand between commanded jumps.
        terminal_hold_duration_s: Final-reference hold included in the preceding
            episode's expected full duration [s].
        retrigger_prepare_duration_s: Duration for which a carried episode holds
            phase zero before advancing its reference clock [s].
        fresh_prepare_duration_s: Duration for which a frame-zero reset holds
            phase zero before advancing its reference clock [s]. This requires
            :paramref:`init_start_prob` to be zero.

    Raises:
        ValueError: If a probability, range, or safety threshold is invalid.
    """
    if not 0.0 <= init_start_prob <= 1.0:
        raise ValueError(f"init_start_prob must lie in [0, 1], got {init_start_prob}.")
    if not isinstance(use_soft_joint_limits, bool):
        raise ValueError("use_soft_joint_limits must be a boolean.")
    if not 0.0 < root_height_range[0] < root_height_range[1]:
        raise ValueError(f"root_height_range must be positive and ordered, got {root_height_range}.")
    thresholds = {
        "max_tilt_rad": max_tilt_rad,
        "max_root_linear_speed": max_root_linear_speed,
        "max_root_angular_speed": max_root_angular_speed,
        "max_joint_speed": max_joint_speed,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds.values()):
        raise ValueError(f"Retrigger safety thresholds must be finite and positive, got {thresholds}.")
    limit_thresholds = {
        "joint_limit_margin": joint_limit_margin,
        "joint_limit_tolerance": joint_limit_tolerance,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in limit_thresholds.values()):
        raise ValueError(f"Joint-limit thresholds must be finite and non-negative, got {limit_thresholds}.")
    durations = {
        "terminal_hold_duration_s": terminal_hold_duration_s,
        "retrigger_prepare_duration_s": retrigger_prepare_duration_s,
        "fresh_prepare_duration_s": fresh_prepare_duration_s,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in durations.values()):
        raise ValueError(f"Retrigger durations must be finite and non-negative, got {durations}.")
    if fresh_prepare_duration_s > 0.0 and init_start_prob > 0.0:
        raise ValueError("fresh_prepare_duration_s requires init_start_prob=0 so every reset starts at frame zero.")

    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    root_pose = asset.data.root_link_pose_w.torch[env_ids].clone()
    root_velocity = asset.data.root_link_vel_w.torch[env_ids].clone()
    joint_pos = asset.data.joint_pos.torch[env_ids].clone()
    joint_velocity = asset.data.joint_vel.torch[env_ids].clone()
    joint_limit_data = asset.data.soft_joint_pos_limits if use_soft_joint_limits else asset.data.joint_pos_limits
    joint_pos_limits = joint_limit_data.torch[env_ids]

    safe = _terminal_state_is_retriggerable(
        root_pose[:, :3],
        root_pose[:, 3:],
        root_velocity,
        joint_pos,
        joint_velocity,
        joint_pos_limits,
        root_height_range=root_height_range,
        max_tilt_rad=max_tilt_rad,
        max_root_linear_speed=max_root_linear_speed,
        max_root_angular_speed=max_root_angular_speed,
        max_joint_speed=max_joint_speed,
        joint_limit_margin=joint_limit_margin,
        joint_limit_tolerance=joint_limit_tolerance,
    )
    timeout = getattr(env, "reset_time_outs", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))[
        env_ids
    ].bool()
    terminated = getattr(env, "reset_terminated", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))[
        env_ids
    ].bool()
    current_step = int(env.common_step_counter)
    previous_retrigger = getattr(
        env,
        "retrigger_reset_mask",
        torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
    )[env_ids]
    if not hasattr(env, "retrigger_last_reset_step"):
        env.retrigger_last_reset_step = torch.full(
            (env.num_envs,),
            current_step,
            device=env.device,
            dtype=torch.long,
        )
        full_episode = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)
    else:
        elapsed_steps = current_step - env.retrigger_last_reset_step[env_ids]
        full_episode = _full_episode_mask(
            elapsed_steps,
            previous_retrigger,
            env.step_dt,
            terminal_hold_duration_s,
            retrigger_prepare_duration_s,
            fresh_prepare_duration_s,
        )
    sampled = _sample_retrigger_mask(
        previous_retrigger,
        retrigger_probability,
        retrigger_after_retrigger_probability,
    )
    retrigger = safe & timeout & ~terminated & full_episode & sampled
    retrigger_env_ids = env_ids[retrigger]
    reset_env_ids = env_ids[~retrigger]

    if len(reset_env_ids) > 0:
        reference_state_initialization(
            env,
            reset_env_ids,
            asset_cfg,
            init_start_prob=init_start_prob,
        )
        if fresh_prepare_duration_s > 0.0:
            env.start_times[reset_env_ids] = -fresh_prepare_duration_s

    if len(retrigger_env_ids) > 0:
        carried_root_pose = root_pose[retrigger]
        carried_root_pose[:, :2] = warp_to_torch(env.scene.env_origins)[retrigger_env_ids, :2]
        carried_root_velocity = root_velocity[retrigger]
        carried_joint_pos = joint_pos[retrigger]
        carried_joint_velocity = joint_velocity[retrigger]
        if zero_retrigger_velocity:
            carried_root_velocity = torch.zeros_like(carried_root_velocity)
            carried_joint_velocity = torch.zeros_like(carried_joint_velocity)

        if not hasattr(env, "start_times"):
            env.start_times = torch.zeros(env.num_envs, device=env.device)
        env.start_times[retrigger_env_ids] = -retrigger_prepare_duration_s
        asset.write_joint_position_to_sim_index(position=carried_joint_pos, env_ids=retrigger_env_ids)
        asset.write_joint_velocity_to_sim_index(velocity=carried_joint_velocity, env_ids=retrigger_env_ids)
        asset.write_root_pose_to_sim_index(root_pose=carried_root_pose, env_ids=retrigger_env_ids)
        asset.write_root_link_velocity_to_sim_index(
            root_velocity=carried_root_velocity,
            env_ids=retrigger_env_ids,
        )

    if not hasattr(env, "retrigger_reset_mask"):
        env.retrigger_reset_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.retrigger_reset_mask[env_ids] = retrigger
    env.retrigger_last_reset_step[env_ids] = current_step


def randomize_contact_compliance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    stiffness_range: tuple[float, float],
    damping_ratio_range: tuple[float, float] = (0.7, 1.4),
    rigid_probability: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Give each environment's collision shapes a randomly sampled compliant contact spring.

    PhysX contacts are rigid by default, so a policy trained without this term only ever meets one
    contact stiffness. Real floors, shoe soles and force plates are not rigid, and the two
    simulators used for sim-to-sim disagree on contact stiffness by an order of magnitude, so the
    stiffness is sampled per environment and held for the run. Damping follows the sampled ratio
    against the environment's own total mass, ``2 * ratio * sqrt(stiffness * mass)``.

    .. attention::
        PhysX stores the compliant spring in the material's restitution slot, so an environment
        given a spring no longer carries a randomized restitution, and any later write through
        :func:`~isaaclab.envs.mdp.randomize_rigid_body_material` erases the spring. Order this term
        after that one, and leave a fraction of environments rigid through
        :paramref:`rigid_probability` if restitution randomization still matters.

    Args:
        env: Environment whose asset is randomized.
        env_ids: Environment indices to randomize. If ``None``, randomizes every environment.
        stiffness_range: Contact stiffness sampling range per collision shape, log-uniform [N/m].
        damping_ratio_range: Contact damping ratio sampling range, 1.0 being critically damped.
        rigid_probability: Fraction of environments left rigid, keeping their randomized restitution.
        asset_cfg: Asset whose collision shapes are randomized.

    Raises:
        ValueError: If a sampling range is not positive and ordered.
        RuntimeError: If the physics backend does not expose compliant contact materials.
    """
    import warp as wp

    if not 0.0 < stiffness_range[0] <= stiffness_range[1]:
        raise ValueError(f"Contact stiffness range must be positive and ordered. Got {stiffness_range}.")
    if not 0.0 < damping_ratio_range[0] <= damping_ratio_range[1]:
        raise ValueError(f"Contact damping ratio range must be positive and ordered. Got {damping_ratio_range}.")
    if not 0.0 <= rigid_probability <= 1.0:
        raise ValueError(f"Rigid probability must lie in [0, 1]. Got {rigid_probability}.")

    asset: Articulation = env.scene[asset_cfg.name]
    view = asset.root_view
    if not hasattr(view, "set_compliant_material_properties"):
        raise RuntimeError(
            "Compliant contact randomization requires a physics backend that exposes compliant "
            f"contact materials; {type(view).__name__} does not."
        )

    materials = wp.to_torch(view.get_material_properties())
    compliant = wp.to_torch(view.get_compliant_material_properties()[0])
    count, num_shapes = compliant.shape[0], compliant.shape[1]
    device = compliant.device

    if env_ids is None:
        env_ids = torch.arange(count, dtype=torch.int32, device=device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.int32, device=device).reshape(-1)
    keep_rigid = torch.rand(len(env_ids), device=device) < rigid_probability
    compliant_env_ids = env_ids[~keep_rigid]
    if len(compliant_env_ids) == 0:
        return

    log_low, log_high = math.log(stiffness_range[0]), math.log(stiffness_range[1])
    stiffness = torch.exp(torch.empty(len(compliant_env_ids), device=device).uniform_(log_low, log_high))
    ratio = torch.empty(len(compliant_env_ids), device=device).uniform_(*damping_ratio_range)
    mass = warp_to_torch(asset.data.default_mass).to(device).sum(dim=1)[compliant_env_ids.long()]
    damping = 2.0 * ratio * torch.sqrt(stiffness * mass)

    data = torch.zeros((count, num_shapes, 4), dtype=torch.float32, device=device)
    data[..., 0:2] = materials[..., 0:2]
    data[compliant_env_ids.long(), :, 2] = stiffness.unsqueeze(1).float()
    data[compliant_env_ids.long(), :, 3] = damping.unsqueeze(1).float()
    # Minimum keeps the softer of the two contacting materials in charge, so the sampled spring
    # governs against the rigid ground plane. Friction keeps PhysX's default average.
    combine = torch.ones((count, num_shapes, 3), dtype=torch.uint8, device=device)
    combine[..., 0] = 0
    view.set_compliant_material_properties(
        wp.from_torch(data.contiguous(), dtype=wp.float32),
        wp.from_torch(combine.contiguous(), dtype=wp.uint8),
        wp.from_torch(compliant_env_ids.contiguous(), dtype=wp.int32),
    )

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settle a MuJoCo robot at static equilibrium before starting a rollout."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

_FORCE_RELATIVE_TOLERANCE = 0.03
_LINEAR_SPEED_TOLERANCE_M_S = 1.0e-2
_ANGULAR_SPEED_TOLERANCE_RAD_S = 1.0e-2
_CONVERGED_DURATION_S = 0.05
_MAX_DURATION_S = 2.0


@dataclass(frozen=True)
class StaticEquilibriumResult:
    """Measurements from a successful static-equilibrium settle."""

    duration_s: float
    static_weight_n: float
    foot_contact_force_n: float
    root_linear_velocity_m_s: np.ndarray
    root_angular_velocity_rad_s: np.ndarray

    @property
    def root_linear_speed_m_s(self) -> float:
        """Residual root linear speed [m/s]."""
        return float(np.linalg.norm(self.root_linear_velocity_m_s))

    @property
    def root_angular_speed_rad_s(self) -> float:
        """Residual root angular speed [rad/s]."""
        return float(np.linalg.norm(self.root_angular_velocity_rad_s))


def ground_contact_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ground_geom_id: int,
    robot_geom_ids: frozenset[int],
) -> np.ndarray:
    """Return the net ground force on selected robot geoms in world axes [N]."""
    result = np.zeros(3, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if ground_geom_id not in (geom1, geom2):
            continue
        wrench_contact = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_id, wrench_contact)
        # Contact-frame axes are rows. The reported force points from geom1 to
        # geom2, so it acts with that sign on geom2 and oppositely on geom1.
        force_world = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3).T @ wrench_contact[:3]
        if geom2 in robot_geom_ids:
            result += force_world
        elif geom1 in robot_geom_ids:
            result -= force_world
    return result


def settle_static_equilibrium(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    root_dof_adr: int,
    joint_qpos_adr: np.ndarray,
    joint_dof_adr: np.ndarray,
    actuator_ids: np.ndarray,
    reference_joint_pos: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    effort_limit: np.ndarray,
    ground_geom_id: int,
    foot_geom_ids: frozenset[int],
    use_implicit_pd: bool = False,
) -> StaticEquilibriumResult:
    """Hold the reference pose under joint PD until the floating base is at rest.

    The convergence conditions must hold continuously for 50 ms. On success,
    the settled state is retained and only the MuJoCo clock is reset to zero.

    Args:
        model: Compiled MuJoCo model.
        data: Mutable MuJoCo simulation state.
        root_dof_adr: First floating-base velocity address.
        joint_qpos_adr: Controlled joint position addresses.
        joint_dof_adr: Controlled joint velocity addresses.
        actuator_ids: Controlled actuator IDs in the same order as the joint arrays.
        reference_joint_pos: Joint PD targets [rad].
        stiffness: Joint proportional gains [N·m/rad].
        damping: Joint derivative gains [N·m·s/rad].
        effort_limit: Symmetric joint effort limits [N·m].
        ground_geom_id: Ground collision geometry ID.
        foot_geom_ids: Foot collision geometry IDs.
        use_implicit_pd: Whether :paramref:`data.ctrl` carries position targets
            for solver-side PD instead of explicit torque commands.

    Returns:
        Final equilibrium measurements. The simulation clock has been reset to zero.

    Raises:
        RuntimeError: If equilibrium is not reached within 2 s.
    """
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    gravity_magnitude = float(np.linalg.norm(gravity))
    static_weight = float(mujoco.mj_getTotalmass(model)) * gravity_magnitude
    if not math.isfinite(static_weight) or static_weight <= 0.0:
        raise ValueError("Static-equilibrium settling requires positive finite gravity and model mass.")
    support_direction = -gravity / gravity_magnitude

    timestep = float(model.opt.timestep)
    stable_steps_required = int(math.ceil(_CONVERGED_DURATION_S / timestep))
    maximum_steps = int(math.floor(_MAX_DURATION_S / timestep + 1.0e-12))
    stable_steps = 0
    foot_contact_force = 0.0
    root_linear_velocity = np.zeros(3, dtype=np.float64)
    root_angular_velocity = np.zeros(3, dtype=np.float64)

    for step in range(1, maximum_steps + 1):
        if use_implicit_pd:
            data.ctrl[actuator_ids] = reference_joint_pos
        else:
            joint_positions = np.asarray(data.qpos[joint_qpos_adr], dtype=np.float64)
            joint_velocities = np.asarray(data.qvel[joint_dof_adr], dtype=np.float64)
            torque = stiffness * (reference_joint_pos - joint_positions) - damping * joint_velocities
            data.ctrl[actuator_ids] = np.clip(torque, -effort_limit, effort_limit)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        root_linear_velocity = np.asarray(data.qvel[root_dof_adr : root_dof_adr + 3], dtype=np.float64).copy()
        root_angular_velocity = np.asarray(data.qvel[root_dof_adr + 3 : root_dof_adr + 6], dtype=np.float64).copy()
        foot_force_world = ground_contact_force(model, data, ground_geom_id, foot_geom_ids)
        foot_contact_force = float(np.dot(foot_force_world, support_direction))
        finite = bool(
            math.isfinite(foot_contact_force) and np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        )
        force_converged = abs(foot_contact_force - static_weight) <= _FORCE_RELATIVE_TOLERANCE * static_weight
        velocity_converged = (
            np.linalg.norm(root_linear_velocity) <= _LINEAR_SPEED_TOLERANCE_M_S
            and np.linalg.norm(root_angular_velocity) <= _ANGULAR_SPEED_TOLERANCE_RAD_S
        )
        stable_steps = stable_steps + 1 if finite and force_converged and velocity_converged else 0
        if stable_steps >= stable_steps_required:
            result = StaticEquilibriumResult(
                duration_s=step * timestep,
                static_weight_n=static_weight,
                foot_contact_force_n=foot_contact_force,
                root_linear_velocity_m_s=root_linear_velocity,
                root_angular_velocity_rad_s=root_angular_velocity,
            )
            data.time = 0.0
            mujoco.mj_forward(model, data)
            return result

    linear_speed = float(np.linalg.norm(root_linear_velocity))
    angular_speed = float(np.linalg.norm(root_angular_velocity))
    raise RuntimeError(
        "MuJoCo failed to reach static equilibrium within "
        f"{_MAX_DURATION_S:.3f} s: foot contact force={foot_contact_force:.3f} N "
        f"(static weight={static_weight:.3f} N, tolerance={100.0 * _FORCE_RELATIVE_TOLERANCE:.1f}%), "
        f"root linear speed={linear_speed:.6f} m/s "
        f"(limit={_LINEAR_SPEED_TOLERANCE_M_S:.6f} m/s), "
        f"root angular speed={angular_speed:.6f} rad/s "
        f"(limit={_ANGULAR_SPEED_TOLERANCE_RAD_S:.6f} rad/s)."
    )

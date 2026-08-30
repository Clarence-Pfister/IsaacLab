# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply Isaac-compatible actuator and joint properties to compiled MuJoCo models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class PhysicsParityConfig:
    """Switches for independently measurable Isaac physics corrections."""

    use_implicit_pd: bool = True
    zero_passive_forces: bool = True
    zero_frictionloss: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PhysicsParityConfig:
        """Build parity settings from parsed command-line arguments."""
        return cls(
            use_implicit_pd=bool(getattr(args, "use_implicit_pd", True)),
            zero_passive_forces=bool(getattr(args, "zero_passive_forces", True)),
            zero_frictionloss=bool(getattr(args, "zero_frictionloss", True)),
        )

    def metadata(self) -> dict[str, bool]:
        """Return JSON-serializable correction settings."""
        return asdict(self)


def add_physics_parity_arguments(parser: argparse.ArgumentParser) -> None:
    """Add default-on, independently switchable physics-parity arguments."""
    group = parser.add_argument_group("Isaac physics parity")

    implicit_pd = group.add_mutually_exclusive_group()
    implicit_pd.add_argument(
        "--use_implicit_pd",
        dest="use_implicit_pd",
        action="store_true",
        help="Use solver-side MuJoCo position-actuator PD (default).",
    )
    implicit_pd.add_argument(
        "--no_use_implicit_pd",
        dest="use_implicit_pd",
        action="store_false",
        help="Use the legacy explicit Python torque-PD path.",
    )

    passive_forces = group.add_mutually_exclusive_group()
    passive_forces.add_argument(
        "--zero_passive_forces",
        dest="zero_passive_forces",
        action="store_true",
        help="Zero compiled-model joint stiffness and damping (default).",
    )
    passive_forces.add_argument(
        "--no_zero_passive_forces",
        dest="zero_passive_forces",
        action="store_false",
        help="Retain passive joint stiffness and damping from the MJCF.",
    )

    frictionloss = group.add_mutually_exclusive_group()
    frictionloss.add_argument(
        "--zero_frictionloss",
        dest="zero_frictionloss",
        action="store_true",
        help="Zero compiled-model Coulomb joint frictionloss (default).",
    )
    frictionloss.add_argument(
        "--no_zero_frictionloss",
        dest="zero_frictionloss",
        action="store_false",
        help="Retain joint frictionloss from the MJCF.",
    )
    parser.set_defaults(use_implicit_pd=True, zero_passive_forces=True, zero_frictionloss=True)


def configure_implicit_pd(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    effort_limit: np.ndarray,
) -> None:
    """Configure position actuators for solver-side PD in policy order.

    Args:
        model: Compiled MuJoCo model.
        actuator_ids: Actuator IDs corresponding to the policy-order arrays.
        stiffness: Joint proportional gains [N·m/rad].
        damping: Joint derivative gains [N·m·s/rad].
        effort_limit: Symmetric joint effort limits [N·m].
    """
    count = len(actuator_ids)
    if any(np.asarray(values).shape != (count,) for values in (stiffness, damping, effort_limit)):
        raise ValueError("Implicit-PD gains and limits must match the actuator ID array.")

    affine_bias = int(mujoco.mjtBias.mjBIAS_AFFINE)
    for policy_index, actuator_id_value in enumerate(actuator_ids):
        actuator_id = int(actuator_id_value)
        kp = float(stiffness[policy_index])
        kd = float(damping[policy_index])
        effort = float(effort_limit[policy_index])
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biastype[actuator_id] = affine_bias
        model.actuator_biasprm[actuator_id, 0] = 0.0
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_forcerange[actuator_id] = (-effort, effort)
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_ctrllimited[actuator_id] = 0


def apply_physics_parity(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    effort_limit: np.ndarray,
    config: PhysicsParityConfig,
    *,
    print_status: bool = True,
) -> None:
    """Apply selected Isaac physics corrections to a compiled MuJoCo model."""
    if config.zero_passive_forces:
        model.jnt_stiffness[:] = 0.0
        model.dof_damping[:] = 0.0
    if config.zero_frictionloss:
        model.dof_frictionloss[:] = 0.0
    if config.use_implicit_pd:
        configure_implicit_pd(model, actuator_ids, stiffness, damping, effort_limit)

    if print_status:
        print("MuJoCo physics parity corrections:")
        print(f"  implicit PD: {'APPLIED' if config.use_implicit_pd else 'DISABLED (explicit Python PD)'}")
        print(f"  zero passive stiffness/damping: {'APPLIED' if config.zero_passive_forces else 'DISABLED'}")
        print(f"  zero frictionloss: {'APPLIED' if config.zero_frictionloss else 'DISABLED'}")

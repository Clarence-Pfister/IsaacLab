# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the filtered joint position action.

Kept apart from the action class itself, mirroring the split upstream makes between
``actions_cfg.py`` and ``joint_actions.py``. Importing the action class binds
``JointPositionAction``, and resolving that name lazy-loads ``Articulation`` and with it USD.
Task configs are resolved before ``SimulationApp`` starts, and USD loaded that early aborts
the process, so only this config module may be reached from the package ``__init__``. The
class itself is loaded from the ``class_type`` string once the app is running.
"""

from __future__ import annotations

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils.configclass import configclass


@configclass
class LowPassJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for filtered joint position targets."""

    class_type: type | str = "{DIR}.actions:LowPassJointPositionAction"
    alpha: float | dict[str, float] = 1.0
    """New-target weight; lower values apply stronger low-pass filtering."""

    min_delay_steps: int = 0
    """Minimum action delay in policy control steps, sampled per environment on reset.

    One policy control step is 0.02 s at the 50 Hz policy rate.
    """

    max_delay_steps: int = 0
    """Maximum action delay in policy control steps, sampled per environment on reset.

    One policy control step is 0.02 s at the 50 Hz policy rate.
    """

    effort_limit_ratio: float | dict[str, float] | None = None
    """Available fraction of actuator effort used to project position targets.

    When configured, the action term recomputes the nearest admissible target at each
    physics step so its instantaneous implicit-PD demand remains inside the envelope.
    A dictionary maps joint-name expressions to ratios; unmatched joints use ``1.0``.
    ``None`` disables projection.
    """

    lower_limit_velocity_lookahead: dict[str, float] | None = None
    """Lower-limit braking lookahead by joint-name expression [s].

    At every physics step, a configured joint target is raised to at least
    ``clip_lower + lookahead * max(-joint_velocity, 0)``. This provides an
    explicit stopping margin before a rapidly extending joint reaches its lower
    command bound. ``None`` disables the projection.
    """

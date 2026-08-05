# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 reference-motion jump task.

Deliberately free of imports. ``isaaclab_tasks`` walks its subpackages when it is imported,
and ``utils/importer.py`` skips plain modules but imports every package, so anything
re-exported here is loaded as a side effect of ``import isaaclab_tasks``. The environment
config builds a ``SimulationCfg`` in a class body, which lazy-loads ``isaaclab.sim`` and with
it USD; loading USD before Kit starts aborts the process inside ``SimulationApp``. The gym
entry points are strings resolved after the app launches, so no re-export is needed.
"""

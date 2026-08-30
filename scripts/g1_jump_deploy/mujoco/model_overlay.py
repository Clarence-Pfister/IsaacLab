# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose the G1 jump MuJoCo overlay without modifying the source MJCF."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import mujoco


def apply_initial_ground_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_qpos_adr: int,
    ground_geom_id: int,
    foot_geom_ids: Iterable[int],
) -> float:
    """Raise the floating base until active foot collision geoms clear the ground.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo data containing the initialized robot pose.
        root_qpos_adr: Position address of the floating-base joint.
        ground_geom_id: Ground plane geom identifier.
        foot_geom_ids: Candidate foot collision geom identifiers.

    Returns:
        Applied vertical floating-base offset [m].

    Raises:
        ValueError: If the ground is not a plane or no supplied foot geom is
            collision-active against it.
        RuntimeError: If the correction does not remove the measured penetration.
    """
    if int(model.geom_type[ground_geom_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
        raise ValueError("Initial ground-clearance correction requires a plane ground geom.")

    ground_contype = int(model.geom_contype[ground_geom_id])
    ground_conaffinity = int(model.geom_conaffinity[ground_geom_id])
    active_foot_geom_ids = tuple(
        geom_id
        for geom_id in foot_geom_ids
        if (int(model.geom_contype[geom_id]) & ground_conaffinity)
        or (ground_contype & int(model.geom_conaffinity[geom_id]))
    )
    if not active_foot_geom_ids:
        raise ValueError("No supplied foot collision geom is active against the ground plane.")

    mujoco.mj_forward(model, data)
    signed_distances = tuple(
        mujoco.mj_geomDistance(model, data, ground_geom_id, geom_id, 1.0e6, None) for geom_id in active_foot_geom_ids
    )
    minimum_distance = min(signed_distances)
    ground_normal_z = float(data.geom_xmat[ground_geom_id].reshape(3, 3)[2, 2])
    if ground_normal_z <= 0.0:
        raise ValueError("Ground plane normal must have a positive world-Z component.")

    height_offset = max(0.0, -minimum_distance / ground_normal_z)
    data.qpos[root_qpos_adr + 2] += height_offset
    mujoco.mj_forward(model, data)

    if height_offset > 0.0:
        corrected_minimum_distance = min(
            mujoco.mj_geomDistance(model, data, ground_geom_id, geom_id, 1.0e6, None)
            for geom_id in active_foot_geom_ids
        )
        if not abs(corrected_minimum_distance) <= 1.0e-9:
            raise RuntimeError(
                "Initial ground-clearance correction left a residual minimum signed distance of "
                f"{corrected_minimum_distance:.3e} m."
            )
    return height_offset


def _required_child(root: ET.Element, tag: str, source: Path) -> ET.Element:
    child = root.find(tag)
    if child is None:
        raise ValueError(f"{source} is missing required <{tag}> element.")
    return child


def _numeric_value(root: ET.Element, name: str, source: Path) -> int:
    element = root.find(f"./custom/numeric[@name='{name}']")
    if element is None or "data" not in element.attrib:
        raise ValueError(f"{source} is missing custom numeric {name!r}.")
    value = float(element.attrib["data"])
    if not value.is_integer() or value < 0:
        raise ValueError(f"{source} custom numeric {name!r} must be a non-negative integer.")
    return int(value)


def compose_model_xml(model_path: Path, overlay_path: Path) -> tuple[str, float]:
    """Return source MJCF with the sim2sim overlay applied in memory.

    Args:
        model_path: Source robot MJCF path.
        overlay_path: Sim2sim overlay MJCF path.

    Returns:
        Composed XML text and its physics timestep [s].

    Raises:
        ValueError: If either document does not have the expected structure.
    """
    model_path = model_path.resolve()
    overlay_path = overlay_path.resolve()
    source_root = ET.parse(model_path).getroot()
    overlay_root = ET.parse(overlay_path).getroot()
    if source_root.tag != "mujoco" or overlay_root.tag != "mujoco":
        raise ValueError("The source model and overlay must both have <mujoco> roots.")
    if source_root.find("option") is not None:
        raise ValueError(
            f"{model_path} now contains <option>; audit and reconcile it with the overlay instead of "
            "silently replacing it."
        )

    option = _required_child(overlay_root, "option", overlay_path)
    try:
        timestep = float(option.attrib["timestep"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{overlay_path} must declare a numeric option timestep.") from exc
    if timestep <= 0.0:
        raise ValueError(f"Overlay timestep must be positive, got {timestep}.")

    compiler = _required_child(source_root, "compiler", model_path)
    meshdir = compiler.attrib.get("meshdir")
    if meshdir:
        mesh_path = Path(meshdir)
        if not mesh_path.is_absolute():
            mesh_path = model_path.parent / mesh_path
        compiler.set("meshdir", str(mesh_path.resolve()))

    # Top-level order is kept conventional even though MuJoCo accepts sections by tag.
    compiler_index = list(source_root).index(compiler)
    source_root.insert(compiler_index + 1, copy.deepcopy(option))

    source_worldbody = _required_child(source_root, "worldbody", model_path)
    overlay_worldbody = _required_child(overlay_root, "worldbody", overlay_path)
    overlay_geoms = list(overlay_worldbody.findall("geom"))
    if len(overlay_geoms) != 1:
        raise ValueError(f"{overlay_path} must contain exactly one top-level ground geom.")

    collision_mode = overlay_root.find("./custom/text[@name='sim2sim_collision_mode']")
    if collision_mode is None or collision_mode.attrib.get("data") != "robot_ground_only":
        raise ValueError(f"{overlay_path} must explicitly request robot_ground_only collision mode.")
    robot_contype = _numeric_value(overlay_root, "sim2sim_robot_contype", overlay_path)
    robot_conaffinity = _numeric_value(overlay_root, "sim2sim_robot_conaffinity", overlay_path)
    ground_contype = int(overlay_geoms[0].attrib.get("contype", "0"))
    ground_conaffinity = int(overlay_geoms[0].attrib.get("conaffinity", "0"))
    robot_self_matches = robot_contype & robot_conaffinity
    ground_matches = (robot_contype & ground_conaffinity) | (ground_contype & robot_conaffinity)
    if robot_self_matches or not ground_matches:
        raise ValueError("Overlay collision masks must reject robot/robot pairs and accept robot/ground pairs.")

    collidable_geom_count = 0
    for geom in source_worldbody.findall(".//geom"):
        contype = int(geom.attrib.get("contype", "1"))
        conaffinity = int(geom.attrib.get("conaffinity", "1"))
        if contype != 0 or conaffinity != 0:
            geom.set("contype", str(robot_contype))
            geom.set("conaffinity", str(robot_conaffinity))
            collidable_geom_count += 1
    if collidable_geom_count == 0:
        raise ValueError(f"{model_path} contains no collidable robot geoms.")

    source_worldbody.insert(0, copy.deepcopy(overlay_geoms[0]))
    overlay_custom = _required_child(overlay_root, "custom", overlay_path)
    source_custom = source_root.find("custom")
    if source_custom is None:
        source_root.append(copy.deepcopy(overlay_custom))
    else:
        for child in overlay_custom:
            source_custom.append(copy.deepcopy(child))

    return ET.tostring(source_root, encoding="unicode"), timestep

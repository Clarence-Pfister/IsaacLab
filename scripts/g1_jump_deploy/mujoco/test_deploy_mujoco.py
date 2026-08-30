# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for G1 MuJoCo deployment order remapping."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from deploy_mujoco import DeploymentManifest, _build_name_permutations
from isaac_policy_rollout import _create_parser as _create_isaac_rollout_parser
from physics_parity import PhysicsParityConfig, apply_physics_parity

_PARITY_MODEL_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="joint" type="hinge" damping="0.05" stiffness="0.1" frictionloss="0.2"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor name="motor" joint="joint"/></actuator>
</mujoco>
"""
_REPO_ROOT = Path(__file__).resolve().parents[3]
_V12_BUNDLE = _REPO_ROOT / "logs" / "bundle_translation_heading500"
_V15_BUNDLE = _REPO_ROOT / "logs" / "g1_jump_deploy_bundle_validated"


def test_isaac_rollout_defaults_to_manifest_goal_midpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["test", "--task", "task", "--manifest", "manifest.json"])
    parser = _create_isaac_rollout_parser()

    assert parser.get_default("goal_pos_x") is None


def test_name_permutations_scatter_and_gather() -> None:
    policy_names = ("joint_b", "joint_d", "joint_a", "joint_c")
    backend_names = ("joint_a", "joint_b", "joint_c", "joint_d")

    policy_from_backend, backend_from_policy = _build_name_permutations(policy_names, backend_names, "test backend")

    np.testing.assert_array_equal(policy_from_backend, (1, 3, 0, 2))
    np.testing.assert_array_equal(backend_from_policy, (2, 0, 3, 1))
    policy_values = np.asarray((10.0, 20.0, 30.0, 40.0))
    backend_values = policy_values[backend_from_policy]
    np.testing.assert_array_equal(backend_values, (30.0, 10.0, 40.0, 20.0))
    np.testing.assert_array_equal(backend_values[policy_from_backend], policy_values)


@pytest.mark.parametrize(
    ("backend_names", "match"),
    [
        (("joint_a", "joint_b", "joint_c", "joint_c"), "must be unique"),
        (("joint_a", "joint_b", "joint_c", "joint_e"), "Missing=.*joint_d.*extra=.*joint_e"),
    ],
)
def test_name_permutations_reject_non_bijections(backend_names: tuple[str, ...], match: str) -> None:
    policy_names = ("joint_a", "joint_b", "joint_c", "joint_d")

    with pytest.raises(ValueError, match=match):
        _build_name_permutations(policy_names, backend_names, "test backend")


@pytest.mark.parametrize("schema_version", ("1.0", "1.1"))
def test_deployment_manifest_requires_schema_v12(tmp_path: Path, schema_version: str) -> None:
    manifest_path = tmp_path / "deploy_manifest.json"
    manifest_path.write_text(f'{{"schema_version": "{schema_version}"}}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"schema 1\.2.*re-export"):
        DeploymentManifest(manifest_path)


def test_deployment_manifest_v13_requires_and_loads_torque_projection(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(_V12_BUNDLE, bundle)
    manifest_path = bundle / "deploy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.3"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="torque_projection"):
        DeploymentManifest(manifest_path)

    manifest["action"]["torque_projection"] = {
        "type": "instantaneous_pd",
        "period_s": 0.002,
        "effort_limit_ratio": [0.6] * 23,
        "formula": (
            "q_target = q + (clip(kp*(q_requested-q)-kd*dq, -ratio*effort_limit, ratio*effort_limit)+kd*dq)/kp"
        ),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = DeploymentManifest(manifest_path)

    np.testing.assert_array_equal(loaded.effort_limit_ratio, np.full(23, 0.6))


def test_deployment_manifest_v14_requires_and_loads_lower_limit_brake(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(_V12_BUNDLE, bundle)
    manifest_path = bundle / "deploy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.4"
    manifest["action"]["torque_projection"] = {
        "type": "instantaneous_pd",
        "period_s": 0.002,
        "effort_limit_ratio": [0.6] * 23,
        "formula": (
            "q_target = q + (clip(kp*(q_requested-q)-kd*dq, -ratio*effort_limit, ratio*effort_limit)+kd*dq)/kp"
        ),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="lower_limit_brake"):
        DeploymentManifest(manifest_path)

    clip = np.asarray(manifest["action"]["clip"], dtype=np.float64)
    lookahead = np.zeros(23)
    lookahead[11:13] = 0.028
    manifest["action"]["lower_limit_brake"] = {
        "type": "velocity_lookahead",
        "period_s": 0.002,
        "position_lower": clip[:, 0].tolist(),
        "position_upper": clip[:, 1].tolist(),
        "velocity_lookahead_s": lookahead.tolist(),
        "formula": ("q_requested = max(q_filtered, min(q_upper, q_lower + t_lookahead*max(-dq, 0)))"),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = DeploymentManifest(manifest_path)

    np.testing.assert_array_equal(loaded.brake_velocity_lookahead, lookahead)


def test_deployment_manifest_v15_requires_physical_position_limits(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(_V12_BUNDLE, bundle)
    manifest_path = bundle / "deploy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.5"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="position_limits"):
        DeploymentManifest(manifest_path)


def test_deployment_manifest_v16_accepts_retrigger_observation_contract(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(_V15_BUNDLE, bundle)
    manifest_path = bundle / "deploy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.6"
    manifest["goal"]["retrigger_indicator"] = {
        "mode": "goal_command_z",
        "fresh_value": 0.0,
        "retrigger_value": 0.25,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = DeploymentManifest(manifest_path)

    assert loaded.schema_version == "1.6"


def test_deployment_manifest_v17_accepts_affine_retrigger_goal_contract(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(_V15_BUNDLE, bundle)
    manifest_path = bundle / "deploy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.7"
    manifest["goal"]["retrigger_indicator"] = {
        "mode": "goal_command_z_affine_pos_x",
        "fresh_value": 0.0,
        "retrigger_value": 0.25,
        "goal_pos_x_scale": 1.5,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = DeploymentManifest(manifest_path)

    assert loaded.schema_version == "1.7"


def test_physics_parity_configures_compiled_model_and_realised_force() -> None:
    model = mujoco.MjModel.from_xml_string(_PARITY_MODEL_XML)
    apply_physics_parity(
        model,
        np.asarray((0,), dtype=np.int32),
        np.asarray((10.0,)),
        np.asarray((2.0,)),
        np.asarray((3.0,)),
        PhysicsParityConfig(),
        print_status=False,
    )

    np.testing.assert_array_equal(model.jnt_stiffness, 0.0)
    np.testing.assert_array_equal(model.dof_damping, 0.0)
    np.testing.assert_array_equal(model.dof_frictionloss, 0.0)
    assert model.actuator_gainprm[0, 0] == 10.0
    np.testing.assert_array_equal(model.actuator_biasprm[0, :3], (0.0, -10.0, -2.0))
    np.testing.assert_array_equal(model.actuator_forcerange[0], (-3.0, 3.0))
    assert model.actuator_forcelimited[0] == 1
    assert model.actuator_ctrllimited[0] == 0

    data = mujoco.MjData(model)
    data.qpos[0] = 0.1
    data.qvel[0] = 0.2
    data.ctrl[0] = 0.3
    mujoco.mj_forward(model, data)
    assert data.actuator_force[0] == pytest.approx(1.6)


def test_physics_parity_corrections_can_be_disabled_individually() -> None:
    model = mujoco.MjModel.from_xml_string(_PARITY_MODEL_XML)
    apply_physics_parity(
        model,
        np.asarray((0,), dtype=np.int32),
        np.asarray((10.0,)),
        np.asarray((2.0,)),
        np.asarray((3.0,)),
        PhysicsParityConfig(
            use_implicit_pd=False,
            zero_passive_forces=False,
            zero_frictionloss=False,
        ),
        print_status=False,
    )

    np.testing.assert_array_equal(model.jnt_stiffness, 0.1)
    np.testing.assert_array_equal(model.dof_damping, 0.05)
    np.testing.assert_array_equal(model.dof_frictionloss, 0.2)
    assert model.actuator_gainprm[0, 0] == 1.0

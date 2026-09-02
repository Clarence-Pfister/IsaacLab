# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for G1 jump MuJoCo log replay."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from model_overlay import compose_model_xml
from replay_log import detect_log_format, load_replay_log, run

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_MODEL = _REPO_ROOT / "data_storage" / "g1_23dof_holo_compat.xml"
_OVERLAY = _SCRIPT_DIR / "model_overlay.xml"


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    composed_xml, _ = compose_model_xml(_MODEL, _OVERLAY)
    return mujoco.MjModel.from_xml_string(composed_xml)


def _joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        name
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)) is not None
    ]


def _write_log(path: Path, format_name: str, model: mujoco.MjModel) -> tuple[str, float]:
    names = list(reversed(_joint_names(model)))
    marked_name = names[3]
    marked_value = 1.234
    qpos = np.zeros((3, len(names)), dtype=np.float64)
    qpos[:, 3] = marked_value
    root_pose = np.tile(np.asarray((0.1, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0)), (3, 1))
    metadata = {"goal": {"pos_x": 0.1}}
    arrays: dict[str, np.ndarray] = {
        "time": np.asarray((0.0, 0.002, 0.004)),
        "qpos": qpos,
    }
    if format_name == "deploy_mujoco":
        metadata.update({"mujoco_joint_names": names, "phase_names": ["IDLE", "FLIGHT"]})
        arrays.update({"pelvis_pose": root_pose, "phase": np.asarray((0, 1, 1))})
    elif format_name == "run_fsm_mujoco":
        metadata["joint_names"] = names
        arrays.update({"pelvis_pose": root_pose, "fsm_state": np.asarray(("STAND", "JUMP", "JUMP"))})
    else:
        metadata["joint_names"] = names
        arrays.update(
            {
                "pelvis_position": root_pose[:, :3],
                "pelvis_quaternion_wxyz": root_pose[:, 3:],
                "fsm_id": np.asarray((801, 801, 1)),
                "user_control": np.asarray((False, True, False)),
                "fixture_active": np.asarray((True, False, False)),
            }
        )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    return marked_name, marked_value


@pytest.mark.parametrize("format_name", ("deploy_mujoco", "run_fsm_mujoco", "sim_run_fsm_g1"))
def test_detects_format_and_maps_joints_by_name(tmp_path: Path, model: mujoco.MjModel, format_name: str) -> None:
    path = tmp_path / f"{format_name}.npz"
    marked_name, marked_value = _write_log(path, format_name, model)

    with np.load(path, allow_pickle=False) as npz:
        assert detect_log_format(set(npz.files)) == format_name
    replay = load_replay_log(path, model)
    data = mujoco.MjData(model)
    replay.apply_frame(data, 1)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, marked_name)
    assert data.qpos[model.jnt_qposadr[joint_id]] == pytest.approx(marked_value)
    np.testing.assert_allclose(data.qpos[:7], (0.1, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0))


def test_headless_check_prints_summary(
    tmp_path: Path, model: mujoco.MjModel, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "deploy_mujoco.npz"
    _write_log(path, "deploy_mujoco", model)
    args = Namespace(
        log=path,
        speed=0.25,
        loop=False,
        start_s=None,
        end_s=None,
        model=_MODEL,
        overlay=_OVERLAY,
        headless_check=True,
    )

    run(args)

    output = capsys.readouterr().out
    assert "Log format: deploy_mujoco" in output
    assert 'Goal: {"pos_x": 0.1}' in output
    assert "Frames: 3; duration: 0.004000 s" in output
    assert "t=0.002000 s: phase=FLIGHT" in output
    assert "Headless check: PASS" in output

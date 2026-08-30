# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained G1 jump policy as a self-contained deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from packaging import version

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

from scripts.g1_jump_deploy.export.contract import goal_command_contract, validate_joint_name_contract

_DEFAULT_TASK = "Isaac-Velocity-Jump-G1-Stage3-v0"
_AGENT_CFG_ENTRY_POINT = "rsl_rl_cfg_entry_point"
_POLICY_FILENAME = "policy.pt"
_ONNX_FILENAME = "policy.onnx"
_MANIFEST_FILENAME = "deploy_manifest.json"
_REFERENCE_PREVIEW_FILENAME = "reference_preview_152x70.npy"
_JUMP_PHASE_FILENAME = "jump_phase_152x6.npy"
_EQUIVALENCE_SAMPLE_COUNT = 1000
_EQUIVALENCE_TOLERANCE = 1.0e-5

# Unitree SDK2's 29-DOF motor order. The manifest slots are derived by looking up the
# runtime policy joint names in this protocol order; the numeric slot list is never
# duplicated from the manifest contract.
_UNITREE_SDK2_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _create_parser() -> argparse.ArgumentParser:
    """Create the exporter command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=_DEFAULT_TASK, help="Registered G1 jump task ID.")
    parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint to export.")
    parser.add_argument("--output-dir", required=True, help="Directory in which to write the deployment bundle.")
    parser.add_argument("--agent", default=_AGENT_CFG_ENTRY_POINT, help="Agent configuration registry entry point.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed used while resolving the task.")
    add_launcher_args(parser)
    return parser


def _tensor_row(value: Any, count: int) -> list[float]:
    """Convert a scalar or resolved tensor into one environment's vector."""
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.repeat(count)
        elif tensor.ndim > 1:
            tensor = tensor[0]
        values = tensor.reshape(-1).tolist()
    elif isinstance(value, (float, int)):
        values = [float(value)] * count
    else:
        raise TypeError(f"Expected a numeric scalar or torch.Tensor, got {type(value)}.")
    if len(values) != count:
        raise ValueError(f"Expected {count} resolved values, got {len(values)}.")
    return [float(item) for item in values]


def _tensor_pairs(value: Any, count: int, name: str) -> list[list[float]]:
    """Convert one environment's resolved per-joint bounds to finite pairs."""
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(value)}.")
    tensor = value.detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.shape != (count, 2):
        raise ValueError(f"Expected {name} to have shape ({count}, 2), got {tuple(tensor.shape)}.")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"Expected {name} to contain only finite bounds.")
    return [[float(lower), float(upper)] for lower, upper in tensor.tolist()]


def _resolved_observation_scale(value: Any, step_dim: int, name: str) -> float | list[float] | None:
    """Serialize a resolved scalar or per-component observation scale."""
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        scale = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    elif isinstance(value, (float, int, tuple, list)):
        scale = np.asarray(value, dtype=np.float64)
    else:
        raise TypeError(f"Observation term {name!r} has unsupported scale type {type(value)}.")
    if not np.all(np.isfinite(scale)):
        raise RuntimeError(f"Observation term {name!r} has a non-finite scale.")
    if scale.ndim == 0:
        return float(scale)
    if scale.shape != (step_dim,):
        raise RuntimeError(
            f"Observation term {name!r} scale must be scalar or have shape ({step_dim},), got {scale.shape}."
        )
    return [float(item) for item in scale]


def _runtime_observation_schema(env) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """Build the fixed observation schema from the resolved policy observation group."""
    expected_names = [
        "joint_pos",
        "joint_vel",
        "goal_remaining",
        "base_ang_vel",
        "projected_gravity",
        "last_action",
        "goal_command",
        "reference_preview",
        "jump_phase",
    ]
    manager = env.observation_manager
    term_names = list(manager.active_terms["policy"])
    if term_names != expected_names:
        raise RuntimeError(f"Policy observation terms do not match deployment schema: {term_names}.")
    if not manager.group_obs_concatenate["policy"]:
        raise RuntimeError("The deployment policy observation group must concatenate its terms.")

    term_dims = manager.group_obs_term_dim["policy"]
    term_cfgs = manager._group_obs_term_cfgs["policy"]
    terms = []
    configs = {}
    offset = 0
    for name, shape, term_cfg in zip(term_names, term_dims, term_cfgs):
        total = math.prod(shape)
        history = int(term_cfg.history_length) if term_cfg.history_length > 0 else 1
        if total % history != 0:
            raise RuntimeError(f"Observation term {name!r} has total size {total}, not divisible by history {history}.")
        step_dim = total // history
        term = {"name": name, "offset": offset, "step_dim": step_dim, "history": history, "total": total}
        scale = _resolved_observation_scale(term_cfg.scale, step_dim, name)
        if scale is not None:
            term["scale"] = scale
        terms.append(term)
        configs[name] = term_cfg
        offset += total

    total_dim = math.prod(manager.group_obs_dim["policy"])
    if offset != total_dim:
        raise RuntimeError(f"Observation term sizes sum to {offset}, but the runtime group dimension is {total_dim}.")
    return total_dim, terms, configs


def _runtime_actuator_schema(robot, joint_names: list[str]) -> dict[str, Any]:
    """Read gains and limits from the resolved implicit actuator objects."""
    from isaaclab.actuators import ImplicitActuator

    properties: dict[str, dict[str, float]] = {}
    for actuator_name, actuator in robot.actuators.items():
        if not isinstance(actuator, ImplicitActuator):
            raise RuntimeError(
                f"Actuator {actuator_name!r} is {type(actuator).__name__}, but deployment requires implicit PD."
            )
        for local_index, joint_name in enumerate(actuator.joint_names):
            if joint_name in properties:
                raise RuntimeError(f"Joint {joint_name!r} is assigned to more than one resolved actuator.")
            properties[joint_name] = {
                "stiffness": float(actuator.stiffness[0, local_index].item()),
                "damping": float(actuator.damping[0, local_index].item()),
                "effort_limit": float(actuator.effort_limit[0, local_index].item()),
                "velocity_limit": float(actuator.velocity_limit[0, local_index].item()),
                "armature": float(actuator.armature[0, local_index].item()),
            }

    if set(properties) != set(joint_names):
        missing = sorted(set(joint_names) - set(properties))
        extra = sorted(set(properties) - set(joint_names))
        raise RuntimeError(f"Resolved actuators do not cover the policy joints. Missing={missing}, extra={extra}.")

    effort_limit = [properties[name]["effort_limit"] for name in joint_names]
    if not all(math.isfinite(item) for item in effort_limit):
        raise RuntimeError("All deployment effort limits must be finite.")
    velocity_limit = [
        value if math.isfinite(value) else None
        for value in (properties[name]["velocity_limit"] for name in joint_names)
    ]
    return {
        "type": "implicit_pd",
        "stiffness": [properties[name]["stiffness"] for name in joint_names],
        "damping": [properties[name]["damping"] for name in joint_names],
        "effort_limit": effort_limit,
        "velocity_limit": velocity_limit,
        "armature": [properties[name]["armature"] for name in joint_names],
    }


def _resolved_joint_names_from_scene_entity(robot, scene_entity_cfg) -> list[str]:
    """Return selected joint names in the resolved SceneEntityCfg index order."""
    if isinstance(scene_entity_cfg.joint_ids, slice):
        return list(robot.joint_names[scene_entity_cfg.joint_ids])
    return [robot.joint_names[index] for index in scene_entity_cfg.joint_ids]


def _generate_reference_data(
    env,
    observation_terms: list[dict[str, Any]],
    observation_cfgs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate deployment tables and derive reference metadata through environment functions."""
    import torch

    preview_cfg = observation_cfgs["reference_preview"]
    phase_cfg = observation_cfgs["jump_phase"]
    preview_func = preview_cfg.func
    phase_func = phase_cfg.func
    episode_steps = int(env.max_episode_length)

    # Both observation functions derive time from these runtime buffers. Preserve them so
    # metadata extraction does not leave the environment in a surprising state.
    if not hasattr(env, "start_times"):
        preview_func(env, **preview_cfg.params)
    original_start_times = env.start_times.clone()
    original_episode_length = env.episode_length_buf.clone()
    try:
        preview_rows = []
        phase_rows = []
        env.start_times.zero_()
        for step in range(episode_steps):
            env.episode_length_buf.fill_(step)
            preview_rows.append(preview_func(env, **preview_cfg.params)[0].detach().cpu())
            phase_rows.append(phase_func(env, **phase_cfg.params)[0].detach().cpu())
        reference_preview = torch.stack(preview_rows).to(dtype=torch.float32).numpy()
        jump_phase = torch.stack(phase_rows).to(dtype=torch.float32).numpy()

        loader = env.motion_loader
        num_frames = int(loader.length)
        reference_fps = float(num_frames / float(env.cfg.episode_length_s))

        # Derive frame phase ranges by querying the same one-hot observation at every
        # reference frame, rather than copying the phase boundaries from constants.py.
        frame_phase_ids = []
        env.episode_length_buf.zero_()
        for frame in range(num_frames):
            env.start_times.fill_(frame / reference_fps)
            phase = phase_func(env, **phase_cfg.params)[0]
            frame_phase_ids.append(int(torch.argmax(phase).item()))

        phase_names_by_range = phase_func.__globals__.get("JUMP_PHASES")
        if not isinstance(phase_names_by_range, dict):
            raise RuntimeError("Unable to resolve jump phase names from the configured observation function.")
        phase_names = list(phase_names_by_range)
        phase_frame_ranges = []
        for phase_id, phase_name in enumerate(phase_names):
            indices = [index for index, value in enumerate(frame_phase_ids) if value == phase_id]
            if not indices or indices != list(range(indices[0], indices[-1] + 1)):
                raise RuntimeError(f"Reference phase {phase_name!r} is empty or non-contiguous.")
            phase_frame_ranges.append([indices[0], indices[-1] + 1])

        # Infer the preview frame offsets by matching its joint-position blocks against
        # the runtime loader. This keeps even these numeric metadata values tied to the
        # active observation implementation.
        env.start_times.zero_()
        preview_at_zero = preview_func(env, **preview_cfg.params)[0]
        preview_dim = next(term["step_dim"] for term in observation_terms if term["name"] == "reference_preview")
        num_preview_blocks, remainder = divmod(preview_dim - 1, int(loader.num_joints))
        if remainder != 0:
            raise RuntimeError("Reference preview dimension is incompatible with the runtime motion joint count.")
        preview_offsets = []
        for block in range(num_preview_blocks):
            start = 1 + block * loader.num_joints
            values = preview_at_zero[start : start + loader.num_joints]
            errors = torch.amax(torch.abs(loader.ref_joint_pos - values.unsqueeze(0)), dim=1)
            preview_offsets.append(int(torch.argmin(errors).item()))
        if preview_offsets != sorted(set(preview_offsets)):
            raise RuntimeError(f"Could not infer unique increasing preview offsets: {preview_offsets}.")
    finally:
        env.start_times.copy_(original_start_times)
        env.episode_length_buf.copy_(original_episode_length)

    source_csv = Path(loader.csv_path).resolve()
    try:
        source_bytes = source_csv.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read the resolved reference motion CSV: {source_csv}.") from exc
    reference = {
        "fps": reference_fps,
        "num_frames": num_frames,
        "phase_names": phase_names,
        "phase_frame_ranges": phase_frame_ranges,
        "preview_offsets_frames": preview_offsets,
        "source_csv": str(source_csv),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "root_frame0": {
            "pos": _tensor_row(loader.ref_root_pos[0], 3),
            "quat_xyzw": _tensor_row(loader.ref_root_quat[0], 4),
        },
    }
    return reference_preview, jump_phase, reference


def _validate_onnx_contract(model, observation_dim: int, action_dim: int) -> None:
    """Validate the emitted ONNX graph's opset, names, and static shapes."""
    import onnx

    onnx.checker.check_model(model)
    default_domain_opsets = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
    if default_domain_opsets != [18]:
        raise RuntimeError(f"Expected ONNX opset 18, got {default_domain_opsets}.")
    if len(model.graph.input) != 1 or model.graph.input[0].name != "obs":
        raise RuntimeError(f"Expected one ONNX input named 'obs', got {[item.name for item in model.graph.input]}.")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "actions":
        raise RuntimeError(
            f"Expected one ONNX output named 'actions', got {[item.name for item in model.graph.output]}."
        )

    def shape(value_info) -> list[int]:
        return [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]

    input_shape = shape(model.graph.input[0])
    output_shape = shape(model.graph.output[0])
    if input_shape != [1, observation_dim] or output_shape != [1, action_dim]:
        raise RuntimeError(
            f"Unexpected ONNX shapes: obs={input_shape}, actions={output_shape}; "
            f"expected [1, {observation_dim}] and [1, {action_dim}]."
        )


def _load_and_validate_onnx_artifact(onnx_path: Path, jit_policy):
    """Load the final ONNX artifact and require all actor weights to be self-contained."""
    import onnx

    # A deployment consumer receives only policy.onnx. Do not let ONNX resolve an
    # external-data sidecar here, even if a stale or accidentally generated one exists.
    model = onnx.load(str(onnx_path), load_external_data=False)
    external_initializers = []
    empty_initializers = []
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.DEFAULT:
            external_initializers.append(initializer.name)
        if not initializer.raw_data:
            empty_initializers.append(initializer.name)
    if external_initializers or empty_initializers:
        raise RuntimeError(
            "ONNX artifact is not self-contained. "
            f"External initializers: {external_initializers}; empty initializers: {empty_initializers}."
        )

    initializer_element_count = sum(math.prod(initializer.dims) for initializer in model.graph.initializer)
    torch_parameter_count = sum(parameter.numel() for parameter in jit_policy.parameters())
    if initializer_element_count != torch_parameter_count:
        raise RuntimeError(
            "ONNX initializer element count does not match the TorchScript actor: "
            f"{initializer_element_count} != {torch_parameter_count}."
        )

    expected_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in jit_policy.parameters()
    )
    file_size = onnx_path.stat().st_size
    minimum_size = 0.8 * expected_parameter_bytes
    maximum_size = 1.5 * expected_parameter_bytes
    if not minimum_size <= file_size <= maximum_size:
        raise RuntimeError(
            f"ONNX file size {file_size} bytes is inconsistent with {expected_parameter_bytes} bytes of "
            f"Torch parameters; expected a size in [{minimum_size:.0f}, {maximum_size:.0f}] bytes."
        )
    print(f"[INFO] Self-contained ONNX artifact: {file_size} bytes, {initializer_element_count} initializer elements.")
    return model


def _run_onnx_batch(model, observations: np.ndarray) -> np.ndarray:
    """Evaluate the fixed-batch ONNX graph over a larger in-memory validation batch."""
    # Linear policy graphs are batch-agnostic, but the deployment file intentionally has a
    # static batch dimension of one. Change only the in-memory model's I/O metadata so the
    # equivalence gate can execute all random vectors efficiently in one call.
    import onnx

    validation_model = onnx.ModelProto()
    validation_model.CopyFrom(model)
    validation_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = observations.shape[0]
    validation_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = observations.shape[0]

    try:
        import onnxruntime as ort
    except ImportError:
        from onnx.reference import ReferenceEvaluator

        print("[INFO] onnxruntime is unavailable; using onnx.reference.ReferenceEvaluator for the equivalence gate.")
        return np.asarray(ReferenceEvaluator(validation_model).run(None, {"obs": observations})[0])

    session = ort.InferenceSession(validation_model.SerializeToString(), providers=["CPUExecutionProvider"])
    return np.asarray(session.run(["actions"], {"obs": observations})[0])


def _numeric_equivalence_gate(
    runner,
    jit_path: Path,
    onnx_path: Path,
    observation_dim: int,
    action_dim: int,
    installed_rsl_rl_version: str,
) -> tuple[float, float]:
    """Require ONNX, TorchScript, and eager inference to be numerically equivalent."""
    import torch

    # Validate the freshly loaded final artifact before evaluating a single random
    # observation. This prevents a temporary external-data sidecar from masking an
    # undeployable policy.onnx file.
    jit_policy = torch.jit.load(str(jit_path), map_location="cpu").eval()
    onnx_model = _load_and_validate_onnx_artifact(onnx_path, jit_policy)
    _validate_onnx_contract(onnx_model, observation_dim, action_dim)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    observations = torch.randn(
        _EQUIVALENCE_SAMPLE_COUNT,
        observation_dim,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )

    eager_policy = runner.get_inference_policy(device="cpu")
    with torch.inference_mode():
        if version.parse(installed_rsl_rl_version) >= version.parse("4.0.0"):
            from tensordict import TensorDict

            obs_groups = list(eager_policy.obs_groups)
            if obs_groups != ["policy"]:
                raise RuntimeError(f"Expected the eager actor to consume only the policy group, got {obs_groups}.")
            eager_actions = eager_policy(TensorDict({"policy": observations}, batch_size=[_EQUIVALENCE_SAMPLE_COUNT]))
        else:
            eager_actions = eager_policy(observations)
        jit_actions = jit_policy(observations)

    eager_actions = eager_actions.detach().cpu()
    jit_actions = jit_actions.detach().cpu()
    expected_shape = (_EQUIVALENCE_SAMPLE_COUNT, action_dim)
    if tuple(eager_actions.shape) != expected_shape or tuple(jit_actions.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected equivalence output shapes: eager={tuple(eager_actions.shape)}, "
            f"jit={tuple(jit_actions.shape)}, expected={expected_shape}."
        )

    onnx_actions = torch.from_numpy(_run_onnx_batch(onnx_model, observations.numpy())).to(dtype=torch.float32)
    if tuple(onnx_actions.shape) != expected_shape:
        raise RuntimeError(f"Unexpected ONNX equivalence output shape {tuple(onnx_actions.shape)}.")

    max_onnx_jit = float(torch.max(torch.abs(onnx_actions - jit_actions)).item())
    max_jit_eager = float(torch.max(torch.abs(jit_actions - eager_actions)).item())

    # Scale the tolerance by the action magnitude actually produced. Float32 error grows with
    # the values being accumulated, so a fixed absolute bound silently tightens as a policy
    # learns to output larger actions: one checkpoint here reached max|a| = 90, where a single
    # ULP is already 7.6e-6 and an honest re-serialisation lands 4 ULP out. The floor of 1.0
    # keeps the bound absolute for policies whose actions stay small, so a near-zero output
    # cannot pass on relative error alone.
    action_magnitude = max(1.0, float(torch.max(torch.abs(jit_actions)).item()))
    tolerance = _EQUIVALENCE_TOLERANCE * action_magnitude
    print(f"[INFO] Numeric equivalence over {_EQUIVALENCE_SAMPLE_COUNT} random observations:")
    print(f"[INFO]   max|onnx - jit|   = {max_onnx_jit:.9e}")
    print(f"[INFO]   max|jit - eager| = {max_jit_eager:.9e}")
    print(f"[INFO]   tolerance         = {tolerance:.9e}  (peak |action| = {action_magnitude:.4f})")
    if max_onnx_jit >= tolerance or max_jit_eager >= tolerance:
        raise RuntimeError(
            f"Numeric equivalence gate failed (required both maxima < {tolerance:.3e} for peak"
            f" action magnitude {action_magnitude:.4f})."
        )
    return max_onnx_jit, max_jit_eager


def _export_policy(runner, output_dir: Path, installed_rsl_rl_version: str) -> None:
    """Export through the same version-dependent path as play_rsl_rl.py."""
    from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

    if version.parse(installed_rsl_rl_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=str(output_dir), filename=_POLICY_FILENAME)
        policy = runner.get_inference_policy(device="cpu")
        export_policy_as_onnx(policy, normalizer=None, path=str(output_dir), filename=_ONNX_FILENAME)
        return

    if version.parse(installed_rsl_rl_version) >= version.parse("2.3.0"):
        policy_nn = runner.alg.policy
    else:
        policy_nn = runner.alg.actor_critic
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(output_dir), filename=_POLICY_FILENAME)
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(output_dir), filename=_ONNX_FILENAME)


def _build_manifest(
    env,
    task: str,
    checkpoint: Path,
    observation_dim: int,
    observation_terms: list[dict[str, Any]],
    observation_cfgs: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Build a deployment manifest entirely from resolved runtime objects."""
    import torch

    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.constants import JOINT_NAMES

    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    command_term = env.command_manager.get_term("jump_goal")
    joint_names = list(robot.joint_names)
    action_joint_names = list(action_term._joint_names)
    if action_joint_names != joint_names:
        raise RuntimeError(
            "The resolved action order differs from the runtime articulation order: "
            f"action={action_joint_names}, articulation={joint_names}."
        )

    for observation_name in ("joint_pos", "joint_vel"):
        scene_entity_cfg = observation_cfgs[observation_name].params["asset_cfg"]
        observation_joint_names = _resolved_joint_names_from_scene_entity(robot, scene_entity_cfg)
        if observation_joint_names != joint_names:
            raise RuntimeError(
                f"The resolved {observation_name} order differs from the runtime articulation order: "
                f"{observation_joint_names}."
            )

    try:
        joint_order_matches_constants = validate_joint_name_contract(joint_names, JOINT_NAMES)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    action_dim = int(action_term.action_dim)
    default_pos = _tensor_row(robot.data.default_joint_pos.torch, action_dim)
    default_vel = _tensor_row(robot.data.default_joint_vel.torch, action_dim)
    action_scale = _tensor_row(action_term._scale, action_dim)
    action_offset = _tensor_row(action_term._offset, action_dim)
    if not np.allclose(action_offset, default_pos, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("Resolved action offsets do not equal the runtime default joint positions.")
    # An absent clip is legitimate and is recorded as null: this task's policy commands
    # position targets past the joint stops on purpose, and bounding them changes the
    # dynamics rather than merely tidying the command. Consumers must handle both.
    if action_term.cfg.clip is None or not hasattr(action_term, "_clip"):
        action_clip = None
    else:
        action_clip = _tensor_pairs(action_term._clip, action_dim, "resolved action clip")
    clipped_to_limits: list[str] = []
    position_limits = _tensor_pairs(robot.data.joint_pos_limits.torch, action_dim, "runtime joint position limits")
    for index, (clip_bounds, limit_bounds) in enumerate(zip(action_clip or [], position_limits)):
        clip_lower, clip_upper = clip_bounds
        limit_lower, limit_upper = limit_bounds
        if clip_lower >= clip_upper:
            raise RuntimeError(
                f"Resolved action clip for joint {joint_names[index]!r} must satisfy low < high, got {clip_bounds}."
            )
        # Clamp rather than refuse. The training clip is default +/- scale, symmetric about the
        # default pose, but the reference motion swings asymmetrically about it, so one side can
        # fall outside the joint's own travel -- left hip pitch reaches -2.693 against a -2.531
        # limit. PhysX clamps at the limit during training regardless, so the reachable set is
        # unchanged; emitting the intersection simply makes the deployed bound honest and stops
        # hardware ever being commanded past a mechanical stop. The result is a subset of what
        # training allowed, never a superset.
        clamped_lower = max(clip_lower, limit_lower)
        clamped_upper = min(clip_upper, limit_upper)
        if clamped_lower >= clamped_upper:
            raise RuntimeError(
                f"Joint {joint_names[index]!r} has an empty action clip after intersecting "
                f"{clip_bounds} with its position limits {limit_bounds}."
            )
        if (clamped_lower, clamped_upper) != (clip_lower, clip_upper):
            clipped_to_limits.append(joint_names[index])
            action_clip[index] = [clamped_lower, clamped_upper]

    if clipped_to_limits:
        print(
            f"[INFO] Action clip narrowed to the joint travel for {len(clipped_to_limits)} joint(s): "
            f"{', '.join(clipped_to_limits)}."
        )

    if hasattr(action_term, "_alpha"):
        filter_alpha = _tensor_row(action_term._alpha, action_dim)
    else:
        filter_alpha = torch.ones(action_dim, dtype=torch.float32).tolist()
    delay_min = int(getattr(action_term.cfg, "min_delay_steps", 0))
    delay_max = int(getattr(action_term.cfg, "max_delay_steps", 0))
    resolved_effort_ratio = getattr(action_term, "_effort_limit_ratio", None)
    torque_projection = None
    schema_version = "1.2"
    if resolved_effort_ratio is not None:
        effort_limit_ratio = _tensor_row(resolved_effort_ratio, action_dim)
        if any(ratio <= 0.0 or ratio > 1.0 for ratio in effort_limit_ratio):
            raise RuntimeError("Resolved torque-projection effort-limit ratios must be in (0, 1].")
        torque_projection = {
            "type": "instantaneous_pd",
            "period_s": float(env.cfg.sim.dt),
            "effort_limit_ratio": effort_limit_ratio,
            "formula": (
                "q_target = q + (clip(kp*(q_requested-q)-kd*dq, -ratio*effort_limit, ratio*effort_limit)+kd*dq)/kp"
            ),
        }
        schema_version = "1.3"

    resolved_velocity_lookahead = getattr(action_term, "_lower_limit_velocity_lookahead", None)
    lower_limit_brake = None
    if resolved_velocity_lookahead is not None:
        velocity_lookahead = _tensor_row(resolved_velocity_lookahead, action_dim)
        if any(value < 0.0 for value in velocity_lookahead):
            raise RuntimeError("Resolved lower-limit velocity lookahead values must be non-negative.")
        if any(value > 0.0 for value in velocity_lookahead):
            if action_clip is None:
                raise RuntimeError("Lower-limit braking requires finite resolved action clips.")
            if torque_projection is None:
                raise RuntimeError("Lower-limit braking requires torque projection.")
            lower_limit_brake = {
                "type": "velocity_lookahead",
                "period_s": float(env.cfg.sim.dt),
                "position_lower": [bounds[0] for bounds in action_clip],
                "position_upper": [bounds[1] for bounds in action_clip],
                "velocity_lookahead_s": velocity_lookahead,
                "formula": ("q_requested = max(q_filtered, min(q_upper, q_lower + t_lookahead*max(-dq, 0)))"),
            }
            schema_version = "1.5"

    # Schema 1.5 separates physical feedback limits from the potentially
    # narrower action-target clip used by deployment safety filters.

    ranges = command_term.cfg.ranges
    goal_ranges = {
        name: [float(value) for value in getattr(ranges, name)] for name in ("pos_x", "pos_y", "roll", "pitch", "yaw")
    }
    goal_remaining_cfg = observation_cfgs["goal_remaining"]
    goal_remaining_function = getattr(goal_remaining_cfg.func, "__name__", "")
    remaining_modes = {
        "obs_goal_remaining": "live",
        "obs_goal_remaining_stale": "flight_frozen",
        "obs_goal_remaining_latched": "latched",
    }
    try:
        remaining_mode = remaining_modes[goal_remaining_function]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported goal_remaining observation function: {goal_remaining_function!r}.") from exc
    freeze_enabled = remaining_mode == "flight_frozen"
    freeze_probability = float(goal_remaining_cfg.params.get("freeze_prob", 0.0))
    drift_std = float(goal_remaining_cfg.params.get("drift_std", 0.0))
    goal_command_function = getattr(observation_cfgs["goal_command"].func, "__name__", "")
    orientation_mode, retrigger_indicator = goal_command_contract(
        goal_command_function,
        observation_cfgs["goal_command"].params,
    )
    if retrigger_indicator is not None:
        schema_version = "1.7" if retrigger_indicator["mode"] == "goal_command_z_affine_pos_x" else "1.6"

    policy_dt = float(env.step_dt)
    sim_dt = float(env.cfg.sim.dt)
    episode_steps = int(env.max_episode_length)
    action_schema = {
        "dim": action_dim,
        "scale": action_scale,
        "offset": action_offset,
        "filter_alpha": filter_alpha,
        "delay_steps": {"min": delay_min, "max": delay_max},
        "clip": action_clip,
        "formula": "q_target = alpha*clip(offset + scale*a_delayed) + (1-alpha)*q_target_prev",
    }
    if torque_projection is not None:
        action_schema["torque_projection"] = torque_projection
    if lower_limit_brake is not None:
        action_schema["lower_limit_brake"] = lower_limit_brake

    manifest = {
        "schema_version": schema_version,
        "task": task,
        "checkpoint": str(checkpoint),
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "joint_order_matches_constants": joint_order_matches_constants,
        "control": {
            "policy_dt": policy_dt,
            "policy_hz": 1.0 / policy_dt,
            "sim_dt": sim_dt,
            "decimation": int(env.cfg.decimation),
            "episode_steps": episode_steps,
            "episode_duration_s": float(env.cfg.episode_length_s),
        },
        "joints": {
            "names": joint_names,
            "unitree_sdk2_slots": [_UNITREE_SDK2_JOINT_ORDER.index(name) for name in joint_names],
            "default_pos": default_pos,
            "default_vel": default_vel,
            "position_limits": position_limits,
        },
        "observation": {
            "total_dim": observation_dim,
            "history_order": "oldest_first",
            "history_layout": "history_major",
            "terms": observation_terms,
        },
        "action": action_schema,
        "actuators": _runtime_actuator_schema(robot, joint_names),
        "reference": reference,
        "goal": {
            "quat_order": "xyzw",
            "ranges": goal_ranges,
            "remaining_mode": remaining_mode,
            "orientation_mode": orientation_mode,
            **({"retrigger_indicator": retrigger_indicator} if retrigger_indicator is not None else {}),
            "flight_freeze": {
                "enabled": freeze_enabled,
                "freeze_prob_trained": freeze_probability,
                "drift_std_trained": drift_std,
            },
        },
        "tables": {
            "reference_preview": _REFERENCE_PREVIEW_FILENAME,
            "jump_phase": _JUMP_PHASE_FILENAME,
        },
    }
    return manifest, joint_order_matches_constants


def _validate_runtime_contract(
    env,
    observation_dim: int,
    observation_terms: list[dict[str, Any]],
    reference_preview: np.ndarray,
    jump_phase: np.ndarray,
) -> None:
    """Fail if the selected task does not implement the fixed deployment contract."""
    action_dim = int(env.action_manager.get_term("joint_pos").action_dim)
    episode_steps = int(env.max_episode_length)
    preview_dim = next(term["step_dim"] for term in observation_terms if term["name"] == "reference_preview")
    phase_dim = next(term["step_dim"] for term in observation_terms if term["name"] == "jump_phase")
    if not math.isclose(float(env.step_dt), 0.02, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"Deployment tables require a 0.02 s policy step, got {env.step_dt}.")
    if observation_dim != 326 or action_dim != 23:
        raise RuntimeError(
            f"Deployment schema requires observation/action dimensions 326/23, got {observation_dim}/{action_dim}."
        )
    if reference_preview.shape != (episode_steps, preview_dim) or reference_preview.shape != (152, 70):
        raise RuntimeError(f"Reference preview must have shape (152, 70), got {reference_preview.shape}.")
    if jump_phase.shape != (episode_steps, phase_dim) or jump_phase.shape != (152, 6):
        raise RuntimeError(f"Jump phase must have shape (152, 6), got {jump_phase.shape}.")
    if reference_preview.dtype != np.float32 or jump_phase.dtype != np.float32:
        raise RuntimeError("Precomputed deployment tables must use float32.")


def _publish_bundle(staging_dir: Path, output_dir: Path) -> None:
    """Publish all validated bundle files from the staging directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        _POLICY_FILENAME,
        _ONNX_FILENAME,
        _MANIFEST_FILENAME,
        _REFERENCE_PREVIEW_FILENAME,
        _JUMP_PHASE_FILENAME,
    ):
        os.replace(staging_dir / filename, output_dir / filename)


def _run_export(env_cfg, agent_cfg, args_cli: argparse.Namespace) -> None:
    """Instantiate the task, export the actor, validate it, and write the bundle."""
    import gymnasium as gym
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1

    # Stage 3 randomizes gains at startup. Deployment needs the nominal resolved
    # actuator controller, not a single random training-domain sample.
    if hasattr(env_cfg.events, "actuator_gains"):
        env_cfg.events.actuator_gains = None

    checkpoint = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}.")
    env_cfg.log_dir = str(checkpoint.parent)
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    try:
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}.")
        print(f"[INFO] Loading checkpoint: {checkpoint}")
        runner.load(str(checkpoint))

        base_env = env.unwrapped
        observation_dim, observation_terms, observation_cfgs = _runtime_observation_schema(base_env)
        reference_preview, jump_phase, reference = _generate_reference_data(
            base_env, observation_terms, observation_cfgs
        )
        _validate_runtime_contract(base_env, observation_dim, observation_terms, reference_preview, jump_phase)
        manifest, joint_order_matches_constants = _build_manifest(
            base_env,
            args_cli.task,
            checkpoint,
            observation_dim,
            observation_terms,
            observation_cfgs,
            reference,
        )

        with tempfile.TemporaryDirectory(prefix=".g1-jump-export-", dir=output_dir.parent) as temporary_directory:
            staging_dir = Path(temporary_directory)
            _export_policy(runner, staging_dir, installed_rsl_rl_version)
            np.save(staging_dir / _REFERENCE_PREVIEW_FILENAME, reference_preview, allow_pickle=False)
            np.save(staging_dir / _JUMP_PHASE_FILENAME, jump_phase, allow_pickle=False)
            with (staging_dir / _MANIFEST_FILENAME).open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, allow_nan=False)
                file.write("\n")
            _publish_bundle(staging_dir, output_dir)
        _numeric_equivalence_gate(
            runner,
            output_dir / _POLICY_FILENAME,
            output_dir / _ONNX_FILENAME,
            observation_dim,
            int(base_env.action_manager.get_term("joint_pos").action_dim),
            installed_rsl_rl_version,
        )

        if not joint_order_matches_constants:
            print(
                "[INFO] Runtime articulation order differs from JOINT_NAMES; "
                "the validated manifest name and SDK-slot mappings preserve the runtime order."
            )
        print(f"[INFO] Deployment bundle written to: {output_dir}")
    finally:
        env.close()


parser = _create_parser()
args_cli, hydra_args = setup_preset_cli(parser)
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    """Resolve the task and run the exporter inside its simulation runtime."""
    with launch_simulation(env_cfg, args_cli):
        _run_export(env_cfg, agent_cfg, args_cli)


if __name__ == "__main__":
    main()

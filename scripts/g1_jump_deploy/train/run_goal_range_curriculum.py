# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train and evaluate the staged G1 longitudinal goal-range curriculum."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path("/workspace/isaaclab")
_ISAAC_PYTHON = Path("/isaac-sim/python.sh")
_EXPERIMENT_NAME = "g1_jump_deploy_v1"
_EXPERIMENT_DIR = _REPO_ROOT / "logs" / "rsl_rl" / _EXPERIMENT_NAME
_SUMMARY_ROOT = _REPO_ROOT / "logs" / "goal_range_curriculum"
_INITIAL_CHECKPOINT = (
    _EXPERIMENT_DIR
    / "2026-08-27_09-40-14_stage2_longitudinal_odometry_smooth_narrow_knee_safe_from823_lr1e7_25_2048"
    / "model_825.pt"
)
_SUCCESS_RATE_KEYS = {0.10: "success_rate_0p10", 0.20: "success_rate_0p20"}
_MINIMUM_UPRIGHT_RATE = 0.99
_MINIMUM_CORRELATION_X = 0.95


@dataclass(frozen=True)
class Stage:
    """One longitudinal range-widening stage."""

    name: str
    range_code: str
    task: str


@dataclass(frozen=True)
class CheckpointRef:
    """A checkpoint and its containing RSL-RL run."""

    path: Path
    run: str
    checkpoint: str


_RANGE_CODES = ("020", "040", "060", "080", "100")
_STAGE_FAMILIES = {
    variant: tuple(
        Stage(
            name=f"range{task_variant.lower()}{range_code}",
            range_code=range_code,
            task=f"Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-Range{task_variant}{range_code}-v0",
        )
        for range_code in (("020", "040") if variant == "contact_trigger" else _RANGE_CODES)
    )
    for variant, task_variant in (("plain", ""), ("contact", "Contact"), ("contact_trigger", "ContactTrigger"))
}
_OVERALL_SUCCESS_TABLE = re.compile(
    r"^[ \t]*Overall success[^\n]*\n(?P<table>.*?)(?:\n[ \t]*\n|\Z)", re.MULTILINE | re.DOTALL
)
_SUCCESS_ROW = re.compile(r"^\s*(0\.10|0\.20)\s+(\d+)\s+(\d+)\s+([0-9.]+)%", re.MULTILINE)
_UPRIGHT_ROW = re.compile(r"upright at episode end .*?:\s*(\d+)/(\d+)\s+\(([0-9.]+)%\)")
_FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RESPONSE_GAIN_ROW = re.compile(rf"response matrix = \[\[({_FLOAT_PATTERN}),")
_RESPONSE_OFFSET_ROW = re.compile(rf"offset \[m\] = \[({_FLOAT_PATTERN}),")
_CORRELATION_ROW = re.compile(rf"same-axis Pearson correlation = \[x\s+({_FLOAT_PATTERN}),")
_CHECKPOINT_ITERATION = re.compile(r"model_(\d+)\.pt$")


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Stages to run in curriculum order, as space- or comma-separated range codes (default: all available).",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(_STAGE_FAMILIES),
        default="plain",
        help="Task family to train (default: plain).",
    )
    parser.add_argument(
        "--iterations_per_stage",
        type=int,
        default=300,
        help="Fine-tuning iterations for every selected stage (default: 300).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the full command sequence without running it.")
    parser.add_argument("--start_from", type=Path, default=_INITIAL_CHECKPOINT, help="Checkpoint for the first stage.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override the G1JumpFineTunePPORunnerCfg learning rate.",
    )
    parser.add_argument(
        "--selection_tolerance_m",
        type=float,
        choices=tuple(_SUCCESS_RATE_KEYS),
        default=0.10,
        help="Goal-error tolerance used to select each stage checkpoint [m] (default: 0.10).",
    )
    parser.add_argument(
        "--minimum_response_gain",
        type=float,
        default=0.85,
        help="Minimum settled-displacement x response gain required for normal selection (default: 0.85).",
    )
    return parser


def _selected_stages(values: list[str] | None, variant: str) -> list[Stage]:
    stages = _STAGE_FAMILIES[variant]
    if values is None:
        return list(stages)
    requested = []
    for value in values:
        requested.extend(item.strip().lower().removeprefix("range") for item in value.split(",") if item.strip())
    known_codes = {stage.range_code for stage in stages}
    unknown = sorted(set(requested) - known_codes)
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; expected a subset of {sorted(known_codes)}.")
    if len(requested) != len(set(requested)):
        raise ValueError("--stages must not contain duplicates.")
    return [stage for stage in stages if stage.range_code in requested]


def _checkpoint_ref(path: Path, *, require_exists: bool) -> CheckpointRef:
    path = path if path.is_absolute() else _REPO_ROOT / path
    if require_exists and not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return CheckpointRef(path=path, run=path.parent.name, checkpoint=path.name)


def _train_command(
    stage: Stage,
    source: CheckpointRef,
    iterations: int,
    run_name: str,
    learning_rate: float | None,
) -> list[str]:
    command = [
        str(_ISAAC_PYTHON),
        "scripts/reinforcement_learning/rsl_rl/train.py",
        "--task",
        stage.task,
        "--headless",
        "--num_envs",
        "2048",
        "--max_iterations",
        str(iterations),
        "--resume",
        "--experiment_name",
        _EXPERIMENT_NAME,
        "--load_run",
        source.run,
        "--load_checkpoint",
        source.checkpoint,
        "--run_name",
        run_name,
    ]
    if learning_rate is not None:
        command.extend(("--learning_rate", repr(learning_rate)))
    return command


def _eval_command(stage: Stage, checkpoint: Path | str) -> list[str]:
    return [
        str(_ISAAC_PYTHON),
        "scripts/g1_jump_deploy/eval/eval_success_rate.py",
        "--task",
        stage.task,
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        "1024",
    ]


def _checkpoint_iteration(path: Path) -> int:
    match = _CHECKPOINT_ITERATION.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def _find_training_run(run_name: str, existing_runs: set[Path]) -> Path:
    candidates = sorted(path for path in _EXPERIMENT_DIR.glob(f"*_{run_name}") if path.is_dir())
    new_candidates = [path for path in candidates if path not in existing_runs]
    if len(new_candidates) != 1:
        raise RuntimeError(f"Expected one new run ending in _{run_name}, found {len(new_candidates)}: {new_candidates}")
    return new_candidates[0]


def _parse_evaluation(output: str, checkpoint: Path) -> dict[str, object]:
    overall_success_match = _OVERALL_SUCCESS_TABLE.search(output)
    overall_success_table = overall_success_match.group("table") if overall_success_match is not None else ""
    success_matches = {
        float(match.group(1)): match.groups()[1:] for match in _SUCCESS_ROW.finditer(overall_success_table)
    }
    upright_match = _UPRIGHT_ROW.search(output)
    response_gain_match = _RESPONSE_GAIN_ROW.search(output)
    response_offset_match = _RESPONSE_OFFSET_ROW.search(output)
    correlation_match = _CORRELATION_ROW.search(output)
    if (
        set(success_matches) != set(_SUCCESS_RATE_KEYS)
        or upright_match is None
        or response_gain_match is None
        or response_offset_match is None
        or correlation_match is None
    ):
        raise RuntimeError(f"Could not parse success, upright, and command-response metrics for {checkpoint}.")
    upright, upright_samples, upright_percent = upright_match.groups()
    upright_count = int(upright)
    upright_sample_count = int(upright_samples)
    evaluation = {
        "checkpoint": str(checkpoint),
        "iteration": _checkpoint_iteration(checkpoint),
        "upright": upright_count,
        "upright_samples": upright_sample_count,
        "upright_rate": upright_count / upright_sample_count,
        "reported_upright_percent": float(upright_percent),
        "response_gain_xx": float(response_gain_match.group(1)),
        "response_offset_x": float(response_offset_match.group(1)),
        "correlation_x": float(correlation_match.group(1)),
    }
    for tolerance, key in _SUCCESS_RATE_KEYS.items():
        successes, samples, reported_percent = success_matches[tolerance]
        success_count = int(successes)
        success_sample_count = int(samples)
        evaluation[key] = success_count / success_sample_count
        evaluation[f"{key}_successes"] = success_count
        evaluation[f"{key}_samples"] = success_sample_count
        evaluation[f"{key}_reported_percent"] = float(reported_percent)
    return evaluation


def _evaluate_checkpoints(stage: Stage, checkpoints: list[Path]) -> list[dict[str, object]]:
    evaluations = []
    for checkpoint in checkpoints:
        command = _eval_command(stage, checkpoint)
        print(shlex.join(command), flush=True)
        result = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(result.stdout, end="")
        evaluations.append(_parse_evaluation(result.stdout, checkpoint))
    return evaluations


def _select_checkpoint(
    evaluations: list[dict[str, object]], selection_tolerance_m: float, minimum_response_gain: float
) -> tuple[dict[str, object], str | None]:
    upright = [item for item in evaluations if item["upright_rate"] >= _MINIMUM_UPRIGHT_RATE]
    if not upright:
        raise RuntimeError("No checkpoint met the required 99% upright rate.")
    eligible = [
        item
        for item in upright
        if item["response_gain_xx"] >= minimum_response_gain and item["correlation_x"] >= _MINIMUM_CORRELATION_X
    ]
    if not eligible:
        selected = max(upright, key=lambda item: (item["response_gain_xx"], item["iteration"]))
        return selected, "no checkpoint met the gain criterion"
    success_rate_key = _SUCCESS_RATE_KEYS[selection_tolerance_m]
    selected = max(eligible, key=lambda item: (item[success_rate_key], item["iteration"]))
    return selected, None


def _write_summary(
    stage: Stage,
    source: CheckpointRef,
    run_dir: Path,
    iterations: int,
    selection_tolerance_m: float,
    minimum_response_gain: float,
    evaluations: list[dict[str, object]],
    selected: dict[str, object],
    selection_note: str | None,
) -> None:
    output_dir = _SUMMARY_ROOT / stage.name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": stage.name,
        "task": stage.task,
        "iterations": iterations,
        "source_checkpoint": str(source.path),
        "training_run": str(run_dir),
        "selection": {
            "goal_tolerance_m": selection_tolerance_m,
            "minimum_upright_rate": _MINIMUM_UPRIGHT_RATE,
            "minimum_response_gain": minimum_response_gain,
            "minimum_correlation_x": _MINIMUM_CORRELATION_X,
            "selected_checkpoint": selected["checkpoint"],
            "success_rate_0p10": selected["success_rate_0p10"],
            "success_rate_0p20": selected["success_rate_0p20"],
            "upright_rate": selected["upright_rate"],
            "response_gain_xx": selected["response_gain_xx"],
            "response_offset_x": selected["response_offset_x"],
            "correlation_x": selected["correlation_x"],
            **({"note": selection_note} if selection_note is not None else {}),
        },
        "evaluations": evaluations,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _dry_run(stages: list[Stage], args: argparse.Namespace) -> None:
    source = _checkpoint_ref(args.start_from, require_exists=False)
    previous = _checkpoint_iteration(source.path) if _CHECKPOINT_ITERATION.fullmatch(source.checkpoint) else "start"
    success_rate_key = _SUCCESS_RATE_KEYS[args.selection_tolerance_m]
    print(
        f"# Select by {success_rate_key} with upright_rate >= {_MINIMUM_UPRIGHT_RATE:.2f}, "
        f"response_gain_xx >= {args.minimum_response_gain:g}, correlation_x >= {_MINIMUM_CORRELATION_X:.2f}"
    )
    for stage in stages:
        run_name = f"{stage.name}_from{previous}"
        print(shlex.join(_train_command(stage, source, args.iterations_per_stage, run_name, args.learning_rate)))
        checkpoint_placeholder = f"<each_saved_checkpoint_in_{run_name}>"
        print(shlex.join(_eval_command(stage, checkpoint_placeholder)))
        source = CheckpointRef(
            path=Path(f"<selected_{stage.name}_checkpoint>"),
            run=f"<selected_{stage.name}_run>",
            checkpoint=f"<selected_{stage.name}_checkpoint>",
        )
        previous = stage.range_code


def main() -> None:
    """Run the requested curriculum stages."""
    parser = _create_parser()
    args = parser.parse_args()
    try:
        stages = _selected_stages(args.stages, args.variant)
    except ValueError as exc:
        parser.error(str(exc))
    if not stages:
        parser.error("--stages must select at least one stage.")
    if args.iterations_per_stage <= 0:
        parser.error("--iterations_per_stage must be positive.")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0):
        parser.error("--learning_rate must be finite and positive.")
    if not math.isfinite(args.minimum_response_gain):
        parser.error("--minimum_response_gain must be finite.")

    if args.dry_run:
        _dry_run(stages, args)
        return

    source = _checkpoint_ref(args.start_from, require_exists=True)
    previous: int | str = (
        _checkpoint_iteration(source.path) if _CHECKPOINT_ITERATION.fullmatch(source.checkpoint) else "start"
    )
    for stage in stages:
        run_name = f"{stage.name}_from{previous}"
        existing_runs = {path for path in _EXPERIMENT_DIR.glob(f"*_{run_name}") if path.is_dir()}
        train_command = _train_command(stage, source, args.iterations_per_stage, run_name, args.learning_rate)
        print(shlex.join(train_command), flush=True)
        subprocess.run(train_command, cwd=_REPO_ROOT, check=True)

        run_dir = _find_training_run(run_name, existing_runs)
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_iteration)
        if not checkpoints:
            raise RuntimeError(f"Training run produced no checkpoints: {run_dir}")
        evaluations = _evaluate_checkpoints(stage, checkpoints)
        selected, selection_note = _select_checkpoint(
            evaluations, args.selection_tolerance_m, args.minimum_response_gain
        )
        _write_summary(
            stage,
            source,
            run_dir,
            args.iterations_per_stage,
            args.selection_tolerance_m,
            args.minimum_response_gain,
            evaluations,
            selected,
            selection_note,
        )
        source = _checkpoint_ref(Path(str(selected["checkpoint"])), require_exists=True)
        previous = stage.range_code


if __name__ == "__main__":
    main()

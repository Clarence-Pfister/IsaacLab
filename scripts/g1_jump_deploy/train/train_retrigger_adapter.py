# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch RSL-RL training with a retrigger-only actor input adapter."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any

from rsl_rl.runners import OnPolicyRunner

from scripts.g1_jump_deploy.train.retrigger_adapter import configure_actor_input_adapter


def main() -> None:
    """Run the standard trainer after constraining resumed actor updates."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--adapter_observation_index", type=int, required=True)
    args, training_args = parser.parse_known_args()
    if "--resume" not in training_args:
        parser.error("--resume is required so the fresh-jump policy is preserved")

    original_load = OnPolicyRunner.load

    def _load_and_constrain(self: OnPolicyRunner, *load_args: Any, **load_kwargs: Any) -> dict:
        infos = original_load(self, *load_args, **load_kwargs)
        weight = configure_actor_input_adapter(self, args.adapter_observation_index)
        print(
            "[INFO]: Retrigger adapter active: "
            f"actor input {args.adapter_observation_index}, "
            f"{weight.shape[0]} mutable actor weights; all other actor values are frozen.",
            flush=True,
        )
        return infos

    OnPolicyRunner.load = _load_and_constrain
    repository_root = Path(__file__).resolve().parents[3]
    training_script = repository_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    sys.argv = [str(training_script), *training_args]
    sys.path.insert(0, str(training_script.parent))
    runpy.run_path(str(training_script), run_name="__main__")


if __name__ == "__main__":
    main()

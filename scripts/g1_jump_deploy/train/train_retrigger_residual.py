# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Initialize and train a repeat-only residual from a base G1 jump actor."""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.retrigger_residual import (
    configure_retrigger_residual_actor,
)

_STANDARD_ON_POLICY_LEARN = OnPolicyRunner.learn


def learn_retrigger_from_full_episode(
    self: OnPolicyRunner,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
) -> Any:
    """Train without artificial initial episode-length randomization.

    A retrigger state is valid only after the actor has executed one complete
    jump-and-stand episode. Randomizing the initial episode counter would make
    the first timeout occur early and can falsely carry a partial trajectory.

    Args:
        self: RSL-RL runner executing the training loop.
        num_learning_iterations: Number of PPO iterations to run.
        init_at_random_ep_len: Ignored request from the standard training entry
            point; retrigger training always starts at episode step zero.

    Returns:
        The standard runner's result.
    """
    del init_at_random_ep_len
    return _STANDARD_ON_POLICY_LEARN(
        self,
        num_learning_iterations,
        init_at_random_ep_len=False,
    )


def load_retrigger_actor_state(
    actor: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Load either a base actor or an already-initialized residual actor.

    Args:
        actor: Residual actor receiving checkpoint weights.
        state_dict: Actor state from a base or residual checkpoint.

    Returns:
        ``"initialized"`` for a base checkpoint or ``"resumed"`` for a
        residual checkpoint.

    Raises:
        RuntimeError: If the checkpoint is not compatible with the actor.
    """
    incompatible = actor.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    residual_keys = {name for name in actor.state_dict() if name.startswith("retrigger_residual.")}
    unexpected = set(incompatible.unexpected_keys)
    if unexpected or missing not in (set(), residual_keys):
        raise RuntimeError(
            "The checkpoint actor is incompatible with the residual actor: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    return "initialized" if missing else "resumed"


def main() -> None:
    """Run the standard trainer after loading only compatible base state."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--retrigger_exploration_std", type=float, default=0.15)
    custom_args, training_args = parser.parse_known_args()
    if "--resume" not in training_args:
        parser.error("--resume is required so a validated base actor can be loaded")

    def _load_retrigger_actor(
        self: OnPolicyRunner,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        del load_cfg, strict
        loaded = torch.load(path, weights_only=False, map_location=map_location or self.device)
        load_mode = load_retrigger_actor_state(self.alg.actor, loaded["actor_state_dict"])
        self.alg.critic.load_state_dict(loaded["critic_state_dict"], strict=True)
        self.current_learning_iteration = int(loaded["iter"])
        configure_retrigger_residual_actor(
            self,
            exploration_std=custom_args.retrigger_exploration_std,
        )
        mutable_parameters = sum(parameter.numel() for parameter in self.alg.actor.retrigger_residual.parameters())
        print(
            f"[INFO]: Repeat residual {load_mode}: "
            f"{mutable_parameters} mutable residual parameters; base actor frozen; "
            f"exploration std={custom_args.retrigger_exploration_std:.3f}.",
            flush=True,
        )
        return loaded.get("infos")

    OnPolicyRunner.load = _load_retrigger_actor
    OnPolicyRunner.learn = learn_retrigger_from_full_episode
    repository_root = Path(__file__).resolve().parents[3]
    training_script = repository_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    sys.argv = [str(training_script), *training_args]
    sys.path.insert(0, str(training_script.parent))
    runpy.run_path(str(training_script), run_name="__main__")


if __name__ == "__main__":
    main()

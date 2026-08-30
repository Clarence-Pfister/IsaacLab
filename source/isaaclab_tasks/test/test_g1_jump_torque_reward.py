# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for torque-aware G1 jump rewards."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump import constants
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp import rewards
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.mdp.rewards import joint_torque_demand_limit


def test_joint_torque_demand_limit_penalizes_unclipped_pd_demand() -> None:
    """Demand beyond the soft effort margin must be visible before actuator clipping."""
    data = SimpleNamespace(
        joint_pos_target=torch.tensor(((1.0, 0.0), (3.0, 0.0))),
        joint_pos=torch.zeros((2, 2)),
        joint_vel=torch.tensor(((0.0, 2.0), (0.0, -5.0))),
        joint_stiffness=torch.full((2, 2), 10.0),
        joint_damping=torch.ones((2, 2)),
        joint_effort_limits=torch.tensor(((20.0, 5.0), (20.0, 5.0))),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        device="cpu",
        num_envs=2,
        start_times=torch.zeros(2),
        episode_length_buf=torch.zeros(2),
        step_dt=0.02,
    )

    cost = joint_torque_demand_limit(
        env,
        soft_ratio=0.8,
        maximum_excess=2.0,
        phase_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )

    # Environment 0 demands [10, -2] N m, both below the soft margin. Environment 1
    # demands [30, +5] N m, giving normalized excesses [0.7, 0.2].
    torch.testing.assert_close(cost, torch.tensor((0.0, 0.53)))


def test_reference_joint_target_deviation_uses_action_scale_and_joint_order(monkeypatch) -> None:
    """A one-scale target error must have the same cost for every joint."""
    joint_names = ("left_hip_pitch_joint", "left_hip_roll_joint")
    scales = torch.tensor([constants.JOINT_ACTION_SCALES[name] for name in joint_names])
    loader = SimpleNamespace(joint_names=joint_names, joint_ids=None)
    loader.get_state = lambda _: (torch.zeros((2, 2)), None, None, None, None, None, None)
    monkeypatch.setattr(rewards, "get_loader", lambda _: loader)
    monkeypatch.setattr(rewards, "get_env_time", lambda _: torch.zeros(2))
    monkeypatch.setattr(rewards, "get_phase_weight", lambda _env, _weights: torch.ones(2))

    # Articulation order is [hip roll, hip pitch], while the reference loader uses
    # [hip pitch, hip roll]. The second environment is one scale away on hip pitch
    # and half a scale away on hip roll.
    joint_pos_target = torch.tensor(((0.0, 0.0), (-0.5 * scales[1], scales[0])))
    robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos_target=joint_pos_target),
        find_joints=lambda _names, preserve_order: ([1, 0], joint_names),
    )
    env = SimpleNamespace(scene={"robot": robot}, device="cpu")

    cost = rewards.reference_joint_target_deviation(
        env,
        phase_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )

    torch.testing.assert_close(cost, torch.tensor((0.0, 0.625)))

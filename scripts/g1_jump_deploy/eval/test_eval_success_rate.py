# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import numpy as np
import torch

from scripts.g1_jump_deploy.eval.eval_success_rate import (
    _FAILURE_TERMS,
    _configure_evaluation,
    _configure_goal_feedback,
    _fit_command_response,
    _pd_torque_demand,
    _success_masks,
)


def test_command_response_fit_recovers_gain_and_offset() -> None:
    commanded_xy = np.asarray(
        (
            (-0.2, -0.15),
            (-0.2, 0.15),
            (0.0, -0.15),
            (0.0, 0.15),
            (0.2, -0.15),
            (0.2, 0.15),
        )
    )
    response_matrix = np.asarray(((0.8, 0.1), (-0.2, 1.1)))
    offset_xy = np.asarray((0.03, -0.02))
    landed_xy = commanded_xy @ response_matrix.T + offset_xy

    fit = _fit_command_response(commanded_xy, landed_xy)

    np.testing.assert_allclose(fit.response_matrix, response_matrix, atol=1.0e-12)
    np.testing.assert_allclose(fit.offset_xy, offset_xy, atol=1.0e-12)
    assert np.all(fit.axis_correlation > 0.9)
    np.testing.assert_allclose(
        fit.mean_absolute_tracking_error_xy,
        np.mean(np.abs(landed_xy - commanded_xy), axis=0),
    )


def test_command_response_fit_marks_collinear_axis_unidentifiable() -> None:
    commanded_x = np.linspace(-0.2, 0.2, 9)
    numerical_noise = 1.0e-5 * np.asarray((1.0, -1.0, 1.0, -1.0, 0.0, -1.0, 1.0, -1.0, 1.0))
    commanded_xy = np.column_stack((commanded_x, 0.01 * commanded_x + numerical_noise))
    landed_xy = np.column_stack((0.8 * commanded_x + 0.03, -0.2 * commanded_x - 0.02))

    fit = _fit_command_response(commanded_xy, landed_xy)

    np.testing.assert_allclose(fit.response_matrix[:, 0], (0.8, -0.2), atol=1.0e-12)
    assert np.all(np.isnan(fit.response_matrix[:, 1]))
    np.testing.assert_allclose(fit.offset_xy, (0.03, -0.02), atol=1.0e-12)


def test_success_requires_yaw_accuracy() -> None:
    result = SimpleNamespace(
        termination_failures={name: np.zeros(2, dtype=np.bool_) for name in _FAILURE_TERMS},
        final_height=np.full(2, 0.8),
        final_tilt_error=np.zeros(2),
        final_goal_error=np.zeros(2),
        final_yaw_error=np.asarray((0.1, 0.3)),
    )

    success = _success_masks(result, height_floor=0.6, tilt_limit_rad=0.5, yaw_limit_rad=0.2)

    for mask in success.values():
        np.testing.assert_array_equal(mask, np.asarray((True, False)))


def test_success_requires_minimum_airborne_time() -> None:
    result = SimpleNamespace(
        termination_failures={name: np.zeros(2, dtype=np.bool_) for name in _FAILURE_TERMS},
        final_height=np.full(2, 0.8),
        final_tilt_error=np.zeros(2),
        final_goal_error=np.zeros(2),
        final_yaw_error=np.zeros(2),
        maximum_airborne_time=np.asarray((0.2, 0.05)),
    )

    success = _success_masks(
        result,
        height_floor=0.6,
        tilt_limit_rad=0.5,
        yaw_limit_rad=0.2,
        minimum_airborne_time_s=0.1,
    )

    for mask in success.values():
        np.testing.assert_array_equal(mask, np.asarray((True, False)))


def test_goal_feedback_override_preserves_task_or_selects_deployable_mode() -> None:
    def obs_goal_remaining_stale():
        pass

    term = SimpleNamespace(func=obs_goal_remaining_stale, params={"freeze_prob": 0.8})
    env_cfg = SimpleNamespace(observations=SimpleNamespace(policy=SimpleNamespace(goal_remaining=term)))

    assert _configure_goal_feedback(env_cfg, "task") == "flight_frozen"
    assert term.func is obs_goal_remaining_stale
    assert _configure_goal_feedback(env_cfg, "latched") == "latched"
    assert term.func.__name__ == "obs_goal_remaining_latched"
    assert term.params == {}


def test_deterministic_evaluation_disables_contact_compliance() -> None:
    noise_terms = {
        name: SimpleNamespace(noise=object())
        for name in ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "goal_remaining")
    }
    policy = SimpleNamespace(enable_corruption=True, **noise_terms)
    events = SimpleNamespace(
        physics_material=object(),
        robot_mass=object(),
        pelvis_com=object(),
        actuator_gains=object(),
        contact_compliance=object(),
        push_robot=object(),
        reset_to_reference=SimpleNamespace(
            params={"roll_range": (-0.1, 0.1), "pitch_range": (-0.1, 0.1), "lin_vel_range": (-0.1, 0.1)}
        ),
    )
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=0),
        seed=None,
        commands=SimpleNamespace(jump_goal=SimpleNamespace(debug_vis=True)),
        sim=SimpleNamespace(device="cuda:0"),
        events=events,
        actions=SimpleNamespace(joint_pos=SimpleNamespace(min_delay_steps=0, max_delay_steps=2)),
        observations=SimpleNamespace(policy=policy),
    )
    args = SimpleNamespace(num_envs=4, seed=7, device=None, no_randomization=True)

    disabled = _configure_evaluation(env_cfg, args)

    assert events.contact_compliance is None
    assert "startup event events.contact_compliance" in disabled


def test_pd_torque_demand_is_measured_before_effort_clipping() -> None:
    demand = _pd_torque_demand(
        joint_pos_target=torch.tensor([[1.0, -1.0]]),
        joint_pos=torch.tensor([[0.2, -0.4]]),
        joint_vel=torch.tensor([[0.1, -0.2]]),
        stiffness=torch.tensor([[10.0, 20.0]]),
        damping=torch.tensor([[2.0, 3.0]]),
    )

    torch.testing.assert_close(demand, torch.tensor([[7.8, -11.4]]))

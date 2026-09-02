# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the G1 goal-range curriculum driver."""

from pathlib import Path

import pytest

from scripts.g1_jump_deploy.train.run_goal_range_curriculum import _parse_evaluation


def test_parse_evaluation_uses_overall_success_table() -> None:
    output = """Overall success (95% Wilson binomial confidence interval):
goal tol [m]  successes   samples      rate              95% CI
        0.10       2290      4096    55.91% [ 54.38%,  57.42%]
        0.15       3035      4096    74.10% [ 72.73%,  75.42%]
        0.20       3727      4096    90.99% [ 90.08%,  91.83%]
        0.30       3727      4096    90.99% [ 90.08%,  91.83%]

Settled displacement response to the commanded displacement:
  [landed_x]   [gain_xx gain_xy] [command_x]   [bias_x]
  [landed_y] = [gain_yx gain_yy] [command_y] + [bias_y]
  response matrix = [[+0.915, n/a], [+0.035, n/a]]
  n/a denotes a command axis that was not independently excited
  offset [m] = [-0.0973, -0.0066]
  same-axis Pearson correlation = [x +0.769, y n/a]
  mean absolute tracking error [m] = [x 0.1030, y 0.0085]
  planar tracking-error norm: p50=0.0808 m, p90=0.1823 m, p95=0.2687 m

Stability, independent of goal accuracy:
  upright at episode end (height > 0.60 m, tilt <= 30 deg): 4096/4096 (100.00%)
  hard fall (base_contact or bad_orientation): 0/4096 (0.00%)
  airborne for >= 0.100 s: 4096/4096 (100.00%)
  maximum airborne time: p50=0.260 s, p05=0.254 s; peak pelvis height p50=0.813 m

  Of the episodes that FAIL each goal tolerance, how many still end upright:
  goal tol [m]  failures   upright   upright %
          0.10      1806      1806     100.00%
          0.15      1061      1061     100.00%
          0.20       369       369     100.00%
          0.30       151       151     100.00%
"""

    evaluation = _parse_evaluation(output, Path("model_1124.pt"))

    assert evaluation["success_rate_0p10"] == pytest.approx(2290 / 4096)
    assert evaluation["success_rate_0p10_successes"] == 2290
    assert evaluation["success_rate_0p10_samples"] == 4096
    assert evaluation["success_rate_0p10_reported_percent"] == 55.91
    assert evaluation["success_rate_0p20"] == pytest.approx(3727 / 4096)

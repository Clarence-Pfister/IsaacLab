# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validated ONNX Runtime wrapper for deployed G1 jump policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxPolicy:
    """Execute a fixed-shape ONNX policy and retain its latest inputs.

    Args:
        path: ONNX policy path.
        observation_dim: Required policy observation dimension.
        action_dim: Required policy action dimension.

    Raises:
        FileNotFoundError: If :paramref:`path` does not exist.
        RuntimeError: If ONNX Runtime is unavailable.
        ValueError: If the model does not have the required single-input,
            single-output interface.
    """

    def __init__(self, path: str | Path, observation_dim: int, action_dim: int):
        policy_path = Path(path).resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Policy execution requires onnxruntime.") from exc
        self._session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("ONNX policy must have exactly one input and one output.")
        if inputs[0].shape != [1, observation_dim] or outputs[0].shape != [1, action_dim]:
            raise ValueError(
                f"ONNX policy shapes must be [1, {observation_dim}]/[1, {action_dim}], "
                f"got {inputs[0].shape}/{outputs[0].shape}."
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self.last_observation = np.zeros(observation_dim, dtype=np.float32)
        self.last_action = np.zeros(action_dim, dtype=np.float64)

    def warm_up(self) -> None:
        """Initialize inference kernels before a timed control loop."""
        self(np.zeros(self._observation_dim, dtype=np.float32))
        self.last_observation.fill(0.0)
        self.last_action.fill(0.0)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Evaluate one observation.

        Args:
            observation: Policy observation, shape ``(observation_dim,)``.

        Returns:
            Raw policy action, shape ``(action_dim,)``.

        Raises:
            ValueError: If the observation or model output is non-finite or
                has the wrong shape.
        """
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != (self._observation_dim,) or not np.all(np.isfinite(observation_array)):
            raise ValueError(f"Policy observation must contain {self._observation_dim} finite values.")
        output = self._session.run(
            [self._output_name],
            {self._input_name: observation_array[np.newaxis, :]},
        )[0]
        action = np.asarray(output, dtype=np.float64).reshape(-1)
        if action.shape != (self._action_dim,) or not np.all(np.isfinite(action)):
            raise ValueError(f"ONNX policy returned invalid action shape or values: {action.shape}.")
        self.last_observation = observation_array.copy()
        self.last_action = action.copy()
        return action

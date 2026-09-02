# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage1-Deploy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage1DeployEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Translation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployTranslationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Uniform-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalUniformEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalSmoothEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-Narrow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalSmoothNarrowEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


for range_name in ("020", "040", "060", "080", "100"):
    gym.register(
        id=f"Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-Range{range_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.jump.jump_env_cfg:"
                f"G1JumpStage2DeployLongitudinalSmoothRange{range_name}EnvCfg"
            ),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
        },
    )


for range_name in ("020", "040", "060", "080", "100"):
    gym.register(
        id=f"Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-RangeContact{range_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.jump.jump_env_cfg:"
                f"G1JumpStage2DeployLongitudinalSmoothRangeContact{range_name}EnvCfg"
            ),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
        },
    )


for range_name in ("020", "040"):
    gym.register(
        id=f"Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-RangeContactTrigger{range_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.jump.jump_env_cfg:"
                f"G1JumpStage2DeployLongitudinalSmoothRangeContactTrigger{range_name}EnvCfg"
            ),
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
        },
    )


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-Narrow-Handoff-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalSmoothNarrowHandoffEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Smooth-Narrow-Repeat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalSmoothNarrowRepeatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Odometry-Smooth-Narrow-Repeat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalOdometrySmoothNarrowRepeatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Latched-Smooth-Narrow-Repeat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalLatchedSmoothNarrowRepeatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Latched-Smooth-Narrow-Direct-Repeat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalLatchedSmoothNarrowDirectRepeatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Latched-Smooth-Narrow-Damped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalLatchedSmoothNarrowDampedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Latched-Smooth-Narrow-Commandable-Repeat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id=(
        "Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Latched-Smooth-Narrow-"
        "Commandable-Repeat-Strong-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatStrongEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id=(
        "Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Latched-Smooth-Narrow-"
        "Commandable-Repeat-Dense-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatDenseEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id=(
        "Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Latched-Smooth-Narrow-"
        "Commandable-Repeat-Retrigger-Aware-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerAwareEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id=(
        "Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Latched-Smooth-Narrow-"
        "Commandable-Repeat-Retrigger-Goal-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerGoalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id=(
        "Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Latched-Smooth-Narrow-"
        "Commandable-Repeat-Retrigger-Residual-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:"
            "G1JumpStage2DeployLongitudinalLatchedSmoothNarrowCommandableRepeatRetriggerChainEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpRetriggerResidualPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Odometry-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalOdometryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Odometry-Smooth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalOdometrySmoothEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Odometry-Smooth-Narrow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalOdometrySmoothNarrowEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Odometry-Smooth-Target-Safe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalOdometrySmoothTargetSafeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Contact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalContactEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Deploy-Longitudinal-Odometry-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.jump.jump_env_cfg:G1JumpStage2DeployLongitudinalOdometryRobustEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Wide-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2WideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Wide-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2WideEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Wide-Land-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2WideLandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage2-Wide-Land-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage2WideLandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage3-Deploy-Translation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage3DeployTranslationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage3-Deploy-Longitudinal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage3DeployLongitudinalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpFineTunePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage3-Narrow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage3NarrowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage3-Narrow-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage3NarrowEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Jump-G1-Stage3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jump.jump_env_cfg:G1JumpStage3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JumpPPORunnerCfg",
    },
)

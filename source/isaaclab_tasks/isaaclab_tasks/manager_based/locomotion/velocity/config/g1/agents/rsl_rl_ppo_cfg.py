# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import preset


@configclass
class G1JumpRetriggerResidualModelCfg(RslRlMLPModelCfg):
    """Actor configuration with a repeat-only residual branch."""

    class_name: str = (
        "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.retrigger_residual:RetriggerResidualMLPModel"
    )
    retrigger_observation_index: int = 245
    residual_hidden_dims: list[int] = [512, 512]


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    # Newton needs ~1.7x the PPO iterations to match PhysX on G1. PhysX saturates near iter 3000
    # (reward ≈ +18, ep_len ≈ 980) and does not meaningfully improve on either metric past that —
    # reward oscillates +16 to +19 through iter 7500, ep_len stays flat. Newton reaches the same
    # (reward, ep_len) quality at iter 5000 (+16 / 984). Comparing reward alone is misleading:
    # ep_len confirms the robot is stable in both cases. The gap is sample-efficiency, not a
    # ceiling — no physics or reward tuning closes it.
    max_iterations = preset(default=3000, newton=5000)
    save_interval = 50
    experiment_name = "g1_rough"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.actor.hidden_dims = [256, 128, 128]
        self.critic.hidden_dims = [256, 128, 128]


@configclass
class G1JumpPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_jump"
        self.max_iterations = 100000
        # Deployment acceptance can regress while aggregate reward continues to rise. Keep
        # enough intermediate candidates to select on deterministic accuracy and saturation
        # without retaining a full 109 MB checkpoint after every policy update.
        self.save_interval = 100
        # The critic reads its own observation group, which carries the base linear
        # velocity the actor is denied because the robot cannot measure it. Without this
        # the critic falls back to the actor group and the asymmetry silently does nothing.
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}

        self.actor.hidden_dims = [1024, 1024, 1024, 1024, 1024, 1024]
        self.critic.hidden_dims = [1024, 1024, 1024, 1024]
        # A hard environment-side clip allowed the previous actor mean to drift as far as
        # +/-59: every value past the clip produced the same target and therefore the same
        # reward. The smooth transformed distribution keeps both training samples and the
        # exported deterministic actor inside the normalized action range by construction.
        self.actor.distribution_cfg.class_name = (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.jump.distributions:TanhGaussianDistribution"
        )
        self.actor.distribution_cfg.init_std = 0.5
        self.num_steps_per_env = 32
        self.algorithm.learning_rate = 5e-5
        self.algorithm.num_mini_batches = 4
        # The transformed distribution has no analytic entropy, and the earlier entropy
        # bonus actively drove actions into saturation after the task reward converged.
        # Exploration still comes from the learned latent standard deviation.
        self.algorithm.entropy_coef = 0.0


@configclass
class G1JumpFineTunePPORunnerCfg(G1JumpPPORunnerCfg):
    """Conservative optimizer settings for deployment-policy curriculum transitions."""

    def __post_init__(self):
        super().__post_init__()

        # A changed observation or reward contract can move the deterministic policy long
        # before aggregate stochastic reward reveals the regression. Keep these transitions
        # fixed and slow; acceptance is selected from deterministic checkpoint evaluations.
        self.save_interval = 25
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.schedule = "fixed"


@configclass
class G1JumpRetriggerResidualPPORunnerCfg(G1JumpFineTunePPORunnerCfg):
    """Fine-tune a gated residual while retaining the complete base actor."""

    def __post_init__(self):
        super().__post_init__()

        base_actor = self.actor
        self.actor = G1JumpRetriggerResidualModelCfg(
            hidden_dims=list(base_actor.hidden_dims),
            activation=base_actor.activation,
            obs_normalization=base_actor.obs_normalization,
            distribution_cfg=base_actor.distribution_cfg,
        )
        self.algorithm.learning_rate = 5.0e-5

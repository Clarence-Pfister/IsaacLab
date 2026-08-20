# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import preset


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
        self.save_interval = 500
        # The critic reads its own observation group, which carries the base linear
        # velocity the actor is denied because the robot cannot measure it. Without this
        # the critic falls back to the actor group and the asymmetry silently does nothing.
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}

        self.actor.hidden_dims = [1024, 1024, 1024, 1024, 1024, 1024]
        self.critic.hidden_dims = [1024, 1024, 1024, 1024]
        self.num_steps_per_env = 32
        self.algorithm.learning_rate = 5e-5
        self.algorithm.num_mini_batches = 4
        # Down from the 0.008 inherited from the rough-terrain config. The loss is
        # surrogate + value - entropy_coef * entropy, so once the task gradient is spent the
        # entropy bonus is the only term left that can still improve the objective, and PPO
        # collects it by widening the action distribution. Late in the wide stage 2 run that
        # cost 5% of the return: mean_std rose 1.63 -> 2.15 while every velocity-sensitive
        # term fell (joint velocity tracking -51%, angular rate -41%) and every position term
        # held flat, which is jitter rather than a policy that forgot the task.
        self.algorithm.entropy_coef = 0.002

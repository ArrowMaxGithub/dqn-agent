import json
import os

import torch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from dqn_agent import DQNMaskedRLModule, DQNTorchLearner
from durak_env import DurakEnv
from random_agent import RandomMaskedRLModule


def env_creator(cfg=None):
    return ParallelPettingZooEnv(DurakEnv())


register_env(
    "custom-cardgame-v1",
    env_creator,
)


def set_epsilon(epsilon: float, algo) -> None:
    algo.env_runner_group.foreach_env_runner(
        lambda w: w.module["p0"].model_config.update({"epsilon": epsilon})
    )


def save_parameters(path, params):
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/parameters.json", "w") as f:
        json.dump(params, f, indent=True)


def policy_mapping_fn(aid, *args, **kwargs):
    assert aid in ("Player 1", "Player 2"), f"Unexpected agent ID: {aid!r}"
    return "p0" if aid == "Player 1" else "opponent"


def dqn_config(params) -> DQNConfig:
    params["distributed_batch_size"] = params["train_batch_size"] // (
        params["num_env_runners"] * params["num_envs_per_env_runner"]
    )

    params["steps_per_iteration"] = (
        params["num_env_runners"]
        * params["num_envs_per_env_runner"]
        * params["distributed_batch_size"]
    )

    params["warmup_iterations"] = (
        params["num_steps_sampled_before_learning_starts"]
        // params["steps_per_iteration"]
    )

    return (
        DQNConfig()
        .debugging(log_level="ERROR")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env="custom-cardgame-v1",
            disable_env_checking=True,
        )
        .multi_agent(
            policies={"p0", "opponent"},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["p0"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "p0": RLModuleSpec(
                        module_class=DQNMaskedRLModule,
                    ),
                    "opponent": RLModuleSpec(
                        module_class=RandomMaskedRLModule,
                        inference_only=True,
                    ),
                }
            )
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1 if torch.cuda.is_available() else 0,
        )
        .env_runners(
            num_env_runners=params["num_env_runners"],
            num_envs_per_env_runner=params["num_envs_per_env_runner"],
        )
        .training(
            learner_class=DQNTorchLearner,
            replay_buffer_config={
                "type": "MultiAgentEpisodeReplayBuffer",
                "capacity": params["replay_buffer_capacity"],
            },
            lr=params["learning_rate"],
            double_q=params["double_q"],
            train_batch_size_per_learner=params["train_batch_size"],
            num_steps_sampled_before_learning_starts=params[
                "num_steps_sampled_before_learning_starts"
            ],
            target_network_update_freq=params["target_network_update_freq"],
            td_error_loss_fn=params["td_error_loss_fn"],
            n_step=params["n_step"],
            adam_epsilon=params["adam_epsilon"],
            grad_clip=params["grad_clip"],
            tau=params["tau"],
            gamma=params["gamma"],
            grad_clip_by=params["grad_clip_by"],
            training_intensity=params["training_intensity"],
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=params["num_eval_env_runners"],
            evaluation_duration_unit="episodes",
            evaluation_duration=params["eval_episodes"],
        )
    )

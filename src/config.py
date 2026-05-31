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
from self_play import SelfPlayCallback, OpponentPool, policy_mapping_fn_creator


def env_creator(cfg=None):
    return ParallelPettingZooEnv(DurakEnv())


register_env(
    "durak-v1",
    env_creator,
)


def set_epsilon(epsilon: float, algorithm) -> None:
    algorithm.env_runner_group.foreach_env_runner(
        lambda w: w.module["dqn"].model_config.update({"epsilon": epsilon})
    )


def save_parameters(path, params):
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/parameters.json", "w") as f:
        json.dump(params, f, indent=True)


def dqn_config(params: dict, opponents: dict, checkpoint_path: str) -> DQNConfig:
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

    tmp_env = env_creator()

    rl_module_specs = {
        policy_id: RLModuleSpec(
            module_class=module,
            inference_only=True,
            observation_space=tmp_env.observation_space,
            action_space=tmp_env.action_space,
        )
        for policy_id, module in opponents.items()
    }

    rl_module_specs["dqn"] = RLModuleSpec(
        module_class=DQNMaskedRLModule,
    )

    opponent_pool = OpponentPool(
        capacity=params["self_play_capacity"],
        recency_bias=params["self_play_recency_bias"],
        opponents=opponents.keys(),
    )
    train_callback = SelfPlayCallback(
        interval=params["self_play_interval"],
        warmup_iterations=params["warmup_iterations"],
        self_play_confidence=params["self_play_confidence"],
        checkpoint_path=checkpoint_path,
        opponent_pool=opponent_pool,
    )

    train_policy_mapping_fn = policy_mapping_fn_creator(opponent_pool, False)
    eval_policy_mapping_fn = policy_mapping_fn_creator(opponent_pool, True)

    config = (
        DQNConfig()
        .debugging(log_level="ERROR")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env="durak-v1",
            disable_env_checking=True,
        )
        .multi_agent(
            policies=["dqn"],
            policy_mapping_fn=train_policy_mapping_fn,
            policies_to_train=["dqn"],
        )
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=rl_module_specs))
        .learners(
            learner_class=DQNTorchLearner,
            num_learners=1,
            num_gpus_per_learner=1 if torch.cuda.is_available() else 0,
        )
        .env_runners(
            num_env_runners=params["num_env_runners"],
            num_envs_per_env_runner=params["num_envs_per_env_runner"],
        )
        .training(
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
            evaluation_interval=params["eval_interval"],
            evaluation_num_env_runners=params["num_eval_env_runners"],
            evaluation_duration_unit="episodes",
            evaluation_duration=params["eval_episodes"],
        )
        .callbacks(
            on_train_result=lambda algorithm, metrics_logger, result: (
                train_callback.on_train_result(
                    algorithm=algorithm, metrics_logger=metrics_logger, result=result
                )
            )
        )
    )

    algorithm = config.build_algo()

    # Override policies used during eval: Only static baselines
    algorithm.eval_env_runner_group.foreach_env_runner(
        lambda env_runner: env_runner.config.multi_agent(
            policy_mapping_fn=eval_policy_mapping_fn,
        ),
        local_env_runner=True,
    )

    return algorithm

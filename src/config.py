import json
import os
from pathlib import Path

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from dqn_agent import DQNMaskedRLModule, DQNTorchLearner
from durak_env import DurakEnv


class OpponentPool:
    def __init__(self, capacity: int, opponents: list[str]):
        self.buffer = [module_id for module_id in opponents]
        self.capacity = capacity
        print(f"Initialized opponent pool with {len(self.buffer)} opponents")

    def opponents(self) -> list[str]:
        return list(self.buffer)

    def num_opponents(self) -> int:
        return len(self.buffer)

    def add(self, module_id):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)

        print(f"Added {module_id} to the opponent pool")

        self.buffer.append(module_id)

    def sample(self):
        return np.random.choice(self.buffer)


def env_creator(cfg=None):
    return ParallelPettingZooEnv(DurakEnv())


register_env(
    "custom-cardgame-v1",
    env_creator,
)


def set_epsilon(epsilon: float, algo) -> None:
    algo.env_runner_group.foreach_env_runner(
        lambda w: w.module["dqn"].model_config.update({"epsilon": epsilon})
    )


def policy_mapping_fn_creator(opponent_pool: OpponentPool):
    def policy_mapping_fn(aid, *args, **kwargs):
        assert aid in ("Player 1", "Player 2"), f"Unexpected agent ID: {aid!r}"
        return "dqn" if aid == "Player 1" else opponent_pool.sample()

    return policy_mapping_fn


def save_parameters(path, params):
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/parameters.json", "w") as f:
        json.dump(params, f, indent=True)


class SelfPlayCallback(RLlibCallback):
    def __init__(
        self,
        interval: int,
        checkpoint_path: str,
        opponent_pool: OpponentPool,
    ):
        super().__init__()
        self.next_version = 1
        self.save_interval = interval
        self.checkpoint_path = checkpoint_path
        self.opponent_pool = opponent_pool

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        result,
        metrics_logger,
        **kwargs,
    ):
        iteration = result["training_iteration"]
        if iteration == 0 or iteration % self.save_interval != 0:
            return

        version_path = Path(
            f"{self.checkpoint_path}/version_{self.next_version}"
        ).resolve()
        print(f"Saving version {self.next_version} to {version_path}")

        algorithm.save(version_path)

        module_id = f"dqn_{self.next_version}"
        self.next_version += 1

        self.opponent_pool.add(module_id=module_id)

        policy_mapping_fn = policy_mapping_fn_creator(self.opponent_pool)

        local_worker = algorithm.env_runner
        multi_rl_module = local_worker.module
        module = multi_rl_module["dqn"]

        algorithm.add_module(
            module_id=module_id,
            module_spec=RLModuleSpec.from_module(module),
        )
        algorithm.set_state(
            {
                "learner_group": {
                    "learner": {
                        "rl_module": {
                            module_id: multi_rl_module[module_id].get_state(),
                        }
                    }
                }
            }
        )

        algorithm.env_runner_group.foreach_env_runner(
            lambda env_runner: env_runner.config.multi_agent(
                policy_mapping_fn=policy_mapping_fn,
            ),
            local_env_runner=True,
        )

        algorithm.env_runner_group.sync_weights(policies=[module_id])

        result["opponent_pool_size"] = self.opponent_pool.num_opponents()


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
        params["self_play_capacity"], opponents=opponents.keys()
    )
    train_callback = SelfPlayCallback(
        interval=params["self_play_interval"],
        checkpoint_path=checkpoint_path,
        opponent_pool=opponent_pool,
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
            policies=["dqn"],
            policy_mapping_fn=policy_mapping_fn_creator(opponent_pool),
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
            evaluation_interval=1,
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

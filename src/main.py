import logging
import os
import warnings

import ray
import torch
from prettytable import PrettyTable
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.tune.registry import register_env

from dqn_agent import DQNMaskedRLModule
from durak_env import DurakEnv
from random_agent import RandomMaskedRLModule

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
logging.getLogger("ray").setLevel(logging.ERROR)
ray.init(logging_level=logging.ERROR, configure_logging=True, ignore_reinit_error=True)


def env_creator(cfg):
    return ParallelPettingZooEnv(DurakEnv())


register_env(
    "custom-cardgame-v1",
    env_creator,
)


def main():
    if torch.cuda.is_available():
        print("GPU supported")
    else:
        print("NO GPU SUPPORT")

    learning_rate = 1e-5
    episodes = 1024  # 1 Episode = 1 Full game
    num_env_runners = 16
    num_envs_per_env_runner = 32
    replay_buffer_capacity = 65536 * 16
    initial_epsilon = 1.0
    final_epsilon = 0.1
    dueling = False
    double_q = False
    train_batch_size = 2048

    steps_to_final_epsilon = episodes * train_batch_size

    config = (
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
            policies={"p0", "random"},
            policy_mapping_fn=lambda aid, *args, **kwargs: "p0",
            policies_to_train=["p0"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "p0": RLModuleSpec(
                        module_class=DQNMaskedRLModule,
                        model_config={"action_dist_class": TorchCategorical},
                    ),
                    "random": RLModuleSpec(
                        module_class=RandomMaskedRLModule,
                        model_config={"action_dist_class": TorchCategorical},
                    ),
                }
            )
        )
        .resources(num_gpus=1)
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
        )
        .training(
            replay_buffer_config={
                "type": "MultiAgentEpisodeReplayBuffer",
                "capacity": replay_buffer_capacity,
            },
            lr=learning_rate,
            epsilon=[(0, initial_epsilon), (steps_to_final_epsilon, final_epsilon)],
            dueling=dueling,
            double_q=double_q,
            train_batch_size=train_batch_size,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=16,
            evaluation_duration_unit="episodes",
            evaluation_duration=64,
            evaluation_config=DQNConfig.overrides(
                policy_mapping_fn=lambda aid, *args, **kwargs: (
                    "p0" if aid == "Player 1" else "random"
                )
            ),
        )
    )

    algo = config.build_algo()
    for e in range(episodes):
        results = algo.train()
        episode_return_mean = results["evaluation"]["env_runners"][
            "agent_episode_returns_mean"
        ]["Player 1"]
        print(f"Episode {e} return mean vs random: {episode_return_mean:.3f}")


def print_epoch_results(epoch_results):
    table = PrettyTable()
    table.field_names = ["Pairing", "Epoch", "Wins", "Draws", "Losses"]
    for (epoch, a0, a1), r in epoch_results.items():
        wins = f"{r[0]:.2f}"
        draws = f"{r[1]:.2f}"
        losses = f"{r[2]:.2f}"
        table.add_row([f"{a0} vs {a1}", epoch, wins, draws, losses])
    print(table)


def print_results(results):
    table = PrettyTable()
    table.field_names = ["Pairing", "Wins", "Draws", "Losses"]
    for (a0, a1), r in results.items():
        wins = f"{r[0]:.2f}"
        draws = f"{r[1]:.2f}"
        losses = f"{r[2]:.2f}"
        table.add_row([f"{a0} vs {a1}", wins, draws, losses])
    print(table)


if __name__ == "__main__":
    main()

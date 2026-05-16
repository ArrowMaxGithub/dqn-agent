import logging
import os
import warnings

import ray
import torch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.utils.metrics import EVALUATION_RESULTS, ENV_RUNNER_RESULTS

from dqn_agent import DQNMaskedRLModule
from durak_env import DurakEnv
from random_agent import RandomMaskedRLModule

from pathlib import Path
import datetime

from tqdm import tqdm

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

    learning_rate = 0.0001
    iterations = 16
    num_env_runners = 16
    num_envs_per_env_runner = 8
    replay_buffer_capacity = 65536 * 2
    dueling = True
    double_q = True
    train_batch_size = 4096

    decay_steps = num_env_runners * num_envs_per_env_runner * iterations

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
                        model_config={
                            "initial_epsilon": 0.5,
                            "final_epsilon": 0.01,
                            "decay_steps": decay_steps,
                        },
                    ),
                    "random": RLModuleSpec(
                        module_class=RandomMaskedRLModule,
                    ),
                }
            )
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1 if torch.cuda.is_available() else 0,
        )
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
            dueling=dueling,
            double_q=double_q,
            train_batch_size_per_learner=train_batch_size,
            num_steps_sampled_before_learning_starts=0,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=16,
            evaluation_duration_unit="episodes",
            evaluation_duration=10000,
            evaluation_config=DQNConfig.overrides(
                policy_mapping_fn=lambda aid, *args, **kwargs: (
                    "p0" if aid == "Player 1" else "random"
                )
            ),
        )
    )

    algo = config.build_algo()

    # For each iteration => train_batch_size environment steps are generated
    # These are distributed across all env_runners
    # Each env_runner distributes its allocated steps to their environments
    #
    # For batch_size = 2048, env_runner = 16, envs_per_runner = 32:
    # Each runner gets: 2048 / 16 = 128 steps
    # Each env advances by: 128 / 32 = 4 steps
    pbar = tqdm(range(iterations))
    for i in pbar:
        results = algo.train()
        eval_runners = results.get(EVALUATION_RESULTS, {}).get(ENV_RUNNER_RESULTS, {})
        agent_returns = eval_runners.get("agent_episode_returns_mean", {})
        episode_return_mean = agent_returns.get("Player 1", 0.0)
        pbar.set_description(f"Avg vs rand: {episode_return_mean:.3f}")

    timestamp = datetime.datetime.now()
    path = Path(f"./checkpoints/dqn/{timestamp}").resolve()
    os.makedirs(path, exist_ok=True)
    algo.save(path)


if __name__ == "__main__":
    main()

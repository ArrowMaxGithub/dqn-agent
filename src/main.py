import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS
from ray.tune.registry import register_env
from tqdm import tqdm

from dqn_agent import DQNAgent, DQNMaskedRLModule
from durak_env import DurakEnv
from random_agent import RandomAgent, RandomMaskedRLModule

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
logging.getLogger("ray").setLevel(logging.ERROR)
ray.init(logging_level=logging.ERROR, configure_logging=True, ignore_reinit_error=True)


def main():
    if torch.cuda.is_available():
        print("GPU supported")
    else:
        print("NO GPU SUPPORT")

    experiment_name = "2026_05_17_n_step_10"
    learning_rate = 1e-4
    iterations = 512
    num_env_runners = 16
    num_envs_per_env_runner = 8
    replay_buffer_capacity = 65536 * 16
    dueling = True
    double_q = True
    train_batch_size = 2048
    num_steps_sampled_before_learning_starts = 65536 * 4
    target_network_update_freq = train_batch_size * 4
    td_error_loss_fn = "huber"
    n_step = 10
    adam_epsilon = 1e-3
    grad_clip = 10.0
    tau = 1.0

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
            policies={"p0", "opponent"},
            policy_mapping_fn=lambda aid, *args, **kwargs: (
                "p0" if aid == "Player 1" else "opponent"
            ),
            policies_to_train=["p0"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "p0": RLModuleSpec(
                        module_class=DQNMaskedRLModule,
                        model_config={"epsilon": 1.0},
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
            num_steps_sampled_before_learning_starts=num_steps_sampled_before_learning_starts,
            epsilon=1.0,  # Set during training iteration
            target_network_update_freq=target_network_update_freq,
            td_error_loss_fn=td_error_loss_fn,
            n_step=n_step,
            adam_epsilon=adam_epsilon,
            grad_clip=grad_clip,
            tau=tau,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=16,
            evaluation_duration_unit="episodes",
            evaluation_duration=100,
            evaluation_config=DQNConfig.overrides(
                policy_mapping_fn=lambda aid, *args, **kwargs: (
                    "p0" if aid == "Player 1" else "opponent"
                ),
                epsilon=0.0,
            ),
        )
    )

    algo = config.build_algo()
    algo.env_runner_group.foreach_env_runner(
        lambda w: w.module["p0"].model_config.update({"epsilon": 1.0})
    )

    steps_per_iteration = (
        num_env_runners
        * num_envs_per_env_runner
        * (train_batch_size // (num_env_runners * num_envs_per_env_runner))
    )
    warmup_iterations = num_steps_sampled_before_learning_starts // steps_per_iteration

    set_epsilon(epsilon=1.0, algo=algo)

    mean_rewards = []
    epsilons = []

    algo_path = Path(f"./checkpoints/dqn/{experiment_name}").resolve()
    plots_path = Path(f"./checkpoints/dqn/{experiment_name}/plots").resolve()
    os.makedirs(algo_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    pbar = tqdm(range(warmup_iterations))
    for i in pbar:
        results = algo.train()
        eval_runners = results.get(EVALUATION_RESULTS, {}).get(ENV_RUNNER_RESULTS, {})
        agent_returns = eval_runners.get("agent_episode_returns_mean", {})
        mean = agent_returns.get("Player 1", 0.0)
        mean_rewards.append(mean)
        epsilons.append(1.0)
        save_plot(
            f"{plots_path}/mean_reward.svg", warmup_iterations, mean_rewards, epsilons
        )
        pbar.set_description(f"Warmup vs rand: eps: {1.0} wins: {mean:.3f}")

    try:
        pbar = tqdm(range(iterations))
        for i in pbar:
            epsilon = max(0.05, 1.0 - (1.0 - 0.05) * (i / iterations / 0.67))
            set_epsilon(epsilon=epsilon, algo=algo)
            results = algo.train()

            eval_runners = results.get(EVALUATION_RESULTS, {}).get(
                ENV_RUNNER_RESULTS, {}
            )
            agent_returns = eval_runners.get("agent_episode_returns_mean", {})
            mean = agent_returns.get("Player 1", 0.0)
            mean_rewards.append(mean)
            epsilons.append(epsilon)
            save_plot(
                f"{plots_path}/mean_reward.svg",
                warmup_iterations,
                mean_rewards,
                epsilons,
            )
            pbar.set_description(f"Avg vs rand: eps: {epsilon:.3f} wins: {mean:.3f}")

        algo.save(algo_path)

    finally:
        save_plot(
            f"{plots_path}/mean_reward.svg", warmup_iterations, mean_rewards, epsilons
        )


def save_plot(path, warmup_iterations, mean_rewards, epsilons):
    length = min(len(mean_rewards), len(epsilons))
    xs = np.array([i - warmup_iterations for i in range(length)])
    ys = np.array(mean_rewards[:length])
    es = np.array(epsilons[:length])

    plt.plot(xs, ys, label="mean reward")
    plt.plot(xs, es, label="epsilon")
    plt.savefig(path)


def validate():
    dqn = DQNAgent.load("checkpoints/dqn/test")
    rand = RandomAgent(36)

    results = validate(
        env_factory=raw_env_creator, agents=(dqn, rand), n_episodes=10000
    )
    wins = results[0]
    losses = results[2]
    print(f"{wins:.2f}% | {losses:.2f}%")


def set_epsilon(epsilon: float, algo) -> None:
    algo.env_runner_group.foreach_env_runner(
        lambda w: w.module["p0"].model_config.update({"epsilon": epsilon})
    )
    if algo.eval_env_runner_group is not None:
        algo.eval_env_runner_group.foreach_env_runner(
            lambda w: w.module["p0"].model_config.update({"epsilon": 0.0})
        )


def raw_env_creator(cfg=None):
    return DurakEnv()


def env_creator(cfg=None):
    return ParallelPettingZooEnv(DurakEnv())


register_env(
    "custom-cardgame-v1",
    env_creator,
)

if __name__ == "__main__":
    main()

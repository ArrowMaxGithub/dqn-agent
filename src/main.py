import time

import torch
from prettytable import PrettyTable

from dqn_agent import DQNAgent
from durak_env import DurakEnv
from q_agent import QAgent
from random_agent import RandomAgent
from test import test_all, test_all_checkpoints
import ray
import logging
import warnings
import os
import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
logging.getLogger("ray").setLevel(logging.ERROR)
ray.init(logging_level=logging.ERROR, configure_logging=True, ignore_reinit_error=True)


def env_factory() -> DurakEnv:
    return DurakEnv()


def main():
    if torch.cuda.is_available():
        print("GPU supported")
    else:
        print("NO GPU SUPPORT")

    epochs = 10
    episodes_per_epoch = 8192
    episodes_test = 1000
    n_steps_total = epochs * episodes_per_epoch
    train_batch_size = 4096

    tmp_env = DurakEnv()
    dqn = DQNAgent().new(
        passing_action=tmp_env.passing_action,
        learning_rate=1e-5,
        n_steps_total=n_steps_total,
        train_batch_size=train_batch_size,
        num_steps_sampled_before_learning_starts=4096,  # Initialize with 1 full batch
        replay_buffer_capacity=65536 * 16,
        num_env_runners=16,
        num_envs_per_env_runner=32,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        dueling=False,
        double_q=False,
    )
    rand = RandomAgent(passing_action=tmp_env.passing_action)

    pairings = ((dqn, rand),)
    self_train = ((dqn, episodes_per_epoch * 64),)
    _timestamp = datetime.datetime.now()

    total_start = time.perf_counter()
    print("Cross table with untrained agents")
    print_results(test_all(env_factory, pairings, episodes_test))
    elapsed = time.perf_counter() - total_start
    print(f"Cross table completed after {elapsed:.2f}s")
    print("-" * 16)

    training_start = time.perf_counter()
    print("Starting training")
    for agent, n_episodes in self_train:
        label = agent.get_label()
        print(f"{label} vs {label}")
        for epoch in range(epochs):
            print(f"Learning epoch {epoch}")
            agent.train(env_factory, n_episodes)
            # path = f"./checkpoints/{timestamp}/{label}/{epoch}/"
            # print(f"Saving agent to {path}")
            # agent.save(path)
            print_results(test_all(env_factory, pairings, episodes_test))

    elapsed = time.perf_counter() - training_start
    print(f"Training completed after: {elapsed:.2f}s")

    elapsed = time.perf_counter() - total_start
    print(f"Total runtime: {elapsed:.2f}s")


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

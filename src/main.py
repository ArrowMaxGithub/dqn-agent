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
    episodes_per_epoch = 4096
    episodes_test = 10000
    episodes_final = 10000
    n_steps_total_q = epochs * episodes_per_epoch * 2
    n_steps_total_dqn = n_steps_total_q * 64
    train_batch_size = 1024

    tmp_env = DurakEnv()
    q = QAgent().new(
        passing_action=tmp_env.passing_action,
        n_action_space=tmp_env.n_action_space,
        learning_rate=1e-5,
        n_steps_total=n_steps_total_q,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        discount_factor=0.95,
        illegal_mask=-1e34,
    )
    dqn = DQNAgent().new(
        passing_action=tmp_env.passing_action,
        learning_rate=1e-5,
        n_steps_total=n_steps_total_dqn,
        train_batch_size=train_batch_size,
        num_steps_sampled_before_learning_starts=65536,
        num_env_runners=32,
        num_envs_per_env_runner=32,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        dueling=False,
        double_q=False,
    )
    rand = RandomAgent(passing_action=tmp_env.passing_action)

    full_pairings = ((q, q), (q, dqn), (dqn, dqn), (q, rand), (dqn, rand), (rand, rand))
    self_train = (
        (q, episodes_per_epoch),
        (dqn, episodes_per_epoch * 256),
    )
    test_pairings = ((q, dqn), (q, rand), (dqn, rand))

    start = time.perf_counter()
    total_start = start
    print("Cross table with untrained agents")
    print_results(test_all(env_factory, full_pairings, episodes_test))
    elapsed = time.perf_counter() - start
    print(f"Cross table completed after {elapsed:.2f}s")
    print("-" * 16)

    start_training = time.perf_counter()

    print("Starting training")
    for agent, n_episodes in self_train:
        start = time.perf_counter()
        label = agent.get_label()
        print(f"{label} vs {label}")
        for epoch in range(epochs):
            print(f"Learning epoch {epoch}")
            agent.train(env_factory, n_episodes)
            path = f"./checkpoints/{label}/{epoch}/"
            print(f"Saving agent to {path}")
            agent.save(path)
        elapsed = time.perf_counter() - start
        print(f"Training completed after {elapsed:.2f}s")
        print()

    elapsed = time.perf_counter() - start_training
    print(f"Training completed after {elapsed:.2f}s")
    print("-" * 16)

    start_testing = time.perf_counter()
    print("Testing agent checkpoints")
    for pairing in test_pairings:
        print_epoch_results(
            test_all_checkpoints(env_factory, pairing, epochs, episodes_test)
        )

    elapsed = time.perf_counter() - start_testing
    print(f"Testing completed after {elapsed:.2f}s")
    print("-" * 16)

    start_final = time.perf_counter()
    print("Cross table with final checkpoints")
    print_results(test_all(env_factory, full_pairings, episodes_final))
    elapsed = time.perf_counter() - start_final
    print(f"Cross table completed after {elapsed:.2f}s")
    print("-" * 16)

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

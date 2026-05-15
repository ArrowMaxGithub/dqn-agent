import torch

from prettytable import PrettyTable

from dqn_agent import DQNAgent
from durak_env import DurakEnv
from q_agent import QAgent
from random_agent import RandomAgent
from test import cross
from train import train
import time


def env_factory() -> DurakEnv:
    return DurakEnv()


def main():
    if torch.cuda.is_available():
        print("GPU supported")
    else:
        print("NO GPU SUPPORT")

    epochs = 5
    episodes_per_epoch = 1024
    episodes_test = 1000
    episodes_final = 1000
    n_steps_total = epochs * episodes_per_epoch
    train_batch_size = 256

    tmp_env = DurakEnv()
    agents = [
        QAgent().new(
            passing_action=tmp_env.passing_action,
            n_action_space=tmp_env.n_action_space,
            learning_rate=0.001,
            n_steps_total=n_steps_total,
            initial_epsilon=1.0,
            final_epsilon=0.1,
            discount_factor=0.95,
            illegal_mask=-1e34,
        ),
        DQNAgent().new(
            passing_action=tmp_env.passing_action,
            learning_rate=0.001,
            n_steps_total=n_steps_total,
            train_batch_size=train_batch_size,
            num_env_runners=8,
            num_envs_per_env_runner=32,
            initial_epsilon=1.0,
            final_epsilon=0.1,
            dueling=False,
            double_q=False,
        ),
        RandomAgent(passing_action=tmp_env.passing_action),
    ]

    full_pairings = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))
    learn_pairings = (
        (0, 0, episodes_per_epoch),
        (1, 1, episodes_per_epoch // train_batch_size),
    )
    test_pairings = ((0, 1), (0, 2), (1, 2))

    start = time.perf_counter()
    total_start = start
    print("Cross table with untrained agents")
    print_cross_results(cross(env_factory, agents, full_pairings, episodes_test))
    elapsed = time.perf_counter() - start
    print(f"Cross table completed after {elapsed}s")
    print("-" * 16)

    print("Starting training")
    for a0, a1, n_episodes in learn_pairings:
        print(f"{agents[a0].get_label()} vs {agents[a1].get_label()}")

        for epoch in range(epochs):
            print(f"Learning epoch {epoch}")
            train(env_factory, (agents[a0], agents[a1]), n_episodes)
            label = agents[a0].get_label()
            path = f"./checkpoints/{label}/{epoch}/"
            print(f"Saving agent to {path}")
            agents[a0].save(path)
        print()

    elapsed = time.perf_counter() - start
    print(f"Training completed after {elapsed}s")
    print("-" * 16)

    start = time.perf_counter()
    print("Testing agent checkpoints")
    for a0, a1 in test_pairings:
        for epoch in range(epochs):
            label_0 = agents[a0].get_label()
            path_0 = f"./checkpoints/{label_0}/{epoch}/"
            agent_0 = load_agent(label_0, path_0)

            label_1 = agents[a1].get_label()
            path_1 = f"./checkpoints/{label_1}/{epoch}/"
            agent_1 = load_agent(label_1, path_1)

            agents = (agent_0, agent_1)

            print(f"Results after epoch {epoch}")
            print_cross_results(
                cross(env_factory, agents, test_pairings, episodes_test)
            )

    elapsed = time.perf_counter() - start
    print(f"Testing completed after {elapsed}s")
    print("-" * 16)

    start = time.perf_counter()
    print("Cross table with final checkpoints")
    print_cross_results(cross(env_factory, agents, full_pairings, episodes_final))
    elapsed = time.perf_counter() - start
    print(f"Cross table completed after {elapsed}s")
    print("-" * 16)

    elapsed = time.perf_counter() - total_start
    print(f"Total runtime: {elapsed}s")


def print_cross_results(results):
    table = PrettyTable()
    table.field_names = ["Pairing", "Wins", "Draws", "Losses"]
    for (a0, a1), r in results.items():
        wins = f"{r[0]:.2f}"
        draws = f"{r[1]:.2f}"
        losses = f"{r[2]:.2f}"
        table.add_row([f"{a0} vs {a1}", wins, draws, losses])
    print(table)


def load_agent(label, path):
    match label:
        case "QAgent":
            return QAgent.load(path)
        case "DQNAgent":
            return DQNAgent.load(path)


if __name__ == "__main__":
    main()

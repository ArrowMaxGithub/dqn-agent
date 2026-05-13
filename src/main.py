import torch
from prettytable import PrettyTable

from dqn_agent import DQNAgent
from env import Cardgame
from q_agent import QAgent
from random_agent import RandomAgent
from test import cross
from train import train


def main():
    if torch.cuda.is_available():
        print("GPU supported")
    else:
        assert False, "No GPU support"

    epochs = 4
    episodes_per_epoch = 2048
    episodes_test = 1_000
    n_steps_total = epochs * episodes_per_epoch
    train_batch_size = 2048

    num_cards = 8
    num_hand_cards = 4

    env = Cardgame(num_cards=num_cards, num_hand_cards=num_hand_cards)
    agents = [
        QAgent(
            passing_action=env.passing_action,
            n_action_space=env.n_action_space,
            learning_rate=0.001,
            n_steps_total=n_steps_total,
            initial_epsilon=1.0,
            final_epsilon=0.1,
            discount_factor=0.95,
            illegal_mask=-1e34,
        ),
        DQNAgent(
            passing_action=env.passing_action,
            learning_rate=0.001,
            n_steps_total=n_steps_total,
            train_batch_size=train_batch_size,
            initial_epsilon=1.0,
            final_epsilon=0.1,
            dueling=False,
            double_q=False,
        ),
        RandomAgent(passing_action=env.passing_action),
    ]

    full_pairings = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))
    learn_pairings = (
        (0, 0, episodes_per_epoch),
        (1, 1, episodes_per_epoch // train_batch_size),
    )
    test_pairings = ((0, 1), (0, 2), (1, 2))

    print("Untrained agents")
    print_cross_results(cross(env, agents, full_pairings, episodes_test))

    for epoch in range(epochs):
        print(f"Starting epoch {epoch} | after {episodes_per_epoch * epoch} iterations")

        for a0, a1, n_episodes in learn_pairings:
            print(f"{agents[a0].get_label()} vs {agents[a1].get_label()}")
            train((agents[a0], agents[a1]), env, n_episodes)

        print(f"Results after epoch {epoch}")
        print_cross_results(cross(env, agents, test_pairings, episodes_test))


def print_cross_results(results):
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

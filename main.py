from agent import QAgent
from agent import RandomAgent
from train import train
from test import cross
from env import Cardgame


def print_cross_results(results):
    for (a0, a1), r in results.items():
        wins = r[0]
        draws = r[1]
        losses = r[2]
        pairing = f"{a0} vs {a1}"
        print(f"{pairing:<16}: {wins:.2f} | {draws:.2f} | {losses:.2f}")
    print("-" * 16)


def main():
    learning_rate = 0.001
    epochs = 10
    episodes_per_epoch = 1_000
    episodes_test = 10_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (epochs * episodes_per_epoch) / 2
    final_epsilon = 0.1

    num_cards = 8
    num_hand_cards = 4

    env = Cardgame(num_cards=num_cards, num_hand_cards=num_hand_cards)
    agents = [
        QAgent(
            env.passing_action,
            env.n_action_space,
            learning_rate,
            start_epsilon,
            epsilon_decay,
            final_epsilon,
        ),
        QAgent(
            env.passing_action,
            env.n_action_space,
            learning_rate,
            start_epsilon,
            epsilon_decay,
            final_epsilon,
        ),
        RandomAgent(env.passing_action),
    ]

    print("Untrained agents")
    pairings = ((0, 1), (0, 2), (2, 2))
    print_cross_results(cross(env, agents, pairings, episodes_test))

    for epoch in range(epochs):
        print(f"Starting epoch {epoch} | after {episodes_per_epoch * epoch} iterations")
        train(agents, env, episodes_per_epoch)
        print(f"Results after epoch {epoch}")
        pairings = ((0, 1), (0, 2))
        print_cross_results(cross(env, agents, pairings, episodes_test))


if __name__ == "__main__":
    main()

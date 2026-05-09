# N Karten: 0 bis N
# 2 Spieler
# X Handkarten zu Beginn aufgenommen

# Ein zufälliger Spieler beginnt und legt eine Karte aus
# Der nächste Spieler muss diese Karte schlagen
# Wenn die Karte erfolgreich geschlagen wurde, wechseln die Rollen

# Das Spiel endet, wenn eine Karte nicht geschlagen werden kann => Verteidiger verliert
# Wenn beide Spieler keine Karten mehr haben, endet das Spiel unentschieden

# ActionSpace: Discrete[1] = Eine Karte mit Wert X ausspielen
# ObsSpace: MultiDiscrete[8, 4] = 8 Karten mit 4 mögliche States:
#     Unknown
#     My_Card
#     Active_Card
#     Discarded

import gymnasium as gym
from typing import Optional
from qagent import QAgent
import train
import test


class Cardgame(gym.Env):
    def __init__(self, num_cards=8, num_hand_cards=3):
        assert num_hand_cards * 2 <= num_cards
        self.num_cards = num_cards
        self.num_hand_cards = num_hand_cards
        self.observation_space = gym.spaces.MultiDiscrete([4] * num_cards)
        self.action_space = gym.spaces.Discrete(num_cards)

    def _get_info(self):
        return {"phase": self.phase}

    def _get_obs(self):
        states = [0] * self.num_cards
        for card in self.discard:
            states[card] = 3

        for card in self.agent_cards:
            states[card] = 1

        if self.attacking_card:
            states[self.attacking_card] = 2

        if self.defending_card:
            states[self.defending_card] = 2

        return states

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        deck = [i for i in range(self.num_cards)]
        agent_cards = []
        opponent_cards = []

        while len(agent_cards) < self.num_hand_cards:
            card = deck.pop(self.np_random.integers(0, len(deck), dtype=int))
            agent_cards.append(card)

        while len(opponent_cards) < self.num_hand_cards:
            card = deck.pop(self.np_random.integers(0, len(deck), dtype=int))
            opponent_cards.append(card)

        self.deck = deck
        self.agent_cards = agent_cards
        self.opponent_cards = opponent_cards
        self.discard = []
        self.attacking_card = None
        self.defending_card = None
        self.attacker = self.np_random.integers(0, 2, dtype=int)
        self.defender = 0 if self.attacker == 1 else 1
        self.phase = "reset"

        if self.attacker == 1:
            self._handle_opponent_attack()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        self.phase = "continue"

        if self.attacker == 0:
            self._handle_agent_attack(action)
            self._handle_opponent_defense()
            if self.defending_card is None:
                self.phase = "won"
            else:
                self._end_turn()
                self._handle_opponent_attack()
                if self.attacking_card is None:
                    self.phase = "remis"

        elif self.defender == 0:
            self._handle_agent_defense(action)
            if self.defending_card is None:
                self.phase = "lost"
            elif len(self.agent_cards) == 0 and len(self.opponent_cards) == 0:
                self.phase = "remis"
            else:
                self._end_turn()

        match self.phase:
            case "won":
                terminated, reward = True, +1
            case "lost":
                terminated, reward = True, -1
            case "remis":
                terminated, reward = True, 0
            case _:
                terminated, reward = False, 0

        truncated = False
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def get_legal_mask(self):
        mask = [0] * self.num_cards
        for card in self.agent_cards:
            if self.attacker == 0:
                mask[card] = 1
                continue

            if self.defender == 0:
                mask[card] = int(card > self.attacking_card)

        return mask

    def _handle_agent_attack(self, action):
        if action is None:
            return

        if action in self.agent_cards:
            self.agent_cards.remove(action)
            self.attacking_card = action
        else:
            print("illegal attack")

    def _handle_agent_defense(self, action):
        if action is None:
            return

        if action in self.agent_cards and action > self.attacking_card:
            self.agent_cards.remove(action)
            self.defending_card = action
        else:
            print("illegal defense")

    def _handle_opponent_attack(self):
        if len(self.opponent_cards) == 0:
            return

        card = self.opponent_cards.pop(
            self.np_random.integers(0, len(self.opponent_cards), dtype=int)
        )
        self.attacking_card = card

    def _handle_opponent_defense(self):
        valid_cards = [
            card for card in self.opponent_cards if card > self.attacking_card
        ]
        if len(valid_cards) == 0:
            return

        card = valid_cards.pop(self.np_random.integers(0, len(valid_cards), dtype=int))
        self.opponent_cards.remove(card)
        self.defending_card = card

    def _end_turn(self):
        self.attacker, self.defender = self.defender, self.attacker
        self.discard.append(self.attacking_card)
        self.discard.append(self.defending_card)
        self.attacking_card = None
        self.defending_card = None


def main():
    gym.register(id="custom_cardgame_v0", entry_point=Cardgame, max_episode_steps=10)

    learning_rate = 0.001
    epochs = 100
    episodes_per_epoch = 100
    episodes_test = 10_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (epochs * episodes_per_epoch)
    final_epsilon = 0.0

    env = gym.make("custom_cardgame_v0", num_cards=8, num_hand_cards=3)

    agent = QAgent(
        env,
        learning_rate,
        start_epsilon,
        epsilon_decay,
        final_epsilon,
    )

    print("Untrained agent")
    test.test_agent(agent, env, episodes_test)
    print("-" * 16)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch} with {episodes_per_epoch} iterations")
        train.train(agent, env, episodes_per_epoch)
        test.test_agent(agent, env, episodes_test)
        print("-" * 16)


if __name__ == "__main__":
    main()

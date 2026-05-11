# N cards: (1 to N] same-suit
# 2 players
# 2 cards are dealt per player at the start of the game.

# Rounds:
# Attacking player plays a card.
# Defending player must defend with a higher card.
# Defender loses when a played attack card cannot be defended.
# Game ends in a draw if both players hold no cards.

# ActionSpace: Discrete[N + 1] = Play one of the N cards or Pass(None) - Pass is Action[0]
# ObservationSpace: MultiDiscrete[N, 4] = N cards with 4 one of possible states:
#   0: Unknown - Card might be in the deck or in opponent's hand
#   1: My Card
#   2: In Play - This card has been played as attack or defense card
#   3: Discarded

from pettingzoo import AECEnv
from pettingzoo.test import api_test
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class Status(IntEnum):
    Unknown = 0
    MyCard = 1
    InPlay = 2
    Discarded = 3


@dataclass(frozen=True)
class Card:
    value: int
    index: int


class Cardgame(AECEnv):
    metadata = {
        "name": "custom_cardgame_v1",
    }

    def __init__(self, num_cards=8, num_hand_cards=3):
        assert num_hand_cards * 2 <= num_cards
        self.num_cards = num_cards
        self.num_hand_cards = num_hand_cards
        self.possible_agents = [f"agent_{i}" for i in range(2)]
        self.passing_action = np.int64(0)
        self.n_action_space = self.num_cards + 1
        self.full_deck = [
            Card(value=i + 1, index=i) for i in range(self.num_cards)
        ]  # Cards with values 1..=8 to allow easy conversion to passing_action = None/0

    def reset(self, seed=None, options=None):
        self.winner = None
        self.agents = list(self.possible_agents)
        self.deck = list(self.full_deck)

        np.random.shuffle(self.deck)
        self.agents_cards = {}
        for agent in self.agents:
            hand = self.deck[: self.num_hand_cards]
            self.agents_cards[agent] = hand
            self.deck = self.deck[self.num_hand_cards :]

        self.discard = []
        self.attacking_card = None
        self.defending_card = None
        [self.attacking_agent, self.defending_agent] = np.random.choice(
            self.agents, size=2, replace=False
        )
        self.agent_selection = self.attacking_agent
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.observations = {
            agent: np.array([None] * self.num_cards) for agent in self.agents
        }
        self.infos = {agent: {"action_mask": None} for agent in self.agents}
        self.observation_spaces = {
            agent: gym.spaces.MultiDiscrete([len(Status)] * self.num_cards)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: gym.spaces.Discrete(self.num_cards + 1) for agent in self.agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        self._update_agent_data(agent)
        return np.array(self.observations[agent])

    def step(self, action):
        # action None: Agent dropped or acknowledged previous term / trunc
        # action 0: Agent passes turn
        # action X @ [1..num_cards + 1]: Agent plays card with value X

        self._clear_rewards()
        agent = self.agent_selection

        if action is None:
            self._remove_agent(agent)
            self.agent_selection = (
                self.defending_agent
                if agent == self.attacking_agent
                else self.attacking_agent
            )
            return

        if agent == self.attacking_agent:
            self._handle_attack(agent, action)
            self.agent_selection = self.defending_agent
        else:
            self._handle_defense(agent, action)
            self._defense_reward(agent)
            self._end_turn()
            self.agent_selection = self.attacking_agent

        for agent in self.agents:
            self._update_agent_data(agent)

        self._accumulate_rewards()

    def render(self):
        pass

    def close(self):
        pass

    def _get_winner(self):
        return self.winner

    def _update_agent_data(self, agent):
        cards = self.agents_cards[agent]
        obs = np.array([Status.Unknown] * self.num_cards)

        for card in self.discard:
            obs[card.index] = Status.Discarded

        for card in cards:
            obs[card.index] = Status.MyCard

        if self.attacking_card:
            obs[self.attacking_card.index] = Status.InPlay

        if self.defending_card:
            obs[self.defending_card.index] = Status.InPlay

        attack_card_value = self.attacking_card.value if self.attacking_card else 0
        card_values = np.array([card.value for card in self.full_deck])
        legal = (obs == Status.MyCard.value) & (card_values > attack_card_value)
        mask = np.concatenate(([1], legal.astype(np.int8))).astype(np.int8)

        self.observations[agent] = obs
        self.infos[agent]["action_mask"] = mask

    def _handle_attack(self, agent, action):
        if action is None or action == self.passing_action:
            return

        card = Card(value=action, index=action - 1)

        assert self.attacking_card is None
        assert card in self.agents_cards[agent]

        self.agents_cards[agent].remove(card)
        self.attacking_card = card

    def _handle_defense(self, agent, action):
        if action is None or action == self.passing_action:
            return

        card = Card(value=action, index=action - 1)

        assert self.defending_card is None
        assert card in self.agents_cards[agent]

        if self.attacking_card is None:
            return

        assert card.value > self.attacking_card.value

        self.agents_cards[agent].remove(card)
        self.defending_card = card

    # Switch roles and discard attacking/defending cards
    def _end_turn(self):
        if self.attacking_card is not None:
            self.discard.append(self.attacking_card)
            self.attacking_card = None

        if self.defending_card is not None:
            self.discard.append(self.defending_card)
            self.defending_card = None

        self.attacking_agent, self.defending_agent = (
            self.defending_agent,
            self.attacking_agent,
        )

    # Reward, termination, truncation after defense action
    def _defense_reward(self, agent):
        atk_reward, atk_terminated, atk_truncated = 0, False, False
        def_reward, def_terminated, def_truncated = 0, False, False

        # Attacker failed to attack => Attacker lost by default, no reward for defender
        if self.attacking_card is None:
            atk_reward = -1
            atk_terminated = True
            def_reward = 0
            def_terminated = True
            self.winner = None
            print("Warning: Attacker failed to play a card")

        # Defender failed to defend => Attacker won, Defender lost
        elif self.attacking_card is not None and self.defending_card is None:
            atk_reward = +1
            atk_terminated = True
            def_reward = -1
            def_terminated = True
            self.winner = self.attacking_agent

        # Neither player has any cards left => Draw, small reward for both
        elif all(len(cards) == 0 for cards in self.agents_cards.values()):
            atk_reward = 0.1
            atk_terminated = True
            def_reward = 0.1
            def_terminated = True
            self.winner = None

        self.rewards[self.attacking_agent] = atk_reward
        self.terminations[self.attacking_agent] = atk_terminated
        self.truncations[self.attacking_agent] = atk_truncated

        self.rewards[self.defending_agent] = def_reward
        self.terminations[self.defending_agent] = def_terminated
        self.truncations[self.defending_agent] = def_truncated

    def _remove_agent(self, agent):
        self.agents.remove(agent)
        self.rewards.pop(agent)
        self._cumulative_rewards.pop(agent)
        self.terminations.pop(agent)
        self.truncations.pop(agent)
        self.observations.pop(agent)
        self.infos.pop(agent)


if __name__ == "__main__":
    env = Cardgame()
    api_test(env, verbose_progress=True)

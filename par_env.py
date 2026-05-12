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

from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from copy import deepcopy


@dataclass(frozen=True)
class Status(IntEnum):
    Unknown = 0
    MyCard = 1
    InPlay = 2
    Discarded = 3


class Cardgame(ParallelEnv):
    metadata = {
        "name": "custom_cardgame_v1",
    }

    def __init__(self, num_cards=8, num_hand_cards=3):
        assert num_hand_cards * 2 <= num_cards
        self.num_cards = num_cards
        self.num_hand_cards = num_hand_cards
        self.possible_agents = [f"agent_{i}" for i in range(2)]
        self.n_action_space = self.num_cards + 1
        self.passing_action = self.num_cards
        self.full_deck = [i for i in range(self.num_cards)]
        self.observation_spaces = {
            agent: gym.spaces.Dict(
                {
                    "observations": gym.spaces.MultiDiscrete(
                        [len(Status)] * self.num_cards
                    ),
                    "action_mask": gym.spaces.MultiBinary(self.num_cards + 1),
                }
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Discrete(self.num_cards + 1)
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
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
            agent: {
                "observations": None,
                "action_mask": None,
            }
            for agent in self.agents
        }
        self.infos = {agent: {} for agent in self.agents}
        state = self._update_agents_data()

        return (state[0], state[4])

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, actions):
        self._clear_rewards()

        for agent, action in actions.items():
            if agent != self.agent_selection:
                continue

            if agent == self.attacking_agent:
                self._handle_attack(agent, action)
                next_active_player = self.defending_agent
            else:
                self._handle_defense(agent, action)
                self._handle_rewards(agent)
                self._end_turn()
                next_active_player = self.attacking_agent

        state = self._end_of_cycle(next_active_player)

        return state

    def render(self):
        pass

    def close(self):
        pass

    def state(self):
        return np.array([])

    def _get_winner(self):
        return self.winner

    def _end_of_cycle(self, next_active_player):
        self._accumulate_rewards()
        state = self._update_agents_data()

        for agent in list(self.agents):
            if self.terminations[agent] or self.truncations[agent]:
                self._remove_agent(agent)

        if len(self.agents) > 0:
            self.agent_selection = (
                next_active_player
                if next_active_player in self.agents
                else self.agents[0]
            )

        return state

    def _update_agents_data(self):
        for agent in self.agents:
            cards = self.agents_cards[agent]
            obs = np.array([Status.Unknown] * self.num_cards)

            for card in self.discard:
                obs[card] = Status.Discarded

            for card in cards:
                obs[card] = Status.MyCard

            if self.attacking_card:
                obs[self.attacking_card] = Status.InPlay

            if self.defending_card:
                obs[self.defending_card] = Status.InPlay

            attack_card_value = self.attacking_card if self.attacking_card else -1
            card_values = np.array([card for card in self.full_deck])
            legal = (obs == Status.MyCard) & (card_values > attack_card_value)
            mask = np.concatenate((legal, [1])).astype(np.int8)

            self.observations[agent]["observations"] = obs
            self.observations[agent]["action_mask"] = mask

        return (
            deepcopy(self.observations),
            deepcopy(self.rewards),
            deepcopy(self.terminations),
            deepcopy(self.truncations),
            deepcopy(self.infos),
        )

    def _handle_attack(self, agent, action):
        if action is None or action == self.passing_action:
            return

        assert self.attacking_card is None
        assert action in self.agents_cards[agent]

        self.agents_cards[agent].remove(action)
        self.attacking_card = action

    def _handle_defense(self, agent, action):
        if action is None or action == self.passing_action:
            return

        assert self.defending_card is None
        assert action in self.agents_cards[agent]

        if self.attacking_card is None:
            return

        assert action > self.attacking_card

        self.agents_cards[agent].remove(action)
        self.defending_card = action

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

    # Reward, termination, truncation after full turn
    def _handle_rewards(self, agent):
        atk_reward, atk_terminated, atk_truncated = 0, False, False
        def_reward, def_terminated, def_truncated = 0, False, False

        if self.attacking_card is not None and self.defending_card is None:
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

    def _clear_rewards(self):
        for agent in self.agents:
            self.rewards[agent] = 0

    def _accumulate_rewards(self):
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]


if __name__ == "__main__":
    env = Cardgame()
    parallel_api_test(env)

from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum, Enum

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test

from durak import Card, CardColor, CardValue, GameState


@dataclass()
class Status(IntEnum):
    Unknown = 0
    MyCard = 1
    OpponentCard = 2
    Attack = 3
    Defense = 4
    Discarded = 5


class Phase(Enum):
    Attack = 0
    Defend = 1
    ThrowIn = 2
    Take = 3


class DurakEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "durak_card_game_v0"}

    def __init__(self):
        self.gamestate = GameState()
        self.gamestate.setup(2)
        self.num_cards = len(CardColor) * len(CardValue)
        self.possible_agents = [player.name for player in self.gamestate.players]
        self.n_action_space = self.num_cards + 1
        self.passing_action = self.num_cards
        self.observation_spaces = {
            agent: gym.spaces.Dict(
                {
                    "observations": gym.spaces.MultiDiscrete(
                        [len(Status)] * (self.num_cards + 1)  # All cards plus trump
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

    def reset(self, *, seed=None, options=None):
        np.random.seed(seed)
        self.gamestate.setup(2)
        self.agents = list(self.possible_agents)
        self.agent_obs = {
            agent: np.array([Status.Unknown] * self.num_cards) for agent in self.agents
        }
        trump_card = self.gamestate.draw_pile[0]
        self.tracked_cards = {trump_card}
        self.agent_selection = self.gamestate._find_first_attacker()
        self.next_player = self.agent_selection
        self.phase = Phase.Attack
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminateds = {agent: False for agent in self.agents}
        self.truncateds = {agent: False for agent in self.agents}
        self.observations = {
            agent: {
                "observations": None,
                "action_mask": None,
            }
            for agent in self.agents
        }
        self.infos = {agent: {} for agent in self.agents}
        state = self._update_agents_data()

        return state[0], state[4]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, actions):
        self._clear_rewards()

        for agent, action in actions.items():
            if agent != self.agent_selection:
                continue

            match self.phase:
                case Phase.Attack:
                    self._handle_attack(agent, action)
                case Phase.Defense:
                    self._handle_defense(agent, action)
                case Phase.ThrowIn:
                    self._handle_throw_in(agent, action)
                case Phase.Take:
                    self._handle_take(agent, action)

        return self._end_of_cycle()

    def render(self):
        pass

    def close(self):
        pass

    def state(self):
        return np.array([])

    def _handle_attack(self, action):
        if action is None or action == self.passing_action:
            self.next_player = self.agents[self.gamestate.defender]
            self.phase = Phase.Attack
            return

        attacker = self.gamestate.players[self.gamestate.attacker]
        card = self._get_card_from_index(action)

        assert card in attacker.hand

        attacker.hand.cards.remove(card)
        self.gamestate.add_attack_card(card)

        self.next_player = self.agents[self.gamestate.defender]
        self.phase = Phase.Defend

    def _handle_throw_in(self, action):
        if action is None or action == self.passing_action:
            self.next_player = self.agents[self.gamestate.defender]
            self.phase = Phase.Take
            return

        attacker = self.gamestate.players[self.gamestate.defender]
        card = self._get_card_from_index(action)

        assert card in attacker.hand

        attacker.hand.cards.remove(card)
        self.gamestate.add_attack_card(card)

        self.next_player = self.agents[self.gamestate.attacker]
        self.phase = Phase.ThrowIn

    def _handle_defense(self, action):
        if action is None or action == self.passing_action:
            self.next_player = self.agents[self.gamestate.attacker]
            self.phase = Phase.ThrowIn
            return

        defender = self.gamestate.players[self.gamestate.defender]
        card = self._get_card_from_index(action)

        assert card in defender.hand

        defender.hand.cards.remove(card)
        self.gamestate.add_defense_card(card)

        self.next_player = self.agents[self.gamestate.attacker]
        self.phase = Phase.Attack

    def _handle_take(self):
        defender = self.gamestate.players[self.gamestate.defender]
        defender.hand.cards.extend(self.gamestate.collect_table_cards())

        self.next_player = self.agents[self.gamestate.attacker]
        self.phase = Phase.Attack

    def _get_winner(self):
        return self.gamestate.winner_index()

    def _end_of_cycle(self):
        winner = self._get_winner()

        if winner:
            winning_agent = self.agents[winner]
            losing_agent = self.agents[(winner + 1) % 2]

            self.rewards[winning_agent] = 1
            self.terminateds[winning_agent] = True

            self.rewards[losing_agent] = -1
            self.terminateds[losing_agent] = True

        self._accumulate_rewards()
        state = self._update_agents_data()

        for agent in list(self.agents):
            if self.terminateds[agent] or self.truncateds[agent]:
                self._remove_agent(agent)

        self.agent_selection = self.next_player

        return state

    def _update_agents_data(self):
        for pair in self.gamestate.table:
            self.tracked_cards.add(pair.attack)
            if pair.defense:
                self.tracked_cards.add(pair.defense)

        for i, agent in enumerate(self.agents):
            self._update_agent_data(i, agent)

        return (
            deepcopy(self.observations),
            deepcopy(self.rewards),
            deepcopy(self.terminateds),
            deepcopy(self.truncateds),
            deepcopy(self.infos),
        )

    def _update_agent_data(self, i, agent):
        cards = self.gamestate.players[i].hand.cards
        opponent_cards = self.gamestate.players[(i + 1) % 2].hand.cards
        obs = self.agent_obs[agent]

        # Set player hand cards - may yet contain untracked cards
        indices = [self._get_index_from_card(card) for card in cards]
        obs[indices] = Status.MyCard

        # Set already shown cards
        for card in self.tracked_cards:
            index = self._get_index_from_card(card)
            if card in opponent_cards:
                obs[index] = Status.OpponentCard
            elif card in cards:
                obs[index] = Status.MyCard
            else:
                obs[index] = Status.Discarded  # Gets overriden for in-play cards

        # Set in-play cards
        for pair in self.gamestate.table:
            index = self._get_index_from_card(pair.attack)
            obs[index] = Status.Attack
            if pair.defense:
                index = self._get_index_from_card(pair.defense)
                obs[index] = Status.Defense

        # obs now contains all available card information for 'agent'
        # The only unknown cards should be the cards in the deck and any card the opponent has not shown yet

        # Get legal cards
        if i == self.gamestate.attacker:
            legal_cards = (
                self.gamestate.LegalAttackCards(cards=cards)
                if self.gamestate.can_add_more_attack_cards()
                else []
            )
        else:
            legal_cards = self.gamestate.LegalDefenseCards(cards=cards)

        # Set action mask to legal cards
        action_mask = np.zeros(self.num_cards + 1, dtype=np.int8)
        indices = [self._get_index_from_card(card) for card in legal_cards]
        action_mask[indices] = 1
        action_mask[self.passing_action] = 1

        # Must play a first attack card
        if len(self.gamestate.table) == 0 and i == self.gamestate.attacker:
            action_mask[self.passing_action] = 0

        if not any(action_mask):
            raise ValueError(f"No legal actions available to {agent}")

        self.observations[agent]["observations"] = obs
        self.observations[agent]["action_mask"] = action_mask

    def _remove_agent(self, agent):
        self.agents.remove(agent)
        self.rewards.pop(agent)
        self._cumulative_rewards.pop(agent)
        self.terminateds.pop(agent)
        self.truncateds.pop(agent)
        self.observations.pop(agent)
        self.infos.pop(agent)

    def _clear_rewards(self):
        for agent in self.agents:
            self.rewards[agent] = 0

    def _accumulate_rewards(self):
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

    def _get_index_from_card(self, card: Card) -> int:
        # [Spades[6..Ace], Clubs[6..Ace], Hearts[6..Ace], Diamonds[6..Ace]]
        return (card.color.value) * len(CardValue) + (card.value.value - 6)

    def _get_card_from_index(self, index: int) -> Card | None:
        if index == self.passing_action:
            return None

        color = CardColor(index // len(CardValue))
        value = CardValue(index % len(CardValue) + 6)
        return Card(value=value, color=color)


if __name__ == "__main__":
    import itertools

    env = DurakEnv()
    parallel_api_test(env)

    cards = [
        Card(value=value, color=color)
        for value, color in itertools.product(CardValue, CardColor)
    ]
    indices = [env._get_index_from_card(card) for card in cards]

    for card, index in zip(cards, indices):
        assert card == env._get_card_from_index(index)
        assert index == env._get_index_from_card(card)

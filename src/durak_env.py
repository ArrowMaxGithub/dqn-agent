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
    InDeck = 5
    Discarded = 6


class Phase(Enum):
    Attack = 0
    Defense = 1
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
                        [len(Status)]
                        * (self.num_cards + 1)  # All cards plus trump color
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
        self.trump_card = self.gamestate.draw_pile[0]
        self.tracked_cards = {self.trump_card}
        self.agent_selection = self.agents[self.gamestate._find_first_attacker()]
        self.next_player = self.agent_selection
        self.phase = Phase.Attack
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminateds = {agent: False for agent in self.agents}
        self.truncateds = {agent: False for agent in self.agents}
        self.observations = {
            agent: {
                "observations": np.array(
                    [Status.Unknown] * (self.num_cards + 1), dtype=np.int8
                ),
                "action_mask": np.zeros(self.num_cards + 1, dtype=np.int8),
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
        # ParallelEnvWrapper calls step once more after all agents are already dead
        if len(self.agents) == 0:
            return (
                self.observations,
                self.rewards,
                self.terminateds,
                self.truncateds,
                self.infos,
            )

        self._clear_rewards()

        for agent, action in actions.items():
            if agent != self.agent_selection:
                continue

            match self.phase:
                case Phase.Attack:
                    self._handle_attack(action)
                case Phase.Defense:
                    self._handle_defense(action)
                case Phase.ThrowIn:
                    self._handle_throw_in(action)
                case Phase.Take:
                    self._handle_take()

        return self._end_of_cycle()

    def render(self):
        pass

    def close(self):
        pass

    def state(self):
        return np.array([])

    def _handle_attack(self, action):
        attacker = self.gamestate.players[self.gamestate.attacker]
        if action is None or action == self.passing_action:
            self.next_player = self.agents[self.gamestate.defender]
            self.phase = Phase.Attack
            self.gamestate.discard_table_cards()
            self.gamestate.refill_hands()
            self.gamestate.swap_roles()
            return

        card = self._get_card_from_index(action)

        assert card in attacker.hand.cards
        assert card in self.gamestate.LegalAttackCards(attacker.hand.cards)

        attacker.hand.cards.remove(card)
        self.gamestate.add_attack_card(card)
        self.tracked_cards.add(card)

        self.next_player = self.agents[self.gamestate.defender]
        self.phase = Phase.Defense

    def _handle_throw_in(self, action):
        attacker = self.gamestate.players[self.gamestate.attacker]
        if action is None or action == self.passing_action:
            self.next_player = self.agents[self.gamestate.defender]
            self.phase = Phase.Take
            return

        card = self._get_card_from_index(action)

        assert card in attacker.hand.cards
        assert card in self.gamestate.LegalAttackCards(attacker.hand.cards)

        attacker.hand.cards.remove(card)
        self.gamestate.add_attack_card(card)
        self.tracked_cards.add(card)

        self.next_player = self.agents[self.gamestate.attacker]
        self.phase = Phase.ThrowIn

    def _handle_defense(self, action):
        defender = self.gamestate.players[self.gamestate.defender]
        if action is None or action == self.passing_action:
            self.next_player = self.agents[self.gamestate.attacker]
            self.phase = Phase.ThrowIn
            return

        card = self._get_card_from_index(action)

        assert card in defender.hand.cards
        assert card in self.gamestate.LegalDefenseCards(defender.hand.cards)

        defender.hand.cards.remove(card)
        self.gamestate.add_defense_card(card)
        self.tracked_cards.add(card)

        self.next_player = self.agents[self.gamestate.attacker]
        self.phase = Phase.Attack

    def _handle_take(self):
        defender = self.gamestate.players[self.gamestate.defender]
        to_take = self.gamestate.collect_table_cards()

        defender.hand.cards.extend(to_take)

        self.gamestate.refill_hands()

        self.next_player = self.agents[self.gamestate.attacker]
        self.phase = Phase.Attack

    def _get_winner(self):
        winner_index = self.gamestate.winner_index()
        if winner_index is not None:
            return self.possible_agents[winner_index]
        else:
            return None

    def _end_of_cycle(self):
        winner = self._get_winner()

        if winner is not None:
            winning_agent = winner
            losing_agent = (
                self.possible_agents[1]
                if winner == self.possible_agents[0]
                else self.possible_agents[0]
            )

            self.rewards[winning_agent] = 1
            self.terminateds[winning_agent] = True

            self.rewards[losing_agent] = -1
            self.terminateds[losing_agent] = True

        self._accumulate_rewards()
        state = self._update_agents_data()

        for agent in self.possible_agents:
            if self.terminateds[agent] or self.truncateds[agent]:
                self._remove_agent(agent)

        self.agent_selection = self.next_player

        return state

    def _update_agents_data(self):
        for pair in self.gamestate.table:
            self.tracked_cards.add(pair.attack)
            if pair.defense:
                self.tracked_cards.add(pair.defense)

        # Unzip in-play cards
        pairs = self.gamestate.table
        attacks = set([pair.attack for pair in pairs])
        defenses = set([pair.defense for pair in pairs])

        for i, agent in enumerate(self.agents):
            self._update_agent_data(i, agent, attacks, defenses)

        return (
            self.observations,
            self.rewards,
            self.terminateds,
            self.truncateds,
            self.infos,
        )

    def _update_agent_data(self, i, agent, attacks, defenses):
        cards = set(self.gamestate.players[i].hand.cards)
        opponent_cards = set(self.gamestate.players[(i + 1) % 2].hand.cards)
        obs = self.observations[agent]["observations"]

        # Set trump color
        obs[-1] = np.int8(self.trump_card.color.value)

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
            elif card == self.trump_card:
                obs[index] = Status.InDeck  # While the trump card is not drawn
            elif card in attacks:
                obs[index] = Status.Attack
            elif card in defenses:
                obs[index] = Status.Defense
            else:
                obs[index] = Status.Discarded  # Gets overriden for in-play cards

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

        if __debug__:
            if not any(action_mask):
                raise ValueError(f"No legal actions available to {agent}")

            for index, mask in enumerate(action_mask[:-1]):
                card = self._get_card_from_index(index)
                if mask == 1:
                    assert card in cards

        self.observations[agent]["observations"] = obs
        self.observations[agent]["action_mask"] = action_mask

    def _remove_agent(self, agent):
        self.agents.remove(agent)

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

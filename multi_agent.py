# N cards: (0 to N] same-suit
# 2 players
# 2 cards are dealt per player at the start of the game.

# Rounds:
# Attacking player plays a card.
# Defending player must defend with a higher card.
# Defender loses when a played attack card cannot be defended.
# Game ends in a draw if both players hold no cards.

# ActionSpace: Discrete[N + 1] = Play one of the N cards or Pass(None)
# ObservationSpace: MultiDiscrete[N, 4] = N cards with 4 one of possible states:
#   0: Unknown - Card might be in the deck or in opponent's hand
#   1: My Card
#   2: In Play - This card has been played as attack or defense card
#   3: Discarded

from pettingzoo import AECEnv
import gymnasium as gym
from collections import defaultdict
import numpy as np
import math
from tqdm import tqdm

WIN = +1.0
DRAW = +0.1
LOSS = -1.0


class Cardgame(AECEnv):
    metadata = {
        "name": "custom_cardgame_v1",
    }

    def __init__(self, num_cards=8, num_hand_cards=3):
        assert num_hand_cards * 2 <= num_cards

        self.num_cards = num_cards
        self.num_hand_cards = num_hand_cards
        self.possible_agents = [0, 1]
        self.agents = [0, 1]
        self.pass_action = self.num_cards

        # Five possible combinations of Bitflags times num_cards
        self.obs_space = gym.spaces.MultiDiscrete([4] * self.num_cards)

        # Play any card or pass
        # Action masking is implemented by reading CardFlags.legally_playable from obs space
        self.action_space = gym.spaces.Discrete(self.num_cards + 1)

    def reset(self, seed=None, options=None):
        self.deck = [i for i in range(self.num_cards)]
        # One entry for agent '0':
        # 0: [2, 5, 3, 4, 6]
        self.agents_cards = {agent: self._deal_cards() for agent in self.agents}
        self.discard = []
        self.attacking_card = None
        self.defending_card = None
        self.attacking_agent = self.agents[np.random.randint(0, 2)]
        self.defending_agent = 1 if self.attacking_agent == 0 else 0
        self.agent_selection = self.attacking_agent

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.observation_spaces = {agent: self.obs_space for agent in self.agents}
        self.action_spaces = {agent: self.action_space for agent in self.agents}
        self.infos = {agent: None for agent in self.agents}

        self.game_result = None

        self.observe(self.attacking_agent)

    def observe(self, agent):
        cards = self.agents_cards[agent]
        obs = [0] * self.num_cards
        for card in self.discard:
            obs[card] = 3

        for card in cards:
            obs[card] = 1

        if self.attacking_card:
            obs[self.attacking_card] = 2

        if self.defending_card:
            obs[self.defending_card] = 2

        self.observations[agent] = obs

        return np.array(self.observations[agent])

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            return

        if agent == self.attacking_agent:
            self._handle_attack(agent, action)
            self._attack_reward(agent)
            self.agent_selection = self.defending_agent
        else:
            self._handle_defense(agent, action)
            self._defense_reward(agent)
            self._end_turn()
            self.agent_selection = self.attacking_agent

        for agent in self.agents:
            self.observe(agent)

    def render(self):
        pass

    def close(self):
        pass

    def _get_agent_data(self, agent):
        obs = self.observations[agent]
        reward = self.rewards[agent]
        termination = self.terminations[agent]
        truncation = self.truncations[agent]
        info = self.infos[agent]

        return obs, reward, termination, truncation, info

    def _handle_attack(self, agent, action):
        assert self.attacking_card is None

        if action is self.pass_action:
            return

        assert action in self.agents_cards[agent]

        self.agents_cards[agent].remove(action)
        self.attacking_card = action

    def _handle_defense(self, agent, action):
        assert self.defending_card is None

        if self.attacking_card is None or action is self.pass_action:
            return

        assert action in self.agents_cards[agent]
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

    # Immediate reward, termination, truncation after attack action
    def _attack_reward(self, agent):
        atk_reward, atk_terminated, atk_truncated = 0, False, False

        self.rewards[self.attacking_agent] = atk_reward
        self.terminations[self.attacking_agent] = atk_terminated
        self.truncations[self.attacking_agent] = atk_truncated

    # Immediate reward, termination, truncation after defense action
    def _defense_reward(self, agent):
        atk_reward, atk_terminated, atk_truncated = 0, False, False
        def_reward, def_terminated, def_truncated = 0, False, False

        # Defender failed to defend => Attacker won, Defender lost
        if self.attacking_card is not None and self.defending_card is None:
            atk_reward = +1
            atk_terminated = True
            def_reward = -1
            def_terminated = True
            self.game_result = self.attacking_agent

        # Neither player has any cards left => Draw, small reward for both
        elif all(len(cards) == 0 for cards in self.agents_cards.values()):
            atk_reward = 0.1
            atk_terminated = True
            def_reward = 0.1
            def_terminated = True
            self.game_result = 0.5

        self.rewards[self.attacking_agent] = atk_reward
        self.terminations[self.attacking_agent] = atk_terminated
        self.truncations[self.attacking_agent] = atk_truncated

        self.rewards[self.defending_agent] = def_reward
        self.terminations[self.defending_agent] = def_terminated
        self.truncations[self.defending_agent] = def_truncated

    def _deal_cards(self) -> list[int]:
        agent_cards = []
        while len(agent_cards) < self.num_hand_cards:
            card = self.deck.pop(np.random.randint(0, len(self.deck)))
            agent_cards.append(card)

        return agent_cards

    # If there is still at least one agent in-play, the game continues
    def _game_over(self) -> bool:
        for agent in self.agents:
            if not (self.terminations[agent] or self.truncations[agent]):
                return False

        return True

    def _get_result_for_player(self, agent):
        if self.game_result == 0.5:
            return 0.5
        elif agent == self.game_result:
            return 1.0
        else:
            return -1.0

    def _get_legal_cards(self, obs):
        # If no card in play is found => I am attacking and any held card is legal to play.
        # Otherwise: I am defending and must play a higher card

        attacking_card = -1
        for i, card in enumerate(obs):
            if card == 2:
                attacking_card = i
                break

        legal_cards = []
        for i, card in enumerate(obs):
            if card == 1 and i > attacking_card:
                legal_cards.append(i)

        return legal_cards


class QAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs, force_exploitation=False):
        legal_cards = self.env._get_legal_cards(obs)

        if len(legal_cards) == 0:
            return self.env.pass_action

        if np.random.random() < self.epsilon and not force_exploitation:
            chosen = np.random.randint(0, len(legal_cards))
            return legal_cards[chosen]
        else:
            obs_entry = tuple(int(o) for o in obs)
            q_values = np.copy(self.q_values[obs_entry])
            for i, card in enumerate(obs):
                if i not in legal_cards:
                    q_values[i] = -math.inf

            return int(np.argmax(q_values))

    def update(
        self,
        last_obs,  # s
        action: int,  # a: s -> s'
        reward: float,
        terminated: bool,
        current_obs,  # s'
    ):
        # Convert list of CardFlags to tuple of ints
        last_obs_entry = tuple(int(o) for o in last_obs)
        current_obs_entry = tuple(int(o) for o in current_obs)

        legal_cards = self.get_legal_cards(current_obs)

        if terminated or len(legal_cards) == 0:
            future_q_value = 0.0
        else:
            # Bellman equation masked over legal cards
            future_q_value = (not terminated) * np.max(
                [self.q_values[current_obs_entry][card] for card in legal_cards]
            )

        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[last_obs_entry][action]

        self.q_values[last_obs_entry][action] = (
            self.q_values[last_obs_entry][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def get_legal_cards(self, obs):
        # If no card in play is found => I am attacking and any held card is legal to play.
        # Otherwise: I am defending and must play a higher card

        attacking_card = -1
        for i, card in enumerate(obs):
            if card == 2:
                attacking_card = i
                break

        legal_cards = []
        for i, card in enumerate(obs):
            if card == 1 and i > attacking_card:
                legal_cards.append(i)

        return legal_cards


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs, force_exploitation=True):
        legal_cards = self.env._get_legal_cards(obs)

        if len(legal_cards) == 0:
            return self.env.pass_action

        chosen = np.random.randint(0, len(legal_cards))
        return legal_cards[chosen]


def train(q_agents, env, n_episodes):
    for episode in tqdm(range(n_episodes)):
        env.reset()
        last_obs = {agent: None for agent in env.agents}
        last_act = {agent: None for agent in env.agents}

        while not env._game_over():
            agent = env.agent_selection
            obs, reward, term, trunc, info = env._get_agent_data(agent)

            if last_act[agent] is not None:
                q_agents[agent].update(
                    last_obs[agent], last_act[agent], reward, term, obs
                )

            action = q_agents[agent].get_action(obs)
            env.step(action)

            last_obs[agent] = obs
            last_act[agent] = action
            q_agents[agent].decay_epsilon()

        for agent in env.agents:
            obs, reward, term, trunc, info = env._get_agent_data(agent)
            if last_act[agent] is not None:
                q_agents[agent].update(
                    last_obs[agent], last_act[agent], reward, term, obs
                )


def test(agents, env, n_episodes=1000, all_pairings=False):
    agents = agents.copy()
    agents.append(RandomAgent(env))
    results = {
        (0, 1): [0.0, 0.0, 0.0],  # Q-Agent-0 vs Q-Agent-1
        (0, 2): [0.0, 0.0, 0.0],  # Q-Agent-0 vs Random
        (1, 2): [0.0, 0.0, 0.0],  # Q-Agent-1 vs Random
    }

    if all_pairings:
        results[(2, 2)] = ([0.0, 0.0, 0.0],)  # Random vs Random for comparison

    for t_agents in results.keys():
        wins = 0
        draws = 0
        losses = 0

        for _ in range(n_episodes):
            env.reset()
            gameover = False

            while not gameover:
                # Poll active agent
                for agent_id in env.agent_iter():
                    t_agent = agents[t_agents[agent_id]]
                    # Can't use env.last() since we need the data twice: Once before stepping and once after stepping
                    obs, reward, term, trunc, info = env._get_agent_data(agent_id)

                    if term or trunc:
                        action = None
                    else:
                        action = t_agent.get_action(obs, force_exploitation=True)

                    env.step(action)  # Advances active agent
                    next_obs, reward, term, trunc, info = env._get_agent_data(agent_id)

                    # If the game concluded, start a new episode
                    if env._game_over():
                        gameover = True
                        break

            game_result = env._get_result_for_player(0)
            match game_result:
                case 1.0:
                    wins += 1
                case 0.5:
                    draws += 1
                case -1.0:
                    losses += 1

        win = wins / n_episodes
        draw = draws / n_episodes
        lose = losses / n_episodes

        results[t_agents] = [win, draw, lose]

    for (a0, a1), result in results.items():
        print(f"{a0} vs {a1}: {result[0]:.2f} | {result[1]:.2f} | {result[2]:.2f}")


def main():
    learning_rate = 0.001
    epochs = 10
    episodes_per_epoch = 100_000
    episodes_test = 10_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (epochs * episodes_per_epoch) / 2
    final_epsilon = 0.1

    num_cards = 8
    num_hand_cards = 3

    env = Cardgame(num_cards=num_cards, num_hand_cards=num_hand_cards)
    agents = [
        QAgent(
            env,
            learning_rate,
            start_epsilon,
            epsilon_decay,
            final_epsilon,
        )
        for a in env.agents
    ]

    print("Untrained agents")
    test(agents, env, episodes_test, all_pairings=True)
    print("-" * 16)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch} | after {episodes_per_epoch * epoch} iterations")
        train(agents, env, episodes_per_epoch)
        print(f"Results after epoch {epoch}")
        test(agents, env, episodes_test)
        print("-" * 16)


if __name__ == "__main__":
    main()

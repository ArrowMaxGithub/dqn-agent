from collections import defaultdict
import gymnasium as gym
import numpy as np
import math


class QAgent:
    def __init__(
        self,
        env: gym.Env,
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

    def get_action(self, obs, mask):
        if not any(mask):
            return None

        if np.random.random() < self.epsilon:
            legal_indices = [i for (i, m) in enumerate(mask) if m == 1]
            chosen = self.env.np_random.integers(0, len(legal_indices), dtype=int)
            return legal_indices[chosen]
        else:
            q_values = np.copy(self.q_values[tuple(obs)])
            for i, m in enumerate(mask):
                if m == 0:
                    q_values[i] = -math.inf

            return int(np.argmax(q_values))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        # Bellman equation
        future_q_value = (not terminated) * np.max(self.q_values[tuple(next_obs)])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[tuple(obs)][action]

        self.q_values[tuple(obs)][action] = (
            self.q_values[tuple(obs)][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

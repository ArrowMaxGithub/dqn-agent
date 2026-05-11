from collections import defaultdict
import numpy as np


class QAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        illegal_mask: float = -1e9,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.illegal_mask = illegal_mask
        self.training_error = []

    def get_action(self, obs, info, force_exploitation=False):
        obs_key = self.obs_key(obs)
        mask = info["action_mask"]
        illegal_mask = (1 - mask) * self.illegal_mask

        if not any(mask):
            return self.env.pass_action

        if np.random.random() < self.epsilon and not force_exploitation:
            legal_actions = np.where(mask == 1)[0]
            chosen = np.random.randint(0, len(legal_actions))
            return legal_actions[chosen]
        else:
            return int(np.argmax(self.q_values[obs_key] + illegal_mask))

    def update(
        self,
        last_obs,  # s
        action: int,  # a: s -> s'
        reward: float,
        terminated: bool,
        obs,  # s'
        info,
    ):
        mask = info["action_mask"]
        illegal_mask = (1 - mask) * self.illegal_mask

        last_obs_key = self.obs_key(last_obs)
        obs_key = self.obs_key(obs)

        if terminated:
            future_q_value = 0.0
        else:
            masked_future_q_values = self.q_values[obs_key] + illegal_mask
            future_q_value = np.max(masked_future_q_values)

        target = reward + self.discount_factor * future_q_value
        temporal_diff = target - self.q_values[last_obs_key][action]

        self.q_values[last_obs_key][action] += self.lr * temporal_diff
        self.training_error.append(temporal_diff)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def obs_key(self, obs):
        return tuple(obs["observation"].flatten())

import numpy as np


class QAgent:
    def __init__(
        self,
        passing_action: int,
        n_action_space: int,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        illegal_mask: float,
    ):
        self.passing_action = passing_action
        self.n_action_space = n_action_space
        self.q_values = {}
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.illegal_mask = illegal_mask
        self.training_error = []

    def get_label(self):
        return "QAgent"

    def get_action(self, obs_dict, force_exploitation=False):
        mask = obs_dict["action_mask"]
        obs = obs_dict["observations"]

        obs_key = self._obs_key(obs)
        q_values = self._get_q_values(obs_key)

        illegal_mask = (1 - mask) * self.illegal_mask

        if np.random.random() < self.epsilon and not force_exploitation:
            legal_actions = np.where(mask == 1)[0]
            return np.random.choice(legal_actions)
        else:
            return int(np.argmax(q_values + illegal_mask))

    def update(
        self,
        obs_dict,  # s
        action: int,  # a: s -> s'
        reward: float,
        terminated: bool,
        next_obs_dict,  # s'
    ):
        mask = next_obs_dict["action_mask"]
        next_obs = next_obs_dict["observations"]
        obs = obs_dict["observations"]

        obs_key = tuple(obs)
        next_obs_key = tuple(next_obs)

        illegal_mask = (1 - mask) * self.illegal_mask
        q_values = self._get_q_values(obs_key)
        next_q_values = self._get_q_values(next_obs_key)

        if terminated:
            future_q_value = 0.0
        else:
            masked_future_q_values = next_q_values + illegal_mask
            future_q_value = np.max(masked_future_q_values)

        target = reward + self.discount_factor * future_q_value
        temporal_diff = target - q_values[action]

        self.q_values[obs_key][action] += self.lr * temporal_diff
        self.training_error.append(temporal_diff)
        self._decay_epsilon()

    def _get_q_values(self, obs_key):
        if obs_key not in self.q_values:
            self.q_values[obs_key] = np.zeros(self.n_action_space)
        return self.q_values[obs_key]

    def _decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def _obs_key(self, obs):
        return tuple(obs)

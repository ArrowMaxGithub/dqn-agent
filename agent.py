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
        discount_factor: float = 0.95,
        illegal_mask: float = -1e9,
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
        obs = obs_dict["observation"]

        obs_key = self.obs_key(obs)
        q_values = self.get_q_values(obs_key)

        illegal_mask = (1 - mask) * self.illegal_mask

        if not any(mask[:-1]):
            return self.passing_action
        else:
            mask[-1] = 0
            illegal_mask[-1] = self.illegal_mask

        if np.random.random() < self.epsilon and not force_exploitation:
            legal_actions = np.where(mask == 1)[0]
            return np.random.choice(legal_actions)
        else:
            return int(np.argmax(q_values + illegal_mask))

    def update(
        self,
        last_obs_dict,  # s
        action: int,  # a: s -> s'
        reward: float,
        terminated: bool,
        obs_dict,  # s'
    ):
        mask = obs_dict["action_mask"]
        obs = obs_dict["observation"]
        last_obs = last_obs_dict["observation"]

        illegal_mask = (1 - mask) * self.illegal_mask

        last_obs_key = self.obs_key(last_obs)
        last_q_values = self.get_q_values(last_obs_key)
        obs_key = self.obs_key(obs)
        q_values = self.get_q_values(obs_key)

        if terminated:
            future_q_value = 0.0
        else:
            masked_future_q_values = q_values + illegal_mask
            future_q_value = np.max(masked_future_q_values)

        target = reward + self.discount_factor * future_q_value
        temporal_diff = target - last_q_values[action]

        self.q_values[last_obs_key][action] += self.lr * temporal_diff
        self.training_error.append(temporal_diff)

    def get_q_values(self, obs_key):
        if obs_key not in self.q_values:
            self.q_values[obs_key] = np.zeros(self.n_action_space)
        return self.q_values[obs_key]

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def obs_key(self, obs):
        return tuple(obs)


class RandomAgent:
    def __init__(
        self,
        passing_action: int,
    ):
        self.passing_action = passing_action

    def get_label(self):
        return "Random"

    def get_action(self, obs_dict, force_exploitation=True):
        mask = obs_dict["action_mask"]

        if not any(mask[:-1]):
            return self.passing_action
        else:
            mask[-1] = 0

        legal_actions = np.where(mask == 1)[0]
        return np.random.choice(legal_actions)

import numpy as np


class RandomAgent:
    def __init__(
        self,
        passing_action,
    ):
        self.passing_action = passing_action

    def get_label(self):
        return "Random"

    def get_action(self, obs_dict, force_exploitation=True):
        mask = obs_dict["action_mask"]
        legal_actions = np.where(mask == 1)[0]
        return np.random.choice(legal_actions)

    def update(last_obs, action, reward, term, obs): ...

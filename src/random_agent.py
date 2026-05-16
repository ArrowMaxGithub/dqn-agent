import numpy as np
import torch
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns


class RandomAgent:
    def __init__(
        self,
        passing_action,
    ):
        self.passing_action = passing_action

    def get_label(self):
        return "RandomAgent"

    def get_action(self, obs_dict, force_exploitation=True):
        mask = obs_dict["action_mask"]
        legal_actions = np.where(mask == 1)[0]
        return np.random.choice(legal_actions)

    def update(last_obs, action, reward, term, obs): ...


class RandomMaskedRLModule(TorchRLModule):
    def setup(self):
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def _forward_inference(self, batch, **kwargs):
        return self._common_forward(batch)

    def _forward_exploration(self, batch, **kwargs):
        return self._common_forward(batch)

    def _forward_train(self, batch, **kwargs):
        return self._common_forward(batch)

    def _common_forward(self, batch, obs_key=Columns.OBS):
        mask = batch[obs_key]["action_mask"]
        random = torch.rand_like(mask, dtype=torch.float32)

        inf_mask = (1 - mask) * -1e9
        legal = random + inf_mask

        return {
            Columns.ACTION_DIST_INPUTS: legal,
        }

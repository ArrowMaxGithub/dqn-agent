import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.inference_only_api import InferenceOnlyAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from interfaces import AgentInterface

INVALID_MASK = -1e8


class RandomAgent(AgentInterface):
    def __init__(
        self,
        passing_action,
    ):
        self.passing_action = passing_action

    def GetName(self) -> str:
        return "RandomAgent"

    def GetAction(self, obs_dict: dict) -> int:
        mask = obs_dict["action_mask"]
        legal_actions = np.where(mask == 1)[0]
        return np.random.choice(legal_actions)


class RandomMaskedRLModule(InferenceOnlyAPI, TorchRLModule):
    def setup(self):
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def get_non_inference_attributes(self):
        return []

    def _forward_inference(self, batch, **kwargs):
        return self._common_forward(batch)

    def _forward_exploration(self, batch, **kwargs):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        mask = batch[Columns.OBS]["action_mask"]
        noise = torch.rand_like(mask, dtype=torch.float32)
        inf_mask = (1 - mask) * INVALID_MASK
        actions = torch.argmax(noise + inf_mask, dim=-1)
        return {Columns.ACTIONS: actions}

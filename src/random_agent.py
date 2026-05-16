import numpy as np
import torch
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.algorithms.dqn.dqn_learner import (
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_TARGET_NEXT_PREDS,
)


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
        outputs = self._common_forward(batch, obs_key=Columns.OBS)
        if Columns.NEXT_OBS in batch:
            next_outputs = self._common_forward(batch, obs_key=Columns.NEXT_OBS)
            outputs[QF_NEXT_PREDS] = next_outputs[QF_PREDS]
            outputs[QF_TARGET_NEXT_PREDS] = next_outputs[QF_PREDS]

        return outputs

    def _common_forward(self, batch, obs_key=Columns.OBS):
        mask = batch[obs_key]["action_mask"]
        q_values = torch.rand_like(mask, dtype=torch.float32)

        inf_mask = (1 - mask) * -1e34
        masked_q_values = q_values + inf_mask

        return {
            QF_PREDS: masked_q_values,
            Columns.ACTION_DIST_INPUTS: masked_q_values,
        }

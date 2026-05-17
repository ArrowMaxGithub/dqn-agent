import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.dqn.dqn_learner import (
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_TARGET_NEXT_PREDS,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.core.rl_module.apis.target_network_api import TargetNetworkAPI
from ray.rllib.core.rl_module.torch import TorchRLModule


class DQNAgent:
    def save(self, path):
        algo_path = Path(path).resolve()
        os.makedirs(os.path.dirname(algo_path), exist_ok=True)
        self.algo.save_to_path(algo_path.as_uri())

    def load(path):
        algo_path = Path(path).resolve().as_uri()
        agent = DQNAgent()
        agent.algo = Algorithm.from_checkpoint(algo_path)
        return agent

    def get_label(self):
        return "DQNAgent"

    def get_action(self, obs_dict):
        module = self.algo.get_module("p0")
        batch_input = self._get_batch_input(obs_dict)
        output = module.forward_inference(batch_input)
        return torch.argmax(output[Columns.ACTION_DIST_INPUTS], dim=-1).item()

    def _get_batch_input(self, obs_dict, next_obs_dict=None):
        batch_input = {
            Columns.OBS: self._transform_obs_dict(obs_dict=obs_dict),
        }

        if next_obs_dict:
            batch_input[Columns.NEXT_OBS] = self._transform_obs_dict(
                obs_dict=next_obs_dict
            )

        return batch_input

    def _transform_obs_dict(self, obs_dict):
        obs_tensor = (
            torch.from_numpy(np.array(obs_dict["observations"])).long().unsqueeze(0)
        )
        mask_tensor = (
            torch.from_numpy(np.array(obs_dict["action_mask"])).float().unsqueeze(0)
        )

        trans_obs_dict = {"observations": obs_tensor, "action_mask": mask_tensor}
        return trans_obs_dict


class DQNMaskedRLModule(TargetNetworkAPI, TorchRLModule):
    def setup(self):
        obs_shape = self.config.observation_space["observations"].shape
        num_states = self.config.observation_space["observations"].nvec.max()
        num_outputs = self.config.action_space.n
        input_dim = obs_shape[0] * num_states

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

        self.make_target_networks()

    def make_target_networks(self) -> None:
        self.target_net = make_target_network(self.net)

    def get_target_network_pairs(self):
        return [
            (self.net, self.target_net),
        ]

    def forward_target(self, batch):
        with torch.no_grad():
            obs = batch[Columns.OBS]["observations"].long()
            flat_obs = self._get_flat_obs(obs)
            q_values = self.target_net(flat_obs)

            return {QF_PREDS: q_values}

    def _forward_inference(self, batch, **kwargs):
        return self._common_forward(batch)

    def _forward_exploration(self, batch, **kwargs):
        mask = batch[Columns.OBS]["action_mask"]
        outputs = self._common_forward(batch)
        epsilon = self.model_config["epsilon"]

        if torch.rand(1).item() < epsilon:
            random = torch.rand_like(mask, dtype=torch.float32)
            inf_mask = (1 - mask) * -1e8
            return {
                Columns.ACTION_DIST_INPUTS: random + inf_mask,
                QF_PREDS: outputs[QF_PREDS],
            }

        return outputs

    def compute_q_values(self, batch):
        obs = batch[Columns.OBS]["observations"].long()
        flat_obs = self._get_flat_obs(obs)
        return {QF_PREDS: self.net(flat_obs)}

    def compute_advantage_distribution(self, batch):
        return self.compute_q_values(batch)

    def _forward_train(self, batch, **kwargs):
        outputs = self._common_forward(batch)

        if Columns.NEXT_OBS in batch:
            next_batch = {Columns.OBS: batch[Columns.NEXT_OBS]}

            online_next = self.compute_q_values(next_batch)
            target_next = self.forward_target(next_batch)

            outputs[QF_NEXT_PREDS] = online_next[QF_PREDS]
            outputs[QF_TARGET_NEXT_PREDS] = target_next[QF_PREDS]

        return outputs

    def _common_forward(self, batch):
        obs = batch[Columns.OBS]["observations"].long()
        mask = batch[Columns.OBS]["action_mask"]
        flat_obs = self._get_flat_obs(obs)
        q_values = self.net(flat_obs)

        inf_mask = (1 - mask) * -1e8
        masked_q_values = q_values + inf_mask

        return {
            Columns.ACTION_DIST_INPUTS: masked_q_values,
            QF_PREDS: q_values,
        }

    def _get_flat_obs(self, obs):
        nvec = torch.tensor(
            self.config.observation_space["observations"].nvec, device=obs.device
        )
        num_classes = int(nvec.max())
        one_hot = F.one_hot(obs, num_classes=num_classes).float()
        return one_hot.view(obs.shape[0], -1)

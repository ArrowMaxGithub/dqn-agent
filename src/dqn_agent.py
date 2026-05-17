import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

        self.embedding = nn.Embedding(num_states, num_states)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )

        self.make_target_networks()

    def make_target_networks(self) -> None:
        self.target_embedding = make_target_network(self.embedding)
        self.target_net = make_target_network(self.net)

    def get_target_network_pairs(self):
        return [
            (self.embedding, self.target_embedding),
            (self.net, self.target_net),
        ]

    def forward_target(self, batch):
        with torch.no_grad():
            obs = batch[Columns.OBS]["observations"].long()
            mask = batch[Columns.OBS]["action_mask"]

            embedded = self.target_embedding(obs)
            flat_obs = embedded.view(obs.shape[0], -1)
            q_values = self.target_net(flat_obs)

            inf_mask = (1 - mask) * -1e8
            return {QF_PREDS: q_values + inf_mask}

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
            }

        return outputs

    def _forward_train(self, batch, **kwargs):
        outputs = self._common_forward(batch)
        outputs[QF_PREDS] = outputs[Columns.ACTION_DIST_INPUTS]

        if Columns.NEXT_OBS in batch:
            next_batch = {Columns.OBS: batch[Columns.NEXT_OBS]}

            online_output = self._common_forward(next_batch)
            target_output = self.forward_target(next_batch)

            outputs[QF_TARGET_NEXT_PREDS] = target_output[QF_PREDS]
            outputs[QF_NEXT_PREDS] = online_output[Columns.ACTION_DIST_INPUTS]

        return outputs

    def _common_forward(self, batch):
        obs = batch[Columns.OBS]["observations"].long()
        mask = batch[Columns.OBS]["action_mask"]

        embedded = self.embedding(obs)
        flat_obs = embedded.view(obs.shape[0], -1)
        q_values = self.net(flat_obs)

        inf_mask = (1 - mask) * -1e8
        masked_q_values = q_values + inf_mask

        return {
            Columns.ACTION_DIST_INPUTS: masked_q_values,
        }

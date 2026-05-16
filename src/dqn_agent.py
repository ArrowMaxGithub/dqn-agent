import numpy as np
import torch
import torch.nn as nn
from ray.rllib.algorithms.dqn.dqn_learner import (
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_TARGET_NEXT_PREDS,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule

from ray.rllib.algorithms.algorithm import Algorithm
import os
from pathlib import Path
import json

from tqdm import tqdm

from ray.rllib.core.rl_module.apis.target_network_api import TargetNetworkAPI
from ray.rllib.core.learner.utils import make_target_network


class DQNAgent:
    def save(self, path):
        parameters_path = Path(f"{path}/agent_parameters.json").resolve()
        algo_path = Path(path).resolve().as_uri()
        os.makedirs(os.path.dirname(parameters_path), exist_ok=True)

        parameters = {
            "passing_action": self.passing_action,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
        }
        with open(parameters_path, "w") as f:
            json.dump(parameters, f)

        self.algo.save_to_path(algo_path)

    def load(path):
        parameters_path = Path(f"{path}/agent_parameters.json").resolve()
        algo_path = Path(path).resolve().as_uri()

        with open(parameters_path, "r") as f:
            parameters = json.load(f)

        agent = DQNAgent()
        agent.passing_action = parameters["passing_action"]
        agent.epsilon = parameters["epsilon"]
        agent.epsilon_decay = parameters["epsilon_decay"]
        agent.final_epsilon = parameters["final_epsilon"]
        agent.algo = Algorithm.from_checkpoint(algo_path)

        return agent

    def get_label(self):
        return "DQNAgent"

    def train(self, env_factory, n_episodes):
        print(f"Current epsilon: {self.epsilon:.2f}")
        batches = n_episodes // self.algo.config.train_batch_size
        for batch in tqdm(range(batches)):
            self.algo.train()
            self.epsilon = max(
                self.epsilon - self.epsilon_decay,
                self.final_epsilon,
            )

    def get_action(self, obs_dict, force_exploitation=False):
        if np.random.random() < self.epsilon and not force_exploitation:
            mask = obs_dict["action_mask"]
            valid_actions = np.where(mask == 1)[0]
            return np.random.choice(valid_actions)
        else:
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

        self.initial_epsilon = self.model_config.get("initial_epsilon", 1.0)
        self.final_epsilon = self.model_config.get("final_epsilon", 0.1)
        self.decay_steps = self.model_config.get("decay_steps", 10000)
        self.step_counter = 0

        self.embedding = nn.Embedding(num_states, num_states)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
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
        obs = batch[Columns.OBS]["observations"].long()
        mask = batch[Columns.OBS]["action_mask"]

        embedded = self.target_embedding(obs)
        flat_obs = embedded.view(obs.shape[0], -1)
        q_values = self.target_net(flat_obs)

        inf_mask = (1 - mask) * -1e9
        return {QF_PREDS: q_values + inf_mask}

    def _forward_inference(self, batch, **kwargs):
        return self._common_forward(batch)

    def _forward_exploration(self, batch, **kwargs):
        batch_size = batch[Columns.OBS]["observations"].shape[0]
        outputs = self._common_forward(batch)
        mask = batch[Columns.OBS]["action_mask"]

        decay = self.step_counter / self.decay_steps
        decay_delta = self.initial_epsilon - self.final_epsilon
        epsilon = max(self.final_epsilon, self.initial_epsilon - decay * decay_delta)
        self.step_counter += batch_size

        if np.random.rand() < epsilon:
            random = torch.rand_like(mask, dtype=torch.float32)
            inf_mask = (1 - mask) * -1e9
            outputs[QF_PREDS] = torch.zeros_like(random, dtype=torch.float32)
            outputs[Columns.ACTION_DIST_INPUTS] = random + inf_mask

        return outputs

    def _forward_train(self, batch, **kwargs):
        outputs = self._common_forward(batch, obs_key=Columns.OBS)

        if Columns.NEXT_OBS in batch:
            next_batch = {Columns.OBS: batch[Columns.NEXT_OBS]}
            target_output = self.forward_target(next_batch)

            outputs[QF_TARGET_NEXT_PREDS] = target_output[QF_PREDS]
            outputs[QF_NEXT_PREDS] = outputs[QF_PREDS]

        return outputs

    def _common_forward(self, batch, obs_key=Columns.OBS):
        obs = batch[obs_key]["observations"].long()
        mask = batch[obs_key]["action_mask"]

        embedded = self.embedding(obs)
        flat_obs = embedded.view(obs.shape[0], -1)
        q_values = self.net(flat_obs)

        inf_mask = (1 - mask) * -1e9
        masked_q_values = q_values + inf_mask

        return {
            QF_PREDS: masked_q_values,
            Columns.ACTION_DIST_INPUTS: masked_q_values,
        }

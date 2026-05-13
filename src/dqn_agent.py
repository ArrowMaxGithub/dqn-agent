import numpy as np
import torch
import torch.nn as nn
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_learner import (
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_TARGET_NEXT_PREDS,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.tune.registry import register_env

from env import Cardgame


def env_creator(cfg):
    env = Cardgame()
    return ParallelPettingZooEnv(env)


tmp_env = env_creator(None)
obs_space = tmp_env.observation_space["agent_0"]
act_space = tmp_env.action_space["agent_0"]

register_env("custom-cardgame-v1", env_creator)


class DQNAgent:
    def __init__(
        self,
        passing_action,
        learning_rate,
        n_steps_total,
        train_batch_size,
        initial_epsilon,
        final_epsilon,
        dueling,
        double_q,
    ):
        self.passing_action = passing_action
        self.epsilon = initial_epsilon
        self.epsilon_decay = (self.epsilon - final_epsilon) / 2 * n_steps_total
        self.final_epsilon = final_epsilon

        config = (
            DQNConfig()
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(
                env="custom-cardgame-v1",
                disable_env_checking=True,
            )
            .multi_agent(
                policies={"p0": (None, obs_space, act_space, {})},
                policy_mapping_fn=lambda aid, *args, **kwargs: "p0",
            )
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=MaskedRLModule,
                    model_config={"action_dist_class": TorchCategorical},
                )
            )
            .resources(num_gpus=1)
            .env_runners(num_env_runners=16, num_envs_per_env_runner=4)
            .training(
                replay_buffer_config={
                    "type": "MultiAgentEpisodeReplayBuffer",
                    "capacity": 500_000,
                },
                lr=learning_rate,
                epsilon=[(0, 1.0), (n_steps_total, final_epsilon)],
                dueling=dueling,
                double_q=double_q,
                train_batch_size=train_batch_size,
                training_intensity=1.0,
            )
        )

        self.algo = config.build()
        self.module = self.algo.get_module("p0")

    def get_action(self, obs_dict, force_exploitation=False):

        if np.random.random() < self.epsilon and not force_exploitation:
            mask = obs_dict["action_mask"]
            valid_actions = np.where(mask == 1)[0]
            return np.random.choice(valid_actions)
        else:
            batch_input = self._get_batch_input(obs_dict)
            output = self.module.forward_inference(batch_input)
            return torch.argmax(output[Columns.ACTION_DIST_INPUTS], dim=-1).item()

    def update(self, obs, action, reward, term, next_obs):
        self.algo.train()

    def get_label(self):
        return "DQNAgent"

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

    def _decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class MaskedRLModule(TorchRLModule):
    def setup(self):
        obs_shape = self.config.observation_space["observations"].shape
        num_states = self.config.observation_space["observations"].nvec.max()
        num_outputs = self.config.action_space.n

        input_dim = obs_shape[0] * num_states

        self.embedding = nn.Embedding(num_states, num_states)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

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

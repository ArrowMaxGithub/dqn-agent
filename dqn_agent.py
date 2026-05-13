import torch
import torch.nn as nn
import torch.nn.functional as F
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

from ray.rllib.algorithms.algorithm import Algorithm

from par_env import Cardgame

import numpy as np


def env_creator(cfg):
    env = Cardgame()
    return ParallelPettingZooEnv(env)


tmp_env = env_creator(None)
obs_space = tmp_env.observation_space["agent_0"]
act_space = tmp_env.action_space["agent_0"]

register_env("custom-cardgame-v1", env_creator)


class DQNAgent:
    def __init__(self, checkpoint_path, passing_action):
        self.algo = Algorithm.from_checkpoint(checkpoint_path)
        self.module = self.algo.get_module("p0")
        self.module.eval()
        self.passing_action = passing_action

    def get_action(self, obs_dict, force_exploitation=False):
        obs_tensor = (
            torch.from_numpy(np.array(obs_dict["observations"])).long().unsqueeze(0)
        )
        mask_tensor = (
            torch.from_numpy(np.array(obs_dict["action_mask"])).float().unsqueeze(0)
        )

        nested_obs_dict = {"observations": obs_tensor, "action_mask": mask_tensor}
        batch_input = {"obs": nested_obs_dict}

        with torch.no_grad():
            if force_exploitation:
                output = self.module.forward_inference(batch_input)
            else:
                output = self.module.forward_exploration(batch_input)

        return output["actions"][0].item()

    def get_label(self):
        return "DQNAgent"


class MaskedRLModule(TorchRLModule):
    def setup(self):
        obs_shape = self.config.observation_space["observations"].shape
        num_states = self.config.observation_space["observations"].nvec.max()
        num_outputs = self.config.action_space.n
        input_dim = obs_shape[0] * num_states

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

    def _forward_inference(self, batch, **kwargs):
        outputs = self._common_forward(batch)
        outputs["actions"] = torch.argmax(outputs["action_dist_inputs"], dim=-1)
        return outputs

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
        obs_batch = batch[obs_key]
        obs = obs_batch["observations"]
        mask = obs_batch["action_mask"]
        num_states = self.config.observation_space["observations"].nvec.max()

        obs_one_hot = F.one_hot(obs.long(), num_classes=num_states).float()
        flat_obs = obs_one_hot.view(obs.shape[0], -1)
        logits = self.net(flat_obs)

        inf_mask = torch.clamp((1 - mask) * -1e34, min=-1e34)
        masked_logits = logits + inf_mask

        return {
            QF_PREDS: masked_logits,
            Columns.ACTION_DIST_INPUTS: masked_logits,
        }


if __name__ == "__main__":
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
        .env_runners(num_env_runners=16)
        .training(
            replay_buffer_config={
                "type": "MultiAgentEpisodeReplayBuffer",
                "capacity": 50000,
            },
            lr=0.001,
            train_batch_size=256,
        )
    )

    algo = config.build()
    for i in range(16):
        result = algo.train()
        print(f"Iteration: {i}")
    algo.save_to_path("/home/max/dev/dqn-agent/checkpoints")

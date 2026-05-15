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
from durak_env import DurakEnv

from ray.rllib.algorithms.algorithm import Algorithm
import pickle
import os
from pathlib import Path


def env_creator(cfg):
    return ParallelPettingZooEnv(DurakEnv())


tmp_env = env_creator(None)
player_id = tmp_env.possible_agents[0]
obs_space = tmp_env.observation_space[player_id]
act_space = tmp_env.action_space[player_id]

register_env(
    "custom-cardgame-v1",
    env_creator,
)


class DQNAgent:
    def new(
        self,
        passing_action,
        learning_rate,
        n_steps_total,
        train_batch_size,
        num_env_runners,
        num_envs_per_env_runner,
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
            .env_runners(
                num_env_runners=num_env_runners,
                num_envs_per_env_runner=num_envs_per_env_runner,
            )
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
        return self

    def save(self, path):
        parameters_path = Path(f"{path}/agent_parameters.pkl").resolve()
        algo_path = Path(path).resolve().as_uri()
        os.makedirs(os.path.dirname(parameters_path), exist_ok=True)

        parameters = {
            "passing_action": self.passing_action,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
        }
        with open(parameters_path, "wb") as f:
            pickle.dump(parameters, f)

        self.algo.save_to_path(algo_path)

    def load(path):
        parameters_path = Path(f"{path}/agent_parameters.pkl").resolve()
        algo_path = Path(path).resolve().as_uri()

        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        algo = Algorithm.from_checkpoint(algo_path)

        agent = DQNAgent()
        agent.passing_action = parameters["passing_action"]
        agent.epsilon = parameters["epsilon"]
        agent.passepsilon_decaying_action = parameters["epsilon_decay"]
        agent.final_epsilon = parameters["final_epsilon"]
        agent.algo = algo

        return agent

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

    def update(self, obs=None, action=None, reward=None, term=None, next_obs=None):
        print("dqn train")
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


def bench():
    import time

    from tqdm import tqdm

    epochs = 5
    train_batch_size = 1024
    n_steps_total = epochs * train_batch_size

    env = DurakEnv()

    agent = DQNAgent().new(
        passing_action=env.passing_action,
        learning_rate=0.001,
        n_steps_total=n_steps_total,
        train_batch_size=train_batch_size,
        num_env_runners=16,
        num_envs_per_env_runner=32,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        dueling=False,
        double_q=False,
    )

    start = time.perf_counter_ns()
    for e in tqdm(range(epochs)):
        agent.update()
    end = time.perf_counter_ns()
    delta = end - start
    delta_s = delta / 1000 / 1000 / 1000
    avg_s = delta_s / epochs
    avg_ms_per_step = delta / 1000 / 1000 / train_batch_size
    print(
        f"Total time: {delta_s:.2f}s | Avg: {avg_s:.2f}s | Avg-per-step: {avg_ms_per_step:.2f}ms"
    )


if __name__ == "__main__":
    bench()

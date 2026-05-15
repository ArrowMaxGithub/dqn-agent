import numpy as np
import os
from pathlib import Path
import json
import msgpack
from tqdm import tqdm


class QAgent:
    def new(
        self,
        passing_action: int,
        n_action_space: int,
        learning_rate: float,
        n_steps_total: int,
        initial_epsilon: float,
        final_epsilon: float,
        discount_factor: float,
        illegal_mask: float,
    ):
        self.passing_action = passing_action
        self.n_action_space = n_action_space
        self.q_values = {}
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = (self.epsilon - final_epsilon) / (2 * n_steps_total)
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.illegal_mask = illegal_mask
        return self

    def save(self, path):
        q_valuespath = Path(f"{path}/agent.pkl").resolve()
        parameters_path = Path(f"{path}/agent_parameters.json").resolve()
        os.makedirs(os.path.dirname(q_valuespath), exist_ok=True)

        parameters = {
            "passing_action": self.passing_action,
            "n_action_space": self.n_action_space,
            "lr": self.lr,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
            "discount_factor": self.discount_factor,
            "illegal_mask": self.illegal_mask,
        }

        with open(parameters_path, "w") as f:
            json.dump(parameters, f)

        with open(q_valuespath, "wb") as f:
            f.write(
                msgpack.packb(self.q_values, default=encode_numpy, use_bin_type=True)
            )

    def load(path):
        parameters_path = Path(f"{path}/agent_parameters.json").resolve()
        path = Path(f"{path}/agent.pkl").resolve()

        with open(parameters_path, "r") as f:
            parameters = json.load(f)

        with open(path, "rb") as f:
            q_values = msgpack.unpackb(f.read(), object_hook=decode_numpy, raw=False)

        agent = QAgent()
        agent.passing_action = parameters["passing_action"]
        agent.n_action_space = parameters["n_action_space"]
        agent.lr = parameters["lr"]
        agent.epsilon = parameters["epsilon"]
        agent.epsilon_decay = parameters["epsilon_decay"]
        agent.final_epsilon = parameters["final_epsilon"]
        agent.discount_factor = parameters["discount_factor"]
        agent.illegal_mask = parameters["illegal_mask"]
        agent.q_values = q_values

        return agent

    def get_label(self):
        return "QAgent"

    def train(self, env_factory, n_episodes):
        env = env_factory()
        agents = (self, self)
        agents_dict = {
            agent_id: agent for agent_id, agent in zip(env.possible_agents, agents)
        }

        actions = {agent_id: None for agent_id in env.possible_agents}

        for _ in tqdm(range(n_episodes)):
            obss, infos = env.reset()

            while env.agents:
                for agent_id in env.agents:
                    actions[agent_id] = agents_dict[agent_id].get_action(obss[agent_id])

                last_obss = dict(obss)
                obss, rewards, terms, truncs, infos = env.step(actions)

                for agent_id, agent in agents_dict.items():
                    last_obs = last_obss[agent_id]
                    action = actions[agent_id]
                    reward = rewards[agent_id]
                    term = terms[agent_id]
                    obs = obss[agent_id]

                    agent.update(last_obs, action, reward, term, obs)

    def get_action(self, obs_dict, force_exploitation=False):
        mask = obs_dict["action_mask"]
        obs = obs_dict["observations"]

        obs_key = self._obs_key(obs)
        q_values = self._get_q_values(obs_key)

        illegal_mask = (1 - mask) * self.illegal_mask

        if np.random.random() < self.epsilon and not force_exploitation:
            legal_actions = np.where(mask == 1)[0]
            return np.random.choice(legal_actions)
        else:
            max_q = np.max(q_values + illegal_mask)
            equal = np.where((q_values + illegal_mask) == max_q)[0]
            return int(np.random.choice(equal))

    def update(
        self,
        obs_dict,  # s
        action: int,  # a: s -> s'
        reward: float,
        terminated: bool,
        next_obs_dict,  # s'
    ):
        mask = next_obs_dict["action_mask"]
        next_obs = next_obs_dict["observations"]
        obs = obs_dict["observations"]

        obs_key = self._obs_key(obs)
        next_obs_key = self._obs_key(next_obs)

        illegal_mask = (1 - mask) * self.illegal_mask
        q_values = self._get_q_values(obs_key)
        next_q_values = self._get_q_values(next_obs_key)

        if terminated:
            future_q_value = 0.0
        else:
            masked_future_q_values = next_q_values + illegal_mask
            future_q_value = np.max(masked_future_q_values)

        target = reward + self.discount_factor * future_q_value
        temporal_diff = target - q_values[action]

        self.q_values[obs_key][action] += self.lr * temporal_diff
        self._decay_epsilon()

    def _get_q_values(self, obs_key):
        if obs_key not in self.q_values:
            self.q_values[obs_key] = np.zeros(self.n_action_space)
        return self.q_values[obs_key]

    def _decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def _obs_key(self, obs):
        return obs.tobytes()


def encode_numpy(obj):
    return {
        b"__nd__": True,
        b"data": obj.tobytes(),
        b"dtype": str(obj.dtype),
        b"shape": obj.shape,
    }


def decode_numpy(obj):
    if b"__nd__" in obj:
        return np.frombuffer(obj[b"data"], dtype=obj[b"dtype"]).reshape(obj[b"shape"])
    return obj

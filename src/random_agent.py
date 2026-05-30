import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.inference_only_api import InferenceOnlyAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from interfaces import AgentInterface

from agent_utils import create_batch_from_dict

from durak_env import DurakEnv

INVALID_MASK = -1e8


class RandomAgent(AgentInterface):
    def __init__(
        self,
        passing_action,
    ):
        self.passing_action = passing_action
        self.module = RandomMaskedRLModule()

    def GetName(self) -> str:
        return "RandomAgent"

    def GetAction(self, obs_dict: dict) -> int:
        batch = create_batch_from_dict(obs_dict=obs_dict)
        return self.module.forward_inference(batch)[Columns.ACTIONS].item()


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
        inf_mask = ~mask * INVALID_MASK
        actions = torch.argmax(noise + inf_mask, dim=-1)
        return {Columns.ACTIONS: actions}


def test():
    env = DurakEnv()
    agents_dict = {
        agent_id: RandomAgent(env.passing_action) for agent_id in env.possible_agents
    }

    obss, infos = env.reset()
    while env.agents:
        actions = {
            agent_id: agents_dict[agent_id].GetAction(obss[agent_id])
            for agent_id in obss.keys()
        }

        obss, rewards, terms, truncs, infos = env.step(actions)

    winner = env._get_winner()
    print(f"Winner: {winner}")


if __name__ == "__main__":
    test()

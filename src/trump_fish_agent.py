import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.inference_only_api import InferenceOnlyAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from interfaces import AgentInterface

from durak_env import (
    DurakEnv,
    get_index_from_card,
)

from agent_utils import (
    batch_size,
    create_batch_from_dict,
    get_legal_cards,
    get_game_info,
    get_sorted_cards,
    partition_trump_cards,
)


class TrumpFishAgent(AgentInterface):
    def __init__(
        self,
        passing_action,
    ):
        self.passing_action = passing_action
        self.module = TrumpFishRLModule()

    def GetName(self) -> str:
        return "TrumpFishAgent"

    # Get agent action by forwarding the call to the RLModule
    def GetAction(self, obs_dict: dict) -> int:
        # Reconstruct batch shape
        batch = create_batch_from_dict(obs_dict=obs_dict)

        # Forward to the internal module to avoid redundant implementation
        return self.module.forward_inference(batch)[Columns.ACTIONS].item()


class TrumpFishRLModule(InferenceOnlyAPI, TorchRLModule):
    def setup(self):
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def get_non_inference_attributes(self):
        return []

    def _forward_inference(self, batch, **kwargs):
        return self._common_forward(batch)

    def _forward_exploration(self, batch, **kwargs):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        # batch may contain data from many batched games at once
        # actions returns 1 action per batch-entry
        actions = []

        for b in range(batch_size(batch=batch)):
            legal_cards = get_legal_cards(batch=batch, batch_index=b)

            if len(legal_cards) == 0:
                actions.append(36)
                continue

            game_info = get_game_info(batch=batch, batch_index=b)
            trump = game_info["trump"]
            sorted_cards = get_sorted_cards(legal_cards, trump=trump)
            non_trumps, trumps = partition_trump_cards(sorted_cards, trump=trump)

            if game_info["is_attacker"]:
                if len(non_trumps) > 0:
                    card = non_trumps[-1]  # Highest non-trump
                else:
                    card = trumps[0]  # Lowest trump

                actions.append(get_index_from_card(card))

            else:
                actions.append(get_index_from_card(sorted_cards[0]))  # Lowest card

        return {Columns.ACTIONS: torch.tensor(actions)}


def test():
    env = DurakEnv()
    agents_dict = {
        agent_id: TrumpFishAgent(env.passing_action) for agent_id in env.possible_agents
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

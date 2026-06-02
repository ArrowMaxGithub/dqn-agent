# interpolation_agent.py

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.inference_only_api import InferenceOnlyAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from interfaces import AgentInterface

from durak_env import DurakEnv, get_index_from_card

from agent_utils import (
    batch_size,
    create_batch_from_dict,
    get_game_info,
    get_legal_cards,
    get_sorted_cards,
)

PASSING_ACTION = 36
MAX_DRAW_PILE_SIZE = 24  # 36 Karten - 2 * 6 Starthandkarten


class InterpolationAgent(AgentInterface):
    def __init__(self, passing_action: int = PASSING_ACTION):
        self.passing_action = passing_action
        self.module = InterpolationRLModule()

    def GetName(self) -> str:
        return "InterpolationAgent"

    def GetAction(self, obs_dict: dict) -> int:
        batch = create_batch_from_dict(obs_dict=obs_dict)
        return self.module.forward_inference(batch)[Columns.ACTIONS].item()


class InterpolationRLModule(InferenceOnlyAPI, TorchRLModule):
    def setup(self):
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def get_non_inference_attributes(self):
        return []

    def _forward_inference(self, batch, **kwargs):
        return self._common_forward(batch)

    def _forward_exploration(self, batch, **kwargs):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        actions = []

        for b in range(batch_size(batch=batch)):
            legal_cards = get_legal_cards(batch=batch, batch_index=b)

            if len(legal_cards) == 0:
                actions.append(PASSING_ACTION)
                continue

            game_info = get_game_info(batch=batch, batch_index=b)
            trump = game_info["trump"]

            sorted_cards = get_sorted_cards(legal_cards, trump=trump)

            progress = self._game_progress(game_info)

            selected_index = round(progress * (len(sorted_cards) - 1))
            selected_card = sorted_cards[selected_index]

            actions.append(get_index_from_card(selected_card))

        return {
            Columns.ACTIONS: torch.tensor(
                actions,
                dtype=torch.long,
                device=batch[Columns.OBS]["action_mask"].device,
            )
        }

    def _game_progress(self, game_info: dict) -> float:
        draw_pile = game_info["draw_pile"]

        progress = 1.0 - (draw_pile / MAX_DRAW_PILE_SIZE)

        return max(0.0, min(1.0, progress))


def test():
    env = DurakEnv()

    agents_dict = {
        agent_id: InterpolationAgent(env.passing_action)
        for agent_id in env.possible_agents
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

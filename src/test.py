from tqdm import tqdm

from adapter import AgentInterface


def test(
    env_factory: callable,
    agent: AgentInterface,
    opponent: AgentInterface,
    n_episodes: int,
) -> (float, float, float):
    env = env_factory()

    wins = 0
    draws = 0
    losses = 0

    agent_id = env.possible_agents[0]
    opponent_id = env.possible_agents[1]

    agents_dict = {
        agent_id: agent,
        opponent_id: opponent,
    }

    for _ in tqdm(range(n_episodes)):
        obss, infos = env.reset()

        while env.agents:
            actions = {
                agent_id: agents_dict[agent_id].get_action(obss[agent_id])
                for agent_id in obss.keys()
            }

            obss, rewards, terms, truncs, infos = env.step(actions)

        winner = env._get_winner()
        if winner == agent_id:
            wins += 1
        elif winner == opponent_id:
            losses += 1
        elif winner is None:
            draws += 1

    return (
        wins / n_episodes,
        draws / n_episodes,
        losses / n_episodes,
    )

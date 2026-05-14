from tqdm import tqdm


def test(env, agents, n_episodes) -> (float, float, float):
    assert len(agents) == len(env.possible_agents)

    wins = 0
    draws = 0
    losses = 0

    agents_dict = {
        agent_id: agent for agent_id, agent in zip(env.possible_agents, agents)
    }
    agent_0_id = env.possible_agents[0]
    agent_1_id = env.possible_agents[1]

    for _ in tqdm(range(n_episodes)):
        obss, infos = env.reset()

        while env.agents:
            actions = {
                agent_id: agents_dict[agent_id].get_action(
                    obss[agent_id], force_exploitation=True
                )
                for agent_id in env.agents
            }

            obss, rewards, terms, truncs, infos = env.step(actions)

        winner = env._get_winner()
        if winner == agent_0_id:
            wins += 1
        elif winner == agent_1_id:
            losses += 1
        elif winner is None:
            draws += 1

    return (
        wins / n_episodes,
        draws / n_episodes,
        losses / n_episodes,
    )


def cross(env, agents, pairings, n_episodes):
    results = {}

    for a0, a1 in pairings:
        agent_0 = agents[a0]
        agent_1 = agents[a1]
        results[(agent_0.get_label(), agent_1.get_label())] = test(
            env, (agent_0, agent_1), n_episodes
        )

    return results

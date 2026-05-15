from tqdm import tqdm
from q_agent import QAgent
from dqn_agent import DQNAgent
from random_agent import RandomAgent


def test(env_factory, agents, n_episodes) -> (float, float, float):
    env = env_factory()
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


def test_all(env_factory, pairings, n_episodes):
    results = {}

    for a0, a1 in pairings:
        results[(a0.get_label(), a1.get_label())] = test(
            env_factory, (a0, a1), n_episodes
        )

    return results


def test_all_with_epoch(env_factory, pairings, n_episodes):
    results = {}

    for epoch, (a0, a1) in enumerate(pairings):
        results[(epoch, a0.get_label(), a1.get_label())] = test(
            env_factory, (a0, a1), n_episodes
        )

    return results


def test_all_checkpoints(env_factory, pairing, epochs, n_episodes):
    (a0, a1) = pairing
    agents_0 = load_all_checkpoints(a0.get_label(), epochs)
    agents_1 = load_all_checkpoints(a1.get_label(), epochs)
    pairings = zip(agents_0, agents_1)
    return test_all_with_epoch(env_factory, pairings, n_episodes)


def load_all_checkpoints(label, epochs):
    agents = []
    for epoch in range(epochs):
        path = f"./checkpoints/{label}/{epoch}/"
        agent = load_agent(label, path)
        agents.append(agent)
    return agents


def load_agent(label, path):
    agent = None
    match label:
        case "QAgent":
            agent = QAgent.load(path)
        case "DQNAgent":
            agent = DQNAgent.load(path)
        case "RandomAgent":
            agent = RandomAgent(36)

    if agent is None:
        raise ValueError(f"Failed to load agent from {path}")

    return agent

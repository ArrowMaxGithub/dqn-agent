from pettingzoo import ParallelEnv
from tqdm import tqdm


def train(agents, env: ParallelEnv, n_episodes):
    assert len(agents) == env.max_num_agents

    agents_dict = {
        agent_id: agent for agent_id, agent in zip(env.possible_agents, agents)
    }

    for _ in tqdm(range(n_episodes)):
        obss, infos = env.reset()

        while env.agents:
            actions = {
                agent_id: agents_dict[agent_id].get_action(obss[agent_id])
                for agent_id in env.agents
            }

            last_obss = dict(obss)
            obss, rewards, terms, truncs, infos = env.step(actions)

            for agent_id, agent in agents_dict.items():
                last_obs = last_obss[agent_id]
                action = actions[agent_id]
                reward = rewards[agent_id]
                term = terms[agent_id]
                obs = obss[agent_id]

                agent.update(last_obs, action, reward, term, obs)

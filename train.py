from tqdm import tqdm
from pettingzoo import AECEnv


def train(agents, env: AECEnv, n_episodes):
    agents_dict = {
        agent_id: agent for agent_id, agent in zip(env.possible_agents, agents)
    }

    for _ in tqdm(range(n_episodes)):
        env.reset()

        last_act = {agent_id: None for agent_id in env.agents}
        last_obs = {agent_id: None for agent_id in env.agents}

        for agent_id in env.agent_iter():
            agent = agents_dict[agent_id]
            obs, reward, term, trunc, info = env.last()

            if last_act[agent_id] is not None:
                agent.update(last_obs[agent_id], last_act[agent_id], reward, term, obs)

            if term or trunc:
                action = None
            else:
                action = agent.get_action(obs)

            env.step(action)

            last_obs[agent_id] = obs
            last_act[agent_id] = action
            agent.decay_epsilon()

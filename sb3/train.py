from tqdm import tqdm
from pettingzoo import AECEnv


def train(q_agents, env: AECEnv, n_episodes):
    for episode in tqdm(range(n_episodes)):
        env.reset()

        last_act = {agent: None for agent in env.agents}
        last_obs = {agent: None for agent in env.agents}

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            if last_act[agent] is not None:
                q_agents[agent].update(last_obs, last_act, reward, term, obs)

            if term or trunc:
                action = None
            else:
                action = q_agents[agent].get_action(obs, info)
            env.step(action)

            last_obs[agent] = obs
            last_act[agent] = action
            q_agents[agent].decay_epsilon()

        for agent in env.agents:
            obs, reward, term, trunc, info = env._get_agent_data(agent)
            if last_act[agent] is not None:
                q_agents[agent].update(
                    last_obs[agent], last_act[agent], reward, term, obs
                )

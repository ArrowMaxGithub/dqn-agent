from tqdm import tqdm


def train(agent, env, n_episodes):
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            legal = env.unwrapped.get_legal_mask()
            action = agent.get_action(obs, legal)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

import numpy as np


def test_agent(agent, env, num_episodes=1000):
    total_rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            legal = env.unwrapped.get_legal_mask()
            action = agent.get_action(obs, legal)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    draw_rate = np.mean(np.array(total_rewards) == 0)
    lose_rate = np.mean(np.array(total_rewards) < 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win: {win_rate:.1%} | Draw: {draw_rate:.1%} | Lose: {lose_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

from pathlib import Path

import torch
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS
from tqdm import tqdm

from config import dqn_config, save_parameters, set_epsilon
from epsilon_decay import EpsilonDecay


def main():
    print(f"GPU supported: {torch.cuda.is_available()}")

    experiment_name = "2026_05_19"
    params = {
        "learning_rate": 1e-4,
        "iterations": 16384,
        "epsilon_schedule": "linear",
        "epsilon_decay": 0.67,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "num_env_runners": 16,
        "num_envs_per_env_runner": 8,
        "replay_buffer_capacity": 65536 * 16,
        "double_q": True,
        "train_batch_size": 2048,
        "num_steps_sampled_before_learning_starts": 65536,
        "target_network_update_freq": 4,
        "td_error_loss_fn": "huber",
        "n_step": 5,
        "adam_epsilon": 1e-3,
        "grad_clip": 4.0,
        "grad_clip_by": "global_norm",
        "tau": 0.005,
        "gamma": 0.99,
        "training_intensity": 1.0,
        "num_eval_env_runners": 16,
        "eval_episodes": 100,
    }

    config = dqn_config(params=params)
    algo = config.build_algo()
    epsilon_decay = EpsilonDecay.from_params(params=params)
    checkpoint_path = Path(f"./checkpoints/{experiment_name}").resolve()

    save_parameters(checkpoint_path, params)

    try:
        epsilon = epsilon_decay.get(0)
        set_epsilon(epsilon=epsilon, algo=algo)
        warmup(algo=algo, iterations=params["warmup_iterations"])
        train(
            algo=algo,
            epsilon_decay=epsilon_decay,
            iterations=params["iterations"],
        )

    except KeyboardInterrupt:
        print("Exiting...")

    except BaseException as e:
        print(f"Exception: {e}")

    finally:
        print(f"Saving model to {checkpoint_path}")
        algo.save(checkpoint_path)


def warmup(algo, iterations):
    pbar = tqdm(range(iterations))
    for i in pbar:
        results = algo.train()

        eval_runners = results.get(EVALUATION_RESULTS, {}).get(ENV_RUNNER_RESULTS, {})
        agent_returns = eval_runners.get("agent_episode_returns_mean", {})
        mean = agent_returns.get("Player 1", 0.0)
        pbar.set_description(f"Warmup | wins: {mean:.3f}")


def train(algo, epsilon_decay, iterations):
    pbar = tqdm(range(iterations))
    for i in pbar:
        epsilon = epsilon_decay.get(i)
        set_epsilon(epsilon=epsilon, algo=algo)

        results = algo.train()

        eval_runners = results.get(EVALUATION_RESULTS, {}).get(ENV_RUNNER_RESULTS, {})
        agent_returns = eval_runners.get("agent_episode_returns_mean", {})
        mean = agent_returns.get("Player 1", 0.0)
        pbar.set_description(f"Training | Epsilon: {epsilon:.3f} Wins: {mean:.3f}")


if __name__ == "__main__":
    main()

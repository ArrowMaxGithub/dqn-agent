import os

from pathlib import Path

import torch
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import dqn_config, save_parameters, set_epsilon
from epsilon_decay import EpsilonDecay

from datetime import datetime


def main():
    print(f"GPU supported: {torch.cuda.is_available()}")

    experiment_name = f"{datetime.now()}_trump_fish"
    params = {
        "learning_rate": 1e-4,
        "iterations": 1024,
        "epsilon_schedule": "linear",
        "epsilon_decay": 0.67,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "num_env_runners": 16,
        "num_envs_per_env_runner": 8,
        "replay_buffer_capacity": 65536 * 64,
        "double_q": True,
        "train_batch_size": 2048,
        "num_steps_sampled_before_learning_starts": 65536 * 16,
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

    checkpoint_path = Path(f"./checkpoints/dqn/{experiment_name}").resolve()
    os.makedirs(checkpoint_path, exist_ok=True)

    log_dir = Path(f"./ray_results/dqn/{experiment_name}").resolve()
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    save_parameters(checkpoint_path, params)

    try:
        epsilon = epsilon_decay.get(0)
        set_epsilon(epsilon=epsilon, algo=algo)
        warmup(algo=algo, iterations=params["warmup_iterations"])
        train(
            w=writer,
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
        algo.stop()
        algo.save(checkpoint_path)


def warmup(algo, iterations):
    pbar = tqdm(range(iterations))
    for i in pbar:
        results = algo.train()

        eval_runners = results.get(EVALUATION_RESULTS, {}).get(ENV_RUNNER_RESULTS, {})
        agent_returns = eval_runners.get("agent_episode_returns_mean", {})
        mean = agent_returns.get("Player 1", 0.0)
        pbar.set_description(f"Warmup | wins: {mean:.3f}")


def train(w, algo, epsilon_decay, iterations):
    pbar = tqdm(range(iterations))
    for i in pbar:
        epsilon = epsilon_decay.get(i)
        set_epsilon(epsilon=epsilon, algo=algo)

        result = algo.train()

        eval_runners = result.get(EVALUATION_RESULTS, {}).get(ENV_RUNNER_RESULTS, {})
        agent_returns = eval_runners.get("agent_episode_returns_mean", {})
        agent_returns_mean = agent_returns.get("Player 1", 0.0)
        w.add_scalar("eval/agent_episode_returns_mean", agent_returns_mean, i)

        em = result.get("env_runners", {})
        w.add_scalar("env/episode_len_mean", em.get("episode_len_mean", 0), i)

        lm = result.get("learners", {}).get("p0", {})
        w.add_scalar("learner/total_loss", lm.get("total_loss", 0), i)
        w.add_scalar("learner/td_error", lm.get("td_error_mean", 0), i)
        w.add_scalar("learner/qf_loss", lm.get("qf_loss", 0), i)
        w.add_scalar("learner/qf_max", lm.get("qf_max", 0), i)
        w.add_scalar("learner/qf_mean", lm.get("qf_mean", 0), i)
        w.add_scalar("learner/qf_min", lm.get("qf_min", 0), i)

        pbar.set_description(
            f"Training | Epsilon: {epsilon:.3f} Wins: {agent_returns_mean:.3f}"
        )


if __name__ == "__main__":
    main()

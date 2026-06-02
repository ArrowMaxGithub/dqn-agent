import os
import math

from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import dqn_config, save_parameters, set_epsilon
from epsilon_decay import EpsilonDecay

from datetime import datetime

from trump_fish_agent import TrumpFishRLModule
from random_agent import RandomMaskedRLModule
from interpolation_agent import InterpolationRLModule


def main():
    print(f"GPU supported: {torch.cuda.is_available()}")

    experiment_name = f"selfplay_{datetime.now()}"
    params = {
        "learning_rate": 1e-4,
        "iterations": 8192,
        "epsilon_schedule": "linear",
        "epsilon_decay": 0.67,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.1,
        "num_env_runners": 16,
        "num_envs_per_env_runner": 8,
        "replay_buffer_capacity": 65536 * 32,
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
        "eval_episodes": 1000,
        "eval_interval": 8,
        "self_play_capacity": 64,
        "self_play_interval": 64,
        "self_play_recency_bias": 0.25,
        "self_play_confidence": 0.05,
    }

    opponents = {
        "random": RandomMaskedRLModule,
        "trump_fish": TrumpFishRLModule,
        "interpolation": InterpolationRLModule,
    }

    checkpoint_path = Path(f"./checkpoints/{experiment_name}").resolve()
    os.makedirs(checkpoint_path, exist_ok=True)

    algorithm = dqn_config(
        params=params, opponents=opponents, checkpoint_path=checkpoint_path
    )
    epsilon_decay = EpsilonDecay.from_params(params=params)

    log_dir = Path(f"./ray_results/{experiment_name}").resolve()
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Saving training parameters to {checkpoint_path}")
    save_parameters(checkpoint_path, params)

    try:
        epsilon = epsilon_decay.get(0)
        set_epsilon(epsilon=epsilon, algorithm=algorithm)
        warmup(algorithm=algorithm, iterations=params["warmup_iterations"])
        train(
            w=writer,
            algorithm=algorithm,
            epsilon_decay=epsilon_decay,
            iterations=params["iterations"],
        )

    except KeyboardInterrupt:
        print("Exiting...")

    except BaseException as e:
        print(f"Exception: {e}")

    finally:
        final_path = Path(f"{checkpoint_path}/final").resolve()
        print(f"Saving final version to {final_path}")
        algorithm.save(final_path)
        algorithm.stop()


def warmup(algorithm, iterations):
    pbar = tqdm(range(iterations))
    for i in pbar:
        algorithm.train()
        pbar.set_description("Warmup")


def train(w, algorithm, epsilon_decay, iterations):
    last_eval = None
    pbar = tqdm(range(iterations))
    for i in pbar:
        epsilon = epsilon_decay.get(i)
        set_epsilon(epsilon=epsilon, algorithm=algorithm)

        result = algorithm.train()

        eval_runners = result.get("evaluation", {}).get("env_runners", {})
        agent_returns = eval_runners.get("agent_episode_returns_mean", {})
        if "Player 1" in agent_returns:
            return_mean = agent_returns["Player 1"]
            if not math.isnan(return_mean):
                w.add_scalar("eval/agent_episode_returns_mean", return_mean, i)
                last_eval = return_mean

        em = result.get("env_runners", {})
        w.add_scalar("env/episode_len_mean", em.get("episode_len_mean", 0), i)

        lm = result.get("learners", {}).get("dqn", {})
        w.add_scalar("learner/total_loss", lm.get("total_loss", 0), i)
        w.add_scalar("learner/td_error", lm.get("td_error_mean", 0), i)
        w.add_scalar("learner/qf_loss", lm.get("qf_loss", 0), i)
        w.add_scalar("learner/qf_max", lm.get("qf_max", 0), i)
        w.add_scalar("learner/qf_mean", lm.get("qf_mean", 0), i)
        w.add_scalar("learner/qf_min", lm.get("qf_min", 0), i)

        if "opponent_pool_size" in result:
            w.add_scalar(
                "self_play/opponent_pool_size", result["opponent_pool_size"], i
            )

        if last_eval is None:
            pbar.set_description(f"Training | Epsilon: {epsilon:.3f} Last eval: None")
        else:
            pbar.set_description(
                f"Training | Epsilon: {epsilon:.3f} Last eval: {last_eval:.3f}"
            )


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


class OpponentPool:
    def __init__(self, capacity: int, recency_bias: float, opponents: list[str]):
        self.buffer = [module_id for module_id in opponents]
        self.baseline_buffer = self.buffer.copy()
        self.capacity = capacity
        self.recency_bias = recency_bias
        print(f"Initialized opponent pool with {len(self.buffer)} opponents")

    def opponents(self) -> list[str]:
        return list(self.buffer)

    def num_opponents(self) -> int:
        return len(self.buffer)

    def add(self, module_id: str):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)

        self.buffer.append(module_id)

    def sample(self):
        old_pool = (1.0 - self.recency_bias) / (len(self.buffer) - 1)
        recent_pool = self.recency_bias

        p = [old_pool for a in self.buffer]
        p[-1] = recent_pool

        return np.random.choice(a=self.buffer, p=p)

    def baseline_sample(self):
        return np.random.choice(a=self.baseline_buffer)


def policy_mapping_fn_creator(opponent_pool: OpponentPool, only_baseline: bool):
    def policy_mapping_fn(aid, *args, **kwargs):
        assert aid in ("Player 1", "Player 2"), f"Unexpected agent ID: {aid}"
        if aid == "Player 1":
            return "dqn"
        elif only_baseline:
            return opponent_pool.baseline_sample()
        else:
            return opponent_pool.sample()

    return policy_mapping_fn


class SelfPlayCallback(RLlibCallback):
    def __init__(
        self,
        interval: int,
        warmup_iterations: int,
        self_play_confidence: float,
        checkpoint_path: str,
        opponent_pool: OpponentPool,
    ):
        super().__init__()

        self.next_version = 1
        self.save_interval = interval
        self.warmup_iterations = warmup_iterations
        self.self_play_confidence = self_play_confidence
        self.checkpoint_path = checkpoint_path
        self.opponent_pool = opponent_pool
        self.last_win_rate = 0.0

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        result,
        metrics_logger,
        **kwargs,
    ):
        iteration = result["training_iteration"]
        if iteration <= self.warmup_iterations or iteration % self.save_interval != 0:
            return

        if not self._evaluate_learner(result):
            return

        self._save_version(algorithm)
        self._add_version(algorithm)

        result["opponent_pool_size"] = self.opponent_pool.num_opponents()

    def _evaluate_learner(self, result) -> bool:
        eval_runners = result.get("evaluation", {}).get("env_runners", {})
        agent_returns = eval_runners.get("agent_episode_returns_mean", {})
        version_eval = agent_returns.get("Player 1", 0.0)

        win_rate = 0.5 * (version_eval + 1)

        diff = win_rate - self.last_win_rate
        if self.last_win_rate == 0.0:
            significant = diff >= self.self_play_confidence
        else:
            quot = win_rate / self.last_win_rate
            significant = quot >= (1.0 + self.self_play_confidence)

        if significant:
            print(
                f"Improvement {(self.last_win_rate * 100.0):.1f}% -> {(win_rate * 100.0):.1f}%"
            )
            self.last_win_rate = win_rate
            return True
        else:
            print(
                f"No improvement: {(self.last_win_rate * 100.0):.1f}% -> {(win_rate * 100.0):.1f}%"
            )
            return False

    def _save_version(self, algorithm):
        version_path = Path(
            f"{self.checkpoint_path}/version_{self.next_version}"
        ).resolve()
        print(f"Saving version {self.next_version} to {version_path}")

        algorithm.save(version_path)

    def _add_version(self, algorithm):
        print(f"Adding new version {self.next_version}")
        # Increment version and add to opponent pool
        module_id = f"dqn_{self.next_version}"
        self.next_version += 1
        self.opponent_pool.add(module_id=module_id)

        # Generate new polcy_mapping_fn to capture changes to opponent_pool
        train_policy_mapping_fn = policy_mapping_fn_creator(self.opponent_pool, False)
        eval_policy_mapping_fn = policy_mapping_fn_creator(self.opponent_pool, True)

        # Add new version to algorithm
        local_worker = algorithm.env_runner
        multi_rl_module = local_worker.module
        module = multi_rl_module["dqn"]
        algorithm.add_module(
            module_id=module_id,
            module_spec=RLModuleSpec.from_module(module),
        )

        # Copy weigths from current version
        algorithm.set_state(
            {
                "learner_group": {
                    "learner": {
                        "rl_module": {
                            module_id: multi_rl_module[module_id].get_state(),
                        }
                    }
                }
            }
        )

        # Propagate changes to env_runners
        algorithm.env_runner_group.foreach_env_runner(
            lambda env_runner: env_runner.config.multi_agent(
                policy_mapping_fn=train_policy_mapping_fn,
            ),
            local_env_runner=True,
        )
        algorithm.env_runner_group.sync_weights(policies=[module_id])

        # Propagate changes to eval_runners
        algorithm.eval_env_runner_group.foreach_env_runner(
            lambda env_runner: env_runner.config.multi_agent(
                policy_mapping_fn=eval_policy_mapping_fn,
            ),
            local_env_runner=True,
        )
        algorithm.eval_env_runner_group.sync_weights(policies=[module_id])

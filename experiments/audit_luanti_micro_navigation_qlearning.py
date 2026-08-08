"""Public-vision tabular Q-learning lower bound for Luanti micro-navigation.

The feature extractor only measures the relative positions of the rendered
agent and goal tiles in the current 11x11 observation.  This deliberately
tests the reward/action learning loop after the raw-tensor MLP PPO failure; it
does not validate visual representation learning or any PLWM mechanism.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.luanti_reward_only_env import DEFAULT_BRIDGE_ROOT, GymVisionOnlyLuanti, VISION_SHAPE


State = tuple[int, int]
ACTION_COUNT = 4
GOAL_NOT_VISIBLE: State = (99, 99)


@dataclass(frozen=True)
class Evaluation:
    episodes: int
    successes: int
    mean_return: float
    mean_steps: float
    action_counts: list[int]

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes


@dataclass(frozen=True)
class Report:
    protocol: str
    executor: str
    training_episodes: int
    max_steps: int
    training_seed_range: list[int]
    heldout_seed_range: list[int]
    random_evaluation: Evaluation
    q_learning_evaluation: Evaluation
    learned_states: int
    accepted_public_feature_baseline: bool
    artifact: str


def public_goal_delta(observation: np.ndarray) -> State:
    """Return only the relative goal location encoded in the public frame."""
    if not isinstance(observation, np.ndarray) or observation.shape != VISION_SHAPE or observation.dtype != np.uint8:
        raise ValueError("expected the public Luanti vision tensor")
    agents = np.argwhere(observation[:, :, 2] == 1)
    goals = np.argwhere(observation[:, :, 3] == 1)
    if agents.shape != (1, 2):
        raise ValueError("public micro-navigation frame must contain one agent")
    if goals.shape == (0, 2):
        return GOAL_NOT_VISIBLE
    if goals.shape != (1, 2):
        raise ValueError("public micro-navigation frame must contain at most one goal")
    return int(goals[0, 0] - agents[0, 0]), int(goals[0, 1] - agents[0, 1])


def _evaluate_random(*, episodes: int, seed_start: int, max_steps: int, bridge_root: Path) -> Evaluation:
    environment = GymVisionOnlyLuanti(seed=seed_start, max_steps=max_steps, bridge_root=bridge_root)
    generator = random.Random(972_337)
    returns, steps, successes, action_counts = [], [], 0, [0] * ACTION_COUNT
    try:
        for episode in range(episodes):
            _observation, info = environment.reset(seed=seed_start + episode)
            if info != {}:
                raise RuntimeError("random baseline received non-public info")
            total = 0.0
            for step in range(max_steps):
                action = generator.randrange(ACTION_COUNT)
                _observation, reward, terminated, truncated, info = environment.step(action)
                if info != {}:
                    raise RuntimeError("random baseline received non-public info")
                action_counts[action] += 1
                total += reward
                if terminated or truncated:
                    successes += int(terminated and reward > 0.0)
                    steps.append(step + 1)
                    break
            else:
                steps.append(max_steps)
            returns.append(total)
    finally:
        environment.close()
    return Evaluation(episodes, successes, float(np.mean(returns)), float(np.mean(steps)), action_counts)


def _greedy_action(values: np.ndarray) -> int:
    """Stable tie breaking prevents evaluation from using random exploration."""
    return int(np.flatnonzero(values == values.max())[0])


def _evaluate_q(
    table: dict[State, np.ndarray], *, episodes: int, seed_start: int, max_steps: int, bridge_root: Path,
) -> Evaluation:
    environment = GymVisionOnlyLuanti(seed=seed_start, max_steps=max_steps, bridge_root=bridge_root)
    returns, steps, successes, action_counts = [], [], 0, [0] * ACTION_COUNT
    try:
        for episode in range(episodes):
            observation, info = environment.reset(seed=seed_start + episode)
            if info != {}:
                raise RuntimeError("Q evaluation received non-public info")
            total = 0.0
            for step in range(max_steps):
                state = public_goal_delta(observation)
                action = _greedy_action(table.get(state, np.zeros(ACTION_COUNT, dtype=np.float32)))
                observation, reward, terminated, truncated, info = environment.step(action)
                if info != {}:
                    raise RuntimeError("Q evaluation received non-public info")
                action_counts[action] += 1
                total += reward
                if terminated or truncated:
                    successes += int(terminated and reward > 0.0)
                    steps.append(step + 1)
                    break
            else:
                steps.append(max_steps)
            returns.append(total)
    finally:
        environment.close()
    return Evaluation(episodes, successes, float(np.mean(returns)), float(np.mean(steps)), action_counts)


def run(
    *,
    training_episodes: int = 128,
    max_steps: int = 10,
    evaluation_episodes: int = 16,
    training_seed_start: int = 6_300_000,
    heldout_seed_start: int = 6_400_000,
    action_seed: int = 932_337,
    bridge_root: Path = DEFAULT_BRIDGE_ROOT,
    artifact: Path | None = None,
) -> Report:
    if (
        training_episodes < 32 or max_steps < 4 or evaluation_episodes < 8
        or not 6_000_000 <= training_seed_start < 7_000_000
        or not 6_000_000 <= heldout_seed_start < 7_000_000
        or set(range(training_seed_start, training_seed_start + training_episodes)).intersection(
            range(heldout_seed_start, heldout_seed_start + evaluation_episodes)
        )
    ):
        raise ValueError("invalid public-feature Q-learning contract")
    generator = random.Random(action_seed)
    table: dict[State, np.ndarray] = {}
    environment = GymVisionOnlyLuanti(seed=training_seed_start, max_steps=max_steps, bridge_root=bridge_root)
    try:
        for episode in range(training_episodes):
            observation, info = environment.reset(seed=training_seed_start + episode)
            if info != {}:
                raise RuntimeError("Q training received non-public info")
            epsilon = 0.35 - 0.30 * episode / max(1, training_episodes - 1)
            for _step in range(max_steps):
                state = public_goal_delta(observation)
                values = table.setdefault(state, np.zeros(ACTION_COUNT, dtype=np.float32))
                action = generator.randrange(ACTION_COUNT) if generator.random() < epsilon else _greedy_action(values)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                if info != {}:
                    raise RuntimeError("Q training received non-public info")
                if terminated or truncated:
                    target = float(reward)
                else:
                    next_values = table.setdefault(public_goal_delta(next_observation), np.zeros(ACTION_COUNT, dtype=np.float32))
                    target = float(reward) + 0.95 * float(next_values.max())
                values[action] += 0.35 * (target - float(values[action]))
                observation = next_observation
                if terminated or truncated:
                    break
    finally:
        environment.close()
    random_evaluation = _evaluate_random(
        episodes=evaluation_episodes, seed_start=heldout_seed_start,
        max_steps=max_steps, bridge_root=bridge_root,
    )
    q_evaluation = _evaluate_q(
        table, episodes=evaluation_episodes, seed_start=heldout_seed_start,
        max_steps=max_steps, bridge_root=bridge_root,
    )
    target = artifact or ROOT / "artifacts" / "luanti-micro-navigation-qlearning-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "luanti_micro_navigation_public_feature_qlearning_v1: real Luanti visible-goal micro-navigation. A tabular Q learner receives only the current public 11x11 vision-derived agent-to-goal row/column delta (or a fixed public goal-not-visible state), selected action, scalar environment reward, terminal/truncation, and empty Gym info. It receives no bridge state, task seed, teacher trajectory, external planner, shaped reward, or PLWM state. This validates the reward-action course lower bound only, not raw visual representation or PLWM.",
        "local-cpu", training_episodes, max_steps,
        list(range(training_seed_start, training_seed_start + training_episodes)),
        list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)),
        random_evaluation, q_evaluation, len(table),
        q_evaluation.success_rate >= 0.75 and q_evaluation.success_rate >= random_evaluation.success_rate + 0.25,
        str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-episodes", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=16)
    parser.add_argument("--training-seed-start", type=int, default=6_300_000)
    parser.add_argument("--heldout-seed-start", type=int, default=6_400_000)
    parser.add_argument("--action-seed", type=int, default=932_337)
    parser.add_argument("--bridge-root", type=Path, default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument("--artifact", type=Path, default=None)
    arguments = parser.parse_args()
    print(json.dumps(asdict(run(
        training_episodes=arguments.training_episodes, max_steps=arguments.max_steps,
        evaluation_episodes=arguments.evaluation_episodes,
        training_seed_start=arguments.training_seed_start,
        heldout_seed_start=arguments.heldout_seed_start,
        action_seed=arguments.action_seed, bridge_root=arguments.bridge_root,
        artifact=arguments.artifact,
    )), indent=2))

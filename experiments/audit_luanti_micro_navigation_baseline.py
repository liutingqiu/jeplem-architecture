"""Validate a conventional PPO lower bound on visible-goal Luanti micro-navigation.

This establishes that the new reward-only curriculum is learnable before any
PLWM mechanism is evaluated on it.  It is explicitly not an architecture
comparison and uses only the same public Gym vision/action/reward boundary.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from plwm.luanti_reward_only_env import DEFAULT_BRIDGE_ROOT, GymVisionOnlyLuanti


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
    device: str
    total_timesteps: int
    max_steps: int
    training_seed_start: int
    heldout_seed_range: list[int]
    random_evaluation: Evaluation
    ppo_evaluation: Evaluation
    accepted_standard_baseline: bool
    artifact: str


def _evaluate_random(*, episodes: int, seed_start: int, max_steps: int, bridge_root: Path) -> Evaluation:
    environment = GymVisionOnlyLuanti(seed=seed_start, max_steps=max_steps, bridge_root=bridge_root)
    generator = random.Random(962_337)
    returns, steps, successes = [], [], 0
    actions = [0, 0, 0, 0]
    try:
        for episode in range(episodes):
            _observation, info = environment.reset(seed=seed_start + episode)
            if info != {}:
                raise RuntimeError("micro-navigation random baseline received non-public info")
            total = 0.0
            for step in range(max_steps):
                action = generator.randrange(4)
                _observation, reward, terminated, truncated, info = environment.step(action)
                if info != {}:
                    raise RuntimeError("micro-navigation random baseline received non-public info")
                actions[action] += 1
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
    return Evaluation(episodes, successes, float(np.mean(returns)), float(np.mean(steps)), actions)


def _evaluate_ppo(model: object, *, episodes: int, seed_start: int, max_steps: int, bridge_root: Path) -> Evaluation:
    environment = GymVisionOnlyLuanti(seed=seed_start, max_steps=max_steps, bridge_root=bridge_root)
    returns, steps, successes = [], [], 0
    actions = [0, 0, 0, 0]
    try:
        for episode in range(episodes):
            observation, info = environment.reset(seed=seed_start + episode)
            if info != {}:
                raise RuntimeError("micro-navigation PPO evaluation received non-public info")
            total = 0.0
            for step in range(max_steps):
                action, _state = model.predict(observation, deterministic=True)  # type: ignore[attr-defined]
                selected = int(action)
                if not 0 <= selected < 4:
                    raise RuntimeError("PPO emitted an invalid public action")
                observation, reward, terminated, truncated, info = environment.step(selected)
                if info != {}:
                    raise RuntimeError("micro-navigation PPO evaluation received non-public info")
                actions[selected] += 1
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
    return Evaluation(episodes, successes, float(np.mean(returns)), float(np.mean(steps)), actions)


def run(
    *,
    total_timesteps: int = 1_024,
    max_steps: int = 10,
    evaluation_episodes: int = 16,
    training_seed_start: int = 6_100_000,
    heldout_seed_start: int = 6_200_000,
    seed: int = 912_337,
    bridge_root: Path = DEFAULT_BRIDGE_ROOT,
    device: str | torch.device | None = None,
    artifact: Path | None = None,
) -> Report:
    if (
        total_timesteps < 512 or max_steps < 4 or evaluation_episodes < 8
        or not 6_000_000 <= training_seed_start < 7_000_000
        or not 6_000_000 <= heldout_seed_start < 7_000_000
        or training_seed_start == heldout_seed_start
    ):
        raise ValueError("invalid micro-navigation standard-baseline contract")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    resolved = resolve_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    heldout = list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes))
    random_evaluation = _evaluate_random(
        episodes=evaluation_episodes, seed_start=heldout_seed_start,
        max_steps=max_steps, bridge_root=bridge_root,
    )
    vector = DummyVecEnv([lambda: GymVisionOnlyLuanti(
        seed=training_seed_start, max_steps=max_steps, bridge_root=bridge_root,
    )])
    try:
        model = PPO(
            "MlpPolicy", vector, device=str(resolved), seed=seed,
            n_steps=64, batch_size=64, n_epochs=4, learning_rate=2.5e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        ppo_evaluation = _evaluate_ppo(
            model, episodes=evaluation_episodes, seed_start=heldout_seed_start,
            max_steps=max_steps, bridge_root=bridge_root,
        )
    finally:
        vector.close()
    target = artifact or ROOT / "artifacts" / "luanti-micro-navigation-sb3-baseline-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "luanti_micro_navigation_sb3_baseline_v1: real Luanti public visible-goal micro-navigation; only 11x11 local vision, selected four-direction action, scalar environment reward, termination/truncation, and empty Gym info are available. Goal is visible, controls are not inverted, and reaching it produces the sole terminal environment reward. Standard PPO validation only, not a PLWM architecture experiment.",
        str(resolved), total_timesteps, max_steps, training_seed_start, heldout,
        random_evaluation, ppo_evaluation,
        ppo_evaluation.success_rate >= 0.75 and ppo_evaluation.success_rate >= random_evaluation.success_rate + 0.25,
        str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_024)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=16)
    parser.add_argument("--training-seed-start", type=int, default=6_100_000)
    parser.add_argument("--heldout-seed-start", type=int, default=6_200_000)
    parser.add_argument("--seed", type=int, default=912_337)
    parser.add_argument("--bridge-root", type=Path, default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    arguments = parser.parse_args()
    print(json.dumps(asdict(run(
        total_timesteps=arguments.total_timesteps, max_steps=arguments.max_steps,
        evaluation_episodes=arguments.evaluation_episodes,
        training_seed_start=arguments.training_seed_start,
        heldout_seed_start=arguments.heldout_seed_start, seed=arguments.seed,
        bridge_root=arguments.bridge_root, device=arguments.device,
        artifact=arguments.artifact,
    )), indent=2))

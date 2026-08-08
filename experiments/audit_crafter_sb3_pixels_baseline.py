"""Standard PPO baseline on the strictly pixels-only Crafter boundary.

This is deliberately not a PLWM experiment. Stable-Baselines3 is used to
validate a conventional public interaction baseline after the small custom PPO
sentinel collapsed. The Gym wrapper returns an empty info dictionary, so SB3
cannot access semantic maps, inventory, achievements, player position, or
reward decomposition.
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

from plwm.crafter_env import GymPixelsOnlyCrafter
from plwm.device import resolve_device


@dataclass
class Evaluation:
    episodes: int
    mean_return: float
    mean_steps: float


@dataclass
class Report:
    protocol: str
    device: str
    total_timesteps: int
    episode_length: int
    training_seed: int
    training_world_seed: int
    heldout_seed_range: list[int]
    random_evaluation: Evaluation
    ppo_evaluation: Evaluation
    accepted_standard_baseline: bool
    artifact: str


def _evaluate_random(*, episodes: int, seed_start: int, length: int) -> Evaluation:
    rng = random.Random(612_337)
    returns, steps = [], []
    environment = GymPixelsOnlyCrafter(seed=seed_start, length=length)
    try:
        for episode in range(episodes):
            observation, _info = environment.reset(seed=seed_start + episode)
            total = 0.0
            for step in range(length):
                observation, reward, terminated, truncated, _info = environment.step(rng.randrange(17))
                total += reward
                if terminated or truncated:
                    steps.append(step + 1)
                    break
            else:
                steps.append(length)
            returns.append(total)
    finally:
        environment.close()
    return Evaluation(episodes, float(np.mean(returns)), float(np.mean(steps)))


def _evaluate_ppo(model: object, *, episodes: int, seed_start: int, length: int) -> Evaluation:
    returns, steps = [], []
    environment = GymPixelsOnlyCrafter(seed=seed_start, length=length)
    try:
        for episode in range(episodes):
            observation, _info = environment.reset(seed=seed_start + episode)
            total = 0.0
            for step in range(length):
                action, _state = model.predict(observation, deterministic=True)  # type: ignore[attr-defined]
                observation, reward, terminated, truncated, _info = environment.step(int(action))
                total += reward
                if terminated or truncated:
                    steps.append(step + 1)
                    break
            else:
                steps.append(length)
            returns.append(total)
    finally:
        environment.close()
    return Evaluation(episodes, float(np.mean(returns)), float(np.mean(steps)))


def run(
    *, device: str | torch.device | None = None, total_timesteps: int = 16_384,
    episode_length: int = 64, evaluation_episodes: int = 16,
    seed: int = 611_337, training_world_seed: int = 100_000,
    heldout_seed_start: int = 20_000_000,
    artifact: Path | None = None,
) -> Report:
    if total_timesteps < 512 or episode_length < 8 or evaluation_episodes < 2 or training_world_seed < 0 or heldout_seed_start < 0:
        raise ValueError("invalid SB3 Crafter baseline budget")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    resolved = resolve_device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    heldout = heldout_seed_start
    random_evaluation = _evaluate_random(episodes=evaluation_episodes, seed_start=heldout, length=episode_length)
    vector = VecTransposeImage(DummyVecEnv([lambda: GymPixelsOnlyCrafter(seed=training_world_seed, length=episode_length)]))
    try:
        model = PPO(
            "CnnPolicy", vector, device=str(resolved), seed=seed,
            n_steps=512, batch_size=64, n_epochs=4, learning_rate=2.5e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        ppo_evaluation = _evaluate_ppo(model, episodes=evaluation_episodes, seed_start=heldout, length=episode_length)
    finally:
        vector.close()
    target = artifact or ROOT / "artifacts" / "crafter-sb3-pixels-baseline-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "crafter_sb3_pixels_baseline_v1: Stable-Baselines3 PPO through GymPixelsOnlyCrafter. Observation is 64x64 uint8 RGB; action is 0..16; reward is scalar environment reward; Gym info is always {}. Semantic, inventory, achievements, player position, hidden state, teacher actions, and future observations are unavailable.",
        str(resolved), total_timesteps, episode_length, seed, training_world_seed,
        list(range(heldout, heldout + evaluation_episodes)),
        random_evaluation, ppo_evaluation,
        ppo_evaluation.mean_return >= random_evaluation.mean_return + 0.25,
        str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--total-timesteps", type=int, default=16_384)
    parser.add_argument("--episode-length", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=611_337)
    parser.add_argument("--training-world-seed", type=int, default=100_000)
    parser.add_argument("--heldout-seed-start", type=int, default=20_000_000)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, total_timesteps=args.total_timesteps,
        episode_length=args.episode_length, evaluation_episodes=args.evaluation_episodes,
        seed=args.seed, training_world_seed=args.training_world_seed,
        heldout_seed_start=args.heldout_seed_start,
        artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted_standard_baseline:
        raise SystemExit("standard Crafter PPO baseline did not exceed random reference")

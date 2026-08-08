"""Collect a fresh, leakage-restricted Luanti dynamics corpus for pretraining.

This is not a controller experiment.  It records only the public causal
sequence available to a future learner: current local vision, its sampled
action, scalar environment reward, termination, and resulting local vision.
The symbolic bridge state and Gym ``info`` never enter the corpus.
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


ACTION_COUNT = 4


@dataclass(frozen=True)
class CorpusReport:
    protocol: str
    episodes: int
    max_steps: int
    seed_range: list[int]
    transitions: int
    successes: int
    artifact: str
    metadata: str


def collect(
    environment: GymVisionOnlyLuanti,
    *,
    episodes: int,
    max_steps: int,
    seed_start: int,
    action_seed: int,
) -> dict[str, np.ndarray]:
    """Collect random public transitions without inspecting any environment state."""
    if episodes < 1 or max_steps < 1 or seed_start < 0:
        raise ValueError("invalid public dynamics collection parameters")
    generator = random.Random(action_seed)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    episode_indices: list[int] = []
    step_indices: list[int] = []
    for episode in range(episodes):
        observation, info = environment.reset(seed=seed_start + episode)
        if info != {} or observation.shape != VISION_SHAPE or observation.dtype != np.uint8:
            raise RuntimeError("public Luanti collector received an invalid Gym boundary")
        for step in range(max_steps):
            action = generator.randrange(ACTION_COUNT)
            next_observation, reward, done, cutoff, info = environment.step(action)
            if (
                info != {}
                or next_observation.shape != VISION_SHAPE
                or next_observation.dtype != np.uint8
                or isinstance(reward, bool)
                or not isinstance(reward, (int, float))
            ):
                raise RuntimeError("public Luanti collector received an invalid transition")
            observations.append(observation.copy())
            actions.append(action)
            rewards.append(float(reward))
            next_observations.append(next_observation.copy())
            terminated.append(bool(done))
            truncated.append(bool(cutoff))
            episode_indices.append(episode)
            step_indices.append(step)
            observation = next_observation
            if done or cutoff:
                break
    return {
        "observations": np.asarray(observations, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.uint8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.uint8),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "episode_indices": np.asarray(episode_indices, dtype=np.int32),
        "step_indices": np.asarray(step_indices, dtype=np.int16),
    }


def run(
    *,
    episodes: int = 16,
    max_steps: int = 40,
    seed_start: int = 4_700_000,
    action_seed: int = 812_337,
    bridge_root: Path = DEFAULT_BRIDGE_ROOT,
    artifact: Path | None = None,
    metadata: Path | None = None,
) -> CorpusReport:
    if episodes < 2 or max_steps < 1 or not 4_000_000 <= seed_start < 5_000_000:
        raise ValueError("invalid fresh Luanti pretraining corpus contract")
    target = artifact or ROOT / "artifacts" / "luanti-public-dynamics-pretrain-corpus.npz"
    metadata_target = metadata or target.with_suffix(".json")
    environment = GymVisionOnlyLuanti(seed=seed_start, max_steps=max_steps, bridge_root=bridge_root)
    try:
        corpus = collect(
            environment, episodes=episodes, max_steps=max_steps,
            seed_start=seed_start, action_seed=action_seed,
        )
    finally:
        environment.close()
    if corpus["observations"].shape != corpus["next_observations"].shape or corpus["observations"].ndim != 4:
        raise RuntimeError("public dynamics corpus has invalid vision tensors")
    target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target, **corpus)
    report = CorpusReport(
        "luanti_public_dynamics_pretrain_v1: fresh random exploration in real Luanti; corpus contains only 11x11 local vision, sampled action, scalar environment reward, terminal/truncation, and next local vision. It excludes symbolic bridge state, task seed from learner arrays, teacher trajectories, external planner actions, shaped reward, and Gym info.",
        episodes, max_steps, list(range(seed_start, seed_start + episodes)),
        int(corpus["actions"].shape[0]), int(corpus["terminated"].sum()),
        str(target), str(metadata_target),
    )
    metadata_target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=4_700_000)
    parser.add_argument("--action-seed", type=int, default=812_337)
    parser.add_argument("--bridge-root", type=Path, default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    arguments = parser.parse_args()
    print(json.dumps(asdict(run(
        episodes=arguments.episodes, max_steps=arguments.max_steps,
        seed_start=arguments.seed_start, action_seed=arguments.action_seed,
        bridge_root=arguments.bridge_root, artifact=arguments.artifact,
        metadata=arguments.metadata,
    )), indent=2))

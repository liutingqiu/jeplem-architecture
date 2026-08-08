"""Verify the public Crafter pixels-only interaction contract."""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.crafter_env import PixelsOnlyCrafter


@dataclass
class Report:
    protocol: str
    crafter_version: str
    episodes: int
    episode_length: int
    action_count: int
    observation_shape: list[int]
    random_mean_reward: float
    all_rgb_only: bool
    accepted: bool
    artifact: str


def run(*, episodes: int = 8, episode_length: int = 128, artifact: Path | None = None) -> Report:
    if episodes < 1 or episode_length < 1:
        raise ValueError("episodes and episode_length must be positive")
    import importlib.metadata

    rewards, shapes, action_counts = [], [], []
    rng = random.Random(101_337)
    for index in range(episodes):
        environment = PixelsOnlyCrafter(seed=10_000_000 + index, length=episode_length)
        observation = environment.reset()
        total = 0.0
        for _step in range(episode_length):
            transition = environment.step(rng.randrange(environment.action_count))
            observation = transition.observation
            total += transition.reward
            if transition.done:
                break
        rewards.append(total); shapes.append(tuple(observation.shape)); action_counts.append(environment.action_count)
    target = artifact or ROOT / "artifacts" / "crafter-pixels-contract-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "crafter_pixels_contract_v1: public Crafter interaction exposes only 64x64 RGB sensor frames, selected action, scalar environment reward, and terminal status; semantic map, inventory, and achievements are discarded before agent code.",
        importlib.metadata.version("crafter"), episodes, episode_length, action_counts[0], list(shapes[0]),
        float(np.mean(rewards)), all(shape == (64, 64, 3) for shape in shapes) and len(set(action_counts)) == 1,
        all(shape == (64, 64, 3) for shape in shapes) and len(set(action_counts)) == 1 and action_counts[0] == 17,
        str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("Crafter pixels-only contract did not meet its gate")

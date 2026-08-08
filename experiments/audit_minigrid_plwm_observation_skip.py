"""Test whether a current-observation readout path fixes PLWM control collapse."""
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

try:
    from .train_minigrid_memory_actor_critic import Evaluation, PLWMActorCritic, TransformerActorCritic, _train, evaluate, evaluate_random
except ImportError:
    from train_minigrid_memory_actor_critic import Evaluation, PLWMActorCritic, TransformerActorCritic, _train, evaluate, evaluate_random


ENVIRONMENT = "MiniGrid-Empty-5x5-v0"


@dataclass
class Result:
    name: str
    parameters: int
    train_successes: int
    evaluation: Evaluation


@dataclass
class Report:
    protocol: str
    device: str
    updates: int
    episodes_per_update: int
    random: Evaluation
    results: list[Result]
    artifact: str


def run(*, device: str | torch.device | None = None, updates: int = 30, episodes_per_update: int = 24, evaluation_episodes: int = 64, artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(13_037); np.random.seed(13_037); random.seed(13_037)
    factories = {"plwm_full": PLWMActorCritic, "plwm_observation_skip": lambda: PLWMActorCritic(observation_skip=True), "transformer": TransformerActorCritic}
    results = []
    for name, factory in factories.items():
        model = factory().to(resolved)
        _episodes, successes = _train(model, updates=updates, episodes_per_update=episodes_per_update, start_seed=10_000, device=resolved, learning_rule="ppo_external_reward_v1", ppo_epochs=4, entropy_coefficient=0.02, action_mode="movement_only", environment=ENVIRONMENT)
        evaluation = evaluate(model, episodes=evaluation_episodes, start_seed=1_000_000, device=resolved, action_mode="movement_only", environment=ENVIRONMENT)
        results.append(Result(name, sum(p.numel() for p in model.parameters()), successes, evaluation))
    target = artifact or ROOT / "artifacts" / "minigrid-plwm-observation-skip-report.json"
    report = Report("minigrid_plwm_observation_skip_v1: same public navigation contract and budget; optional raw current-observation readout added only to the PLWM output path", str(resolved), updates, episodes_per_update, evaluate_random(episodes=evaluation_episodes, start_seed=1_000_000, action_mode="movement_only", environment=ENVIRONMENT), results, str(target))
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(asdict(run(device=args.device, updates=args.updates, episodes_per_update=args.episodes_per_update, evaluation_episodes=args.evaluation_episodes, artifact=args.artifact)), indent=2))

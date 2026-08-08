"""Check whether causal self-prediction improves reward-only PLWM control.

The auxiliary target is each already-observed event.  It is never available at
action time and is kept separate from external reward and architecture claims.
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

try:
    from .train_minigrid_memory_actor_critic import (
        Evaluation, PLWMActorCritic, TransformerActorCritic, _train,
        evaluate, evaluate_random,
    )
except ImportError:
    from train_minigrid_memory_actor_critic import (
        Evaluation, PLWMActorCritic, TransformerActorCritic, _train,
        evaluate, evaluate_random,
    )


ENVIRONMENT = "MiniGrid-Empty-5x5-v0"


@dataclass
class Result:
    name: str
    parameters: int
    predictive_coefficient: float
    training_successes: int
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


def run(
    *,
    device: str | torch.device | None = None,
    updates: int = 30,
    episodes_per_update: int = 24,
    evaluation_episodes: int = 64,
    predictive_coefficient: float = 1.0,
    artifact: Path | None = None,
) -> Report:
    if predictive_coefficient <= 0.0:
        raise ValueError("predictive_coefficient must be positive for this audit")
    resolved = resolve_device(device)
    torch.manual_seed(13_037); np.random.seed(13_037); random.seed(13_037)
    factories = (
        ("plwm_reward_only", PLWMActorCritic, 0.0),
        ("plwm_predictive_auxiliary", PLWMActorCritic, predictive_coefficient),
        ("transformer_reward_only", TransformerActorCritic, 0.0),
    )
    results: list[Result] = []
    for name, factory, coefficient in factories:
        model = factory().to(resolved)
        _episodes, successes = _train(
            model, updates=updates, episodes_per_update=episodes_per_update,
            start_seed=10_000, device=resolved,
            learning_rule="ppo_external_reward_v1", ppo_epochs=4,
            entropy_coefficient=0.02, action_mode="movement_only",
            environment=ENVIRONMENT, world_model_coefficient=coefficient,
        )
        results.append(Result(
            name=name,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            predictive_coefficient=coefficient,
            training_successes=successes,
            evaluation=evaluate(
                model, episodes=evaluation_episodes, start_seed=1_000_000,
                device=resolved, action_mode="movement_only", environment=ENVIRONMENT,
            ),
        ))
    target = artifact or ROOT / "artifacts" / "minigrid-plwm-predictive-auxiliary-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_plwm_predictive_auxiliary_v1: reward-only public navigation PPO; PLWM auxiliary target is the current already-observed event predicted from its preceding state, without teacher actions, oracle paths, future observations, seed answers, distance shaping, or inference-time targets",
        str(resolved), updates, episodes_per_update,
        evaluate_random(episodes=evaluation_episodes, start_seed=1_000_000, action_mode="movement_only", environment=ENVIRONMENT),
        results, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--predictive-coefficient", type=float, default=1.0)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(asdict(run(
        device=args.device, updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        predictive_coefficient=args.predictive_coefficient, artifact=args.artifact,
    )), indent=2))

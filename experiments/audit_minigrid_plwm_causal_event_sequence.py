"""Compare merged versus separately sequenced action/reward events in PLWM."""
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
OBSERVATION_MODE = "causal_action_reward"


@dataclass
class Result:
    name: str
    parameters: int
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
    artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(13_037); np.random.seed(13_037); random.seed(13_037)
    factories = (
        ("plwm_merged_action_observation", PLWMActorCritic),
        ("plwm_separate_action_event", lambda: PLWMActorCritic(separate_causal_event=True)),
        ("transformer_merged_action_observation", TransformerActorCritic),
    )
    results: list[Result] = []
    for name, factory in factories:
        model = factory().to(resolved)
        _episodes, successes = _train(
            model, updates=updates, episodes_per_update=episodes_per_update,
            start_seed=10_000, device=resolved, learning_rule="ppo_external_reward_v1",
            ppo_epochs=4, entropy_coefficient=0.02, action_mode="movement_only",
            environment=ENVIRONMENT, observation_mode=OBSERVATION_MODE,
        )
        results.append(Result(
            name=name,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            training_successes=successes,
            evaluation=evaluate(
                model, episodes=evaluation_episodes, start_seed=1_000_000,
                device=resolved, action_mode="movement_only", environment=ENVIRONMENT,
                observation_mode=OBSERVATION_MODE,
            ),
        ))
    target = artifact or ROOT / "artifacts" / "minigrid-plwm-causal-event-sequence-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_plwm_causal_event_sequence_v1: each model gets the same prior sampled action and observed reward; the PLWM sequence condition processes that event before its next sensory vector. No teacher, future observation, seed answer, oracle path, or shaped reward is present.",
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
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(asdict(run(
        device=args.device, updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes, artifact=args.artifact,
    )), indent=2))

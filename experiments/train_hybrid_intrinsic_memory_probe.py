"""Diagnostic: does explicit exploration (intrinsic count bonus) unlock genuine
object-identity memory on MemoryS7, measured by the counterfactual swap probe?

The plain held-out evaluation is known to be defeated by the turn-right-then-
forward route policy (~27/64 for every controller).  This run trains the
hybrid controller with the project's intrinsic-count learning rule and reports
the paired swap probe: only worlds solved in both the original and the swapped
layout count as memory-driven success.
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
        HybridPLWMTransformerActorCritic, _train, evaluate, evaluate_random,
    )
    from .train_jeplem_hybrid_curriculum import evaluate_memory_probe
except ImportError:
    from train_minigrid_memory_actor_critic import (
        HybridPLWMTransformerActorCritic, _train, evaluate, evaluate_random,
    )
    from train_jeplem_hybrid_curriculum import evaluate_memory_probe


BOOTSTRAP_ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
MEMORY_ENVIRONMENT = "MiniGrid-MemoryS7-v0"


@dataclass
class Report:
    protocol: str
    device: str
    learning_rule: str
    intrinsic_bonus: float
    experiment_seed: int
    evaluation: object
    memory_probe: object
    accepted: bool
    artifact: str


def run(
    *,
    device: str | torch.device | None = None,
    bootstrap_updates: int = 30,
    memory_updates: int = 60,
    episodes_per_update: int = 24,
    evaluation_episodes: int = 64,
    experiment_seed: int = 41_123,
    train_seed_start: int = 10_000,
    heldout_seed_start: int = 1_000_000,
    intrinsic_bonus: float = 0.05,
    artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    model = HybridPLWMTransformerActorCritic().to(resolved)
    seed_start = train_seed_start
    for environment, action_mode, updates in (
        (BOOTSTRAP_ENVIRONMENT, "movement_only", bootstrap_updates),
        (MEMORY_ENVIRONMENT, "all", memory_updates),
    ):
        _episodes, _successes = _train(
            model, updates=updates, episodes_per_update=episodes_per_update,
            start_seed=seed_start, device=resolved,
            learning_rule="ppo_intrinsic_count_v1", ppo_epochs=4,
            entropy_coefficient=0.02, intrinsic_bonus=intrinsic_bonus,
            action_mode=action_mode, environment=environment,
            observation_mode="observation_only",
        )
        seed_start += updates * episodes_per_update
    evaluation = evaluate(
        model, episodes=evaluation_episodes, start_seed=heldout_seed_start,
        device=resolved, action_mode="all", environment=MEMORY_ENVIRONMENT,
    )
    memory_probe = evaluate_memory_probe(
        model, episodes=evaluation_episodes, start_seed=heldout_seed_start,
        device=resolved, action_mode="all", environment=MEMORY_ENVIRONMENT,
    )
    accepted = memory_probe.both >= int(evaluation_episodes * 0.40)
    target = artifact or ROOT / "artifacts" / "hybrid-intrinsic-memory-probe-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "hybrid_intrinsic_memory_probe_v1: HybridPLWMTransformerActorCritic trained with the intrinsic-count learning rule (ppo_intrinsic_count_v1, bonus 0.05) through the two-stage curriculum, then evaluated on held-out MemoryS7 worlds with the paired counterfactual swap probe; only worlds solved in both the original and swapped layouts count as memory-driven success; no teacher action, path oracle, seed answer, or reward shaping.",
        str(resolved), "ppo_intrinsic_count_v1", intrinsic_bonus, experiment_seed,
        evaluation, memory_probe, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap-updates", type=int, default=30)
    parser.add_argument("--memory-updates", type=int, default=60)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--intrinsic-bonus", type=float, default=0.05)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, bootstrap_updates=args.bootstrap_updates,
        memory_updates=args.memory_updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, intrinsic_bonus=args.intrinsic_bonus,
        artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("intrinsic exploration did not unlock memory-driven success on MemoryS7")

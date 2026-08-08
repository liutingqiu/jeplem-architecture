"""Replicate the explicit intrinsic-exploration condition across unseen worlds."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .train_minigrid_memory_actor_critic import (
        GRUActorCritic, NoMemoryActorCritic, PLWMActorCritic,
        TransformerActorCritic, Evaluation, _train, evaluate, evaluate_random,
    )
except ImportError:  # Direct execution from the experiments directory.
    from train_minigrid_memory_actor_critic import (
        GRUActorCritic, NoMemoryActorCritic, PLWMActorCritic,
        TransformerActorCritic, Evaluation, _train, evaluate, evaluate_random,
    )
from plwm.device import resolve_device


ROOT = Path(__file__).resolve().parents[1]
HELDOUT_STARTS = (1_000_000, 2_000_000, 3_000_000)


@dataclass
class SplitResult:
    start_seed: int
    random: Evaluation
    plwm: Evaluation
    no_memory: Evaluation
    gru: Evaluation
    transformer: Evaluation


@dataclass
class AuditReport:
    protocol: str
    device: str
    updates: int
    episodes_per_update: int
    evaluation_episodes: int
    intrinsic_bonus: float
    splits: list[SplitResult]
    accepted: bool
    artifact: str


def run(
    *,
    device: str | torch.device | None = None,
    updates: int = 8,
    episodes_per_update: int = 24,
    evaluation_episodes: int = 32,
    intrinsic_bonus: float = 0.05,
    artifact: Path | None = None,
) -> AuditReport:
    if updates < 1 or episodes_per_update < 2 or evaluation_episodes < 1 or intrinsic_bonus < 0.0:
        raise ValueError("invalid audit budget")
    resolved = resolve_device(device)
    torch.manual_seed(13_037)
    random.seed(13_037)
    np.random.seed(13_037)
    models = {
        "plwm": PLWMActorCritic().to(resolved),
        "no_memory": NoMemoryActorCritic().to(resolved),
        "gru": GRUActorCritic().to(resolved),
        "transformer": TransformerActorCritic().to(resolved),
    }
    for model in models.values():
        _train(
            model,
            updates=updates,
            episodes_per_update=episodes_per_update,
            start_seed=10_000,
            device=resolved,
            learning_rule="ppo_intrinsic_count_v1",
            ppo_epochs=4,
            entropy_coefficient=0.02,
            intrinsic_bonus=intrinsic_bonus,
        )
    splits = []
    for start_seed in HELDOUT_STARTS:
        evaluations = {name: evaluate(model, episodes=evaluation_episodes, start_seed=start_seed, device=resolved) for name, model in models.items()}
        splits.append(SplitResult(start_seed, evaluate_random(episodes=evaluation_episodes, start_seed=start_seed), **evaluations))
    accepted = all(
        split.plwm.success_rate >= 0.55
        and split.plwm.success_rate >= split.no_memory.success_rate + 0.10
        and split.plwm.success_rate >= split.gru.success_rate + 0.05
        and split.plwm.success_rate >= split.transformer.success_rate + 0.05
        for split in splits
    )
    target = artifact or ROOT / "artifacts" / "minigrid-intrinsic-generalization-audit-report.json"
    report = AuditReport(
        "minigrid_intrinsic_generalization_v1: three disjoint held-out seed blocks; PPO with an explicit 0.05 local-observation visit-count bonus during training only; success and acceptance use environment reward only; no teacher, oracle, future observation, seed answer, or distance shaping",
        str(resolved), updates, episodes_per_update, evaluation_episodes, intrinsic_bonus, splits, accepted, str(target),
    )
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=32)
    parser.add_argument("--intrinsic-bonus", type=float, default=0.05)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device,
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        intrinsic_bonus=args.intrinsic_bonus,
        artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("intrinsic exploration audit did not meet the architecture gate")

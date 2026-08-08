"""Predeclared reward-only curriculum for a PLWM/Transformer hybrid controller.

This is a materially different control candidate, not another PLWM head
variation.  A tactical causal Transformer processes the visible event prefix;
an independently switchable PLWM side state contributes only a persistent
belief/workspace/memory feature.  Both stages use sampled actions and scalar
environment reward only.  The first stage is public short-range navigation;
the second introduces the delayed-memory task.  Final evaluation uses unseen
MemoryS7 worlds, so success in the first stage is not itself a result.
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
        HybridPLWMTransformerActorCritic, GRUActorCritic, NoMemoryActorCritic,
        TransformerActorCritic, _train, evaluate, evaluate_random,
    )
except ImportError:
    from train_minigrid_memory_actor_critic import (
        HybridPLWMTransformerActorCritic, GRUActorCritic, NoMemoryActorCritic,
        TransformerActorCritic, _train, evaluate, evaluate_random,
    )


BOOTSTRAP_ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
MEMORY_ENVIRONMENT = "MiniGrid-MemoryS7-v0"


@dataclass
class CurriculumStage:
    name: str
    environment: str
    action_mode: str
    updates: int
    episodes_per_update: int
    seed_start: int
    train_episodes: int
    train_successes: int


@dataclass
class ControllerResult:
    name: str
    parameters: int
    active_parameters: int
    active_plwm_side: bool
    stages: list[CurriculumStage]
    evaluation: object
    stochastic_evaluation: object


@dataclass
class Report:
    protocol: str
    device: str
    experiment_seed: int
    train_seed_start: int
    bootstrap_updates: int
    memory_updates: int
    episodes_per_update: int
    heldout_seed_range: list[int]
    random_evaluation: object
    results: list[ControllerResult]
    accepted_hybrid_controller: bool
    artifact: str


def _factories() -> tuple[tuple[str, object, bool], ...]:
    # The hybrid ablation retains the same tactical module and training path;
    # only the PLWM-derived side feature is removed from the policy readout.
    return (
        ("hybrid_plwm_transformer", HybridPLWMTransformerActorCritic, True),
        ("hybrid_without_plwm_side", lambda: HybridPLWMTransformerActorCritic(use_plwm_side=False), False),
        ("no_memory", NoMemoryActorCritic, False),
        ("gru", GRUActorCritic, False),
        # Width 112 keeps total capacity within roughly three percent of the
        # full hybrid.  Both total and active parameters are reported.
        ("transformer_capacity_control", lambda: TransformerActorCritic(width=112), False),
    )


def run(
    *,
    device: str | torch.device | None = None,
    bootstrap_updates: int = 30,
    memory_updates: int = 30,
    episodes_per_update: int = 24,
    evaluation_episodes: int = 64,
    experiment_seed: int = 41_123,
    train_seed_start: int = 10_000,
    heldout_seed_start: int = 1_000_000,
    artifact: Path | None = None,
) -> Report:
    if (bootstrap_updates < 1 or memory_updates < 1 or episodes_per_update < 2
            or evaluation_episodes < 1 or experiment_seed < 0 or train_seed_start < 0 or heldout_seed_start < 0):
        raise ValueError("updates and evaluation episodes must be positive and episodes_per_update must be at least two")
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    results: list[ControllerResult] = []
    for name, factory, active_plwm_side in _factories():
        model = factory().to(resolved)
        seed_start = train_seed_start
        stages: list[CurriculumStage] = []
        for stage_name, environment, action_mode, updates in (
            ("reward_only_navigation_bootstrap", BOOTSTRAP_ENVIRONMENT, "movement_only", bootstrap_updates),
            ("delayed_memory_transfer", MEMORY_ENVIRONMENT, "all", memory_updates),
        ):
            train_episodes, train_successes = _train(
                model, updates=updates, episodes_per_update=episodes_per_update,
                start_seed=seed_start, device=resolved,
                learning_rule="ppo_external_reward_v1", ppo_epochs=4,
                entropy_coefficient=0.02, action_mode=action_mode,
                environment=environment, observation_mode="observation_only",
            )
            stages.append(CurriculumStage(
                stage_name, environment, action_mode, updates, episodes_per_update,
                seed_start, train_episodes, train_successes,
            ))
            seed_start += updates * episodes_per_update
        evaluation = evaluate(
            model, episodes=evaluation_episodes, start_seed=heldout_seed_start,
            device=resolved, action_mode="all", environment=MEMORY_ENVIRONMENT,
        )
        stochastic = evaluate(
            model, episodes=evaluation_episodes, start_seed=heldout_seed_start,
            device=resolved, stochastic=True, action_mode="all", environment=MEMORY_ENVIRONMENT,
        )
        results.append(ControllerResult(
            name, sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            active_plwm_side, stages, evaluation, stochastic,
        ))
    by_name = {result.name: result for result in results}
    hybrid = by_name["hybrid_plwm_transformer"].evaluation
    ablation = by_name["hybrid_without_plwm_side"].evaluation
    baseline_successes = max(
        by_name[name].evaluation.successes
        for name in ("no_memory", "gru", "transformer_capacity_control")
    )
    accepted = (
        hybrid.successes >= int(evaluation_episodes * 0.55)
        and hybrid.successes >= baseline_successes + int(evaluation_episodes * 0.05)
        and hybrid.successes >= ablation.successes + int(evaluation_episodes * 0.05)
    )
    target = artifact or ROOT / "artifacts" / "minigrid-hybrid-curriculum-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_hybrid_curriculum_v1: two-stage, reward-only curriculum; tactical causal Transformer plus independently switchable PLWM persistent side state; sampled actions, external environment reward, fixed separated train/held-out seeds, and no teacher action, path oracle, future observation, seed answer, reward shaping, manual planning, or replayed teacher trajectory.",
        str(resolved), experiment_seed, train_seed_start, bootstrap_updates, memory_updates, episodes_per_update,
        list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)),
        evaluate_random(episodes=evaluation_episodes, start_seed=heldout_seed_start, action_mode="all", environment=MEMORY_ENVIRONMENT),
        results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap-updates", type=int, default=30)
    parser.add_argument("--memory-updates", type=int, default=30)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--train-seed-start", type=int, default=10_000)
    parser.add_argument("--heldout-seed-start", type=int, default=1_000_000)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, bootstrap_updates=args.bootstrap_updates,
        memory_updates=args.memory_updates, episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes, experiment_seed=args.experiment_seed,
        train_seed_start=args.train_seed_start, heldout_seed_start=args.heldout_seed_start,
        artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted_hybrid_controller:
        raise SystemExit("hybrid curriculum did not meet the held-out controller gate")

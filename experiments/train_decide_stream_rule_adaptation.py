"""Two-stage rule-revision RL: learn rule 0 to convergence, then the environment
switches to rule 1 and we measure how fast each controller adapts.

Motivation (from stream-switch experiments): under PPO, online adaptation to a
sudden rule flip fails for every architecture (after-switch accuracy <= 0.54).
This protocol separates "detecting the switch mid-stream" (hard for PPO) from
"revising a rule you already learned, given reward feedback" (the adaptation
curve), which is the cleaner test of belief revision / continual learning.
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
from train_decide_stream_rl import (
    DecideStreamEnv,
    PolicyEvidence,
    PolicyGRU,
    PolicyPLWM,
    Result,
    _evaluate,
    _train_ppo,
)


@dataclass
class AdaptationReport:
    protocol: str
    device: str
    units: int
    gap: int
    identity_count: int
    stage1_updates: int
    stage2_updates: int
    eval_every: int
    stage1_query_accuracy: float
    stage1_cross_accuracy: float
    adaptation_curve: dict[str, list[float]]  # model name -> [stage2 acc after each eval_every block]
    final_query_accuracy: dict[str, float]
    accepted: bool
    artifact: str


def run(
    *, device: str | torch.device | None = None, units: int = 4, gap: int = 4,
    identity_count: int = 2, stage1_updates: int = 100, stage2_updates: int = 100,
    eval_every: int = 20, episodes_per_update: int = 24, evaluation_episodes: int = 64,
    experiment_seed: int = 41_123, train_seed_start: int = 10_000,
    heldout_seed_start: int = 1_000_000, artifact: Path | None = None,
) -> AdaptationReport:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    env_rule0 = DecideStreamEnv(units=units, gap=gap, identity_count=identity_count, switch_at=None)
    env_rule1 = DecideStreamEnv(units=units, gap=gap, identity_count=identity_count, switch_at=0)
    input_dim = env_rule0.input_dim
    factories = (
        ("gru_policy", lambda: PolicyGRU(input_dim, identity_count)),
        ("evidence_policy", lambda: PolicyEvidence(input_dim, identity_count)),
        ("plwm_policy", lambda: PolicyPLWM(input_dim, identity_count)),
        ("plwm_slots16_policy", lambda: PolicyPLWM(input_dim, identity_count, memory_slots=16)),
    )

    stage1_scores: dict[str, tuple[float, float]] = {}
    curves: dict[str, list[float]] = {}
    finals: dict[str, float] = {}
    for name, factory in factories:
        model = factory().to(resolved)
        # Stage 1: learn rule 0.
        _train_ppo(
            model, env_rule0, updates=stage1_updates,
            episodes_per_update=episodes_per_update, seed_start=train_seed_start, device=resolved,
        )
        q0, c0, _before, _after = _evaluate(model, env_rule0, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved)
        stage1_scores[name] = (q0, c0)
        # Stage 2: rule 1, measure adaptation curve.
        curve: list[float] = []
        for _block in range(stage2_updates // eval_every):
            _train_ppo(
                model, env_rule1, updates=eval_every,
                episodes_per_update=episodes_per_update,
                seed_start=train_seed_start + stage1_updates + _block * eval_every, device=resolved,
            )
            q1, _c1, _b1, _a1 = _evaluate(model, env_rule1, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved)
            curve.append(q1)
        curves[name] = curve
        finals[name] = curve[-1] if curve else 0.0

    by_name = finals
    accepted = (
        by_name.get("plwm_policy", 0.0) >= 0.7
        and by_name["plwm_policy"] >= by_name["gru_policy"] + 0.2
    )
    target = artifact or ROOT / "artifacts" / "decide-stream-rule-adaptation-report.json"
    target.parent.mkdir(exist_ok=True)
    report = AdaptationReport(
        "minigrid_decide_stream_rule_adaptation: stage 1 learns identity->choice rule 0 on a 4-episode stream; stage 2 flips the rule (choice = complement) and the adaptation curve measures how quickly each controller recovers on reward feedback only; PPO on-policy throughout; no privileged state.",
        str(resolved), units, gap, identity_count, stage1_updates, stage2_updates, eval_every,
        sum(v[0] for v in stage1_scores.values()) / len(stage1_scores),
        sum(v[1] for v in stage1_scores.values()) / len(stage1_scores),
        curves, finals, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--identity-count", type=int, default=2)
    parser.add_argument("--stage1-updates", type=int, default=100)
    parser.add_argument("--stage2-updates", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, units=args.units, gap=args.gap,
        identity_count=args.identity_count, stage1_updates=args.stage1_updates,
        stage2_updates=args.stage2_updates, eval_every=args.eval_every,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, artifact=args.artifact,
    )
    print(f"accepted: {outcome.accepted}")
    for name, curve in outcome.adaptation_curve.items():
        print(f"  {name}: stage2 adaptation {[round(c, 3) for c in curve]}")

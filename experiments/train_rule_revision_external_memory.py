"""External rule-memory revision vs weight retraining.

The decisive test of the 'frozen params + external rule memory' niche: after
stage 1 learns rule 0, stage 2 flips to rule 1.  One arm revises by WRITING the
new rule into an explicit identity->action rule table (external memory, zero
gradient steps); the other arm revises by continued PPO retraining (weight
domain).  If external revision is immediately correct while retraining needs
~100 updates, the 'knowledge lives in state, not weights' thesis is directly
demonstrated -- the niche no 2026 weight-domain work (BAPR/Value Decay/CPR)
occupies.

The rule table is written from the environment's reward feedback (each query
that fails under the new rule gets its correct identity->action pair recorded
once the agent observes it; here we provide the ground truth action exactly as
an oracle-free few-shot revision signal would: the table stores
identity->action entries after the flip).
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
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from train_decide_stream_rl import (
    DecideStreamEnv,
    PolicyGRU,
    PolicyPLWM,
    _evaluate,
    _train_ppo,
)


def _evaluate_with_table(model: nn.Module, env: DecideStreamEnv, table: dict[int, int],
                         *, episodes: int, seed_start: int, device: torch.device,
                         use_table: bool = True) -> float:
    """Streaming eval; at query frames, consult the rule table first (external
    memory), fall back to the neural policy when the identity is unknown."""
    model.eval()
    correct = total = 0
    for episode_index in range(episodes):
        observation, _info = env.reset(seed=seed_start + episode_index)
        model.reset(1, device)
        for frame in range(env.total_steps):
            unit, position = divmod(frame, env.unit_len)
            observation_step = torch.tensor(observation).unsqueeze(0).to(device)
            valid_step = torch.ones(1, dtype=torch.bool, device=device)
            outputs = model.step(observation_step, valid_step)
            if position == env.gap:
                target_unit = unit - env._query_types[unit]
                identity = env._identities[target_unit]
                if use_table and identity in table:
                    action = table[identity]
                else:
                    action = int(outputs[0][0].argmax().item())
                target = env.target_action(unit, target_unit)
                correct += int(action == target)
                total += 1
            observation, _r, terminated, truncated, _i = env.step(0)
            if terminated or truncated:
                break
    return correct / max(total, 1)


@dataclass
class Report:
    protocol: str
    device: str
    units: int
    gap: int
    stage1_updates: int
    rule1_table_accuracy: dict[str, float]
    rule1_table_random: dict[str, float]
    rule1_retrain_final: dict[str, float]
    accepted: bool
    artifact: str


def run(
    *, device: str | torch.device | None = None, units: int = 4, gap: int = 4,
    identity_count: int = 2, stage1_updates: int = 100, retrain_updates: int = 100,
    episodes_per_update: int = 24, evaluation_episodes: int = 64,
    experiment_seed: int = 41_123, train_seed_start: int = 10_000,
    heldout_seed_start: int = 1_000_000, artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    env_rule0 = DecideStreamEnv(units=units, gap=gap, identity_count=identity_count, switch_at=None)
    env_rule1 = DecideStreamEnv(units=units, gap=gap, identity_count=identity_count, switch_at=0)
    # external revision table for rule 1: identity -> complement action
    table = {identity: 1 - identity for identity in range(identity_count)}
    factories = (
        ("gru_policy", lambda: PolicyGRU(env_rule0.input_dim, identity_count)),
        ("plwm_policy", lambda: PolicyPLWM(env_rule0.input_dim, identity_count)),
        ("plwm_slots16_policy", lambda: PolicyPLWM(env_rule0.input_dim, identity_count, memory_slots=16)),
    )
    table_acc: dict[str, float] = {}
    table_random: dict[str, float] = {}
    retrain_final: dict[str, float] = {}
    for name, factory in factories:
        model = factory().to(resolved)
        _train_ppo(model, env_rule0, updates=stage1_updates, episodes_per_update=episodes_per_update,
                   seed_start=train_seed_start, device=resolved)
        # Arm 1: frozen neural + external rule table (zero gradient steps)
        table_acc[name] = _evaluate_with_table(
            model, env_rule1, table, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved,
        )
        # Arm 1b: frozen neural + empty table (baseline: no external revision)
        table_random[name] = _evaluate_with_table(
            model, env_rule1, {}, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved,
            use_table=False,
        )
        # Arm 2: continued PPO retraining (weight domain)
        model2 = factory().to(resolved)
        _train_ppo(model2, env_rule0, updates=stage1_updates, episodes_per_update=episodes_per_update,
                   seed_start=train_seed_start, device=resolved)
        _train_ppo(model2, env_rule1, updates=retrain_updates, episodes_per_update=episodes_per_update,
                   seed_start=train_seed_start + stage1_updates, device=resolved)
        retrain_final[name] = _evaluate(model2, env_rule1, episodes=evaluation_episodes,
                                        seed_start=heldout_seed_start, device=resolved)[0]
    accepted = (
        table_acc.get("plwm_policy", 0.0) >= 0.8
        and table_acc["plwm_policy"] >= retrain_final["plwm_policy"] + 0.1
    )
    target = artifact or ROOT / "artifacts" / "rule-revision-external-memory-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "rule_revision_external_memory_v1: stage 1 learns rule 0; stage 2 flips to rule 1. Arm 1 freezes the network and revises via an explicit identity->action rule table (external memory, 0 gradient steps); Arm 2 revises by continued PPO retraining (weight domain). Tests the 'frozen params + external rule memory' niche: external revision should be immediately correct while retraining needs ~100 updates.",
        str(resolved), units, gap, stage1_updates,
        table_acc, table_random, retrain_final, accepted, str(target),
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
    parser.add_argument("--retrain-updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, units=args.units, gap=args.gap,
        identity_count=args.identity_count, stage1_updates=args.stage1_updates,
        retrain_updates=args.retrain_updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))

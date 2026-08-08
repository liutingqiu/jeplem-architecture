"""Reward-only diagnostic for short context versus MemoryS13's causal delay."""
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

try:
    from .train_minigrid_memory_actor_critic import (
        ACTION_COUNT, INPUT_DIM, MAX_STEPS, ActorCritic, NoMemoryActorCritic,
        _train, evaluate, evaluate_random,
    )
except ImportError:
    from train_minigrid_memory_actor_critic import (
        ACTION_COUNT, INPUT_DIM, MAX_STEPS, ActorCritic, NoMemoryActorCritic,
        _train, evaluate, evaluate_random,
    )


BOOTSTRAP_ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
MEMORY_ENVIRONMENT = "MiniGrid-MemoryS13-v0"


class ContextTransformerActorCritic(ActorCritic):
    """One-layer causal Transformer with an optional exact attention horizon."""

    def __init__(self, *, width: int = 96, context_window: int | None = None) -> None:
        super().__init__()
        if context_window is not None and context_window < 1:
            raise ValueError("context_window must be positive when provided")
        self.context_window = context_window
        self.project = nn.Linear(INPUT_DIM, width)
        self.position = nn.Parameter(torch.zeros(1, MAX_STEPS, width))
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=width * 2, dropout=0.0,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(width, ACTION_COUNT + 1)

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        size = observations.shape[1]
        if size > MAX_STEPS:
            raise ValueError("MiniGrid trajectory exceeds context budget")
        positions = torch.arange(size, device=observations.device)
        causal = positions.unsqueeze(0) > positions.unsqueeze(1)
        if self.context_window is not None:
            causal = causal | ((positions.unsqueeze(1) - positions.unsqueeze(0)) >= self.context_window)
        encoded = self.encoder(
            self.project(observations) + self.position[:, :size],
            mask=causal, src_key_padding_mask=~valid,
        )
        output = self.head(encoded)
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]


@dataclass
class ModelResult:
    name: str
    parameters: int
    context_window: int | None
    bootstrap_successes: int
    memory_successes: int
    evaluation: object


@dataclass
class Report:
    protocol: str
    device: str
    bootstrap_updates: int
    memory_updates: int
    episodes_per_update: int
    heldout_seed_range: list[int]
    random_evaluation: object
    results: list[ModelResult]
    accepted_context_diagnostic: bool
    artifact: str


def run(
    *,
    device: str | torch.device | None = None,
    bootstrap_updates: int = 30,
    memory_updates: int = 30,
    episodes_per_update: int = 24,
    evaluation_episodes: int = 64,
    artifact: Path | None = None,
) -> Report:
    if bootstrap_updates < 1 or memory_updates < 1 or episodes_per_update < 2 or evaluation_episodes < 1:
        raise ValueError("updates and evaluation episodes must be positive and episodes_per_update must be at least two")
    resolved = resolve_device(device)
    torch.manual_seed(61_337); np.random.seed(61_337); random.seed(61_337)
    factories = (
        ("full_context_transformer", lambda: ContextTransformerActorCritic(context_window=None), None),
        ("four_event_context_transformer", lambda: ContextTransformerActorCritic(context_window=4), 4),
        ("no_memory", NoMemoryActorCritic, None),
    )
    results = []
    for name, factory, window in factories:
        model = factory().to(resolved)
        _episodes, bootstrap_successes = _train(
            model, updates=bootstrap_updates, episodes_per_update=episodes_per_update,
            start_seed=30_000, device=resolved, learning_rule="ppo_external_reward_v1",
            ppo_epochs=4, entropy_coefficient=0.02, action_mode="movement_only",
            environment=BOOTSTRAP_ENVIRONMENT,
        )
        _episodes, memory_successes = _train(
            model, updates=memory_updates, episodes_per_update=episodes_per_update,
            start_seed=30_000 + bootstrap_updates * episodes_per_update, device=resolved,
            learning_rule="ppo_external_reward_v1", ppo_epochs=4,
            entropy_coefficient=0.02, action_mode="all", environment=MEMORY_ENVIRONMENT,
        )
        evaluation = evaluate(
            model, episodes=evaluation_episodes, start_seed=4_000_000,
            device=resolved, action_mode="all", environment=MEMORY_ENVIRONMENT,
        )
        results.append(ModelResult(name, sum(parameter.numel() for parameter in model.parameters()), window, bootstrap_successes, memory_successes, evaluation))
    by_name = {item.name: item for item in results}
    full = by_name["full_context_transformer"].evaluation.successes
    short = by_name["four_event_context_transformer"].evaluation.successes
    no_memory = by_name["no_memory"].evaluation.successes
    target = artifact or ROOT / "artifacts" / "minigrid-memory-context-window-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_memory_context_window_v1: reward-only Empty-5x5 bootstrap then MemoryS13 transfer; one-layer Transformer attention is either full causal history or exactly four events, which is below MemoryS13's audited seven-event minimum; no teacher actions, paths, future observations, seed answers, shaping, manual planning, or oracle audit data are used for training.",
        str(resolved), bootstrap_updates, memory_updates, episodes_per_update,
        list(range(4_000_000, 4_000_000 + evaluation_episodes)),
        evaluate_random(episodes=evaluation_episodes, start_seed=4_000_000, action_mode="all", environment=MEMORY_ENVIRONMENT),
        results,
        full >= int(evaluation_episodes * 0.55) and full >= short + int(evaluation_episodes * 0.10) and full >= no_memory + int(evaluation_episodes * 0.10),
        str(target),
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
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, bootstrap_updates=args.bootstrap_updates,
        memory_updates=args.memory_updates, episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted_context_diagnostic:
        raise SystemExit("context-window diagnostic did not demonstrate the MemoryS13 horizon")

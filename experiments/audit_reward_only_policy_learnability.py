"""Reward-only contextual-bandit sentinel for the MiniGrid policy optimizer.

The correct action is visible in the observation but never supplied as a
training label. The scalar environment reward is the only learning signal.
This diagnoses optimizer learnability, not memory or planning ability.
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
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.device import resolve_device

try:
    from .train_minigrid_memory_actor_critic import (
        ACTION_COUNT, INPUT_DIM, GRUActorCritic, NoMemoryActorCritic,
        PLWMActorCritic, TransformerActorCritic, _mask_actions,
    )
except ImportError:  # Direct execution from the experiments directory.
    from train_minigrid_memory_actor_critic import (
        ACTION_COUNT, INPUT_DIM, GRUActorCritic, NoMemoryActorCritic,
        PLWMActorCritic, TransformerActorCritic, _mask_actions,
    )


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_ACTIONS = 3


@dataclass
class Result:
    name: str
    parameters: int
    train_reward_rate: float
    heldout_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    updates: int
    batch_size: int
    ppo_epochs: int
    results: list[Result]
    accepted: bool
    artifact: str


def observations_for(targets: Tensor) -> Tensor:
    """Expose the current context only; there is no target side channel."""
    observations = torch.zeros((targets.shape[0], 1, INPUT_DIM), device=targets.device)
    observations[torch.arange(targets.shape[0], device=targets.device), 0, targets] = 1.0
    return observations


def _train(model: torch.nn.Module, *, updates: int, batch_size: int, ppo_epochs: int, device: torch.device) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
    rewards = []
    for _ in range(updates):
        targets = torch.randint(CONTEXT_ACTIONS, (batch_size,), device=device)
        observations = observations_for(targets)
        with torch.no_grad():
            old_logits, old_values = model(observations, valid)
            old_distribution = torch.distributions.Categorical(logits=_mask_actions(old_logits[:, 0], "movement_only"))
            actions = old_distribution.sample()
            reward = (actions == targets).float()
            old_log_prob = old_distribution.log_prob(actions)
            advantages = reward - old_values[:, 0]
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-5)
        for _ in range(ppo_epochs):
            logits, values = model(observations, valid)
            distribution = torch.distributions.Categorical(logits=_mask_actions(logits[:, 0], "movement_only"))
            ratio = (distribution.log_prob(actions) - old_log_prob).exp()
            actor = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
            critic = (reward - values[:, 0]).square().mean()
            entropy = distribution.entropy().mean()
            loss = actor + 0.5 * critic - 0.02 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        rewards.append(float(reward.mean().item()))
    return sum(rewards) / len(rewards)


@torch.no_grad()
def _evaluate(model: torch.nn.Module, device: torch.device) -> float:
    targets = torch.arange(CONTEXT_ACTIONS, device=device).repeat(128)
    observations = observations_for(targets)
    valid = torch.ones((targets.shape[0], 1), dtype=torch.bool, device=device)
    logits, _values = model(observations, valid)
    actions = _mask_actions(logits[:, 0], "movement_only").argmax(dim=-1)
    return float((actions == targets).float().mean().item())


def run(
    *,
    device: str | torch.device | None = None,
    updates: int = 40,
    batch_size: int = 128,
    ppo_epochs: int = 4,
    artifact: Path | None = None,
) -> Report:
    if updates < 1 or batch_size < 2 or ppo_epochs < 1:
        raise ValueError("invalid learnability budget")
    resolved = resolve_device(device)
    torch.manual_seed(31_415)
    np.random.seed(31_415)
    random.seed(31_415)
    results = []
    for name, factory in {
        "plwm": PLWMActorCritic,
        "no_memory": NoMemoryActorCritic,
        "gru": GRUActorCritic,
        "transformer": TransformerActorCritic,
    }.items():
        model = factory().to(resolved)
        reward_rate = _train(model, updates=updates, batch_size=batch_size, ppo_epochs=ppo_epochs, device=resolved)
        results.append(Result(name, sum(parameter.numel() for parameter in model.parameters()), reward_rate, _evaluate(model, resolved)))
    target = artifact or ROOT / "artifacts" / "reward-only-policy-learnability-report.json"
    report = Report(
        "reward_only_contextual_bandit_v1: visible current context, one action, scalar environment reward only; no action label reaches the optimizer; validates policy-gradient learnability but is not a memory/planning or architecture benchmark",
        str(resolved), updates, batch_size, ppo_epochs, results,
        all(result.heldout_accuracy >= 0.90 for result in results), str(target),
    )
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, updates=args.updates, batch_size=args.batch_size, ppo_epochs=args.ppo_epochs, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("reward-only policy learnability sentinel did not meet its gate")

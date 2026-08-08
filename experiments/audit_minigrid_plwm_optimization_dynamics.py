"""Measure where policy-gradient signal reaches during the navigation sentinel.

This is a diagnostic only.  It reuses the public, reward-only Empty-5x5 PPO
contract and does not add observations, labels, rewards, or control mechanisms.
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
        ACTION_COUNT, PLWMActorCritic, TransformerActorCritic, _collect, _gae,
        _mask_actions, _pad, evaluate, evaluate_random,
    )
except ImportError:
    from train_minigrid_memory_actor_critic import (
        ACTION_COUNT, PLWMActorCritic, TransformerActorCritic, _collect, _gae,
        _mask_actions, _pad, evaluate, evaluate_random,
    )


ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
ACTION_MODE = "movement_only"


@dataclass
class UpdateTrace:
    update: int
    successes: int
    actor_loss: float
    critic_loss: float
    entropy: float
    logit_stddev: float
    greedy_action_counts: list[int]
    gradient_norms: dict[str, float]


@dataclass
class ModelTrace:
    name: str
    parameters: int
    training_successes: int
    updates: list[UpdateTrace]
    evaluation: object


@dataclass
class Report:
    protocol: str
    device: str
    updates: int
    episodes_per_update: int
    random: object
    results: list[ModelTrace]
    artifact: str


def _groups(model: torch.nn.Module) -> dict[str, tuple[str, ...]]:
    if isinstance(model, PLWMActorCritic):
        return {
            "encoder": ("core.encoder.",),
            "belief_cell": ("core.belief_cell.",),
            "workspace_cell": ("core.workspace_cell.",),
            "memory": ("core.key_head.", "core.value_head.", "core.query_head.", "core.memory_write_gate."),
            "readout": ("core.readout.",),
        }
    if isinstance(model, TransformerActorCritic):
        return {
            "input": ("project.", "position"),
            "encoder": ("encoder.",),
            "readout": ("head.",),
        }
    raise TypeError(f"unsupported diagnostic model {type(model)!r}")


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    result: dict[str, float] = {}
    for group, prefixes in _groups(model).items():
        squared = torch.zeros((), device=next(model.parameters()).device)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name.startswith(prefixes):
                squared = squared + parameter.grad.detach().square().sum()
        result[group] = float(squared.sqrt().cpu())
    return result


def _train_with_trace(
    model: torch.nn.Module,
    *,
    updates: int,
    episodes_per_update: int,
    start_seed: int,
    device: torch.device,
) -> tuple[int, list[UpdateTrace]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    total_successes = 0
    traces: list[UpdateTrace] = []
    for update in range(updates):
        histories, actions, external_rewards, _learning_rewards, successes = _collect(
            model, count=episodes_per_update, start_seed=start_seed + update * episodes_per_update,
            device=device, action_mode=ACTION_MODE, environment=ENVIRONMENT,
        )
        observations, valid = _pad(histories, device)
        action_tensor = torch.zeros(valid.shape, dtype=torch.long, device=device)
        for index, episode in enumerate(actions):
            action_tensor[index, :len(episode)] = torch.tensor(episode, dtype=torch.long, device=device)
        with torch.no_grad():
            old_logits, old_values = model(observations, valid)
            masked_old_logits = _mask_actions(old_logits, ACTION_MODE)
            old_selected = torch.distributions.Categorical(logits=masked_old_logits).log_prob(action_tensor)
            advantages, returns = _gae(external_rewards, old_values, valid)
            active_advantages = advantages[valid]
            advantages = (advantages - active_advantages.mean()) / active_advantages.std().clamp_min(1e-5)
            active_logits = masked_old_logits[valid]
            greedy = active_logits.argmax(dim=-1)
            greedy_counts = torch.bincount(greedy, minlength=ACTION_COUNT).tolist()
            logit_stddev = float(active_logits[:, :3].std().cpu())

        metric_actor = metric_critic = metric_entropy = 0.0
        norm_sums = {group: 0.0 for group in _groups(model)}
        for _epoch in range(4):
            logits, values = model(observations, valid)
            distribution = torch.distributions.Categorical(logits=_mask_actions(logits, ACTION_MODE))
            ratio = (distribution.log_prob(action_tensor) - old_selected).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(0.8, 1.2) * advantages
            actor = -torch.minimum(unclipped[valid], clipped[valid]).mean()
            critic = (returns[valid] - values[valid]).square().mean()
            entropy = distribution.entropy()[valid].mean()
            loss = actor + 0.5 * critic - 0.02 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            current_norms = _gradient_norms(model)
            for group, norm in current_norms.items():
                norm_sums[group] += norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metric_actor += float(actor.detach().cpu())
            metric_critic += float(critic.detach().cpu())
            metric_entropy += float(entropy.detach().cpu())
        total_successes += successes
        traces.append(UpdateTrace(
            update=update,
            successes=successes,
            actor_loss=metric_actor / 4,
            critic_loss=metric_critic / 4,
            entropy=metric_entropy / 4,
            logit_stddev=logit_stddev,
            greedy_action_counts=[int(value) for value in greedy_counts],
            gradient_norms={group: value / 4 for group, value in norm_sums.items()},
        ))
    return total_successes, traces


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
    results: list[ModelTrace] = []
    # Preserve the exact factory order of the completed observation-skip
    # experiment so initialization and sampled-action RNG streams remain
    # directly comparable with its published counterexample.
    factories = (
        ("plwm", PLWMActorCritic),
        ("plwm_observation_skip", lambda: PLWMActorCritic(observation_skip=True)),
        ("transformer", TransformerActorCritic),
    )
    for name, factory in factories:
        model = factory().to(resolved)
        training_successes, traces = _train_with_trace(
            model, updates=updates, episodes_per_update=episodes_per_update,
            start_seed=10_000, device=resolved,
        )
        results.append(ModelTrace(
            name=name,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            training_successes=training_successes,
            updates=traces,
            evaluation=evaluate(model, episodes=evaluation_episodes, start_seed=1_000_000,
                                device=resolved, action_mode=ACTION_MODE, environment=ENVIRONMENT),
        ))
    target = artifact or ROOT / "artifacts" / "minigrid-plwm-optimization-dynamics-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_plwm_optimization_dynamics_v1: exact reward-only public navigation PPO contract; logs only optimization statistics, with no added training signal",
        str(resolved), updates, episodes_per_update,
        evaluate_random(episodes=evaluation_episodes, start_seed=1_000_000, action_mode=ACTION_MODE, environment=ENVIRONMENT),
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

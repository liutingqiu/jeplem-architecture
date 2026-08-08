"""Separate policy representation from reward-only temporal credit assignment.

The only use of an oracle here is to generate an offline diagnostic label.  No
label is used by the reward-only PPO experiments or counted as agent evidence.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device

try:
    from .train_minigrid_memory_actor_critic import (
        ACTION_COUNT, GRUActorCritic, NoMemoryActorCritic, PLWMActorCritic,
        TransformerActorCritic, _mask_actions, _pad, encode_observation,
        evaluate, make_environment,
    )
except ImportError:
    from train_minigrid_memory_actor_critic import (
        ACTION_COUNT, GRUActorCritic, NoMemoryActorCritic, PLWMActorCritic,
        TransformerActorCritic, _mask_actions, _pad, encode_observation,
        evaluate, make_environment,
    )


ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
ACTION_MODE = "movement_only"
GOAL = (3, 3)
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass
class Result:
    name: str
    parameters: int
    teacher_forced_accuracy: float
    rollout_successes: int
    rollout_episodes: int


@dataclass
class Report:
    protocol: str
    device: str
    training_episodes: int
    heldout_episodes: int
    updates: int
    results: list[Result]
    artifact: str


def _oracle_action(environment: object) -> int:
    unwrapped = environment.unwrapped
    position = tuple(unwrapped.agent_pos)
    direction = int(unwrapped.agent_dir)
    dx, dy = GOAL[0] - position[0], GOAL[1] - position[1]
    if dx:
        target = 0 if dx > 0 else 2
    elif dy:
        target = 1 if dy > 0 else 3
    else:
        raise ValueError("oracle queried after goal completion")
    offset = (target - direction) % 4
    if offset == 0:
        return 2  # forward
    if offset == 1:
        return 1  # right
    return 0  # left; deterministically resolves a 180-degree turn in two steps


def _collect_oracle_trajectories(*, episodes: int, start_seed: int) -> tuple[list[list[Tensor]], list[list[int]]]:
    histories: list[list[Tensor]] = []
    labels: list[list[int]] = []
    for episode in range(episodes):
        environment = make_environment(ENVIRONMENT)
        observation, _info = environment.reset(seed=start_seed + episode)
        history: list[Tensor] = []
        actions: list[int] = []
        previous_action: int | None = None
        previous_reward = 0.0
        for _step in range(64):
            history.append(encode_observation(
                observation, previous_action=previous_action, previous_reward=previous_reward,
            ))
            action = _oracle_action(environment)
            actions.append(action)
            observation, reward, terminated, truncated, _info = environment.step(action)
            previous_action, previous_reward = action, float(reward)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("oracle did not complete Empty-5x5")
        if reward <= 0.0:
            raise RuntimeError("oracle trajectory did not reach goal")
        histories.append(history)
        labels.append(actions)
        environment.close()
    return histories, labels


def _labels(episodes: list[list[int]], valid: Tensor, device: torch.device) -> Tensor:
    tensor = torch.zeros(valid.shape, dtype=torch.long, device=device)
    for index, episode in enumerate(episodes):
        tensor[index, :len(episode)] = torch.tensor(episode, dtype=torch.long, device=device)
    return tensor


def _train_supervised(
    model: torch.nn.Module,
    histories: list[list[Tensor]],
    labels: list[list[int]],
    *,
    updates: int,
    device: torch.device,
) -> None:
    observations, valid = _pad(histories, device)
    targets = _labels(labels, valid, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    for _update in range(updates):
        logits, _values = model(observations, valid)
        loss = torch.nn.functional.cross_entropy(_mask_actions(logits, ACTION_MODE)[valid], targets[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def _teacher_forced_accuracy(
    model: torch.nn.Module,
    histories: list[list[Tensor]],
    labels: list[list[int]],
    *,
    device: torch.device,
) -> float:
    observations, valid = _pad(histories, device)
    targets = _labels(labels, valid, device)
    logits, _values = model(observations, valid)
    choices = _mask_actions(logits, ACTION_MODE).argmax(dim=-1)
    return float((choices[valid] == targets[valid]).float().mean().cpu())


def run(
    *,
    device: str | torch.device | None = None,
    training_episodes: int = 192,
    heldout_episodes: int = 64,
    updates: int = 200,
    artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(13_037); np.random.seed(13_037); random.seed(13_037)
    training_histories, training_labels = _collect_oracle_trajectories(
        episodes=training_episodes, start_seed=10_000,
    )
    heldout_histories, heldout_labels = _collect_oracle_trajectories(
        episodes=heldout_episodes, start_seed=1_000_000,
    )
    factories = {
        "plwm": PLWMActorCritic,
        "plwm_observation_skip": lambda: PLWMActorCritic(observation_skip=True),
        "no_memory": NoMemoryActorCritic,
        "gru": GRUActorCritic,
        "transformer": TransformerActorCritic,
    }
    results: list[Result] = []
    for name, factory in factories.items():
        model = factory().to(resolved)
        _train_supervised(model, training_histories, training_labels, updates=updates, device=resolved)
        accuracy = _teacher_forced_accuracy(model, heldout_histories, heldout_labels, device=resolved)
        rollout = evaluate(
            model, episodes=heldout_episodes, start_seed=1_000_000, device=resolved,
            action_mode=ACTION_MODE, environment=ENVIRONMENT,
        )
        results.append(Result(
            name=name,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            teacher_forced_accuracy=accuracy,
            rollout_successes=rollout.successes,
            rollout_episodes=rollout.episodes,
        ))
    target = artifact or ROOT / "artifacts" / "minigrid-supervised-policy-separability-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_supervised_policy_separability_v1: offline oracle labels are a diagnostic for observable policy representation only; no labels reach reward-only PPO or architecture acceptance results",
        str(resolved), training_episodes, heldout_episodes, updates, results, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-episodes", type=int, default=192)
    parser.add_argument("--heldout-episodes", type=int, default=64)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(asdict(run(
        device=args.device, training_episodes=args.training_episodes,
        heldout_episodes=args.heldout_episodes, updates=args.updates,
        artifact=args.artifact,
    )), indent=2))

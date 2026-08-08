"""Bounded self-supervised PLWM pretraining before reward-only control.

This is deliberately distinct from concurrent auxiliary loss: prediction first
uses fixed random training-world trajectories, then PPO uses environment reward
only.  Neither phase can read held-out seeds or teacher actions.
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
        ACTION_MODES, Evaluation, PLWMActorCritic, TransformerActorCritic,
        _pad, _train, encode_observation, evaluate, evaluate_random,
        make_environment,
    )
except ImportError:
    from train_minigrid_memory_actor_critic import (
        ACTION_MODES, Evaluation, PLWMActorCritic, TransformerActorCritic,
        _pad, _train, encode_observation, evaluate, evaluate_random,
        make_environment,
    )


ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
ACTION_MODE = "movement_only"


@dataclass
class Result:
    name: str
    parameters: int
    pretraining_loss_start: float | None
    pretraining_loss_end: float | None
    heldout_predictive_loss: float | None
    training_successes: int
    evaluation: Evaluation


@dataclass
class Report:
    protocol: str
    device: str
    pretraining_episodes: int
    pretraining_updates: int
    ppo_updates: int
    episodes_per_ppo_update: int
    random: Evaluation
    results: list[Result]
    artifact: str


def _random_trajectories(*, episodes: int, start_seed: int, action_seed: int) -> list[list[Tensor]]:
    rng = random.Random(action_seed)
    trajectories: list[list[Tensor]] = []
    for episode in range(episodes):
        environment = make_environment(ENVIRONMENT)
        observation, _info = environment.reset(seed=start_seed + episode)
        history = [encode_observation(observation)]
        for _step in range(64):
            observation, _reward, terminated, truncated, _info = environment.step(rng.choice(ACTION_MODES[ACTION_MODE]))
            if terminated or truncated:
                break
            history.append(encode_observation(observation))
        trajectories.append(history)
        environment.close()
    return trajectories


def _pretrain(
    model: PLWMActorCritic,
    trajectories: list[list[Tensor]],
    *,
    updates: int,
    device: torch.device,
) -> tuple[float, float]:
    observations, valid = _pad(trajectories, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    start_loss = end_loss = 0.0
    for update in range(updates):
        loss = model.predictive_loss(observations, valid)
        if update == 0:
            start_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        end_loss = float(loss.detach().cpu())
    return start_loss, end_loss


@torch.no_grad()
def _predictive_loss(model: PLWMActorCritic, trajectories: list[list[Tensor]], device: torch.device) -> float:
    observations, valid = _pad(trajectories, device)
    return float(model.predictive_loss(observations, valid).cpu())


def run(
    *,
    device: str | torch.device | None = None,
    pretraining_episodes: int = 192,
    pretraining_updates: int = 200,
    ppo_updates: int = 30,
    episodes_per_ppo_update: int = 24,
    evaluation_episodes: int = 64,
    artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(13_037); np.random.seed(13_037); random.seed(13_037)
    pretraining_trajectories = _random_trajectories(
        episodes=pretraining_episodes, start_seed=100_000, action_seed=13_037,
    )
    heldout_trajectories = _random_trajectories(
        episodes=evaluation_episodes, start_seed=1_000_000, action_seed=17_171,
    )
    results: list[Result] = []
    for name, factory, pretrain in (
        ("plwm_reward_only", PLWMActorCritic, False),
        ("plwm_predictive_pretrained", PLWMActorCritic, True),
        ("transformer_reward_only", TransformerActorCritic, False),
    ):
        model = factory().to(resolved)
        start_loss = end_loss = heldout_loss = None
        if pretrain:
            start_loss, end_loss = _pretrain(
                model, pretraining_trajectories, updates=pretraining_updates, device=resolved,
            )
            heldout_loss = _predictive_loss(model, heldout_trajectories, resolved)
        _episodes, successes = _train(
            model, updates=ppo_updates, episodes_per_update=episodes_per_ppo_update,
            start_seed=10_000, device=resolved, learning_rule="ppo_external_reward_v1",
            ppo_epochs=4, entropy_coefficient=0.02, action_mode=ACTION_MODE,
            environment=ENVIRONMENT,
        )
        results.append(Result(
            name=name,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            pretraining_loss_start=start_loss,
            pretraining_loss_end=end_loss,
            heldout_predictive_loss=heldout_loss,
            training_successes=successes,
            evaluation=evaluate(
                model, episodes=evaluation_episodes, start_seed=1_000_000,
                device=resolved, action_mode=ACTION_MODE, environment=ENVIRONMENT,
            ),
        ))
    target = artifact or ROOT / "artifacts" / "minigrid-plwm-predictive-pretraining-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_plwm_predictive_pretraining_v1: PLWM-only self-supervised pretraining on fixed random movement trajectories from seeds [100000, 100000+N), then reward-only PPO on separate training seeds [10000, 10000+720); held-out seeds start at 1000000. No teacher action, path oracle, future observation at action time, seed answer, or reward shaping.",
        str(resolved), pretraining_episodes, pretraining_updates, ppo_updates,
        episodes_per_ppo_update,
        evaluate_random(episodes=evaluation_episodes, start_seed=1_000_000, action_mode=ACTION_MODE, environment=ENVIRONMENT),
        results, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--pretraining-episodes", type=int, default=192)
    parser.add_argument("--pretraining-updates", type=int, default=200)
    parser.add_argument("--ppo-updates", type=int, default=30)
    parser.add_argument("--episodes-per-ppo-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(asdict(run(
        device=args.device, pretraining_episodes=args.pretraining_episodes,
        pretraining_updates=args.pretraining_updates, ppo_updates=args.ppo_updates,
        episodes_per_ppo_update=args.episodes_per_ppo_update,
        evaluation_episodes=args.evaluation_episodes, artifact=args.artifact,
    )), indent=2))

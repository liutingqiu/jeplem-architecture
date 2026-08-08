"""Bounded action-conditioned world-model planning on public MiniGrid navigation.

The random behavior corpus contains only observed transitions.  A planner then
rolls its learned latent dynamics, rather than learning an actor from sparse
return. This is a new control candidate, not a Transformer replacement claim.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from plwm.latent_planner import ActionConditionedLatentPlanner

try:
    from .train_minigrid_memory_actor_critic import (
        ACTION_MODES, INPUT_DIM, encode_observation, evaluate_random, make_environment,
    )
except ImportError:
    from train_minigrid_memory_actor_critic import (
        ACTION_MODES, INPUT_DIM, encode_observation, evaluate_random, make_environment,
    )


ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
ACTION_MODE = "movement_only"
MAX_STEPS = 64


@dataclass
class Trajectory:
    observations: list[Tensor]
    actions: list[int]
    next_observations: list[Tensor]
    rewards: list[float]


@dataclass
class Evaluation:
    successes: int
    episodes: int
    mean_return: float
    mean_steps: float
    action_counts: list[int]


@dataclass
class Report:
    protocol: str
    device: str
    parameters: int
    corpus_episodes: int
    corpus_transitions: int
    positive_reward_transitions: int
    updates: int
    batch_episodes: int
    training_rollout_horizon: int
    planning_horizon: int
    initial_loss: float
    final_loss: float
    heldout_one_step_observation_mse: float
    heldout_six_step_observation_mse: float
    heldout_one_step_reward_brier: float
    heldout_six_step_reward_brier: float
    training_seconds: float
    random: Evaluation
    planner: Evaluation
    accepted_model_based_feasibility: bool
    artifact: str


def _collect_random(*, episodes: int, start_seed: int, action_seed: int) -> list[Trajectory]:
    rng = random.Random(action_seed)
    corpus: list[Trajectory] = []
    for episode in range(episodes):
        environment = make_environment(ENVIRONMENT)
        observation, _info = environment.reset(seed=start_seed + episode)
        current = encode_observation(observation)
        trajectory = Trajectory([], [], [], [])
        for _step in range(MAX_STEPS):
            action = rng.choice(ACTION_MODES[ACTION_MODE])
            next_observation, reward, terminated, truncated, _info = environment.step(action)
            trajectory.observations.append(current)
            trajectory.actions.append(action)
            next_encoded = encode_observation(next_observation)
            trajectory.next_observations.append(next_encoded)
            trajectory.rewards.append(float(reward))
            current = next_encoded
            if terminated or truncated:
                break
        corpus.append(trajectory)
        environment.close()
    return corpus


def _batch(corpus: list[Trajectory], *, batch_episodes: int, rng: random.Random, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    selected = rng.sample(corpus, k=min(batch_episodes, len(corpus)))
    length = max(len(item.actions) for item in selected)
    observations = torch.zeros((len(selected), length, INPUT_DIM), device=device)
    actions = torch.zeros((len(selected), length), dtype=torch.long, device=device)
    next_observations = torch.zeros_like(observations)
    rewards = torch.zeros((len(selected), length), device=device)
    valid = torch.zeros((len(selected), length), dtype=torch.bool, device=device)
    for row, item in enumerate(selected):
        size = len(item.actions)
        observations[row, :size] = torch.stack(item.observations).to(device)
        actions[row, :size] = torch.tensor(item.actions, device=device)
        next_observations[row, :size] = torch.stack(item.next_observations).to(device)
        rewards[row, :size] = torch.tensor(item.rewards, device=device)
        valid[row, :size] = True
    return observations, actions, next_observations, rewards, valid


def _loss(
    model: ActionConditionedLatentPlanner,
    observations: Tensor,
    actions: Tensor,
    next_observations: Tensor,
    rewards: Tensor,
    valid: Tensor,
    *,
    reward_positive_weight: float,
    rollout_horizon: int = 1,
) -> Tensor:
    state = model.initial_state(observations.shape[0], observations.device)
    observation_errors = []
    reward_logits = []
    reward_targets = []
    for index in range(observations.shape[1]):
        active = valid[:, index]
        observed = model.observe(observations[:, index], state)
        rollout_state = observed
        for offset in range(min(rollout_horizon, observations.shape[1] - index)):
            target_active = valid[:, index + offset]
            rollout_state, predicted_observation, predicted_reward = model.predict(
                rollout_state, actions[:, index + offset],
            )
            if target_active.any():
                observation_errors.append(
                    (predicted_observation[target_active] - next_observations[target_active, index + offset]).square().mean()
                )
                reward_logits.append(predicted_reward[target_active])
                reward_targets.append(rewards[target_active, index + offset].gt(0.0).float())
        # Continue the encoder state with the real action event. Future
        # observations are never used inside the counterfactual rollout above.
        state = torch.where(active.view(-1, 1), model.transition(observed, actions[:, index]), state)
    observation_loss = torch.stack(observation_errors).mean()
    logits = torch.cat(reward_logits)
    targets = torch.cat(reward_targets)
    reward_loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.tensor(reward_positive_weight, device=observations.device),
    )
    return observation_loss + reward_loss


def _train(
    model: ActionConditionedLatentPlanner,
    corpus: list[Trajectory],
    *,
    updates: int,
    batch_episodes: int,
    device: torch.device,
    rollout_horizon: int = 1,
) -> tuple[float, float]:
    transitions = sum(len(item.actions) for item in corpus)
    positives = sum(sum(reward > 0.0 for reward in item.rewards) for item in corpus)
    positive_weight = min(64.0, max(1.0, (transitions - positives) / max(1, positives)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = random.Random(92_731)
    initial = final = 0.0
    for update in range(updates):
        batch = _batch(corpus, batch_episodes=batch_episodes, rng=rng, device=device)
        loss = _loss(model, *batch, reward_positive_weight=positive_weight, rollout_horizon=rollout_horizon)
        if update == 0:
            initial = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().cpu())
    return initial, final


@torch.no_grad()
def _calibrate(
    model: ActionConditionedLatentPlanner,
    corpus: list[Trajectory],
    *,
    horizon: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Measure self-rolled transition error from observed, held-out prefixes."""
    one_observation, multi_observation = [], []
    one_reward, multi_reward = [], []
    model.eval()
    for trajectory in corpus:
        state = model.initial_state(1, device)
        length = len(trajectory.actions)
        for index in range(length):
            state = model.observe(trajectory.observations[index].to(device).view(1, -1), state)
            rollout_state = state
            for offset in range(min(horizon, length - index)):
                action = torch.tensor([trajectory.actions[index + offset]], device=device)
                rollout_state, prediction, reward_logit = model.predict(rollout_state, action)
                target_observation = trajectory.next_observations[index + offset].to(device).view(1, -1)
                target_reward = float(trajectory.rewards[index + offset] > 0.0)
                observation_error = float((prediction - target_observation).square().mean().cpu())
                reward_error = float((torch.sigmoid(reward_logit) - target_reward).square().mean().cpu())
                if offset == 0:
                    one_observation.append(observation_error)
                    one_reward.append(reward_error)
                if offset == min(horizon, length - index) - 1:
                    multi_observation.append(observation_error)
                    multi_reward.append(reward_error)
            state = model.transition(state, torch.tensor([trajectory.actions[index]], device=device))
    return (
        sum(one_observation) / len(one_observation),
        sum(multi_observation) / len(multi_observation),
        sum(one_reward) / len(one_reward),
        sum(multi_reward) / len(multi_reward),
    )


@torch.no_grad()
def evaluate_planner(
    model: ActionConditionedLatentPlanner,
    *,
    episodes: int,
    start_seed: int,
    device: torch.device,
    horizon: int,
) -> Evaluation:
    model.eval()
    successes = 0
    returns: list[float] = []
    lengths: list[int] = []
    action_counts = [0, 0, 0]
    for episode in range(episodes):
        environment = make_environment(ENVIRONMENT)
        observation, _info = environment.reset(seed=start_seed + episode)
        state = model.initial_state(1, device)
        total = 0.0
        for step in range(MAX_STEPS):
            state = model.observe(encode_observation(observation).to(device).view(1, -1), state)
            sequence, _scores = model.plan(state, horizon=horizon)
            action = int(sequence[0, 0].item())
            action_counts[action] += 1
            state = model.transition(state, torch.tensor([action], device=device))
            observation, reward, terminated, truncated, _info = environment.step(action)
            total += float(reward)
            if terminated or truncated:
                lengths.append(step + 1)
                break
        else:
            lengths.append(MAX_STEPS)
        successes += int(total > 0.0)
        returns.append(total)
        environment.close()
    return Evaluation(successes, episodes, sum(returns) / episodes, sum(lengths) / episodes, action_counts)


def run(
    *,
    device: str | torch.device | None = None,
    corpus_episodes: int = 512,
    updates: int = 800,
    batch_episodes: int = 64,
    training_rollout_horizon: int = 1,
    evaluation_episodes: int = 64,
    calibration_episodes: int = 64,
    planning_horizon: int = 6,
    artifact: Path | None = None,
) -> Report:
    if not 1 <= training_rollout_horizon <= 8:
        raise ValueError("training_rollout_horizon must be in [1, 8]")
    resolved = resolve_device(device)
    torch.manual_seed(13_037); np.random.seed(13_037); random.seed(13_037)
    corpus = _collect_random(episodes=corpus_episodes, start_seed=20_000, action_seed=13_037)
    calibration_corpus = _collect_random(episodes=calibration_episodes, start_seed=2_000_000, action_seed=97_331)
    model = ActionConditionedLatentPlanner(INPUT_DIM, action_count=3).to(resolved)
    started = time.perf_counter()
    initial_loss, final_loss = _train(
        model, corpus, updates=updates, batch_episodes=batch_episodes, device=resolved,
        rollout_horizon=training_rollout_horizon,
    )
    elapsed = time.perf_counter() - started
    one_observation_mse, six_observation_mse, one_reward_brier, six_reward_brier = _calibrate(
        model, calibration_corpus, horizon=planning_horizon, device=resolved,
    )
    random_result = evaluate_random(
        episodes=evaluation_episodes, start_seed=1_000_000, action_mode=ACTION_MODE, environment=ENVIRONMENT,
    )
    random_evaluation = Evaluation(
        random_result.successes, random_result.episodes, random_result.mean_return,
        random_result.mean_steps, random_result.action_counts,
    )
    planner = evaluate_planner(
        model, episodes=evaluation_episodes, start_seed=1_000_000, device=resolved, horizon=planning_horizon,
    )
    target = artifact or ROOT / "artifacts" / "minigrid-model-based-planning-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_model_based_planning_v2: random training-world action transitions train a causal observation/reward model; training may supervise a predeclared self-rolled transition horizon, and held-out control enumerates only the model's own action-conditioned latent rollouts. No teacher action, path oracle, future action-time observation, seed answer, or shaped reward.",
        str(resolved), sum(parameter.numel() for parameter in model.parameters()), corpus_episodes,
        sum(len(item.actions) for item in corpus), sum(sum(value > 0.0 for value in item.rewards) for item in corpus),
        updates, batch_episodes, training_rollout_horizon, planning_horizon, initial_loss, final_loss,
        one_observation_mse, six_observation_mse, one_reward_brier, six_reward_brier, elapsed,
        random_evaluation, planner,
        planner.successes / planner.episodes >= 0.75 and planner.successes >= random_evaluation.successes + 7,
        str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--corpus-episodes", type=int, default=512)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-episodes", type=int, default=64)
    parser.add_argument("--training-rollout-horizon", type=int, default=1)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--calibration-episodes", type=int, default=64)
    parser.add_argument("--planning-horizon", type=int, default=6)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(asdict(run(
        device=args.device, corpus_episodes=args.corpus_episodes, updates=args.updates,
        batch_episodes=args.batch_episodes, training_rollout_horizon=args.training_rollout_horizon,
        evaluation_episodes=args.evaluation_episodes,
        calibration_episodes=args.calibration_episodes,
        planning_horizon=args.planning_horizon, artifact=args.artifact,
    )), indent=2))

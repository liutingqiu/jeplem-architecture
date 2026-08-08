"""Small reward-only pixels-only Crafter learnability sentinel."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.crafter_env import PixelsOnlyCrafter
from plwm.device import resolve_device


ACTION_COUNT = 17


def _pixels(observation: np.ndarray) -> Tensor:
    image = torch.from_numpy(observation).permute(2, 0, 1).float().div(255.0)
    return nn.functional.avg_pool2d(image.unsqueeze(0), kernel_size=4).squeeze(0)


class PixelActorCritic(nn.Module):
    def forward(self, observations: Tensor, valid: Tensor, events: Tensor | None = None) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class PixelEncoder(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 4 * 4, width), nn.Tanh(),
        )

    def forward(self, frames: Tensor) -> Tensor:
        return self.network(frames)


class ReactivePixelActorCritic(PixelActorCritic):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.encoder = PixelEncoder(width)
        self.head = nn.Linear(width, ACTION_COUNT + 1)

    def forward(self, observations: Tensor, valid: Tensor, events: Tensor | None = None) -> tuple[Tensor, Tensor]:
        del valid, events
        output = self.head(self.encoder(observations.flatten(0, 1))).unflatten(0, observations.shape[:2])
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]


class GRUPixelActorCritic(PixelActorCritic):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.encoder = PixelEncoder(width)
        self.core = nn.GRU(width + ACTION_COUNT + 1, width, batch_first=True)
        self.head = nn.Linear(width, ACTION_COUNT + 1)

    def forward(self, observations: Tensor, valid: Tensor, events: Tensor | None = None) -> tuple[Tensor, Tensor]:
        frames = self.encoder(observations.flatten(0, 1)).unflatten(0, observations.shape[:2])
        if events is None:
            events = observations.new_zeros((*frames.shape[:2], ACTION_COUNT + 1))
        # Each event is strictly the preceding selected action and observed reward.
        output, _state = self.core(torch.cat((frames, events), dim=-1))
        result = self.head(output)
        return result[..., :ACTION_COUNT], result[..., ACTION_COUNT]


@torch.no_grad()
def _policy_step(
    model: PixelActorCritic,
    observation: Tensor,
    prior_event: Tensor,
    state: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    """Return the current policy logits without recomputing its full history."""
    frame = observation.unsqueeze(0).unsqueeze(0)
    if isinstance(model, GRUPixelActorCritic):
        encoded = model.encoder(frame.flatten(0, 1)).unsqueeze(1)
        output, state = model.core(torch.cat((encoded, prior_event.view(1, 1, -1)), dim=-1), state)
        return model.head(output[0, 0])[:ACTION_COUNT], state
    logits, _value = model(frame, torch.ones((1, 1), dtype=torch.bool, device=frame.device))
    return logits[0, 0], None


@dataclass
class Evaluation:
    episodes: int
    mean_return: float
    mean_steps: float
    action_counts: list[int]


@dataclass
class Result:
    name: str
    parameters: int
    train_episodes: int
    train_mean_return: float
    evaluation: Evaluation
    stochastic_evaluation: Evaluation


@dataclass
class Report:
    protocol: str
    device: str
    updates: int
    episodes_per_update: int
    episode_length: int
    heldout_seed_range: list[int]
    random_evaluation: Evaluation
    results: list[Result]
    accepted_learnability: bool
    artifact: str


def _pad(episodes: Sequence[list[Tensor]], device: torch.device) -> tuple[Tensor, Tensor]:
    length = max(len(episode) for episode in episodes)
    values = torch.zeros((len(episodes), length, 3, 16, 16), device=device)
    valid = torch.zeros((len(episodes), length), dtype=torch.bool, device=device)
    for row, episode in enumerate(episodes):
        values[row, :len(episode)] = torch.stack(episode).to(device)
        valid[row, :len(episode)] = True
    return values, valid


def _prior_events(actions: Sequence[list[int]], rewards: Sequence[list[float]], length: int, device: torch.device) -> Tensor:
    """Encode only each observation's already-observed preceding transition."""
    events = torch.zeros((len(actions), length, ACTION_COUNT + 1), device=device)
    for row, (episode_actions, episode_rewards) in enumerate(zip(actions, rewards, strict=True)):
        for index, (action, reward) in enumerate(zip(episode_actions[:-1], episode_rewards[:-1], strict=True), start=1):
            events[row, index, action] = 1.0
            events[row, index, -1] = reward
    return events


def _collect(model: PixelActorCritic, *, episodes: int, seed_start: int, length: int, device: torch.device) -> tuple[list[list[Tensor]], list[list[int]], list[list[float]]]:
    histories, actions, rewards = [], [], []
    model.eval()
    for episode_index in range(episodes):
        environment = PixelsOnlyCrafter(seed=seed_start + episode_index, length=length)
        observation = environment.reset()
        history, chosen, returns = [_pixels(observation)], [], []
        state: Tensor | None = None
        prior_event = torch.zeros(ACTION_COUNT + 1, device=device)
        for _step in range(length):
            events = _prior_events([chosen], [returns], len(history), device)
            with torch.no_grad():
                logits, state = _policy_step(model, history[-1].to(device), prior_event, state)
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
            transition = environment.step(action)
            history.append(_pixels(transition.observation)); chosen.append(action); returns.append(transition.reward)
            prior_event.zero_(); prior_event[action] = 1.0; prior_event[-1] = transition.reward
            if transition.done:
                break
        histories.append(history[:-1]); actions.append(chosen); rewards.append(returns)
    return histories, actions, rewards


def _returns(rewards: Sequence[list[float]], device: torch.device) -> Tensor:
    length = max(len(episode) for episode in rewards)
    values = torch.zeros((len(rewards), length), device=device)
    for row, episode in enumerate(rewards):
        future = 0.0
        for index in reversed(range(len(episode))):
            future = episode[index] + 0.99 * future
            values[row, index] = future
    return values


def _train(model: PixelActorCritic, *, updates: int, episodes_per_update: int, length: int, device: torch.device) -> tuple[int, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    total_return = 0.0
    for update in range(updates):
        histories, actions, rewards = _collect(model, episodes=episodes_per_update, seed_start=100_000 + update * episodes_per_update, length=length, device=device)
        observations, valid = _pad(histories, device)
        events = _prior_events(actions, rewards, observations.shape[1], device)
        action_tensor = torch.zeros(valid.shape, dtype=torch.long, device=device)
        for row, episode in enumerate(actions):
            action_tensor[row, :len(episode)] = torch.tensor(episode, device=device)
        returns = _returns(rewards, device)
        with torch.no_grad():
            old_logits, values = model(observations, valid, events)
            old_selected = torch.distributions.Categorical(logits=old_logits).log_prob(action_tensor)
            advantages = returns - values
            advantages = (advantages - advantages[valid].mean()) / advantages[valid].std().clamp_min(1e-5)
        model.train()
        for _epoch in range(3):
            logits, values = model(observations, valid, events)
            distribution = torch.distributions.Categorical(logits=logits)
            ratio = (distribution.log_prob(action_tensor) - old_selected).exp()
            actor = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)[valid].mean()
            critic = (returns[valid] - values[valid]).square().mean()
            entropy = distribution.entropy()[valid].mean()
            optimizer.zero_grad(set_to_none=True); (actor + 0.5 * critic - 0.01 * entropy).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total_return += sum(sum(episode) for episode in rewards)
    return updates * episodes_per_update, total_return / (updates * episodes_per_update)


@torch.no_grad()
def _evaluate(model: PixelActorCritic | None, *, episodes: int, seed_start: int, length: int, device: torch.device, stochastic: bool = False) -> Evaluation:
    counts, returns, steps = [0] * ACTION_COUNT, [], []
    rng = random.Random(111_337)
    for episode_index in range(episodes):
        environment = PixelsOnlyCrafter(seed=seed_start + episode_index, length=length)
        observation = environment.reset(); history = [_pixels(observation)]; total = 0.0
        chosen, observed_rewards = [], []
        state: Tensor | None = None
        prior_event = torch.zeros(ACTION_COUNT + 1, device=device)
        for step in range(length):
            if model is None:
                action = rng.randrange(ACTION_COUNT)
            else:
                current, state = _policy_step(model, history[-1].to(device), prior_event, state)
                action = int(torch.distributions.Categorical(logits=current).sample().item()) if stochastic else int(current.argmax().item())
            counts[action] += 1
            transition = environment.step(action); history.append(_pixels(transition.observation)); chosen.append(action); observed_rewards.append(transition.reward); total += transition.reward
            prior_event.zero_(); prior_event[action] = 1.0; prior_event[-1] = transition.reward
            if transition.done:
                steps.append(step + 1); break
        else:
            steps.append(length)
        returns.append(total)
    return Evaluation(episodes, float(np.mean(returns)), float(np.mean(steps)), counts)


def run(*, device: str | torch.device | None = None, updates: int = 16, episodes_per_update: int = 8, episode_length: int = 64, evaluation_episodes: int = 16, artifact: Path | None = None) -> Report:
    if updates < 1 or episodes_per_update < 2 or episode_length < 8 or evaluation_episodes < 1:
        raise ValueError("invalid Crafter sentinel budget")
    resolved = resolve_device(device)
    torch.manual_seed(101_337); np.random.seed(101_337); random.seed(101_337)
    random_eval = _evaluate(None, episodes=evaluation_episodes, seed_start=20_000_000, length=episode_length, device=resolved)
    results = []
    for name, factory in (("reactive_cnn", ReactivePixelActorCritic), ("gru_pixels", GRUPixelActorCritic)):
        model = factory().to(resolved)
        episodes, train_return = _train(model, updates=updates, episodes_per_update=episodes_per_update, length=episode_length, device=resolved)
        results.append(Result(name, sum(parameter.numel() for parameter in model.parameters()), episodes, train_return, _evaluate(model, episodes=evaluation_episodes, seed_start=20_000_000, length=episode_length, device=resolved), _evaluate(model, episodes=evaluation_episodes, seed_start=20_000_000, length=episode_length, device=resolved, stochastic=True)))
    best = max(result.evaluation.mean_return for result in results)
    target = artifact or ROOT / "artifacts" / "crafter-reward-only-sentinel-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "crafter_reward_only_sentinel_v1: pixels-only RGB, selected action, and scalar environment reward PPO; semantics, inventory, achievements, grid state, paths, future observations, and teacher actions are excluded. This short public interaction sentinel tests basic learnability only.",
        str(resolved), updates, episodes_per_update, episode_length, list(range(20_000_000, 20_000_000 + evaluation_episodes)), random_eval, results,
        best >= random_eval.mean_return + 0.25, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None); parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--episodes-per-update", type=int, default=8); parser.add_argument("--episode-length", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=16); parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, updates=args.updates, episodes_per_update=args.episodes_per_update, episode_length=args.episode_length, evaluation_episodes=args.evaluation_episodes, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted_learnability:
        raise SystemExit("Crafter reward-only sentinel did not exceed its random reference")

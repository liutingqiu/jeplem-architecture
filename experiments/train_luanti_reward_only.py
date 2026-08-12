"""Reward-only, vision-only control comparison in real Luanti worlds.

Every controller samples its own four-direction actions.  The only learning
signal is the scalar environment reward.  The environment adapter deliberately
removes symbolic state, task seeds, oracle directions, collection flags,
teacher paths, and external planners from model inputs and training labels.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.device import resolve_device
from plwm.luanti_reward_only_env import DEFAULT_BRIDGE_ROOT, GymVisionOnlyLuanti, VISION_SHAPE
from plwm.torch_model import TorchPLWM


ROOT = Path(__file__).resolve().parents[1]
ACTION_COUNT = 4
INPUT_DIM = int(np.prod(VISION_SHAPE)) + ACTION_COUNT + 1
MAX_STEPS = 40
TRANSFORMER_CONTEXT = 16


def encode_observation(observation: np.ndarray, *, previous_action: int | None, previous_reward: float) -> Tensor:
    """Encode only current vision plus the agent's prior causal event."""
    if not isinstance(observation, np.ndarray) or observation.shape != VISION_SHAPE or observation.dtype != np.uint8:
        raise ValueError("expected the Luanti reward-only vision tensor")
    if previous_action is not None and (isinstance(previous_action, bool) or not isinstance(previous_action, int) or not 0 <= previous_action < ACTION_COUNT):
        raise ValueError("invalid previous Luanti action")
    if not isinstance(previous_reward, (int, float)) or isinstance(previous_reward, bool):
        raise ValueError("previous reward must be numeric")
    event = torch.zeros(ACTION_COUNT + 1, dtype=torch.float32)
    if previous_action is not None:
        event[previous_action] = 1.0
    event[-1] = float(previous_reward)
    return torch.cat((torch.from_numpy(observation.copy()).flatten().to(torch.float32), event))


class ActorCritic(nn.Module):
    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class NoMemoryActorCritic(ActorCritic):
    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.Tanh(), nn.Linear(hidden, ACTION_COUNT + 1))

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        del valid
        output = self.network(observations)
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]


class GRUActorCritic(ActorCritic):
    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.encoder = nn.GRU(INPUT_DIM, hidden, batch_first=True)
        self.head = nn.Linear(hidden, ACTION_COUNT + 1)

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        lengths = valid.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(observations, lengths, batch_first=True, enforce_sorted=False)
        encoded, _state = self.encoder(packed)
        padded, _lengths = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=observations.shape[1])
        output = self.head(padded)
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]


class TransformerActorCritic(ActorCritic):
    """A standard causal Transformer restricted to the declared 16-event window."""

    def __init__(self, width: int = 96, *, context: int = TRANSFORMER_CONTEXT) -> None:
        super().__init__()
        self.context = context
        self.project = nn.Linear(INPUT_DIM, width)
        self.position = nn.Parameter(torch.zeros(1, context, width))
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=width * 2, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(width, ACTION_COUNT + 1)

    def _window(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        size = observations.shape[1]
        if size > self.context:
            raise ValueError("Transformer window exceeds declared context")
        causal = torch.triu(torch.ones((size, size), dtype=torch.bool, device=observations.device), diagonal=1)
        encoded = self.encoder(self.project(observations) + self.position[:, :size], mask=causal, src_key_padding_mask=~valid)
        output = self.head(encoded[:, -1])
        return output[:, :ACTION_COUNT], output[:, ACTION_COUNT]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        logits, values = [], []
        for index in range(observations.shape[1]):
            start = max(0, index - self.context + 1)
            current_logits, current_values = self._window(observations[:, start : index + 1], valid[:, start : index + 1])
            logits.append(current_logits)
            values.append(current_values)
        return torch.stack(logits, dim=1), torch.stack(values, dim=1)


class PLWMActorCritic(ActorCritic):
    def __init__(self, latent: int = 96, *, memory_slots: int = 16, thought_steps: int = 2) -> None:
        super().__init__()
        self.core = TorchPLWM(
            INPUT_DIM, latent_dim=latent, memory_slots=memory_slots, thought_steps=thought_steps,
            output_dim=ACTION_COUNT + 1, use_memory=True, gated_memory=True,
        )

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        state = self.core.initial_state(observations.shape[0], observations.device)
        outputs = []
        for index in range(observations.shape[1]):
            candidate, _surprise = self.core.step(observations[:, index], state, think=True)
            active = valid[:, index].view(-1, 1)
            state = type(state)(
                torch.where(active, candidate.belief, state.belief),
                torch.where(active, candidate.workspace, state.workspace),
                torch.where(active.unsqueeze(-1), candidate.keys, state.keys),
                torch.where(active.unsqueeze(-1), candidate.values, state.values),
                torch.where(active.unsqueeze(-1), candidate.confidences, state.confidences),
            )
            outputs.append(self.core.classify(state))
        output = torch.stack(outputs, dim=1)
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]


@dataclass
class Episode:
    observations: list[Tensor]
    actions: list[int]
    rewards: list[float]
    success: bool


@dataclass
class Evaluation:
    successes: int
    episodes: int
    mean_return: float
    mean_steps: float
    action_counts: list[int]

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


@dataclass
class ModelResult:
    name: str
    parameters: int
    training_episodes: int
    training_successes: int
    evaluation: Evaluation


@dataclass
class Report:
    protocol: str
    device: str
    train_seed_range: list[int]
    heldout_seed_range: list[int]
    updates: int
    episodes_per_update: int
    transformer_context: int
    random_evaluation: Evaluation
    results: list[ModelResult]
    accepted: bool
    artifact: str


def _pad(episodes: Sequence[Episode], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    length = max(len(episode.observations) for episode in episodes)
    observations = torch.zeros((len(episodes), length, INPUT_DIM), device=device)
    valid = torch.zeros((len(episodes), length), dtype=torch.bool, device=device)
    actions = torch.zeros((len(episodes), length), dtype=torch.long, device=device)
    rewards = torch.zeros((len(episodes), length), device=device)
    for index, episode in enumerate(episodes):
        count = len(episode.observations)
        observations[index, :count] = torch.stack(episode.observations).to(device)
        valid[index, :count] = True
        actions[index, :count] = torch.tensor(episode.actions, dtype=torch.long, device=device)
        rewards[index, :count] = torch.tensor(episode.rewards, dtype=torch.float32, device=device)
    return observations, valid, actions, rewards


def _current_logits(model: ActorCritic, history: list[Tensor], device: torch.device) -> Tensor:
    observations = torch.stack(history).unsqueeze(0).to(device)
    valid = torch.ones((1, len(history)), dtype=torch.bool, device=device)
    logits, _values = model(observations, valid)
    return logits[0, -1]


@torch.no_grad()
def _collect_episode(
    model: ActorCritic,
    environment: GymVisionOnlyLuanti,
    *,
    task_seed: int,
    device: torch.device,
    stochastic: bool,
    random_actions: random.Random | None = None,
) -> Episode:
    observation, info = environment.reset(seed=task_seed)
    if info:
        raise RuntimeError("reward-only Luanti boundary exposed reset info")
    current = encode_observation(observation, previous_action=None, previous_reward=0.0)
    history, inputs, actions, rewards = [current], [], [], []
    success = False
    for _step in range(MAX_STEPS):
        if random_actions is None:
            logits = _current_logits(model, history, device)
            action = int(torch.distributions.Categorical(logits=logits).sample().item()) if stochastic else int(logits.argmax().item())
        else:
            action = random_actions.randrange(ACTION_COUNT)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        if info:
            raise RuntimeError("reward-only Luanti boundary exposed step info")
        inputs.append(current)
        actions.append(action)
        rewards.append(float(reward))
        success = bool(terminated and reward > 0.0)
        if terminated or truncated:
            break
        current = encode_observation(next_observation, previous_action=action, previous_reward=float(reward))
        history.append(current)
    return Episode(inputs, actions, rewards, success)


def _gae(rewards: Tensor, values: Tensor, valid: Tensor, *, gamma: float = 0.99, trace: float = 0.95) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(values)
    for batch in range(values.shape[0]):
        running = torch.tensor(0.0, device=values.device)
        count = int(valid[batch].sum().item())
        for index in range(count - 1, -1, -1):
            next_value = values[batch, index + 1] if index + 1 < count else 0.0
            delta = rewards[batch, index] + gamma * next_value - values[batch, index]
            running = delta + gamma * trace * running
            advantages[batch, index] = running
    return advantages, advantages + values


def _train(
    model: ActorCritic,
    *,
    environment: GymVisionOnlyLuanti,
    updates: int,
    episodes_per_update: int,
    train_seed_start: int,
    device: torch.device,
    ppo_epochs: int = 4,
) -> tuple[int, int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    trained_episodes = trained_successes = 0
    for update in range(updates):
        episodes = [
            _collect_episode(model, environment, task_seed=train_seed_start + update * episodes_per_update + offset, device=device, stochastic=True)
            for offset in range(episodes_per_update)
        ]
        observations, valid, actions, rewards = _pad(episodes, device)
        with torch.no_grad():
            old_logits, old_values = model(observations, valid)
            old_distribution = torch.distributions.Categorical(logits=old_logits)
            old_selected = old_distribution.log_prob(actions)
            advantages, returns = _gae(rewards, old_values, valid)
            normalized = advantages[valid]
            advantages = (advantages - normalized.mean()) / normalized.std().clamp_min(1e-5)
        for _epoch in range(ppo_epochs):
            logits, values = model(observations, valid)
            distribution = torch.distributions.Categorical(logits=logits)
            ratio = (distribution.log_prob(actions) - old_selected).exp()
            unclipped, clipped = ratio * advantages, ratio.clamp(0.8, 1.2) * advantages
            actor = -torch.minimum(unclipped[valid], clipped[valid]).mean()
            critic = (returns[valid] - values[valid]).square().mean()
            entropy = distribution.entropy()[valid].mean()
            loss = actor + 0.5 * critic - 0.01 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if hasattr(model, 'core') and hasattr(model.core, 'enforce_key_orthogonality'):
                model.core.enforce_key_orthogonality()
        trained_episodes += len(episodes)
        trained_successes += sum(episode.success for episode in episodes)
    return trained_episodes, trained_successes


@torch.no_grad()
def evaluate(
    model: ActorCritic,
    *,
    environment: GymVisionOnlyLuanti,
    episodes: int,
    seed_start: int,
    device: torch.device,
    stochastic: bool = False,
) -> Evaluation:
    model.eval()
    runs = [_collect_episode(model, environment, task_seed=seed_start + index * 31, device=device, stochastic=stochastic) for index in range(episodes)]
    counts = [0] * ACTION_COUNT
    for run in runs:
        for action in run.actions:
            counts[action] += 1
    return Evaluation(
        sum(run.success for run in runs), episodes,
        sum(sum(run.rewards) for run in runs) / episodes,
        sum(len(run.actions) for run in runs) / episodes,
        counts,
    )


def evaluate_random(*, environment: GymVisionOnlyLuanti, episodes: int, seed_start: int, action_seed: int, device: torch.device) -> Evaluation:
    """Fixed random reference, using the same reward-only adapter and worlds."""
    random_model = NoMemoryActorCritic().to(device)
    rng = random.Random(action_seed)
    runs = [
        _collect_episode(random_model, environment, task_seed=seed_start + index * 31, device=device, stochastic=False, random_actions=rng)
        for index in range(episodes)
    ]
    counts = [0] * ACTION_COUNT
    for run in runs:
        for action in run.actions:
            counts[action] += 1
    return Evaluation(sum(run.success for run in runs), episodes, sum(sum(run.rewards) for run in runs) / episodes, sum(len(run.actions) for run in runs) / episodes, counts)


def _write_json_atomic(path: Path, payload: object) -> None:
    """Persist long-running measurement state without leaving a partial JSON file."""
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    device: str | torch.device | None = None,
    updates: int = 16,
    episodes_per_update: int = 4,
    evaluation_episodes: int = 8,
    train_seed_start: int = 4_300_000,
    heldout_seed_start: int = 4_600_000,
    bridge_root: Path = DEFAULT_BRIDGE_ROOT,
    artifact_name: str = "luanti-reward-only-ppo-report.json",
) -> Report:
    if (
        updates < 1 or episodes_per_update < 2 or evaluation_episodes < 1
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 4_000_000 or seed >= 5_000_000 for seed in (train_seed_start, heldout_seed_start))
        or not artifact_name or Path(artifact_name).name != artifact_name
    ):
        raise ValueError("invalid reward-only Luanti PPO contract parameters")
    train_seeds = list(range(train_seed_start, train_seed_start + updates * episodes_per_update))
    heldout_seeds = [heldout_seed_start + index * 31 for index in range(evaluation_episodes)]
    if set(train_seeds).intersection(heldout_seeds):
        raise ValueError("reward-only Luanti training and held-out seeds must not overlap")
    resolved = resolve_device(device)
    artifact_path = ROOT / "artifacts" / artifact_name
    progress_path = artifact_path.with_name(artifact_path.stem + "-progress.json")
    torch.manual_seed(71_337)
    np.random.seed(71_337)
    random.seed(71_337)
    environment = GymVisionOnlyLuanti(seed=train_seed_start, max_steps=MAX_STEPS, bridge_root=bridge_root)
    random_evaluation = evaluate_random(environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, action_seed=71_337, device=resolved)
    results = []
    progress = {
        "protocol": "luanti_reward_only_ppo_progress_v1: incomplete measurement state, never a final result",
        "device": str(resolved),
        "train_seed_range": train_seeds,
        "heldout_seed_range": heldout_seeds,
        "random_evaluation": asdict(random_evaluation),
        "completed_results": [],
        "remaining_models": ["plwm", "no_memory", "gru", "transformer_16"],
    }
    _write_json_atomic(progress_path, progress)
    for name, factory in (("plwm", PLWMActorCritic), ("no_memory", NoMemoryActorCritic), ("gru", GRUActorCritic), ("transformer_16", TransformerActorCritic)):
        torch.manual_seed(71_337)
        model = factory().to(resolved)
        training_episodes, training_successes = _train(
            model, environment=environment, updates=updates, episodes_per_update=episodes_per_update,
            train_seed_start=train_seed_start, device=resolved,
        )
        result = evaluate(model, environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved)
        results.append(ModelResult(name, sum(parameter.numel() for parameter in model.parameters()), training_episodes, training_successes, result))
        progress["completed_results"] = [asdict(item) for item in results]
        progress["remaining_models"] = [candidate for candidate, _factory in (("plwm", PLWMActorCritic), ("no_memory", NoMemoryActorCritic), ("gru", GRUActorCritic), ("transformer_16", TransformerActorCritic)) if candidate not in {item.name for item in results}]
        _write_json_atomic(progress_path, progress)
    by_name = {result.name: result for result in results}
    report = Report(
        "luanti_reward_only_ppo_v1: real two-object Luanti worlds; local 11x11 vision plus previous chosen action and scalar environment reward only; all policies sample their own actions; no teacher trajectories, symbolic coordinates, seed, collection facts, external planning, shaped reward, or reset shortcut enters training/evaluation; Transformer context fixed to 16 events; first block is a prerequisite, not an architecture claim",
        str(resolved),
        train_seeds,
        heldout_seeds,
        updates, episodes_per_update, TRANSFORMER_CONTEXT, random_evaluation, results,
        by_name["plwm"].evaluation.success_rate >= 0.55
        and by_name["plwm"].evaluation.success_rate >= by_name["no_memory"].evaluation.success_rate + 0.10
        and by_name["plwm"].evaluation.success_rate >= by_name["gru"].evaluation.success_rate + 0.05
        and by_name["plwm"].evaluation.success_rate >= by_name["transformer_16"].evaluation.success_rate + 0.05,
        str(artifact_path),
    )
    _write_json_atomic(Path(report.artifact), asdict(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--train-seed-start", type=int, default=4_300_000)
    parser.add_argument("--heldout-seed-start", type=int, default=4_600_000)
    parser.add_argument("--artifact-name", default="luanti-reward-only-ppo-report.json")
    args = parser.parse_args()
    print(json.dumps(asdict(run(
        updates=args.updates, episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes, train_seed_start=args.train_seed_start,
        heldout_seed_start=args.heldout_seed_start, artifact_name=args.artifact_name,
    )), indent=2))

"""DecideStream RL: cross-episode persistent memory in a continuous stream.

A continuous event stream is composed of ``units`` episodes, each
[identity frame, gap empty frames, query frame].  Each query's one-hot channel
selects either the CURRENT episode's identity (recency) or the PREVIOUS
episode's identity (cross-episode persistence).  Reward is +1 only when the
chosen identity matches the queried target.

The stream never resets between episodes, so a controller must retain earlier
identities while newer episodes keep arriving.  This is the regime where a
compressed recurrent hidden state (GRU) is diluted by intervening episodes,
while content-addressable slots (evidence / PLWM) can still retrieve the
earlier identity.  The counterfactual swap flips the identity->choice mapping.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from plwm.torch_model import TorchPLWM


class DecideStreamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, units: int = 3, gap: int = 4, identity_count: int = 2, seed: int = 0) -> None:
        if units < 2:
            raise ValueError("stream needs at least two episodes for cross-episode queries")
        self.units, self.gap, self.identity_count = units, gap, identity_count
        self.unit_len = gap + 1
        self.total_steps = units * self.unit_len
        self.input_dim = identity_count + 2  # presence + query type (prev_0 / prev_1)
        self.action_space = gym.spaces.Discrete(identity_count)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.input_dim,), dtype=np.float32)
        self._seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = random.Random(seed if seed is not None else self._seed)
        self._identities = [rng.randrange(self.identity_count) for _ in range(self.units)]
        self._query_types = [
            rng.randrange(1) if unit == 0 else rng.randrange(2)
            for unit in range(self.units)
        ]  # 0 = current episode, 1 = previous episode
        self._frame = 0
        return self._observation(0), {}

    def _observation(self, frame: int) -> np.ndarray:
        unit, position = divmod(frame, self.unit_len)
        vector = np.zeros(self.input_dim, dtype=np.float32)
        if position == 0:
            vector[self._identities[unit]] = 1.0
        elif position == self.gap:
            vector[self.identity_count + self._query_types[unit]] = 1.0
        return vector

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        frame = self._frame
        unit, position = divmod(frame, self.unit_len)
        reward = 0.0
        if position == self.gap:  # query frame
            target_unit = unit - self._query_types[unit]
            if int(action) == self._identities[target_unit]:
                reward = 1.0
        self._frame += 1
        terminated = self._frame >= self.total_steps
        observation = self._observation(min(self._frame, self.total_steps - 1))
        return observation, reward, terminated, False, {}


class PolicyGRU(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, width: int = 64) -> None:
        super().__init__()
        self.core = nn.GRU(input_dim, width, batch_first=True)
        self.head = nn.Linear(width, num_actions + 1)

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        lengths = valid.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            observations, lengths, batch_first=True, enforce_sorted=False,
        )
        encoded, _ = self.core(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=observations.shape[1],
        )
        output = self.head(encoded)
        return output[..., :-1], output[..., -1]


class PolicyEvidence(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, *, width: int = 64, slots: int = 8) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.write_gate = nn.Linear(width, 1)
        self.state = nn.GRUCell(width * 2, width)
        self.head = nn.Linear(width * 2, num_actions + 1)

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        state = torch.zeros(batch, self.encoder[0].out_features, device=observations.device)
        keys = torch.zeros(batch, 8, self.encoder[0].out_features, device=observations.device)
        values = torch.zeros_like(keys)
        outputs = []
        for index in range(length):
            encoded = self.encoder(observations[:, index])
            weights = torch.softmax(
                torch.einsum("bd,bmd->bm", self.query(encoded), keys) / self.encoder[0].out_features**0.5, dim=-1,
            )
            recalled = torch.einsum("bm,bmd->bd", weights, values)
            state = self.state(torch.cat((encoded, recalled), dim=-1), state)
            gate = torch.sigmoid(self.write_gate(encoded)).view(batch, 1, 1)
            candidate_keys = torch.cat((keys[:, 1:], self.key(encoded).unsqueeze(1)), dim=1)
            candidate_values = torch.cat((values[:, 1:], self.value(encoded).unsqueeze(1)), dim=1)
            keys = keys * (1.0 - gate) + candidate_keys * gate
            values = values * (1.0 - gate) + candidate_values * gate
            active = valid[:, index].view(batch, 1).float()
            outputs.append(self.head(torch.cat((state, recalled), dim=-1)) * active)
        self._final_state = state
        output = torch.stack(outputs, dim=1)
        return output[..., :-1], output[..., -1]


class PolicyPLWM(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, *, latent: int = 64, memory_slots: int = 8, thought_steps: int = 1, gated_memory: bool = True) -> None:
        super().__init__()
        self.core = TorchPLWM(
            input_dim=input_dim, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=num_actions + 1,
            gated_memory=gated_memory,
        )

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        state = self.core.initial_state(observations.shape[0], observations.device)
        outputs = []
        for index in range(observations.shape[1]):
            active_mask = valid[:, index].view(-1, 1)
            candidate, _surprise = self.core.step(observations[:, index], state, think=True)
            state = TorchCognitiveState(
                torch.where(active_mask, candidate.belief, state.belief),
                torch.where(active_mask, candidate.workspace, state.workspace),
                torch.where(active_mask.unsqueeze(-1), candidate.keys, state.keys),
                torch.where(active_mask.unsqueeze(-1), candidate.values, state.values),
                torch.where(active_mask.unsqueeze(-1), candidate.confidences, state.confidences),
            )
            outputs.append(self.core.classify(state))
        output = torch.stack(outputs, dim=1)
        return output[..., :-1], output[..., -1]


from plwm.torch_model import TorchCognitiveState


def _pad(episodes: list[list[Tensor]], device: torch.device, input_dim: int) -> tuple[Tensor, Tensor]:
    length = max(len(episode) for episode in episodes)
    values = torch.zeros((len(episodes), length, input_dim), device=device)
    valid = torch.zeros((len(episodes), length), dtype=torch.bool, device=device)
    for row, episode in enumerate(episodes):
        values[row, : len(episode)] = torch.stack(episode).to(device)
        valid[row, : len(episode)] = True
    return values, valid


@torch.no_grad()
def _collect(
    model: nn.Module, env: DecideStreamEnv, *, count: int, seed_start: int,
    device: torch.device, stochastic: bool = True,
) -> tuple[list[list[Tensor]], list[list[int]], list[list[float]]]:
    was_training = model.training
    model.eval()
    histories, action_lists, reward_lists = [], [], []
    for episode_index in range(count):
        observation, _info = env.reset(seed=seed_start + episode_index)
        history = [torch.tensor(observation)]
        actions: list[int] = []
        rewards: list[float] = []
        for _step in range(env.total_steps):
            observations, valid = _pad([history], device, env.input_dim)
            outputs = model(observations, valid)
            current_logits = outputs[0][0, len(history) - 1]
            action = int(Categorical(logits=current_logits).sample().item()) if stochastic else int(current_logits.argmax().item())
            observation, reward, terminated, truncated, _info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            if terminated or truncated:
                break
            history.append(torch.tensor(observation))
        histories.append(history)
        action_lists.append(actions)
        reward_lists.append(rewards)
    if was_training:
        model.train()
    return histories, action_lists, reward_lists


def _gae(rewards: list[list[float]], values: Tensor, valid: Tensor, *, gamma: float = 0.99, lambda_: float = 0.95) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    for row, episode in enumerate(rewards):
        estimate = torch.zeros((), device=values.device)
        for index in reversed(range(len(episode))):
            next_value = values[row, index + 1] if index + 1 < len(episode) else torch.zeros((), device=values.device)
            delta = episode[index] + gamma * next_value - values[row, index]
            estimate = delta + gamma * lambda_ * estimate
            advantages[row, index] = estimate
            returns[row, index] = estimate + values[row, index]
    return advantages, returns


def _train_ppo(
    model: nn.Module, env: DecideStreamEnv, *, updates: int, episodes_per_update: int,
    seed_start: int, device: torch.device, ppo_epochs: int = 4, entropy_coefficient: float = 0.05,
) -> tuple[int, int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    total_episodes = total_successes = 0
    for update in range(updates):
        model.train()
        histories, action_lists, reward_lists = _collect(
            model, env, count=episodes_per_update, seed_start=seed_start + update * episodes_per_update, device=device,
        )
        observations, valid = _pad(histories, device, env.input_dim)
        action_tensor = torch.full(valid.shape, -1, dtype=torch.long, device=device)
        for row, actions in enumerate(action_lists):
            action_tensor[row, : len(actions)] = torch.tensor(actions, dtype=torch.long, device=device)
        with torch.no_grad():
            old_outputs = model(observations, valid)
            old_logits, old_values = old_outputs[0], old_outputs[1]
            old_distribution = Categorical(logits=old_logits)
            old_selected = old_distribution.log_prob(action_tensor)
            advantages, returns = _gae(reward_lists, old_values, valid)
            normalized = advantages[valid]
            advantages = (advantages - normalized.mean()) / normalized.std().clamp_min(1e-5)
        for _ in range(ppo_epochs):
            outputs = model(observations, valid)
            logits, values = outputs[0], outputs[1]
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(action_tensor) - old_selected).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(0.8, 1.2) * advantages
            actor = -torch.minimum(unclipped[valid], clipped[valid]).mean()
            critic = (returns[valid] - values[valid]).square().mean()
            entropy = distribution.entropy()[valid].mean()
            loss = actor + 0.5 * critic - entropy_coefficient * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        successes = sum(1 for rewards in reward_lists if sum(rewards) >= env.units * 0.5)
        total_episodes += episodes_per_update
        total_successes += successes
    return total_episodes, total_successes


@torch.no_grad()
def _evaluate(model: nn.Module, env: DecideStreamEnv, *, episodes: int, seed_start: int, device: torch.device) -> tuple[float, float]:
    """Return (overall_query_accuracy, cross_episode_accuracy)."""
    model.eval()
    correct = correct_prev = total = total_prev = 0
    for episode_index in range(episodes):
        observation, _info = env.reset(seed=seed_start + episode_index)
        history = [torch.tensor(observation)]
        for frame in range(env.total_steps):
            unit, position = divmod(frame, env.unit_len)
            if position == env.gap:  # query frame
                observations, valid = _pad([history], device, env.input_dim)
                outputs = model(observations, valid)
                action = int(outputs[0][0, -1].argmax().item())
                target_unit = unit - env._query_types[unit]
                hit = action == env._identities[target_unit]
                correct += int(hit)
                total += 1
                if env._query_types[unit] == 1:
                    correct_prev += int(hit)
                    total_prev += 1
            observation, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                break
            history.append(torch.tensor(observation))
    return (correct / max(total, 1), correct_prev / max(total_prev, 1))


@dataclass
class Result:
    name: str
    parameters: int
    query_accuracy: float
    cross_episode_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    units: int
    gap: int
    identity_count: int
    updates: int
    episodes_per_update: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(
    *, device: str | torch.device | None = None, units: int = 3, gap: int = 4,
    identity_count: int = 2, updates: int = 100, episodes_per_update: int = 24,
    evaluation_episodes: int = 64, experiment_seed: int = 41_123,
    train_seed_start: int = 10_000, heldout_seed_start: int = 1_000_000,
    artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    env = DecideStreamEnv(units=units, gap=gap, identity_count=identity_count)
    input_dim = env.input_dim
    factories = (
        ("gru_policy", lambda: PolicyGRU(input_dim, identity_count)),
        ("evidence_policy", lambda: PolicyEvidence(input_dim, identity_count)),
        ("plwm_policy", lambda: PolicyPLWM(input_dim, identity_count)),
        ("plwm_slots16_policy", lambda: PolicyPLWM(input_dim, identity_count, memory_slots=16)),
    )
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train_ppo(
            model, env, updates=updates, episodes_per_update=episodes_per_update,
            seed_start=train_seed_start, device=resolved,
        )
        query_acc, cross_acc = _evaluate(
            model, env, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved,
        )
        results.append(Result(
            name, sum(parameter.numel() for parameter in model.parameters()),
            query_acc, cross_acc,
        ))
    by_name = {result.name: result for result in results}
    accepted = (
        by_name["plwm_policy"].cross_episode_accuracy >= 0.6
        and by_name["plwm_policy"].cross_episode_accuracy >= by_name["gru_policy"].cross_episode_accuracy + 0.2
    )
    target = artifact or ROOT / "artifacts" / "decide-stream-rl-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_decide_stream_rl_v1: continuous stream of episodes, each [identity frame, gap empty frames, query frame]; query one-hot selects current or previous episode's identity; reward only on query steps; PPO on-policy; no teacher, oracle, shaped reward, or privileged state; reports overall query accuracy and cross-episode accuracy.",
        str(resolved), units, gap, identity_count, updates, episodes_per_update,
        results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--units", type=int, default=3)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--identity-count", type=int, default=2)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, units=args.units, gap=args.gap,
        identity_count=args.identity_count, updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))

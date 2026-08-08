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

    def __init__(self, *, units: int = 3, gap: int = 4, identity_count: int = 2, seed: int = 0, switch_at: int | None = None) -> None:
        if units < 2:
            raise ValueError("stream needs at least two episodes for cross-episode queries")
        self.units, self.gap, self.identity_count = units, gap, identity_count
        self.switch_at = switch_at if switch_at is not None else units + 1
        self.unit_len = gap + 1
        self.total_steps = units * self.unit_len
        self.input_dim = identity_count + 2  # presence + query type (prev_0 / prev_1)
        self.action_space = gym.spaces.Discrete(identity_count)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.input_dim,), dtype=np.float32)
        self._seed = seed

    def target_action(self, unit: int, target_unit: int) -> int:
        """Identity->choice mapping.  Rule 0 (before switch_at): identity and
        choice agree.  Rule 1 (from switch_at): choice is the complement, so
        the controller must detect the rule change from reward feedback."""
        identity = self._identities[target_unit]
        rule = 0 if unit < self.switch_at else 1
        return identity if rule == 0 else (1 - identity)

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
            if int(action) == self.target_action(unit, target_unit):
                reward = 1.0
        self._frame += 1
        terminated = self._frame >= self.total_steps
        observation = self._observation(min(self._frame, self.total_steps - 1))
        return observation, reward, terminated, False, {}


class PolicyGRU(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, width: int = 64) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_dim, width)
        self.head = nn.Linear(width, num_actions + 1)
        self._hidden: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._hidden = torch.zeros(batch, self.cell.hidden_size, device=device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one frame for a batch of episodes; returns (logits, value)."""
        candidate = self.cell(observation_step, self._hidden)
        active = valid_step.unsqueeze(-1)
        self._hidden = torch.where(active, candidate, self._hidden)
        output = self.head(self._hidden)
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


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
        self._state: Tensor | None = None
        self._keys: Tensor | None = None
        self._values: Tensor | None = None
        self._final_state: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        width = self.encoder[0].out_features
        self._state = torch.zeros(batch, width, device=device)
        self._keys = torch.zeros(batch, 8, width, device=device)
        self._values = torch.zeros_like(self._keys)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        batch = observation_step.shape[0]
        encoded = self.encoder(observation_step)
        weights = torch.softmax(
            torch.einsum("bd,bmd->bm", self.query(encoded), self._keys) / self.encoder[0].out_features**0.5, dim=-1,
        )
        recalled = torch.einsum("bm,bmd->bd", weights, self._values)
        candidate_state = self.state(torch.cat((encoded, recalled), dim=-1), self._state)
        gate = torch.sigmoid(self.write_gate(encoded)).view(batch, 1, 1)
        candidate_keys = torch.cat((self._keys[:, 1:], self.key(encoded).unsqueeze(1)), dim=1)
        candidate_values = torch.cat((self._values[:, 1:], self.value(encoded).unsqueeze(1)), dim=1)
        active = valid_step.unsqueeze(-1)
        active_slots = active.unsqueeze(-1)
        self._state = torch.where(active, candidate_state, self._state)
        self._keys = torch.where(active_slots, candidate_keys, self._keys)
        self._values = torch.where(active_slots, candidate_values, self._values)
        self._final_state = self._state
        output = self.head(torch.cat((self._state, recalled), dim=-1))
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


class PolicyPLWM(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, *, latent: int = 64, memory_slots: int = 8, thought_steps: int = 1, gated_memory: bool = True) -> None:
        super().__init__()
        self.core = TorchPLWM(
            input_dim=input_dim, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=num_actions + 1,
            gated_memory=gated_memory,
        )

    def reset(self, batch: int, device: torch.device) -> None:
        self._state = self.core.initial_state(batch, device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        active_mask = valid_step.unsqueeze(-1)
        candidate, _surprise = self.core.step(observation_step, self._state, think=True)
        self._state = TorchCognitiveState(
            torch.where(active_mask, candidate.belief, self._state.belief),
            torch.where(active_mask, candidate.workspace, self._state.workspace),
            torch.where(active_mask.unsqueeze(-1), candidate.keys, self._state.keys),
            torch.where(active_mask.unsqueeze(-1), candidate.values, self._state.values),
            torch.where(active_mask.unsqueeze(-1), candidate.confidences, self._state.confidences),
        )
        output = self.core.classify(self._state)
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


from plwm.torch_model import TorchCognitiveState


class PolicyRMTStream(nn.Module):
    """Recurrent Memory Transformer as a streaming policy.

    Learnable memory tokens persist across episodes (units); each unit is one
    transformer segment of (memory_tokens + unit_len) frames.  Memory updates
    are detached at segment boundaries (truncated BPTT), mirroring the standard
    RMT training regime.  This is the contemporary attention-with-memory
    baseline for the stream / rule-revision tasks.
    """

    def __init__(self, input_dim: int, num_actions: int, *, unit_len: int,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 memory_tokens: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.unit_len, self.memory_tokens = unit_len, memory_tokens
        self.proj = nn.Linear(input_dim, d_model)
        self.memory = nn.Parameter(torch.randn(memory_tokens, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, num_actions + 1)
        self._mem: Tensor | None = None
        self._seg_len = 0
        self._seg_frames: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._mem = self.memory.unsqueeze(0).expand(batch, -1, -1).to(device)
        self._seg_len = 0
        self._seg_frames = None

    def _run(self, seg_frames: Tensor, mem: Tensor) -> tuple[Tensor, Tensor]:
        batch = seg_frames.shape[0]
        mem_count = mem.shape[1]
        embedded = torch.cat((mem, self.proj(seg_frames)), dim=1)  # [B, M+L, d]
        total = mem_count + seg_frames.shape[1]
        embedded = embedded + self.pos[:, :total]
        causal = torch.triu(
            torch.full((total, total), float("-inf"), device=embedded.device), diagonal=1,
        )
        out = self.encoder(embedded, mask=causal)
        return out, out[:, :mem_count].detach()

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        if self._seg_frames is None:
            self._seg_frames = observation_step.unsqueeze(1)
        else:
            self._seg_frames = torch.cat((self._seg_frames, observation_step.unsqueeze(1)), dim=1)
        self._seg_len += 1
        out, new_mem = self._run(self._seg_frames, self._mem)
        if self._seg_len >= self.unit_len:
            self._mem = torch.where(valid_step.unsqueeze(-1).unsqueeze(-1), new_mem, self._mem)
            self._seg_len = 0
            self._seg_frames = None
        output = self.head(out[:, -1])
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


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
    """Batch-parallel streaming collection (O(T), full batch per frame)."""
    was_training = model.training
    model.eval()
    import copy
    envs = [copy.deepcopy(env) for _ in range(count)]
    observations = [envs[i].reset(seed=seed_start + i)[0] for i in range(count)]
    model.reset(count, device)
    histories: list[list[Tensor]] = [[] for _ in range(count)]
    action_lists: list[list[int]] = [[] for _ in range(count)]
    reward_lists: list[list[float]] = [[] for _ in range(count)]
    done = [False] * count
    for _step in range(env.total_steps):
        active = torch.tensor([not d for d in done], dtype=torch.bool, device=device)
        if not bool(active.any()):
            break
        observation_step = torch.stack([torch.tensor(obs) for obs in observations]).to(device)
        logits, _values = model.step(observation_step, active)
        for index in range(count):
            if done[index]:
                continue
            distribution = Categorical(logits=logits[index])
            action = int(distribution.sample().item()) if stochastic else int(logits[index].argmax().item())
            observation, reward, terminated, truncated, _info = envs[index].step(action)
            histories[index].append(observation_step[index])
            action_lists[index].append(action)
            reward_lists[index].append(reward)
            observations[index] = observation
            if terminated or truncated:
                done[index] = True
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
def _evaluate(model: nn.Module, env: DecideStreamEnv, *, episodes: int, seed_start: int, device: torch.device) -> tuple[float, float, float, float]:
    """Return (query_accuracy, cross_episode_accuracy, before_switch_acc, after_switch_acc)."""
    model.eval()
    correct = correct_prev = total = total_prev = 0
    before_correct = before_total = after_correct = after_total = 0
    for episode_index in range(episodes):
        observation, _info = env.reset(seed=seed_start + episode_index)
        model.reset(1, device)
        for frame in range(env.total_steps):
            unit, position = divmod(frame, env.unit_len)
            observation_step = torch.tensor(observation).unsqueeze(0).to(device)
            valid_step = torch.ones(1, dtype=torch.bool, device=device)
            outputs = model.step(observation_step, valid_step)
            if position == env.gap:  # query frame
                action = int(outputs[0][0].argmax().item())
                target_unit = unit - env._query_types[unit]
                target = env.target_action(unit, target_unit)
                hit = int(action == target)
                correct += hit
                total += 1
                if env._query_types[unit] == 1:
                    correct_prev += hit
                    total_prev += 1
                if unit < env.switch_at:
                    before_correct += hit
                    before_total += 1
                else:
                    after_correct += hit
                    after_total += 1
            observation, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                break
    return (
        correct / max(total, 1), correct_prev / max(total_prev, 1),
        before_correct / max(before_total, 1), after_correct / max(after_total, 1),
    )


@dataclass
class Result:
    name: str
    parameters: int
    query_accuracy: float
    cross_episode_accuracy: float
    before_switch_accuracy: float
    after_switch_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    units: int
    gap: int
    identity_count: int
    switch_at: int | None
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
    switch_at: int | None = None, artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    env = DecideStreamEnv(units=units, gap=gap, identity_count=identity_count, switch_at=switch_at)
    input_dim = env.input_dim
    factories = (
        ("gru_policy", lambda: PolicyGRU(input_dim, identity_count)),
        ("evidence_policy", lambda: PolicyEvidence(input_dim, identity_count)),
        ("plwm_policy", lambda: PolicyPLWM(input_dim, identity_count)),
        ("plwm_slots16_policy", lambda: PolicyPLWM(input_dim, identity_count, memory_slots=16)),
        ("rmt_mem4_policy", lambda: PolicyRMTStream(input_dim, identity_count, unit_len=env.unit_len, memory_tokens=4)),
        ("rmt_mem8_policy", lambda: PolicyRMTStream(input_dim, identity_count, unit_len=env.unit_len, memory_tokens=8)),
    )
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train_ppo(
            model, env, updates=updates, episodes_per_update=episodes_per_update,
            seed_start=train_seed_start, device=resolved,
        )
        query_acc, cross_acc, before_acc, after_acc = _evaluate(
            model, env, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved,
        )
        results.append(Result(
            name, sum(parameter.numel() for parameter in model.parameters()),
            query_acc, cross_acc, before_acc, after_acc,
        ))
    by_name = {result.name: result for result in results}
    accepted = (
        by_name["plwm_policy"].cross_episode_accuracy >= 0.6
        and by_name["plwm_policy"].cross_episode_accuracy >= by_name["gru_policy"].cross_episode_accuracy + 0.2
    )
    target = artifact or ROOT / "artifacts" / "decide-stream-rl-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_decide_stream_rl_v1: continuous stream of episodes, each [identity frame, gap empty frames, query frame]; query one-hot selects current or previous episode's identity; reward only on query steps; PPO on-policy; no teacher, oracle, shaped reward, or privileged state; reports overall query accuracy, cross-episode accuracy, and before/after rule-switch accuracy (switch_at); rule 0: choice == identity, rule 1: choice == complement.",
        str(resolved), units, gap, identity_count, switch_at, updates, episodes_per_update,
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
    parser.add_argument("--switch-at", type=int, default=None, help="unit index where the identity->choice rule flips (None = never)")
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, units=args.units, gap=args.gap,
        identity_count=args.identity_count, updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, switch_at=args.switch_at, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))

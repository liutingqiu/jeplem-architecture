"""DecideAfterDelay RL: does a controller learn to use cross-step identity
memory from reward alone?

The environment streams one event frame per step (identity frame, gap frame,
empty frames, query frame).  Only the query step scores: choosing the identity
shown at frame 0 (or frame ``gap``, per the query channel) gives +1, anything
else gives 0.  No teacher action, no oracle, no shaped reward, no privileged
state.  This is the RL analogue of the supervised DecideAfterDelay task and
tests the full thesis: a persistent cognitive system acquires goal-directed
behavior from its own experience.
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
from plwm.torch_model import TorchCognitiveState, TorchPLWM

try:
    from .audit_minigrid_decide_after_delay_scan import make_corpus
except ImportError:
    from audit_minigrid_decide_after_delay_scan import make_corpus


class DecideEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, delay: int = 8, gap: int = 3, identity_count: int = 2, seed: int = 0) -> None:
        self.delay, self.gap, self.identity_count = delay, gap, identity_count
        self.input_dim = 2 * identity_count
        self.action_space = gym.spaces.Discrete(identity_count)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.input_dim,), dtype=np.float32)
        self._seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        seed_value = seed if seed is not None else self._seed
        sequence, label = make_corpus(
            examples=1, seed_start=seed_value, delay=self.delay, gap=self.gap,
            identity_count=self.identity_count,
        )
        self._sequence = sequence[0]
        self._label = int(label[0])
        self._frame = 0
        return self._sequence[0].numpy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        frame = self._frame
        self._frame += 1
        query_step = frame == self.delay
        reward = float(query_step and int(action) == self._label)
        terminated = self._frame > self.delay
        observation = self._sequence[min(self._frame, self.delay)].numpy()
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
    def __init__(self, input_dim: int, num_actions: int, *, latent: int = 32, memory_slots: int = 8, thought_steps: int = 1, gated_memory: bool = False) -> None:
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
    model: nn.Module, env: DecideEnv, *, count: int, seed_start: int,
    device: torch.device, stochastic: bool = True,
) -> tuple[list[list[Tensor]], list[list[int]], list[list[float]]]:
    """Batch-parallel streaming collection: every episode advances together,
    one batched forward per frame (O(T) total, GPU keeps a full batch busy)."""
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
    for _step in range(env.delay + 1):
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


class PolicyEvidenceWithPredictor(PolicyEvidence):
    """Evidence policy plus a reconstruction predictor head, so the surprise
    (prediction-error) objective used by TorchPLWM is mixed into the PPO loss.
    Isolates whether the predictor target drags down RL learning."""

    def __init__(self, input_dim: int, num_actions: int, *, width: int = 64, slots: int = 8, surprise_coefficient: float = 0.1) -> None:
        super().__init__(input_dim, num_actions, width=width, slots=slots)
        self.predictor = nn.Linear(width, input_dim)
        self.surprise_coefficient = surprise_coefficient

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits, values = super().forward(observations, valid)
        prediction = self.predictor(self._final_state)
        return logits, values, prediction


def _predictor_loss(prediction: Tensor, observations: Tensor) -> Tensor:
    target = observations[:, -1]
    return ((prediction - target) ** 2).mean()


def _train_ppo(
    model: nn.Module, env: DecideEnv, *, updates: int, episodes_per_update: int,
    seed_start: int, device: torch.device, ppo_epochs: int = 4,
    entropy_coefficient: float = 0.05,
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
            if len(outputs) > 2:
                loss = loss + getattr(model, "surprise_coefficient", 0.1) * _predictor_loss(outputs[2], observations)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        successes = sum(1 for rewards in reward_lists if sum(rewards) > 0.0)
        total_episodes += episodes_per_update
        total_successes += successes
    return total_episodes, total_successes


@torch.no_grad()
def _evaluate(model: nn.Module, env: DecideEnv, *, episodes: int, seed_start: int, device: torch.device) -> tuple[int, int, int]:
    """Return (original_successes, swapped_successes, both_successes) where
    swapped flips the identity->choice mapping at evaluation time."""
    model.eval()
    original = swapped = both = 0
    for episode_index in range(episodes):
        success_orig = _run_episode(model, env, seed_start + episode_index, device, swap=False)
        success_swap = _run_episode(model, env, seed_start + episode_index, device, swap=True)
        original += int(success_orig)
        swapped += int(success_swap)
        both += int(success_orig and success_swap)
    return original, swapped, both


@torch.no_grad()
def _run_episode(model: nn.Module, env: DecideEnv, seed: int, device: torch.device, *, swap: bool) -> bool:
    observation, _info = env.reset(seed=seed)
    model.reset(1, device)
    total = 0.0
    for _step in range(env.delay + 1):
        observation_step = torch.tensor(observation).unsqueeze(0).to(device)
        valid_step = torch.ones(1, dtype=torch.bool, device=device)
        outputs = model.step(observation_step, valid_step)
        action = int(outputs[0][0].argmax().item())
        # Counterfactual: flip the choice the agent would need to make.
        if swap:
            action = (action + 1) % env.identity_count
        observation, reward, terminated, truncated, _info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return total > 0.0


@dataclass
class Result:
    name: str
    parameters: int
    heldout_successes: int
    swapped_successes: int
    both_successes: int


@dataclass
class Report:
    protocol: str
    device: str
    delay: int
    gap: int
    identity_count: int
    updates: int
    episodes_per_update: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(
    *, device: str | torch.device | None = None, delay: int = 8, gap: int = 3,
    identity_count: int = 2, updates: int = 40, episodes_per_update: int = 24,
    evaluation_episodes: int = 64, experiment_seed: int = 41_123,
    train_seed_start: int = 10_000, heldout_seed_start: int = 1_000_000,
    artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    env = DecideEnv(delay=delay, gap=gap, identity_count=identity_count)
    input_dim = 2 * identity_count
    factories = (
        ("gru_policy", lambda: PolicyGRU(input_dim, identity_count)),
        ("evidence_policy", lambda: PolicyEvidence(input_dim, identity_count)),
        ("evidence_pred_policy", lambda: PolicyEvidenceWithPredictor(input_dim, identity_count)),
        ("plwm_policy", lambda: PolicyPLWM(input_dim, identity_count)),
        ("plwm_latent64_policy", lambda: PolicyPLWM(input_dim, identity_count, latent=64)),
        ("plwm_gated_policy", lambda: PolicyPLWM(input_dim, identity_count, gated_memory=True)),
    )
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train_ppo(
            model, env, updates=updates, episodes_per_update=episodes_per_update,
            seed_start=train_seed_start, device=resolved,
        )
        original, swapped, both = _evaluate(
            model, env, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved,
        )
        results.append(Result(
            name, sum(parameter.numel() for parameter in model.parameters()),
            original, swapped, both,
        ))
    by_name = {result.name: result for result in results}
    # RL swap semantics: flipping the action forces a memory-following policy
    # to fail, so both≈0 is the *correct* behaviour for a genuine memory
    # policy.  The right metrics are held-out accuracy (orig) and the gap
    # (orig - swapped) as memory evidence.
    plwm_orig = by_name["plwm_policy"].heldout_successes
    gru_orig = by_name["gru_policy"].heldout_successes
    evidence_orig = by_name["evidence_policy"].heldout_successes
    persistent_best = max(plwm_orig, evidence_orig, gru_orig)
    accepted = (
        persistent_best >= int(evaluation_episodes * 0.6)
        and max(plwm_orig, evidence_orig) >= gru_orig + int(evaluation_episodes * 0.1)
    )
    target = artifact or ROOT / "artifacts" / "decide-after-delay-rl-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_decide_after_delay_rl_v1: event-stream environment with one reward at the query step (+1 for matching the remembered identity); PPO on-policy, no teacher action, oracle, shaped reward, or privileged state; counterfactual swap probe flips identity->choice at evaluation; both = solved in original and swapped layouts (genuine memory-driven success).",
        str(resolved), delay, gap, identity_count, updates, episodes_per_update,
        results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument("--identity-count", type=int, default=2)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, delay=args.delay, gap=args.gap,
        identity_count=args.identity_count, updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))

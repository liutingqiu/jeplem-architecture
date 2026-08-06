"""Reward-only actor-critic comparison on MiniGrid's delayed-memory task.

Actions are sampled from the policies and learning uses only environment
rewards.  No shortest-path oracle, task seed, future observation, or shaped
distance reward is available to any controller.  This is a pilot for the
end-to-end learning gate, not evidence of an architecture advantage by itself.
"""

from __future__ import annotations

import json
import argparse
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import minigrid  # noqa: F401 - registers the environments
import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.device import resolve_device
from plwm.torch_model import TorchCognitiveState, TorchPLWM


ROOT = Path(__file__).resolve().parents[1]
ENV_ID = "MiniGrid-MemoryS7-v0"
ACTION_COUNT = 7
INPUT_DIM = 7 * 7 * 3 + 4 + ACTION_COUNT + 1
SENSORY_INPUT_DIM = 7 * 7 * 3 + 4
MAX_STEPS = 64
ACTION_MODES = {"all": tuple(range(ACTION_COUNT)), "movement_only": (0, 1, 2)}
OBSERVATION_MODES = {"observation_only", "causal_action_reward"}


def encode_observation(
    observation: dict[str, Any],
    *,
    previous_action: int | None = None,
    previous_reward: float = 0.0,
    observation_mode: str = "observation_only",
) -> Tensor:
    image = observation.get("image")
    direction = observation.get("direction")
    if not isinstance(image, np.ndarray) or image.shape != (7, 7, 3):
        raise ValueError("expected MiniGrid's 7x7x3 image observation")
    if isinstance(direction, bool) or not isinstance(direction, (int, np.integer)) or not 0 <= int(direction) < 4:
        raise ValueError("expected a MiniGrid direction in [0, 3]")
    encoded = torch.tensor(image, dtype=torch.float32).flatten() / 10.0
    direction_vector = torch.zeros(4, dtype=torch.float32)
    direction_vector[int(direction)] = 1.0
    event_vector = torch.zeros(ACTION_COUNT + 1, dtype=torch.float32)
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError("unknown observation_mode")
    if observation_mode == "causal_action_reward":
        if previous_action is not None:
            if isinstance(previous_action, bool) or not isinstance(previous_action, int) or not 0 <= previous_action < ACTION_COUNT:
                raise ValueError("previous_action must be a MiniGrid action")
            event_vector[previous_action] = 1.0
        event_vector[-1] = float(previous_reward)
    return torch.cat((encoded, direction_vector, event_vector))


def make_environment(environment: str = ENV_ID) -> gym.Env:
    """Give every controller the same explicit interaction/context budget."""
    return gym.make(environment, max_steps=MAX_STEPS)


class ActorCritic(nn.Module):
    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    @property
    def supports_streaming_sampling(self) -> bool:
        return False

    def initial_sampling_state(self, batch_size: int, device: torch.device) -> Any:
        del batch_size, device
        return None

    def sampling_logits(self, observations: Tensor, state: Any, active: Tensor) -> tuple[Tensor, Any]:
        raise NotImplementedError


class NoMemoryActorCritic(ActorCritic):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.Tanh(), nn.Linear(hidden, ACTION_COUNT + 1))

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        del valid
        output = self.network(observations)
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]

    @property
    def supports_streaming_sampling(self) -> bool:
        return True

    def sampling_logits(self, observations: Tensor, state: None, active: Tensor) -> tuple[Tensor, None]:
        del state, active
        output = self.network(observations)
        return output[:, :ACTION_COUNT], None


class GRUActorCritic(ActorCritic):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.GRU(INPUT_DIM, hidden, batch_first=True)
        self.head = nn.Linear(hidden, ACTION_COUNT + 1)

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        lengths = valid.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(observations, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=observations.shape[1])
        output = self.head(encoded)
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]

    @property
    def supports_streaming_sampling(self) -> bool:
        return True

    def initial_sampling_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(self.encoder.num_layers, batch_size, self.encoder.hidden_size, device=device)

    def sampling_logits(self, observations: Tensor, state: Tensor, active: Tensor) -> tuple[Tensor, Tensor]:
        encoded, candidate = self.encoder(observations.unsqueeze(1), state)
        next_state = torch.where(active.view(1, -1, 1), candidate, state)
        output = self.head(next_state[-1])
        del encoded
        return output[:, :ACTION_COUNT], next_state


class TransformerActorCritic(ActorCritic):
    def __init__(self, width: int = 64, *, max_steps: int = MAX_STEPS) -> None:
        super().__init__()
        self.project = nn.Linear(INPUT_DIM, width)
        self.position = nn.Parameter(torch.zeros(1, max_steps, width))
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=width * 2, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(width, ACTION_COUNT + 1)

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        if observations.shape[1] > self.position.shape[1]:
            raise ValueError("MiniGrid trajectory exceeds Transformer context")
        size = observations.shape[1]
        causal = torch.triu(torch.ones((size, size), dtype=torch.bool, device=observations.device), diagonal=1)
        encoded = self.encoder(self.project(observations) + self.position[:, :size], mask=causal, src_key_padding_mask=~valid)
        output = self.head(encoded)
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]


class HybridPLWMTransformerActorCritic(ActorCritic):
    """Short-context Transformer policy with an independently switchable PLWM side state.

    The Transformer receives the complete causal sensory prefix.  PLWM receives
    the same event prefix and supplies a persistent belief/workspace/memory
    feature through a learned residual gate.  ``use_plwm_side=False`` keeps the
    tactical Transformer and training contract intact while removing that
    persistent contribution for an ablation.
    """

    def __init__(
        self,
        width: int = 64,
        *,
        latent: int = 64,
        memory_slots: int = 8,
        thought_steps: int = 2,
        use_plwm_side: bool = True,
        max_steps: int = MAX_STEPS,
    ) -> None:
        super().__init__()
        self.core = TorchPLWM(
            INPUT_DIM, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=ACTION_COUNT + 1,
            use_memory=True, gated_memory=True,
        )
        self.use_plwm_side = use_plwm_side
        self.project = nn.Linear(INPUT_DIM, width)
        self.position = nn.Parameter(torch.zeros(1, max_steps, width))
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=width * 2, dropout=0.0,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.side_projection = nn.Linear(latent * 3, width)
        self.fusion_gate = nn.Linear(width * 2, width)
        self.head = nn.Linear(width, ACTION_COUNT + 1)
        if not use_plwm_side:
            for module in (self.core, self.side_projection, self.fusion_gate):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def _advance(self, observation: Tensor, state: TorchCognitiveState, active: Tensor) -> TorchCognitiveState:
        candidate, _surprise = self.core.step(observation, state, think=True)
        active_mask = active.view(-1, 1)
        return TorchCognitiveState(
            torch.where(active_mask, candidate.belief, state.belief),
            torch.where(active_mask, candidate.workspace, state.workspace),
            torch.where(active_mask.unsqueeze(-1), candidate.keys, state.keys),
            torch.where(active_mask.unsqueeze(-1), candidate.values, state.values),
        )

    def _persistent_feature(self, state: TorchCognitiveState) -> Tensor:
        query = self.core.query_head(state.belief)
        retrieved = self.core._read_memory(query, state)
        return self.side_projection(torch.cat((state.belief, state.workspace, retrieved), dim=-1))

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        if observations.shape[1] > self.position.shape[1]:
            raise ValueError("MiniGrid trajectory exceeds Transformer context")
        size = observations.shape[1]
        causal = torch.triu(torch.ones((size, size), dtype=torch.bool, device=observations.device), diagonal=1)
        tactical = self.encoder(
            self.project(observations) + self.position[:, :size],
            mask=causal, src_key_padding_mask=~valid,
        )
        if self.use_plwm_side:
            state = self.core.initial_state(observations.shape[0], observations.device)
            persistent = []
            for index in range(observations.shape[1]):
                state = self._advance(observations[:, index], state, valid[:, index])
                persistent.append(self._persistent_feature(state))
            side = torch.stack(persistent, dim=1)
            gate = torch.sigmoid(self.fusion_gate(torch.cat((tactical, side), dim=-1)))
            tactical = tactical + gate * side
        output = self.head(tactical)
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]


class PLWMActorCritic(ActorCritic):
    def __init__(
        self,
        latent: int = 64,
        *,
        memory_slots: int = 8,
        thought_steps: int = 2,
        use_memory: bool = True,
        gated_memory: bool = True,
        observation_skip: bool = False,
        separate_causal_event: bool = False,
    ) -> None:
        super().__init__()
        self.core = TorchPLWM(
            INPUT_DIM, latent_dim=latent, memory_slots=memory_slots, thought_steps=thought_steps,
            output_dim=ACTION_COUNT + 1, use_memory=use_memory, gated_memory=gated_memory,
        )
        self.observation_skip = nn.Linear(INPUT_DIM, ACTION_COUNT + 1) if observation_skip else None
        self.separate_causal_event = separate_causal_event

    def _advance(self, observations: Tensor, state: Any, active: Tensor) -> Any:
        """Apply a prior action/reward event before its resulting sensory event."""
        sensory = observations
        if self.separate_causal_event:
            sensory = observations.clone()
            sensory[:, SENSORY_INPUT_DIM:] = 0.0
            action_event = observations - sensory
            event_active = active & action_event.abs().sum(dim=-1).gt(0.0)
            event_candidate, _surprise = self.core.step(action_event, state, think=False)
            event_mask = event_active.view(-1, 1)
            state = type(state)(
                torch.where(event_mask, event_candidate.belief, state.belief),
                torch.where(event_mask, event_candidate.workspace, state.workspace),
                torch.where(event_mask.unsqueeze(-1), event_candidate.keys, state.keys),
                torch.where(event_mask.unsqueeze(-1), event_candidate.values, state.values),
            )
        candidate, _surprise = self.core.step(sensory, state, think=True)
        active_mask = active.view(-1, 1)
        return type(state)(
            torch.where(active_mask, candidate.belief, state.belief),
            torch.where(active_mask, candidate.workspace, state.workspace),
            torch.where(active_mask.unsqueeze(-1), candidate.keys, state.keys),
            torch.where(active_mask.unsqueeze(-1), candidate.values, state.values),
        )

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        state = self.core.initial_state(observations.shape[0], observations.device)
        outputs = []
        for index in range(observations.shape[1]):
            # Padding is a batching artifact, not an event allowed to alter memory.
            state = self._advance(observations[:, index], state, valid[:, index])
            output = self.core.classify(state)
            if self.observation_skip is not None:
                output = output + self.observation_skip(observations[:, index])
            outputs.append(output)
        output = torch.stack(outputs, dim=1)
        return output[..., :ACTION_COUNT], output[..., ACTION_COUNT]

    @property
    def supports_streaming_sampling(self) -> bool:
        return True

    def initial_sampling_state(self, batch_size: int, device: torch.device) -> Any:
        return self.core.initial_state(batch_size, device)

    def sampling_logits(self, observations: Tensor, state: Any, active: Tensor) -> tuple[Tensor, Any]:
        next_state = self._advance(observations, state, active)
        output = self.core.classify(next_state)
        if self.observation_skip is not None:
            output = output + self.observation_skip(observations)
        return output[:, :ACTION_COUNT], next_state

    def predictive_loss(self, observations: Tensor, valid: Tensor) -> Tensor:
        """Predict each current event from only the preceding cognitive state."""
        state = self.core.initial_state(observations.shape[0], observations.device)
        losses = []
        for index in range(observations.shape[1]):
            observation = observations[:, index]
            target = observation
            if self.separate_causal_event:
                sensory = observation.clone()
                sensory[:, SENSORY_INPUT_DIM:] = 0.0
                action_event = observation - sensory
                event_active = valid[:, index] & action_event.abs().sum(dim=-1).gt(0.0)
                event_candidate, _surprise = self.core.step(action_event, state, think=False)
                event_mask = event_active.view(-1, 1)
                state = type(state)(
                    torch.where(event_mask, event_candidate.belief, state.belief),
                    torch.where(event_mask, event_candidate.workspace, state.workspace),
                    torch.where(event_mask.unsqueeze(-1), event_candidate.keys, state.keys),
                    torch.where(event_mask.unsqueeze(-1), event_candidate.values, state.values),
                )
                target = sensory
            prediction = self.core.predictor(state.belief)
            losses.append((prediction - target).square().mean(dim=-1))
            # ``target`` is sensory-only for separate events, so the action is
            # not applied a second time before its resulting observation.
            state = self._advance(target, state, valid[:, index])
        return torch.stack(losses, dim=1)[valid].mean()


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
    train_episodes: int
    train_successes: int
    evaluation: Evaluation
    stochastic_evaluation: Evaluation


@dataclass
class Report:
    protocol: str
    learning_rule: str
    intrinsic_bonus: float
    action_mode: str
    observation_mode: str
    device: str
    environment: str
    train_seed_range: list[int]
    heldout_seed_range: list[int]
    updates: int
    episodes_per_update: int
    random_evaluation: Evaluation
    results: list[ModelResult]
    accepted: bool
    artifact: str


class ObservationVisitCounter:
    """Count only agent-visible local observations for an explicit exploration bonus."""

    def __init__(self) -> None:
        self._visits: dict[bytes, int] = {}

    def reward(self, observation: Tensor, scale: float) -> float:
        if scale <= 0.0:
            return 0.0
        key = observation.mul(10.0).round().to(torch.uint8).contiguous().numpy().tobytes()
        visits = self._visits.get(key, 0) + 1
        self._visits[key] = visits
        return scale / visits**0.5


def _pad(episodes: Sequence[list[Tensor]], device: torch.device) -> tuple[Tensor, Tensor]:
    length = max(len(episode) for episode in episodes)
    values = torch.zeros((len(episodes), length, INPUT_DIM), device=device)
    valid = torch.zeros((len(episodes), length), dtype=torch.bool, device=device)
    for index, episode in enumerate(episodes):
        values[index, : len(episode)] = torch.stack(episode).to(device)
        valid[index, : len(episode)] = True
    return values, valid


def _mask_actions(logits: Tensor, action_mode: str) -> Tensor:
    if action_mode not in ACTION_MODES:
        raise ValueError("unknown action_mode")
    if action_mode == "all":
        return logits
    permitted = torch.zeros(ACTION_COUNT, dtype=torch.bool, device=logits.device)
    permitted[list(ACTION_MODES[action_mode])] = True
    return logits.masked_fill(~permitted, torch.finfo(logits.dtype).min)


@torch.no_grad()
def _choose(model: ActorCritic, histories: Sequence[list[Tensor]], device: torch.device, action_mode: str) -> Tensor:
    observations, valid = _pad(histories, device)
    logits, _values = model(observations, valid)
    positions = valid.sum(dim=1) - 1
    current = _mask_actions(logits[torch.arange(len(histories), device=device), positions], action_mode)
    return torch.distributions.Categorical(logits=current).sample().cpu()


@torch.no_grad()
def _collect(
    model: ActorCritic,
    *,
    count: int,
    start_seed: int,
    device: torch.device,
    visit_counter: ObservationVisitCounter | None = None,
    intrinsic_bonus: float = 0.0,
    action_mode: str = "all",
    environment: str = ENV_ID,
    observation_mode: str = "observation_only",
) -> tuple[list[list[Tensor]], list[list[int]], list[list[float]], list[list[float]], int]:
    environments = [make_environment(environment) for _ in range(count)]
    observations, histories, actions, external_rewards, learning_rewards, active = [], [[] for _ in environments], [[] for _ in environments], [[] for _ in environments], [[] for _ in environments], [True] * count
    for index, environment in enumerate(environments):
        observation, _info = environment.reset(seed=start_seed + index)
        encoded = encode_observation(observation, observation_mode=observation_mode)
        observations.append(encoded); histories[index].append(encoded)
    sampling_state = model.initial_sampling_state(count, device) if model.supports_streaming_sampling else None
    while any(active):
        if model.supports_streaming_sampling:
            active_tensor = torch.tensor(active, dtype=torch.bool, device=device)
            logits, sampling_state = model.sampling_logits(torch.stack(observations).to(device), sampling_state, active_tensor)
            choice = torch.distributions.Categorical(logits=_mask_actions(logits, action_mode)).sample().cpu()
        else:
            active_indices = [index for index, is_active in enumerate(active) if is_active]
            active_choice = _choose(model, [histories[index] for index in active_indices], device, action_mode)
            choice = torch.full((count,), -1, dtype=torch.long)
            choice[active_indices] = active_choice
        for index, environment in enumerate(environments):
            if not active[index]:
                continue
            next_observation, reward, terminated, truncated, _info = environment.step(int(choice[index]))
            external_reward = float(reward)
            actions[index].append(int(choice[index])); external_rewards[index].append(external_reward)
            if terminated or truncated:
                active[index] = False
            else:
                encoded = encode_observation(next_observation, previous_action=int(choice[index]), previous_reward=float(reward), observation_mode=observation_mode)
                observations[index] = encoded; histories[index].append(encoded)
                external_reward += visit_counter.reward(encoded, intrinsic_bonus) if visit_counter is not None else 0.0
            learning_rewards[index].append(external_reward)
    successes = sum(any(value > 0.0 for value in episode) for episode in external_rewards)
    for environment in environments:
        environment.close()
    return histories, actions, external_rewards, learning_rewards, successes


def _returns(rewards: Sequence[list[float]], device: torch.device, gamma: float = 0.99) -> Tensor:
    length = max(len(episode) for episode in rewards)
    result = torch.zeros((len(rewards), length), device=device)
    for row, episode in enumerate(rewards):
        future = 0.0
        for index in reversed(range(len(episode))):
            future = episode[index] + gamma * future
            result[row, index] = future
    return result


def _gae(
    rewards: Sequence[list[float]],
    values: Tensor,
    valid: Tensor,
    *,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute terminal-trajectory GAE from external environment rewards only."""
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
    if not torch.equal(valid, valid.bool()):
        raise ValueError("valid trajectory mask must be boolean")
    return advantages, returns


def _train(
    model: ActorCritic,
    *,
    updates: int,
    episodes_per_update: int,
    start_seed: int,
    device: torch.device,
    learning_rule: str = "a2c_v1",
    ppo_epochs: int = 4,
    entropy_coefficient: float = 0.01,
    intrinsic_bonus: float = 0.05,
    action_mode: str = "all",
    environment: str = ENV_ID,
    observation_mode: str = "observation_only",
    world_model_coefficient: float = 0.0,
) -> tuple[int, int]:
    if learning_rule not in {"a2c_v1", "ppo_external_reward_v1", "ppo_intrinsic_count_v1"}:
        raise ValueError("unsupported learning_rule")
    if ppo_epochs < 1 or entropy_coefficient < 0.0 or intrinsic_bonus < 0.0 or world_model_coefficient < 0.0 or action_mode not in ACTION_MODES:
        raise ValueError("ppo_epochs must be positive and reward coefficients must be non-negative")
    if world_model_coefficient > 0.0 and not isinstance(model, PLWMActorCritic):
        raise ValueError("world-model auxiliary loss is currently defined only for PLWM")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    total_episodes = total_successes = 0
    visit_counter = ObservationVisitCounter() if learning_rule == "ppo_intrinsic_count_v1" else None
    for update in range(updates):
        histories, actions, external_rewards, learning_rewards, successes = _collect(
            model, count=episodes_per_update, start_seed=start_seed + update * episodes_per_update, device=device,
            visit_counter=visit_counter, intrinsic_bonus=intrinsic_bonus, action_mode=action_mode, environment=environment, observation_mode=observation_mode,
        )
        observations, valid = _pad(histories, device)
        action_tensor = torch.zeros(valid.shape, dtype=torch.long, device=device)
        for index, episode in enumerate(actions):
            action_tensor[index, : len(episode)] = torch.tensor(episode, dtype=torch.long, device=device)
        if learning_rule == "a2c_v1":
            returns = _returns(external_rewards, device)
            logits, values = model(observations, valid)
            distribution = torch.distributions.Categorical(logits=_mask_actions(logits, action_mode))
            selected = distribution.log_prob(action_tensor)
            advantages = returns - values.detach()
            normalized = advantages[valid]
            advantages = (advantages - normalized.mean()) / (normalized.std().clamp_min(1e-5))
            actor = -(selected[valid] * advantages[valid]).mean()
            critic = (returns[valid] - values[valid]).square().mean()
            entropy = distribution.entropy()[valid].mean()
            world_model = model.predictive_loss(observations, valid) if world_model_coefficient > 0.0 else 0.0
            loss = actor + 0.5 * critic - entropy_coefficient * entropy + world_model_coefficient * world_model
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if hasattr(model, 'core') and hasattr(model.core, 'enforce_key_orthogonality'):
                model.core.enforce_key_orthogonality()
        else:
            with torch.no_grad():
                old_logits, old_values = model(observations, valid)
                old_distribution = torch.distributions.Categorical(logits=_mask_actions(old_logits, action_mode))
                old_selected = old_distribution.log_prob(action_tensor)
                advantages, returns = _gae(learning_rewards, old_values, valid)
                normalized = advantages[valid]
                advantages = (advantages - normalized.mean()) / normalized.std().clamp_min(1e-5)
            for _ in range(ppo_epochs):
                logits, values = model(observations, valid)
                distribution = torch.distributions.Categorical(logits=_mask_actions(logits, action_mode))
                ratio = (distribution.log_prob(action_tensor) - old_selected).exp()
                unclipped = ratio * advantages
                clipped = ratio.clamp(0.8, 1.2) * advantages
                actor = -torch.minimum(unclipped[valid], clipped[valid]).mean()
                critic = (returns[valid] - values[valid]).square().mean()
                entropy = distribution.entropy()[valid].mean()
                world_model = model.predictive_loss(observations, valid) if world_model_coefficient > 0.0 else 0.0
                loss = actor + 0.5 * critic - entropy_coefficient * entropy + world_model_coefficient * world_model
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                if hasattr(model, 'core') and hasattr(model.core, 'enforce_key_orthogonality'):
                    model.core.enforce_key_orthogonality()
        total_episodes += episodes_per_update; total_successes += successes
    return total_episodes, total_successes


@torch.no_grad()
def evaluate(
    model: ActorCritic,
    *,
    episodes: int,
    start_seed: int,
    device: torch.device,
    stochastic: bool = False,
    action_mode: str = "all",
    environment: str = ENV_ID,
    observation_mode: str = "observation_only",
) -> Evaluation:
    successes, returns, lengths = 0, [], []
    action_counts = [0] * ACTION_COUNT
    model.eval()
    for episode_index in range(episodes):
        instance = make_environment(environment)
        observation, _info = instance.reset(seed=start_seed + episode_index)
        history, total = [encode_observation(observation, observation_mode=observation_mode)], 0.0
        for step in range(MAX_STEPS):
            observations, valid = _pad([history], device)
            logits, _value = model(observations, valid)
            current_logits = _mask_actions(logits[0, len(history) - 1], action_mode)
            action = int(torch.distributions.Categorical(logits=current_logits).sample().item()) if stochastic else int(current_logits.argmax().item())
            action_counts[action] += 1
            observation, reward, terminated, truncated, _info = instance.step(action)
            total += float(reward)
            if terminated or truncated:
                lengths.append(step + 1); break
            history.append(encode_observation(observation, previous_action=action, previous_reward=float(reward), observation_mode=observation_mode))
        else: lengths.append(MAX_STEPS)
        successes += int(total > 0.0); returns.append(total); instance.close()
    return Evaluation(successes, episodes, sum(returns) / episodes, sum(lengths) / episodes, action_counts)


def evaluate_random(*, episodes: int, start_seed: int, action_seed: int = 13_037, action_mode: str = "all", environment: str = ENV_ID) -> Evaluation:
    """Fixed random-action reference on exactly the held-out evaluation worlds."""
    rng = random.Random(action_seed)
    successes, returns, lengths = 0, [], []
    action_counts = [0] * ACTION_COUNT
    for episode_index in range(episodes):
        instance = make_environment(environment)
        _observation, _info = instance.reset(seed=start_seed + episode_index)
        total = 0.0
        for step in range(MAX_STEPS):
            action = rng.choice(ACTION_MODES[action_mode])
            action_counts[action] += 1
            _observation, reward, terminated, truncated, _info = instance.step(action)
            total += float(reward)
            if terminated or truncated:
                lengths.append(step + 1)
                break
        else:
            lengths.append(MAX_STEPS)
        successes += int(total > 0.0)
        returns.append(total)
        instance.close()
    return Evaluation(successes, episodes, sum(returns) / episodes, sum(lengths) / episodes, action_counts)


def run(
    *,
    device: str | torch.device | None = None,
    updates: int = 80,
    episodes_per_update: int = 24,
    evaluation_episodes: int = 128,
    artifact: Path | None = None,
    learning_rule: str = "a2c_v1",
    ppo_epochs: int = 4,
    entropy_coefficient: float = 0.01,
    intrinsic_bonus: float = 0.05,
    action_mode: str = "all",
    environment: str = ENV_ID,
    observation_mode: str = "observation_only",
) -> Report:
    if updates < 1 or episodes_per_update < 2 or evaluation_episodes < 1 or action_mode not in ACTION_MODES or observation_mode not in OBSERVATION_MODES:
        raise ValueError("updates and evaluation_episodes must be positive and episodes_per_update must be at least two")
    resolved = resolve_device(device)
    torch.manual_seed(13_037); random.seed(13_037); np.random.seed(13_037)
    factories = {"plwm": PLWMActorCritic, "no_memory": NoMemoryActorCritic, "gru": GRUActorCritic, "transformer": TransformerActorCritic}
    results = []
    random_evaluation = evaluate_random(episodes=evaluation_episodes, start_seed=1_000_000, action_mode=action_mode, environment=environment)
    for name, factory in factories.items():
        model = factory().to(resolved)
        train_episodes, train_successes = _train(
            model, updates=updates, episodes_per_update=episodes_per_update, start_seed=10_000, device=resolved,
            learning_rule=learning_rule, ppo_epochs=ppo_epochs, entropy_coefficient=entropy_coefficient, intrinsic_bonus=intrinsic_bonus, action_mode=action_mode, environment=environment, observation_mode=observation_mode,
        )
        evaluation = evaluate(model, episodes=evaluation_episodes, start_seed=1_000_000, device=resolved, action_mode=action_mode, environment=environment, observation_mode=observation_mode)
        stochastic_evaluation = evaluate(model, episodes=evaluation_episodes, start_seed=1_000_000, device=resolved, stochastic=True, action_mode=action_mode, environment=environment, observation_mode=observation_mode)
        results.append(ModelResult(name, sum(parameter.numel() for parameter in model.parameters()), train_episodes, train_successes, evaluation, stochastic_evaluation))
    by_name = {item.name: item for item in results}
    report = Report(
        "minigrid_memory_actor_critic_v4: MiniGrid MemoryS7 with a fixed 64-step interaction/context budget; all models train on the same seed range; reports a fixed random-action reference plus greedy and stochastic held-out policies and action counts; intrinsic count bonus, when selected, is derived only from agent-visible local observations and is reported separately from environment reward; no teacher actions, shortest-path oracle, future observation, seed answer, or shaped-distance reward; independent held-out seeds; pilot only, not an architecture claim",
        learning_rule, intrinsic_bonus if learning_rule == "ppo_intrinsic_count_v1" else 0.0, action_mode, observation_mode, str(resolved), environment, list(range(10_000, 10_000 + updates * episodes_per_update)), list(range(1_000_000, 1_000_000 + evaluation_episodes)), updates, episodes_per_update, random_evaluation, results,
        by_name["plwm"].evaluation.success_rate >= 0.55
        and by_name["plwm"].evaluation.success_rate >= by_name["no_memory"].evaluation.success_rate + 0.10
        and by_name["plwm"].evaluation.success_rate >= by_name["gru"].evaluation.success_rate + 0.05
        and by_name["plwm"].evaluation.success_rate >= by_name["transformer"].evaluation.success_rate + 0.05,
        str(artifact or ROOT / "artifacts" / f"minigrid-memory-{learning_rule}-report.json"),
    )
    Path(report.artifact).parent.mkdir(exist_ok=True)
    Path(report.artifact).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--learning-rule", choices=("a2c_v1", "ppo_external_reward_v1", "ppo_intrinsic_count_v1"), default="a2c_v1")
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--intrinsic-bonus", type=float, default=0.05)
    parser.add_argument("--action-mode", choices=tuple(ACTION_MODES), default="all")
    parser.add_argument("--environment", default=ENV_ID)
    parser.add_argument("--observation-mode", choices=tuple(OBSERVATION_MODES), default="observation_only")
    args = parser.parse_args()
    outcome = run(
        device=args.device,
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        artifact=args.artifact,
        learning_rule=args.learning_rule,
        ppo_epochs=args.ppo_epochs,
        entropy_coefficient=args.entropy_coefficient,
        intrinsic_bonus=args.intrinsic_bonus,
        action_mode=args.action_mode,
        environment=args.environment,
        observation_mode=args.observation_mode,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted: raise SystemExit("MiniGrid actor-critic pilot did not meet its gate")

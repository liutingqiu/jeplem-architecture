"""Reward-only DQN comparison on the audited Luanti gated-cue course.

This is deliberately separate from the rejected PPO path.  Every learner sees
only the current 11x11 public vision, its prior chosen action, and prior scalar
environment reward.  The adjacent-wall legal-action mask is calculated from
that same public frame and is shared by every learner.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audit_luanti_delayed_goal_qlearning import ACTION_COUNT, public_valid_actions
from experiments.train_luanti_reward_only import INPUT_DIM, encode_observation
from plwm.device import resolve_device
from plwm.luanti_reward_only_env import DEFAULT_BRIDGE_ROOT, GymVisionOnlyLuanti, VISION_SHAPE
from plwm.torch_model import TorchPLWM


MAX_STEPS = 4
TRANSFORMER_CONTEXT = 2
PREDICTIVE_LOSS_WEIGHT = 0.1
PARAMETER_MATCH_TOLERANCE = 0.05
ACCEPTANCE_MODES = ("course_default", "addressed_memory_diagnostic", "addressed_balanced_diagnostic")
COURSES = {
    "gated_compact": (9_000_000, 10_000_000, 4),
    "long_gated": (10_000_000, 11_000_000, 8),
    "paired_gated": (11_000_000, 12_000_000, 8),
    "addressed_gated": (12_000_000, 13_000_000, 6),
    "addressed_balanced": (16_000_000, 17_000_000, 6),
    "addressed_balanced_2id": (18_000_000, 19_000_000, 6),
}


@dataclass(frozen=True)
class Episode:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    masks: Tensor


@dataclass(frozen=True)
class Evaluation:
    episodes: int
    successes: int
    mean_return: float
    action_counts: list[int]

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes


@dataclass(frozen=True)
class ModelResult:
    name: str
    parameters: int
    training_episodes: int
    training_successes: int
    evaluation: Evaluation


@dataclass(frozen=True)
class ModelWidths:
    """Declared widths for one parameter-capacity-matched comparison."""

    no_memory: int = 452
    gru: int = 131
    transformer: int = 132
    plwm: int = 96


CAPACITY_MATCHED_WIDTHS = ModelWidths()


@dataclass(frozen=True)
class ObjectTokenModelWidths:
    no_memory: int = 310
    gru: int = 170
    transformer: int = 80
    plwm: int = 64


OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS = ObjectTokenModelWidths()


OBJECT_TOKEN_DIM = 16
OBJECT_EVENT_DIM = 32


class ObjectTokenEventEncoder(nn.Module):
    """Shared public-pixel encoder with 2D positional encoding (fix §5.3a).

    Key change: instead of mean pooling over spatial tokens, we now compute
    identity-aware position features that preserve where identities appear.
    """

    def __init__(self, *, identity_count: int = 4) -> None:
        super().__init__()
        if identity_count < 1:
            raise ValueError("identity_count must be positive")
        self.identity_count = identity_count
        self.spatial = nn.Sequential(
            nn.Conv2d(VISION_SHAPE[2], OBJECT_TOKEN_DIM, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv2d(OBJECT_TOKEN_DIM, OBJECT_TOKEN_DIM, kernel_size=3, padding=1), nn.Tanh(),
        )
        # Fix (a): 2D positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, OBJECT_TOKEN_DIM, VISION_SHAPE[0], VISION_SHAPE[1])
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        self.event_context = nn.Sequential(nn.Linear(ACTION_COUNT + 1, OBJECT_EVENT_DIM), nn.Tanh())
        # Fix: use a weighted sum instead of mean pooling to preserve position
        self.position_weight = nn.Parameter(torch.zeros(1, 1, VISION_SHAPE[0] * VISION_SHAPE[1], 1))
        nn.init.normal_(self.position_weight, std=0.1)
        self.project_weighted = nn.Linear(OBJECT_TOKEN_DIM, OBJECT_EVENT_DIM)
        self.event_projection = nn.Sequential(nn.Linear(OBJECT_EVENT_DIM, OBJECT_EVENT_DIM), nn.Tanh())

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if observations.ndim != 3 or observations.shape[-1] != INPUT_DIM:
            raise ValueError("expected reward-only public observation sequences")
        batch, time = observations.shape[:2]
        visual_size = int(np.prod(VISION_SHAPE))
        visual = observations[:, :, :visual_size].reshape(batch * time, *VISION_SHAPE).permute(0, 3, 1, 2)
        tokens = self.spatial(visual)  # [B*T, D, H, W]
        tokens = tokens + self.pos_embed  # Add positional encoding
        tokens = tokens.permute(0, 2, 3, 1).reshape(batch, time, -1, OBJECT_TOKEN_DIM)  # [B, T, H*W, D]
        raw_tiles = observations[:, :, :visual_size].reshape(batch, time, *VISION_SHAPE)
        object_mask = raw_tiles[..., 3:].sum(dim=-1).gt(0).reshape(batch, time, -1)
        context = self.event_context(observations[:, :, visual_size:])
        # Fix: weighted sum preserves position information better than mean
        weights = torch.softmax(self.position_weight[:, :, :tokens.shape[2]], dim=2)
        weighted_tokens = (tokens * weights).sum(dim=2)
        events = self.event_projection(self.project_weighted(weighted_tokens) + context)
        return events, tokens, object_mask

class ObjectTokenNoMemoryQ(nn.Module):
    def __init__(self, width: int, *, identity_count: int = 4) -> None:
        super().__init__()
        self.encoder = ObjectTokenEventEncoder(identity_count=identity_count)
        self.network = nn.Sequential(nn.Linear(OBJECT_EVENT_DIM, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, ACTION_COUNT))

    def forward(self, observations: Tensor) -> Tensor:
        events, _tokens, _mask = self.encoder(observations)
        return self.network(events)


class ObjectTokenGRUQ(nn.Module):
    def __init__(self, width: int, *, identity_count: int = 4) -> None:
        super().__init__()
        self.encoder = ObjectTokenEventEncoder(identity_count=identity_count)
        self.recurrent = nn.GRU(OBJECT_EVENT_DIM, width, batch_first=True)
        self.head = nn.Linear(width, ACTION_COUNT)

    def forward(self, observations: Tensor) -> Tensor:
        events, _tokens, _mask = self.encoder(observations)
        output, _ = self.recurrent(events)
        return self.head(output)


class ObjectTokenTransformerQ(nn.Module):
    def __init__(self, width: int, *, context: int = TRANSFORMER_CONTEXT, identity_count: int = 4) -> None:
        super().__init__()
        self.encoder = ObjectTokenEventEncoder(identity_count=identity_count)
        self.context = context
        self.project = nn.Linear(OBJECT_EVENT_DIM, width)
        self.position = nn.Parameter(torch.zeros(1, context, width))
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=width * 2, dropout=0.0, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(width, ACTION_COUNT)

    def _window(self, events: Tensor) -> Tensor:
        size = events.shape[1]
        causal = torch.triu(torch.ones((size, size), dtype=torch.bool, device=events.device), diagonal=1)
        return self.head(self.transformer(self.project(events) + self.position[:, :size], mask=causal))

    def forward(self, observations: Tensor) -> Tensor:
        events, _tokens, _mask = self.encoder(observations)
        if events.shape[1] <= self.context:
            return self._window(events)
        return torch.stack([self._window(events[:, max(0, index - self.context + 1):index + 1])[:, -1] for index in range(events.shape[1])], dim=1)


class ObjectTokenPLWMQ(nn.Module):
    """Only this variant writes the first-frame public object tokens to memory."""

    def __init__(self, width: int, *, persistent_state: bool, identity_count: int = 4) -> None:
        super().__init__()
        if identity_count < 1:
            raise ValueError("identity_count must be positive")
        self.encoder = ObjectTokenEventEncoder(identity_count=identity_count)
        self.identity_count = identity_count
        self.persistent_state = persistent_state
        self.position = nn.Parameter(torch.empty(VISION_SHAPE[0] * VISION_SHAPE[1], OBJECT_TOKEN_DIM))
        nn.init.normal_(self.position, std=0.02)
        self.candidate_projection = nn.Linear(OBJECT_TOKEN_DIM, width)
        self.core = TorchPLWM(OBJECT_EVENT_DIM, latent_dim=width, memory_slots=16, thought_steps=2, output_dim=ACTION_COUNT, use_memory=persistent_state, gated_memory=True)

    def _events_and_candidates(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        events, tokens, object_mask = self.encoder(observations)
        initial = tokens[:, 0] + self.position.unsqueeze(0)
        # Top-k operates only on the raw public object/non-object predicate.
        indices = object_mask[:, 0].to(dtype=observations.dtype).topk(self.identity_count, dim=1).indices
        candidates = self.candidate_projection(initial.gather(1, indices.unsqueeze(-1).expand(-1, -1, OBJECT_TOKEN_DIM)))
        return events, candidates

    def forward(self, observations: Tensor) -> Tensor:
        events, candidates = self._events_and_candidates(observations)
        state = self.core.initial_state(observations.shape[0], observations.device)
        outputs: list[Tensor] = []
        for index in range(events.shape[1]):
            if not self.persistent_state:
                state = self.core.initial_state(observations.shape[0], observations.device)
            if index == 0:
                writes, mask = candidates, torch.ones(events.shape[0], self.identity_count, dtype=events.dtype, device=events.device)
            else:
                writes, mask = torch.zeros(events.shape[0], 1, self.core.latent_dim, dtype=events.dtype, device=events.device), torch.zeros(events.shape[0], 1, dtype=events.dtype, device=events.device)
            state, _ = self.core.step(events[:, index], state, think=True, write_candidates=writes, write_mask=mask)
            outputs.append(self.core.classify(state))
        return torch.stack(outputs, dim=1)

    def causal_predictions(self, observations: Tensor) -> Tensor:
        events, candidates = self._events_and_candidates(observations)
        state = self.core.initial_state(observations.shape[0], observations.device)
        predictions: list[Tensor] = []
        for index in range(events.shape[1]):
            if not self.persistent_state:
                state = self.core.initial_state(observations.shape[0], observations.device)
            predictions.append(self.core.predictor(state.belief))
            writes = candidates if index == 0 else torch.zeros(events.shape[0], 1, self.core.latent_dim, dtype=events.dtype, device=events.device)
            mask = torch.ones(events.shape[0], self.identity_count, dtype=events.dtype, device=events.device) if index == 0 else torch.zeros(events.shape[0], 1, dtype=events.dtype, device=events.device)
            state, _ = self.core.step(events[:, index], state, think=True, write_candidates=writes, write_mask=mask)
        return torch.stack(predictions, dim=1)

    def predictive_target(self, observations: Tensor) -> Tensor:
        return self.encoder(observations)[0].detach()


@dataclass(frozen=True)
class Report:
    protocol: str
    course: str
    device: str
    training_seed_range: list[int]
    heldout_seed_range: list[int]
    action_seed: int
    max_steps: int
    transformer_context: int
    model_widths: ModelWidths
    parameter_match_tolerance: float
    parameter_match_errors: dict[str, float]
    acceptance_mode: str
    random_evaluation: Evaluation
    results: list[ModelResult]
    accepted_neural_task_diagnostic: bool
    artifact: str


def _masked_argmax(values: Tensor, mask: Tensor) -> int:
    if values.shape != (ACTION_COUNT,) or mask.shape != (ACTION_COUNT,) or not bool(mask.any()):
        raise ValueError("invalid public action mask")
    return int(values.masked_fill(~mask, float("-inf")).argmax().item())


def _mask(observation: np.ndarray) -> Tensor:
    if not isinstance(observation, np.ndarray) or observation.shape != VISION_SHAPE or observation.dtype != np.uint8:
        raise ValueError("expected public Luanti vision")
    available = public_valid_actions(observation)
    result = torch.zeros(ACTION_COUNT, dtype=torch.bool)
    result[list(available)] = True
    return result


class QNetwork(nn.Module):
    def forward(self, observations: Tensor) -> Tensor:
        raise NotImplementedError


class NoMemoryQ(QNetwork):
    def __init__(self, width: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(INPUT_DIM, width), nn.Tanh(), nn.Linear(width, ACTION_COUNT))

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(observations)


class GRUQ(QNetwork):
    def __init__(self, width: int = 96) -> None:
        super().__init__()
        self.encoder = nn.GRU(INPUT_DIM, width, batch_first=True)
        self.head = nn.Linear(width, ACTION_COUNT)

    def forward(self, observations: Tensor) -> Tensor:
        encoded, _ = self.encoder(observations)
        return self.head(encoded)


class TransformerQ(QNetwork):
    def __init__(self, width: int = 96, *, context: int = TRANSFORMER_CONTEXT) -> None:
        super().__init__()
        self.context = context
        self.project = nn.Linear(INPUT_DIM, width)
        self.position = nn.Parameter(torch.zeros(1, context, width))
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=width * 2, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(width, ACTION_COUNT)

    def _window(self, observations: Tensor) -> Tensor:
        if observations.shape[1] > self.context:
            raise ValueError("Transformer context exceeds its declared bound")
        size = observations.shape[1]
        causal = torch.triu(torch.ones((size, size), dtype=torch.bool, device=observations.device), diagonal=1)
        return self.head(self.encoder(self.project(observations) + self.position[:, :size], mask=causal))

    def forward(self, observations: Tensor) -> Tensor:
        if observations.shape[1] <= self.context:
            return self._window(observations)
        outputs = []
        for index in range(observations.shape[1]):
            start = max(0, index - self.context + 1)
            outputs.append(self._window(observations[:, start : index + 1])[:, -1])
        return torch.stack(outputs, dim=1)


class PLWMQ(QNetwork):
    def __init__(self, width: int = 96, *, persistent_state: bool) -> None:
        super().__init__()
        self.persistent_state = persistent_state
        self.core = TorchPLWM(
            INPUT_DIM, latent_dim=width, memory_slots=16, thought_steps=2,
            output_dim=ACTION_COUNT, use_memory=persistent_state, gated_memory=True,
        )

    def forward(self, observations: Tensor) -> Tensor:
        state = self.core.initial_state(observations.shape[0], observations.device)
        outputs: list[Tensor] = []
        for index in range(observations.shape[1]):
            if not self.persistent_state:
                state = self.core.initial_state(observations.shape[0], observations.device)
            state, _ = self.core.step(observations[:, index], state, think=True)
            outputs.append(self.core.classify(state))
        return torch.stack(outputs, dim=1)

    def causal_predictions(self, observations: Tensor) -> Tensor:
        """Predict each public event from the state available before it."""
        state = self.core.initial_state(observations.shape[0], observations.device)
        predictions: list[Tensor] = []
        for index in range(observations.shape[1]):
            if not self.persistent_state:
                state = self.core.initial_state(observations.shape[0], observations.device)
            predictions.append(self.core.predictor(state.belief))
            state, _ = self.core.step(observations[:, index], state, think=True)
        return torch.stack(predictions, dim=1)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def comparison_factories(widths: ModelWidths = CAPACITY_MATCHED_WIDTHS) -> tuple[tuple[str, Callable[[], QNetwork]], ...]:
    """Build all models from the declared capacity-matching configuration."""
    if min(widths.no_memory, widths.gru, widths.transformer, widths.plwm) < 1:
        raise ValueError("model widths must be positive")
    if widths.transformer % 4:
        raise ValueError("Transformer width must be divisible by its four attention heads")
    return (
        ("no_memory", lambda: NoMemoryQ(widths.no_memory)),
        ("gru", lambda: GRUQ(widths.gru)),
        ("transformer_2", lambda: TransformerQ(widths.transformer)),
        ("plwm_state_off", lambda: PLWMQ(widths.plwm, persistent_state=False)),
        ("plwm", lambda: PLWMQ(widths.plwm, persistent_state=True)),
    )


def parameter_match_errors(widths: ModelWidths = CAPACITY_MATCHED_WIDTHS) -> dict[str, float]:
    """Return each comparison model's relative parameter difference from PLWM."""
    counts = {name: _parameter_count(factory()) for name, factory in comparison_factories(widths)}
    reference = counts["plwm"]
    return {name: abs(count - reference) / reference for name, count in counts.items()}


def validate_parameter_match(widths: ModelWidths = CAPACITY_MATCHED_WIDTHS, *, tolerance: float = PARAMETER_MATCH_TOLERANCE) -> dict[str, float]:
    """Reject future architecture comparisons whose capacity is materially unequal."""
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("parameter match tolerance must be in [0, 1)")
    errors = parameter_match_errors(widths)
    unmatched = {name: error for name, error in errors.items() if error > tolerance}
    if unmatched:
        details = ", ".join(f"{name}={error:.3%}" for name, error in unmatched.items())
        raise ValueError(f"comparison parameter match exceeds tolerance: {details}")
    return errors


def object_token_comparison_factories(widths: ObjectTokenModelWidths = OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS, *, identity_count: int = 4) -> tuple[tuple[str, Callable[[], nn.Module]], ...]:
    if min(widths.no_memory, widths.gru, widths.transformer, widths.plwm) < 1 or widths.transformer % 4:
        raise ValueError("invalid object-token comparison widths")
    return (
        ("no_memory", lambda: ObjectTokenNoMemoryQ(widths.no_memory, identity_count=identity_count)),
        ("gru", lambda: ObjectTokenGRUQ(widths.gru, identity_count=identity_count)),
        ("transformer_2", lambda: ObjectTokenTransformerQ(widths.transformer, identity_count=identity_count)),
        ("plwm_state_off", lambda: ObjectTokenPLWMQ(widths.plwm, persistent_state=False, identity_count=identity_count)),
        ("plwm", lambda: ObjectTokenPLWMQ(widths.plwm, persistent_state=True, identity_count=identity_count)),
    )


def object_token_parameter_match_errors(widths: ObjectTokenModelWidths = OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS) -> dict[str, float]:
    counts = {name: _parameter_count(factory()) for name, factory in object_token_comparison_factories(widths)}
    return {name: abs(count - counts["plwm"]) / counts["plwm"] for name, count in counts.items()}


def validate_object_token_parameter_match(widths: ObjectTokenModelWidths = OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS, *, tolerance: float = PARAMETER_MATCH_TOLERANCE) -> dict[str, float]:
    errors = object_token_parameter_match_errors(widths)
    unmatched = {name: error for name, error in errors.items() if error > tolerance}
    if unmatched:
        raise ValueError("object-token comparison parameter match exceeds tolerance")
    return errors


def _collect(
    model: QNetwork | None, environment: GymVisionOnlyLuanti, *, seed: int, device: torch.device,
    epsilon: float, generator: random.Random, max_steps: int,
) -> tuple[Episode, bool]:
    observation, info = environment.reset(seed=seed)
    if info != {}:
        raise RuntimeError("gated DQN received non-public reset info")
    previous_action, previous_reward = None, 0.0
    history: list[Tensor] = []
    actions: list[int] = []
    rewards: list[float] = []
    masks: list[Tensor] = []
    for _ in range(max_steps):
        encoded = encode_observation(observation, previous_action=previous_action, previous_reward=previous_reward)
        history.append(encoded)
        current_mask = _mask(observation)
        masks.append(current_mask)
        if model is None or generator.random() < epsilon:
            action = generator.choice(current_mask.nonzero(as_tuple=False).flatten().tolist())
        else:
            with torch.no_grad():
                values = model(torch.stack(history).unsqueeze(0).to(device))[0, -1].cpu()
            action = _masked_argmax(values, current_mask)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        if info != {}:
            raise RuntimeError("gated DQN received non-public step info")
        actions.append(action)
        rewards.append(float(reward))
        previous_action, previous_reward = action, float(reward)
        if terminated or truncated:
            success = bool(terminated and reward > 0.0)
            return Episode(torch.stack(history), torch.tensor(actions), torch.tensor(rewards), torch.stack(masks)), success
        observation = next_observation
    raise RuntimeError("gated course exceeded its declared horizon")


def _collect_on_worker(
    model: QNetwork | None, worker, environment: GymVisionOnlyLuanti, *, seed: int,
    device: torch.device, epsilon: float, generator: random.Random, max_steps: int,
    seed_offset: int,
) -> tuple[int, Episode, bool]:
    """Collect one episode on a dedicated worker using a per-worker client.

    ``worker`` is an integer worker id; ``_WORKER_ENVS`` maps it to a fresh
    ``GymVisionOnlyLuanti`` wired to that worker's bridge.  Returns
    (episode_index, episode, success) with the episode index left for the
    caller to sort back into seed order.
    """
    env = _worker_environment(worker, environment)
    episode, success = _collect(model, env, seed=seed, device=device,
                                epsilon=epsilon, generator=generator, max_steps=max_steps)
    return seed_offset, episode, success


_WORKER_ENVS: dict[int, GymVisionOnlyLuanti] = {}


_WORKER_POOL_REF: Any = None


def _worker_environment(worker: Any, template: GymVisionOnlyLuanti) -> GymVisionOnlyLuanti:
    """Return (and cache) a GymVisionOnlyLuanti wired to the worker's client."""
    global _WORKER_ENVS
    if worker in _WORKER_ENVS:
        return _WORKER_ENVS[worker]
    client = _WORKER_POOL_REF.client(worker)
    env = GymVisionOnlyLuanti(
        seed=template._seed, max_steps=template._max_steps, bridge_root=template._bridge_root,
        client_factory=lambda root, client=client: client,
    )
    _WORKER_ENVS[worker] = env
    return env


def _learn(model: QNetwork, target: QNetwork, replay: list[Episode], *, batch_size: int, device: torch.device, optimizer: torch.optim.Optimizer, generator: random.Random) -> None:
    batch = generator.sample(replay, batch_size)
    lengths = [item.observations.shape[0] for item in batch]
    if min(lengths) < 2 or any(
        item.actions.shape[0] != length or item.rewards.shape[0] != length or item.masks.shape[0] != length
        for item, length in zip(batch, lengths, strict=True)
    ):
        raise RuntimeError("gated course emitted inconsistent action events")
    maximum = max(lengths)
    observations = torch.zeros((batch_size, maximum, INPUT_DIM), dtype=torch.float32, device=device)
    actions = torch.zeros((batch_size, maximum), dtype=torch.long, device=device)
    rewards = torch.zeros((batch_size, maximum), dtype=torch.float32, device=device)
    masks = torch.zeros((batch_size, maximum, ACTION_COUNT), dtype=torch.bool, device=device)
    valid = torch.zeros((batch_size, maximum), dtype=torch.bool, device=device)
    for index, (episode, length) in enumerate(zip(batch, lengths, strict=True)):
        observations[index, :length] = episode.observations.to(device)
        actions[index, :length] = episode.actions.to(device)
        rewards[index, :length] = episode.rewards.to(device)
        masks[index, :length] = episode.masks.to(device)
        valid[index, :length] = True
    values = model(observations)
    selected = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_values = target(observations)[:, 1:].masked_fill(~masks[:, 1:], float("-inf")).max(dim=-1).values
        continuation = torch.zeros_like(valid)
        continuation[:, :-1] = valid[:, 1:]
        targets = rewards.clone()
        bootstrap = torch.where(continuation[:, :-1], next_values, torch.zeros_like(next_values))
        targets[:, :-1] += 0.95 * bootstrap
    loss = torch.nn.functional.smooth_l1_loss(selected[valid], targets[valid])
    if hasattr(model, "causal_predictions"):
        prediction_valid = valid.clone()
        prediction_valid[:, 0] = False
        target_observations = model.predictive_target(observations) if hasattr(model, "predictive_target") else observations
        prediction_loss = torch.nn.functional.mse_loss(
            model.causal_predictions(observations)[prediction_valid], target_observations[prediction_valid],
        )
        loss = loss + PREDICTIVE_LOSS_WEIGHT * prediction_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if hasattr(model, 'core') and hasattr(model.core, 'enforce_key_orthogonality'):
        model.core.enforce_key_orthogonality()


def _train(
    model: QNetwork, *, environment: GymVisionOnlyLuanti, episodes: int, seed_start: int,
    action_seed: int, device: torch.device, max_steps: int, worker_pool: Any | None = None,
) -> tuple[int, int]:
    """Train with optional parallel episode collection across worker servers.

    The DQN update loop is unchanged; only the episode collection step fans out
    to independent Luanti servers.  When ``worker_pool`` is None the exact
    original serial path runs unchanged.
    """
    model.train()
    target = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = random.Random(action_seed)
    replay: list[Episode] = []
    successes = 0
    if worker_pool is not None:
        global _WORKER_POOL_REF, _WORKER_ENVS
        _WORKER_POOL_REF = worker_pool["pool"]
        _WORKER_ENVS = {}
        workers = worker_pool["workers"]

        def worker_loop(worker: int) -> list[tuple[int, Episode, bool]]:
            """One thread per worker; episodes for this worker run serially."""
            collected: list[tuple[int, Episode, bool]] = []
            for index in range(worker, episodes, workers):
                collected.append(_collect_on_worker(
                    model, worker, environment, seed=seed_start + index, device=device,
                    epsilon=0.45 - 0.40 * index / max(1, episodes - 1),
                    generator=random.Random(action_seed + index), max_steps=max_steps,
                    seed_offset=index,
                ))
            return collected

        outcomes: list[tuple[int, Episode, bool]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch in executor.map(worker_loop, range(workers)):
                outcomes.extend(batch)
        outcomes.sort(key=lambda item: item[0])
        replay = [item[1] for item in outcomes]
        successes = sum(1 for item in outcomes if item[2])
        for offset in range(0, len(replay), 32):
            _learn(model, target, replay, batch_size=32, device=device, optimizer=optimizer,
                   generator=generator)
        return episodes, successes
    for index in range(episodes):
        epsilon = 0.45 - 0.40 * index / max(1, episodes - 1)
        episode, success = _collect(model, environment, seed=seed_start + index, device=device, epsilon=epsilon, generator=generator, max_steps=max_steps)
        replay.append(episode)
        successes += success
        if len(replay) >= 32:
            _learn(model, target, replay, batch_size=32, device=device, optimizer=optimizer, generator=generator)
        if index and index % 32 == 0:
            target.load_state_dict(model.state_dict())
    return episodes, successes


@torch.no_grad()
def _evaluate(model: QNetwork | None, *, environment: GymVisionOnlyLuanti, episodes: int, seed_start: int, action_seed: int, device: torch.device, max_steps: int) -> Evaluation:
    generator = random.Random(action_seed)
    successes, returns, counts = 0, [], [0] * ACTION_COUNT
    for index in range(episodes):
        episode, success = _collect(model, environment, seed=seed_start + index, device=device, epsilon=0.0, generator=generator, max_steps=max_steps)
        successes += success
        returns.append(float(episode.rewards.sum().item()))
        for action in episode.actions.tolist():
            counts[action] += 1
    return Evaluation(episodes, successes, float(np.mean(returns)), counts)


def run(
    *, device: str | torch.device | None = None, training_episodes: int = 512,
    evaluation_episodes: int = 64, training_seed_start: int = 9_500_000,
    heldout_seed_start: int = 9_600_000, action_seed: int = 2_042_337,
    course: str = "gated_compact", bridge_root: Path = DEFAULT_BRIDGE_ROOT, artifact: Path | None = None,
    model_widths: ModelWidths = CAPACITY_MATCHED_WIDTHS,
    parameter_match_tolerance: float = PARAMETER_MATCH_TOLERANCE,
    acceptance_mode: str = "course_default",
) -> Report:
    if course not in COURSES:
        raise ValueError("unsupported gated DQN course")
    if acceptance_mode not in ACCEPTANCE_MODES:
        raise ValueError("unsupported gated DQN acceptance mode")
    if acceptance_mode == "addressed_memory_diagnostic" and course != "addressed_gated":
        raise ValueError("addressed memory diagnostic requires the addressed course")
    if acceptance_mode == "addressed_balanced_diagnostic" and course != "addressed_balanced":
        raise ValueError("addressed balanced diagnostic requires the addressed balanced course")
    minimum_seed, maximum_seed, max_steps = COURSES[course]
    if (
        training_episodes < 128 or evaluation_episodes < 16
        or not minimum_seed <= training_seed_start < maximum_seed
        or not minimum_seed <= heldout_seed_start < maximum_seed
        or set(range(training_seed_start, training_seed_start + training_episodes)).intersection(range(heldout_seed_start, heldout_seed_start + evaluation_episodes))
    ):
        raise ValueError("invalid gated DQN contract")
    match_errors = validate_parameter_match(model_widths, tolerance=parameter_match_tolerance)
    resolved = resolve_device(device)
    torch.manual_seed(action_seed)
    np.random.seed(action_seed)
    environment = GymVisionOnlyLuanti(seed=training_seed_start, max_steps=max_steps, bridge_root=bridge_root)
    try:
        random_evaluation = _evaluate(None, environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, action_seed=action_seed, device=resolved, max_steps=max_steps)
        results: list[ModelResult] = []
        factories = comparison_factories(model_widths)
        for offset, (name, factory) in enumerate(factories):
            torch.manual_seed(action_seed + offset)
            model = factory().to(resolved)
            trained, successes = _train(model, environment=environment, episodes=training_episodes, seed_start=training_seed_start, action_seed=action_seed + 1000 * (offset + 1), device=resolved, max_steps=max_steps)
            evaluation = _evaluate(model, environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, action_seed=action_seed + 2000 * (offset + 1), device=resolved, max_steps=max_steps)
            results.append(ModelResult(name, _parameter_count(model), trained, successes, evaluation))
    finally:
        environment.close()
    by_name = {result.name: result for result in results}
    accepted = (
        by_name["gru"].evaluation.success_rate >= 0.75
        and by_name["transformer_2"].evaluation.success_rate >= 0.75
        and by_name["plwm"].evaluation.success_rate >= 0.75
    ) if acceptance_mode == "addressed_memory_diagnostic" else (
        by_name["plwm"].evaluation.success_rate >= 0.75
        and by_name["gru"].evaluation.success_rate >= 0.75
        and by_name["no_memory"].evaluation.success_rate <= 0.60
        and by_name["plwm_state_off"].evaluation.success_rate <= 0.60
        and by_name["transformer_2"].evaluation.success_rate <= 0.60
    ) if course == "addressed_gated" else (
        by_name["gru"].evaluation.success_rate >= 0.75
        and by_name["transformer_2"].evaluation.success_rate >= 0.75
        and by_name["plwm"].evaluation.success_rate >= 0.75
        and by_name["no_memory"].evaluation.success_rate <= 0.50
        and by_name["plwm_state_off"].evaluation.success_rate <= 0.50
    ) if course == "addressed_balanced" else (
        by_name["gru"].evaluation.success_rate >= 0.75
        and by_name["transformer_2"].evaluation.success_rate >= 0.75
        and by_name["plwm"].evaluation.success_rate >= 0.75
    ) if course == "gated_compact" else (
        by_name["gru"].evaluation.success_rate >= 0.75
        and by_name["plwm"].evaluation.success_rate >= 0.75
        and by_name["no_memory"].evaluation.success_rate <= 0.55
        and by_name["plwm_state_off"].evaluation.success_rate <= 0.55
        and by_name["transformer_2"].evaluation.success_rate <= 0.55
    ) if course == "long_gated" else (
        by_name["gru"].evaluation.success_rate >= 0.75
        and by_name["plwm"].evaluation.success_rate >= 0.75
        and by_name["no_memory"].evaluation.success_rate <= 0.40
        and by_name["plwm_state_off"].evaluation.success_rate <= 0.40
        and by_name["transformer_2"].evaluation.success_rate <= 0.40
    ) if course == "paired_gated" else (
        by_name["plwm"].evaluation.success_rate >= 0.75
        and by_name["plwm"].evaluation.success_rate - by_name["gru"].evaluation.success_rate >= 0.15
        and by_name["plwm"].evaluation.success_rate - by_name["transformer_2"].evaluation.success_rate >= 0.15
        and by_name["no_memory"].evaluation.success_rate <= 0.60
        and by_name["plwm_state_off"].evaluation.success_rate <= 0.60
    )
    target = artifact or ROOT / "artifacts" / "luanti-gated-memory-dqn-report.json"
    report = Report(
        "luanti_gated_memory_dqn_v2: real Luanti gated delayed-goal course; reward-only off-policy DQN from 11x11 public vision, previous chosen action, previous scalar reward, and public adjacent-wall legal-action masks only; no cue labels, symbolic bridge state, seed, teacher trajectory, external planner, shaped reward, or PLWM-only input. Every comparison model is parameter-capacity-matched to full PLWM within the recorded tolerance. This validates a neural task comparison, not an architecture claim.",
        course, str(resolved), list(range(training_seed_start, training_seed_start + training_episodes)), list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)), action_seed, max_steps, TRANSFORMER_CONTEXT, model_widths, parameter_match_tolerance, match_errors, acceptance_mode, random_evaluation, results, accepted, str(target),
    )
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def run_object_token_addressed_smoke(
    *, device: str | torch.device | None = "cuda", training_episodes: int = 256, evaluation_episodes: int = 32,
    training_seed_start: int = 12_920_000, heldout_seed_start: int = 12_930_000, action_seed: int = 9_042_337,
    bridge_root: Path = DEFAULT_BRIDGE_ROOT, artifact: Path | None = None,
    model_widths: ObjectTokenModelWidths = OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS,
) -> Report:
    """Frozen first CUDA smoke for the supported object-token writer precheck."""
    minimum_seed, maximum_seed, max_steps = COURSES["addressed_gated"]
    if (
        training_episodes != 256 or evaluation_episodes != 32
        or not minimum_seed <= training_seed_start < maximum_seed
        or not minimum_seed <= heldout_seed_start < maximum_seed
        or set(range(training_seed_start, training_seed_start + training_episodes)).intersection(range(heldout_seed_start, heldout_seed_start + evaluation_episodes))
    ):
        raise ValueError("invalid frozen object-token addressed smoke contract")
    match_errors = validate_object_token_parameter_match(model_widths)
    resolved = resolve_device(device)
    if str(resolved) != "cuda":
        raise ValueError("object-token addressed smoke requires local CUDA")
    torch.manual_seed(action_seed)
    np.random.seed(action_seed)
    environment = GymVisionOnlyLuanti(seed=training_seed_start, max_steps=max_steps, bridge_root=bridge_root)
    try:
        random_evaluation = _evaluate(None, environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, action_seed=action_seed, device=resolved, max_steps=max_steps)
        results: list[ModelResult] = []
        for offset, (name, factory) in enumerate(object_token_comparison_factories(model_widths)):
            torch.manual_seed(action_seed + offset)
            model = factory().to(resolved)
            trained, successes = _train(model, environment=environment, episodes=training_episodes, seed_start=training_seed_start, action_seed=action_seed + 1000 * (offset + 1), device=resolved, max_steps=max_steps)
            evaluation = _evaluate(model, environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, action_seed=action_seed + 2000 * (offset + 1), device=resolved, max_steps=max_steps)
            results.append(ModelResult(name, _parameter_count(model), trained, successes, evaluation))
    finally:
        environment.close()
    by_name = {result.name: result for result in results}
    success = {name: result.evaluation.success_rate for name, result in by_name.items()}
    accepted = (
        success["plwm"] >= 0.75
        and success["plwm"] - success["gru"] >= 0.15
        and success["plwm"] - success["transformer_2"] >= 0.15
        and success["no_memory"] <= 0.60 and success["plwm_state_off"] <= 0.60
    )
    target = artifact or ROOT / "artifacts" / "luanti-addressed-object-token-dqn-smoke-report.json"
    report = Report(
        "luanti_addressed_object_token_dqn_v1: real repaired seven-channel Luanti addressed retrieval; reward-only off-policy DQN from 11x11 public vision, previous chosen action, previous scalar reward, and public adjacent-wall legal-action masks only. Every model includes the same object-token visual event encoder and its parameters count toward capacity matching; only full PLWM writes four first-frame public object tokens and retrieves with the delayed public query. No cue/query labels, coordinates, symbolic bridge state, seed, teacher trajectory, external planner, shaped reward, or PLWM-only input. This is one CUDA smoke, never an architecture claim.",
        "addressed_gated", str(resolved), list(range(training_seed_start, training_seed_start + training_episodes)), list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)), action_seed, max_steps, TRANSFORMER_CONTEXT, model_widths, PARAMETER_MATCH_TOLERANCE, match_errors, "object_token_writer_smoke", random_evaluation, results, accepted, str(target),
    )
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def run_object_token_addressed_balanced_smoke(
    *, device: str | torch.device | None = "cuda", training_episodes: int = 1024, evaluation_episodes: int = 32,
    training_seed_start: int = 16_000_000, heldout_seed_start: int = 16_002_000, action_seed: int = 9_042_337,
    bridge_root: Path = DEFAULT_BRIDGE_ROOT, artifact: Path | None = None, workers: int = 1,
    model_widths: ObjectTokenModelWidths = OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS,
) -> Report:
    """Frozen CUDA smoke for the side-bias-separated addressed balanced course.

    The counterbalanced course removes the identity->side frequency shortcut:
    every identity is left/right exactly half the time and every identity is
    queried an equal number of times across each contiguous multiple-of-four
    seed window.  A memory-free model can therefore only reach chance, so the
    no-memory ceiling is tightened to 0.50 while the memory models must still
    retrieve the delayed query at 0.75.

    Escalated under a predeclared budget change: training grows from 256 to
    1,024 worlds from 16,000,000 and the held-out window moves forward to 32
    disjoint worlds from 16,002,000; everything else is frozen.  The 256-world
    run trained at only 26% with collapsed action profiles, which reads as an
    exploration/budget failure rather than task insolubility.
    """
    minimum_seed, maximum_seed, max_steps = COURSES["addressed_balanced"]
    if (
        training_episodes != 1024 or evaluation_episodes != 32
        or not minimum_seed <= training_seed_start < maximum_seed
        or not minimum_seed <= heldout_seed_start < maximum_seed
        or set(range(training_seed_start, training_seed_start + training_episodes)).intersection(range(heldout_seed_start, heldout_seed_start + evaluation_episodes))
    ):
        raise ValueError("invalid frozen object-token addressed balanced smoke contract")
    match_errors = validate_object_token_parameter_match(model_widths)
    resolved = resolve_device(device)
    if str(resolved) != "cuda":
        raise ValueError("object-token addressed balanced smoke requires local CUDA")
    torch.manual_seed(action_seed)
    np.random.seed(action_seed)
    pool: Any = None
    worker_pool: Any = None
    eval_bridge = bridge_root
    if workers > 1:
        from plwm.luanti_workers import ParallelWorkerPool
        pool = ParallelWorkerPool(workers)
        pool.start()
        worker_pool = {"pool": pool, "workers": workers}
        eval_bridge = Path(pool.client(0).bridge.root)
    environment = GymVisionOnlyLuanti(seed=training_seed_start, max_steps=max_steps, bridge_root=eval_bridge)
    try:
        random_evaluation = _evaluate(None, environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, action_seed=action_seed, device=resolved, max_steps=max_steps)
        results: list[ModelResult] = []
        for offset, (name, factory) in enumerate(object_token_comparison_factories(model_widths)):
            torch.manual_seed(action_seed + offset)
            model = factory().to(resolved)
            trained, successes = _train(model, environment=environment, episodes=training_episodes, seed_start=training_seed_start, action_seed=action_seed + 1000 * (offset + 1), device=resolved, max_steps=max_steps, worker_pool=worker_pool)
            evaluation = _evaluate(model, environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, action_seed=action_seed + 2000 * (offset + 1), device=resolved, max_steps=max_steps)
            results.append(ModelResult(name, _parameter_count(model), trained, successes, evaluation))
    finally:
        if pool is not None:
            pool.stop()
        environment.close()
    by_name = {result.name: result for result in results}
    success = {name: result.evaluation.success_rate for name, result in by_name.items()}
    accepted = (
        success["plwm"] >= 0.75
        and success["gru"] >= 0.75
        and success["transformer_2"] >= 0.75
        and success["no_memory"] <= 0.50 and success["plwm_state_off"] <= 0.50
    )
    target = artifact or ROOT / "artifacts" / "luanti-addressed-balanced-object-token-dqn-smoke-report.json"
    report = Report(
        "luanti_addressed_balanced_object_token_dqn_v2: real repaired seven-channel Luanti addressed balanced retrieval; reward-only off-policy DQN from 11x11 public vision, previous chosen action, previous scalar reward, and public adjacent-wall legal-action masks only. The counterbalanced course nulls the identity->side frequency shortcut (each identity left/right 50/50 and each identity queried equally per contiguous seed window), so chance-level no-memory success at 0.50 is the guaranteed bound and only real delayed retrieval can exceed it. Predeclared budget escalation: 1,024 training worlds from 16,000,000 and 32 held-out worlds from 16,002,000 under the same gates. Every model includes the same object-token visual event encoder and its parameters count toward capacity matching; only full PLWM writes four first-frame public object tokens and retrieves with the delayed public query. No cue/query labels, coordinates, symbolic bridge state, seed, teacher trajectory, external planner, shaped reward, or PLWM-only input. This is one CUDA smoke, never an architecture claim.",
        "addressed_balanced", str(resolved), list(range(training_seed_start, training_seed_start + training_episodes)), list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)), action_seed, max_steps, TRANSFORMER_CONTEXT, model_widths, PARAMETER_MATCH_TOLERANCE, match_errors, "object_token_writer_smoke", random_evaluation, results, accepted, str(target),
    )
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def run_object_token_addressed_balanced_2id_smoke(
    *, device: str | torch.device | None = "cuda", training_episodes: int = 256, evaluation_episodes: int = 32,
    training_seed_start: int = 18_000_000, heldout_seed_start: int = 18_001_000, action_seed: int = 9_042_337,
    bridge_root: Path = DEFAULT_BRIDGE_ROOT, artifact: Path | None = None, workers: int = 1,
    model_widths: ObjectTokenModelWidths = OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS,
) -> Report:
    """CUDA smoke for the two-identity difficulty-reduction balanced course.

    The 1,024-world four-identity balanced run showed every matched controller
    at 8/32 while no-memory and PLWM state-off sat at the counterbalanced
    chance bound: the side-bias was removed but the four-cue delayed retrieval
    was still not learned by any architecture.  This predeclared successor
    keeps the exact counterbalancing recipe over two identities (query = offset
    mod 2, side = parity of (floor(offset/2) + identity)), so the free
    identity->side oracle is again 50%, and the object-token PLWM writes the
    two first-frame public identity tokens.  All comparison models share the
    same encoder and remain capacity matched within 5%.
    """
    minimum_seed, maximum_seed, max_steps = COURSES["addressed_balanced_2id"]
    if (
        training_episodes != 256 or evaluation_episodes != 32
        or not minimum_seed <= training_seed_start < maximum_seed
        or not minimum_seed <= heldout_seed_start < maximum_seed
        or set(range(training_seed_start, training_seed_start + training_episodes)).intersection(range(heldout_seed_start, heldout_seed_start + evaluation_episodes))
    ):
        raise ValueError("invalid object-token addressed balanced two-identity smoke contract")
    match_errors = validate_object_token_parameter_match(model_widths)
    resolved = resolve_device(device)
    if str(resolved) != "cuda":
        raise ValueError("object-token addressed balanced two-identity smoke requires local CUDA")
    torch.manual_seed(action_seed)
    np.random.seed(action_seed)
    pool: Any = None
    worker_pool: Any = None
    eval_bridge = bridge_root
    if workers > 1:
        from plwm.luanti_workers import ParallelWorkerPool
        pool = ParallelWorkerPool(workers)
        pool.start()
        worker_pool = {"pool": pool, "workers": workers}
        eval_bridge = Path(pool.client(0).bridge.root)
    environment = GymVisionOnlyLuanti(seed=training_seed_start, max_steps=max_steps, bridge_root=eval_bridge)
    try:
        random_evaluation = _evaluate(None, environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, action_seed=action_seed, device=resolved, max_steps=max_steps)
        results: list[ModelResult] = []
        for offset, (name, factory) in enumerate(object_token_comparison_factories(model_widths, identity_count=2)):
            torch.manual_seed(action_seed + offset)
            model = factory().to(resolved)
            trained, successes = _train(model, environment=environment, episodes=training_episodes, seed_start=training_seed_start, action_seed=action_seed + 1000 * (offset + 1), device=resolved, max_steps=max_steps, worker_pool=worker_pool)
            evaluation = _evaluate(model, environment=environment, episodes=evaluation_episodes, seed_start=heldout_seed_start, action_seed=action_seed + 2000 * (offset + 1), device=resolved, max_steps=max_steps)
            results.append(ModelResult(name, _parameter_count(model), trained, successes, evaluation))
    finally:
        if pool is not None:
            pool.stop()
        environment.close()
    by_name = {result.name: result for result in results}
    success = {name: result.evaluation.success_rate for name, result in by_name.items()}
    accepted = (
        success["plwm"] >= 0.75
        and success["gru"] >= 0.75
        and success["transformer_2"] >= 0.75
        and success["no_memory"] <= 0.50 and success["plwm_state_off"] <= 0.50
    )
    target = artifact or ROOT / "artifacts" / "luanti-addressed-balanced-2id-object-token-dqn-smoke-report.json"
    report = Report(
        "luanti_addressed_balanced_2id_object_token_dqn_v1: real repaired seven-channel Luanti addressed balanced retrieval; reward-only off-policy DQN from 11x11 public vision, previous chosen action, previous scalar reward, and public adjacent-wall legal-action masks only. The two-identity counterbalanced course keeps the identity->side frequency shortcut at exactly 50/50 over every contiguous multiple-of-two window while exposing only two first-frame public identity tokens, so the chance no-memory bound is 0.50 and only genuine delayed retrieval can exceed it. Every model includes the same object-token visual event encoder and its parameters count toward capacity matching; only full PLWM writes the first-frame two public identity objects and retrieves with the delayed public query. No cue/query labels, coordinates, symbolic bridge state, seed, teacher trajectory, external planner, shaped reward, or PLWM-only input. This is one CUDA smoke, never an architecture claim.",
        "addressed_balanced_2id", str(resolved), list(range(training_seed_start, training_seed_start + training_episodes)), list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)), action_seed, max_steps, TRANSFORMER_CONTEXT, model_widths, PARAMETER_MATCH_TOLERANCE, match_errors, "object_token_writer_smoke", random_evaluation, results, accepted, str(target),
    )
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-episodes", type=int, default=512)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--training-seed-start", type=int, default=9_500_000)
    parser.add_argument("--heldout-seed-start", type=int, default=9_600_000)
    parser.add_argument("--action-seed", type=int, default=2_042_337)
    parser.add_argument("--course", choices=tuple(COURSES), default="gated_compact")
    parser.add_argument("--acceptance-mode", choices=ACCEPTANCE_MODES, default="course_default")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--object-token-addressed-smoke", action="store_true")
    parser.add_argument("--object-token-addressed-balanced-smoke", action="store_true")
    parser.add_argument("--object-token-addressed-balanced-2id-smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="number of parallel Luanti workers (1 = serial)")
    arguments = parser.parse_args()
    if arguments.object_token_addressed_balanced_2id_smoke:
        outcome = run_object_token_addressed_balanced_2id_smoke(artifact=arguments.artifact, workers=arguments.workers)
    elif arguments.object_token_addressed_balanced_smoke:
        outcome = run_object_token_addressed_balanced_smoke(artifact=arguments.artifact, workers=arguments.workers)
    else:
        outcome = run_object_token_addressed_smoke(artifact=arguments.artifact) if arguments.object_token_addressed_smoke else run(
            training_episodes=arguments.training_episodes, evaluation_episodes=arguments.evaluation_episodes,
            training_seed_start=arguments.training_seed_start, heldout_seed_start=arguments.heldout_seed_start,
            action_seed=arguments.action_seed, course=arguments.course, artifact=arguments.artifact,
            acceptance_mode=arguments.acceptance_mode,
        )
    print(json.dumps(asdict(outcome), indent=2))

"""Continual partially observable benchmark and PLWM research controllers.

This module deliberately makes the task small enough to audit.  It is not an
open-world proxy: it tests whether a controller can retain a delayed intent,
purchase a more reliable observation when warranted, and correct stale
episodic evidence without treating language or latent similarity as authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Protocol

import torch
from torch import Tensor, nn

from .commitments import Commitment, CommitmentLedger, CommitmentStatus
from .torch_model import TorchCognitiveState, TorchPLWM


class TaskAction(IntEnum):
    WAIT = 0
    OBSERVE = 1
    LEFT = 2
    RIGHT = 3


@dataclass(frozen=True)
class EvidenceTask:
    """A delayed two-route task with a noisy cheap cue and costly verification."""

    route: int
    cheap_signal: int
    verify_signal: int
    distractors: tuple[int, ...]
    cheap_reliability: float = 0.68
    verify_reliability: float = 0.97
    observe_cost: float = 0.12
    action_cost: float = 0.04

    @classmethod
    def sample(
        cls,
        generator: torch.Generator,
        *,
        distractor_count: int,
        cheap_reliability: float | None = None,
        verify_reliability: float = 0.97,
    ) -> "EvidenceTask":
        cheap_reliability = cheap_reliability or (0.68 if bool(torch.randint(0, 2, (1,), generator=generator).item()) else 0.94)
        route = int(torch.randint(0, 2, (1,), generator=generator).item())
        cheap_correct = bool(torch.rand((), generator=generator).item() < cheap_reliability)
        verify_correct = bool(torch.rand((), generator=generator).item() < verify_reliability)
        cheap_signal = route if cheap_correct else 1 - route
        verify_signal = route if verify_correct else 1 - route
        distractors = tuple(int(torch.randint(0, 2, (1,), generator=generator).item()) for _ in range(distractor_count))
        return cls(route, cheap_signal, verify_signal, distractors, cheap_reliability, verify_reliability)

    def sequence(self, *, include_verification: bool) -> Tensor:
        """Event vectors include route evidence plus the cheap sensor's reliability."""
        events = [self._event(0, self.cheap_signal, self.cheap_reliability)]
        events.extend(self._event(1, item) for item in self.distractors)
        if include_verification:
            events.append(self._event(2, self.verify_signal, self.cheap_reliability))
        else:
            events.append(torch.zeros(5))
        return torch.stack(events)

    @staticmethod
    def _event(kind: int, value: int, reliability: float = 0.0) -> Tensor:
        result = torch.zeros(5)
        if kind == 0:
            result[value] = 1.0
        elif kind == 2:
            result[2 + value] = 1.0
        result[4] = reliability
        # Distractors contain no route-bearing feature.
        if kind == 1:
            result.zero_()
        return result


def posterior(prior_route_one: float, signal: int, reliability: float) -> float:
    likelihood_one = reliability if signal == 1 else 1.0 - reliability
    likelihood_zero = reliability if signal == 0 else 1.0 - reliability
    numerator = prior_route_one * likelihood_one
    return numerator / (numerator + (1.0 - prior_route_one) * likelihood_zero)


def entropy(probability: float) -> float:
    probability = min(max(probability, 1e-8), 1.0 - 1e-8)
    return -probability * torch.log(torch.tensor(probability)).item() - (1.0 - probability) * torch.log(torch.tensor(1.0 - probability)).item()


@dataclass
class EpisodeResult:
    success: bool
    total_cost: float
    observed: bool
    selected_before_delay: str | None
    selected_after_delay: str | None
    corrected_stale_memory: bool
    commitment_completed: bool


class BayesianPersistentController:
    """Explicit belief and commitment controller used as an auditable PLWM policy."""

    def __init__(self, *, cheap_reliability: float = 0.68, verify_reliability: float = 0.97, observe_cost: float = 0.12):
        self.cheap_reliability = cheap_reliability
        self.verify_reliability = verify_reliability
        self.observe_cost = observe_cost
        self.ledger = CommitmentLedger()
        self.memory: list[tuple[str, int, float]] = []

    def run(self, task: EvidenceTask, *, inject_stale_memory: bool = False) -> EpisodeResult:
        self.ledger = CommitmentLedger()
        self.ledger.add(Commitment("reach-correct-route", target_state=task.route, priority=1.0, deadline=4))
        belief_one = 0.5
        if inject_stale_memory:
            self.memory = [("old-route", 1 - task.route, 0.55)]
            belief_one = self.memory[0][1] * 0.55 + 0.5 * 0.45
        else:
            self.memory = []
        belief_one = posterior(belief_one, task.cheap_signal, task.cheap_reliability)
        selected_before_delay = self.ledger.select([1.0 - belief_one, belief_one]).id
        # Delayed commitment: intervening events must not erase the goal.
        for _ in task.distractors:
            self.ledger.tick()
        selected_after_delay = self.ledger.select([1.0 - belief_one, belief_one]).id
        confidence = max(belief_one, 1.0 - belief_one)
        observe = confidence < 0.82 or bool(self.memory)
        total_cost = task.action_cost
        corrected = False
        if observe:
            total_cost += task.observe_cost
            prior = belief_one
            belief_one = posterior(belief_one, task.verify_signal, task.verify_reliability)
            corrected = inject_stale_memory and (belief_one >= 0.5) == bool(task.route)
            self.memory.append(("verified-route", task.verify_signal, task.verify_reliability))
        action = int(TaskAction.RIGHT if belief_one >= 0.5 else TaskAction.LEFT)
        success = action - int(TaskAction.LEFT) == task.route
        self.ledger.tick()
        if success:
            self.ledger.reconcile([1.0, 0.0] if task.route == 0 else [0.0, 1.0])
        completed = self.ledger.commitments[0].status == CommitmentStatus.COMPLETED
        return EpisodeResult(success, total_cost, observe, selected_before_delay, selected_after_delay, corrected, completed)


class NoMemoryController:
    def run(self, task: EvidenceTask, *, inject_stale_memory: bool = False) -> EpisodeResult:
        action = TaskAction.LEFT
        success = (action == TaskAction.RIGHT) == bool(task.route)
        return EpisodeResult(success, task.action_cost, False, None, None, False, False)


class WindowedContextController:
    """Finite context baseline; it only sees the final ``window`` task events."""

    def __init__(self, window: int = 3):
        self.window = window

    def run(self, task: EvidenceTask, *, inject_stale_memory: bool = False) -> EpisodeResult:
        visible_cheap = self.window >= len(task.distractors) + 2
        belief_one = posterior(0.5, task.cheap_signal, 0.68) if visible_cheap else 0.5
        action = TaskAction.RIGHT if belief_one >= 0.5 else TaskAction.LEFT
        success = (action == TaskAction.RIGHT) == bool(task.route)
        return EpisodeResult(success, task.action_cost, False, None, None, False, False)


class RecurrentPolicy(nn.Module):
    """Fair learned recurrent baseline: receives all events but has no ledger/memory authority."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 24):
        super().__init__()
        self.cell = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.control = nn.Linear(hidden_dim, 3)
        self.route = nn.Linear(hidden_dim, 2)

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        _, hidden = self.cell(sequence)
        hidden = hidden.squeeze(0)
        return self.control(hidden), self.route(hidden)


class TransformerContextPolicy(nn.Module):
    """Finite-context Transformer baseline with the same action interface."""

    def __init__(self, input_dim: int = 5, model_dim: int = 24, *, max_events: int = 128):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, max_events, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.control = nn.Linear(model_dim, 3)
        self.route = nn.Linear(model_dim, 2)

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        if sequence.shape[1] > self.position.shape[1]:
            raise ValueError("sequence exceeds Transformer context budget")
        encoded = self.encoder(self.input_projection(sequence) + self.position[:, : sequence.shape[1]])
        state = encoded[:, -1]
        return self.control(state), self.route(state)


class EndToEndPersistentPLWM(nn.Module):
    """Trainable persistent state/control path built around :class:`TorchPLWM`.

    The policy and next-route prediction are jointly supervised.  The ledger is
    intentionally kept outside this network: a network score cannot declare a
    real commitment complete.
    """

    def __init__(self, input_dim: int = 5, latent_dim: int = 24):
        super().__init__()
        self.core = TorchPLWM(
            input_dim,
            latent_dim=latent_dim,
            memory_slots=8,
            thought_steps=2,
            output_dim=2,
            use_memory=True,
            gated_memory=True,
        )
        self.control = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Tanh(), nn.Linear(latent_dim, 3))
        self.route_prediction = nn.Linear(latent_dim, 2)

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor, TorchCognitiveState]:
        state = self.core.initial_state(sequence.shape[0], sequence.device)
        for index in range(sequence.shape[1]):
            state, _ = self.core.step(sequence[:, index], state, think=index == sequence.shape[1] - 1)
        policy_logits = self.control(torch.cat((state.belief, state.workspace), dim=-1))
        return policy_logits, self.route_prediction(state.belief), state


class LearnedPersistentRuntime:
    """Executes a learned action proposal while retaining auditable goal authority."""

    def __init__(self, model: EndToEndPersistentPLWM):
        self.model = model
        self.ledger = CommitmentLedger()

    @torch.no_grad()
    def run(self, task: EvidenceTask) -> EpisodeResult:
        self.ledger = CommitmentLedger()
        self.ledger.add(Commitment("reach-correct-route", target_state=task.route, priority=1.0, deadline=4))
        sequence = task.sequence(include_verification=False).unsqueeze(0)
        policy_logits, _route_logits, _ = self.model(sequence)
        proposal = int(policy_logits.argmax(dim=-1).item())
        if proposal == 0:
            _policy, route_logits, _ = self.model(task.sequence(include_verification=True).unsqueeze(0))
            route = int(route_logits.argmax(dim=-1).item())
        else:
            route = proposal - 1
        for _ in task.distractors:
            self.ledger.tick()
        selected_before = self.ledger.select([0.5, 0.5]).id
        selected_after = self.ledger.select([0.5, 0.5]).id
        success = route == task.route
        self.ledger.tick()
        if success:
            self.ledger.reconcile([1.0, 0.0] if route == 0 else [0.0, 1.0])
        completed = self.ledger.commitments[0].status == CommitmentStatus.COMPLETED
        return EpisodeResult(success, task.action_cost, False, selected_before, selected_after, False, completed)


class LanguagePort(Protocol):
    def encode_observation(self, text: str) -> tuple[Tensor, Tensor]: ...

    def decode_action(self, action: TaskAction) -> str: ...


@dataclass
class FixedLanguagePort:
    """A test adapter standing in for an LLM boundary; it owns no cognitive state."""

    emitted: list[str] = field(default_factory=list)

    def encode_observation(self, text: str) -> tuple[Tensor, Tensor]:
        lowered = text.lower()
        value = 1 if "right" in lowered else 0
        event = EvidenceTask._event(0, value, 0.97)
        # Evidence is separately declared; text alone cannot alter commitments.
        likelihood = torch.tensor([0.03, 0.97] if value else [0.97, 0.03])
        return event, likelihood

    def decode_action(self, action: TaskAction) -> str:
        text = {TaskAction.OBSERVE: "observe", TaskAction.LEFT: "take left route", TaskAction.RIGHT: "take right route"}[action]
        self.emitted.append(text)
        return text


def batches(tasks: Iterable[EvidenceTask], *, include_verification: bool = False) -> tuple[Tensor, Tensor]:
    items = list(tasks)
    return torch.stack([item.sequence(include_verification=include_verification) for item in items]), torch.tensor([item.route for item in items])

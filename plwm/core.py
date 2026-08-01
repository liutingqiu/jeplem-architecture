"""A dependency-free PLWM forward-pass prototype.

This module is deliberately small and inspectable. It is not a trained LLM;
it implements the architectural contract we want to test before selecting a
large neural backend: persistent state, prediction error, recurrent thought,
addressable memory, and a cognitive-action policy.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Sequence


NumberVector = list[float]


def _vector(values: Iterable[float], size: int) -> NumberVector:
    result = [float(value) for value in values]
    if len(result) != size:
        raise ValueError(f"expected vector of size {size}, got {len(result)}")
    return result


def _tanh(values: Sequence[float]) -> NumberVector:
    return [math.tanh(value) for value in values]


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _dot(matrix: list[NumberVector], values: Sequence[float]) -> NumberVector:
    return [sum(weight * value for weight, value in zip(row, values)) for row in matrix]


def _add(*vectors: Sequence[float]) -> NumberVector:
    if not vectors:
        return []
    return [sum(values) for values in zip(*vectors)]


def _scale(values: Sequence[float], factor: float) -> NumberVector:
    return [factor * value for value in values]


def _blend(old: Sequence[float], new: Sequence[float], amount: float) -> NumberVector:
    return _add(_scale(old, 1.0 - amount), _scale(new, amount))


def _mean_abs_error(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / max(1, len(left))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _random_matrix(rng: random.Random, rows: int, columns: int, scale: float = 0.2) -> list[NumberVector]:
    return [[rng.uniform(-scale, scale) for _ in range(columns)] for _ in range(rows)]


class CognitiveAction(str, Enum):
    OBSERVE = "observe"
    THINK = "think"
    RETRIEVE = "retrieve"
    COMMIT = "commit"
    SPEAK = "speak"
    SLEEP = "sleep"


@dataclass
class MemorySlot:
    key: NumberVector
    value: NumberVector
    timestamp: int
    salience: float
    confidence: float
    source: str = "observation"


@dataclass
class PersistentState:
    belief: NumberVector
    goal: NumberVector
    workspace: NumberVector
    time: int = 0
    last_prediction_error: float = 0.0
    pending_commitment: str | None = None
    memory: list[MemorySlot] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["memory"] = [asdict(slot) for slot in self.memory]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PersistentState":
        return cls(
            belief=list(data["belief"]),
            goal=list(data["goal"]),
            workspace=list(data["workspace"]),
            time=int(data["time"]),
            last_prediction_error=float(data["last_prediction_error"]),
            pending_commitment=data.get("pending_commitment"),
            memory=[MemorySlot(**slot) for slot in data.get("memory", [])],
        )


@dataclass(frozen=True)
class PLWMConfig:
    input_dim: int = 4
    latent_dim: int = 16
    memory_capacity: int = 32
    thought_steps: int = 3
    surprise_threshold: float = 0.15
    retrieval_threshold: float = 0.75
    seed: int = 7


@dataclass
class StepResult:
    action: CognitiveAction
    time: int
    prediction_error: float
    thought_steps: int
    retrieved: int
    memory_written: bool
    emission: NumberVector | None


class PLWM:
    """Persistent state machine with a learned-core-shaped forward pass.

    The matrices are seeded random weights for an inspectable architecture
    test. A future torch backend can preserve the same state/action contract.
    """

    def __init__(self, config: PLWMConfig | None = None):
        self.config = config or PLWMConfig()
        rng = random.Random(self.config.seed)
        d = self.config.latent_dim
        i = self.config.input_dim
        self._encoder = _random_matrix(rng, d, i)
        self._goal_encoder = _random_matrix(rng, d, i)
        self._core = _random_matrix(rng, d, d * 3)
        self._workspace_core = _random_matrix(rng, d, d * 2)
        self._predictor = _random_matrix(rng, i, d)
        self._policy = _random_matrix(rng, len(CognitiveAction), d * 3 + 1)
        self._decoder = _random_matrix(rng, i, d * 2)
        self.state = PersistentState(
            belief=[0.0] * d,
            goal=[0.0] * d,
            workspace=[0.0] * d,
        )

    def set_goal(self, goal: Sequence[float], commitment: str | None = None) -> None:
        encoded = _tanh(_dot(self._goal_encoder, _vector(goal, self.config.input_dim)))
        self.state.goal = encoded
        if commitment is not None:
            self.state.pending_commitment = commitment

    def _policy_action(self, observation: NumberVector | None, error: float) -> CognitiveAction:
        if observation is None and error < self.config.surprise_threshold:
            return CognitiveAction.SLEEP
        features = self.state.belief + self.state.workspace + self.state.goal + [error]
        logits = _dot(self._policy, features)
        ranked = sorted(zip(logits, CognitiveAction), reverse=True)
        action = ranked[0][1]
        if error >= self.config.surprise_threshold and action in {
            CognitiveAction.SLEEP,
            CognitiveAction.SPEAK,
        }:
            return CognitiveAction.THINK
        if self.state.pending_commitment and action == CognitiveAction.SLEEP:
            return CognitiveAction.THINK
        return action

    def _write_memory(self, key: NumberVector, value: NumberVector, salience: float, source: str) -> bool:
        if salience < self.config.surprise_threshold:
            return False
        slot = MemorySlot(
            key=list(key),
            value=list(value),
            timestamp=self.state.time,
            salience=min(1.0, salience),
            confidence=min(1.0, 0.5 + salience),
            source=source,
        )
        if len(self.state.memory) < self.config.memory_capacity:
            self.state.memory.append(slot)
            return True
        replacement = min(
            range(len(self.state.memory)),
            key=lambda index: (self.state.memory[index].salience, self.state.memory[index].timestamp),
        )
        self.state.memory[replacement] = slot
        return True

    def _retrieve(self, key: NumberVector) -> list[MemorySlot]:
        candidates = sorted(
            self.state.memory,
            key=lambda slot: _cosine(key, slot.key) * (0.5 + 0.5 * slot.confidence),
            reverse=True,
        )
        return [slot for slot in candidates if _cosine(key, slot.key) >= self.config.retrieval_threshold][:3]

    def step(
        self,
        observation: Sequence[float] | None = None,
        *,
        goal: Sequence[float] | None = None,
        action: CognitiveAction | None = None,
    ) -> StepResult:
        observation_vector = None if observation is None else _vector(observation, self.config.input_dim)
        if goal is not None:
            self.set_goal(goal)

        predicted = _dot(self._predictor, self.state.belief)
        error = 0.0 if observation_vector is None else _mean_abs_error(predicted, observation_vector)
        self.state.time += 1
        self.state.last_prediction_error = error

        encoded = [0.0] * self.config.latent_dim if observation_vector is None else _tanh(
            _dot(self._encoder, observation_vector)
        )
        core_input = self.state.belief + self.state.goal + encoded
        proposed_belief = _tanh(_dot(self._core, core_input))
        update_rate = 0.1 if observation_vector is None else min(0.9, 0.2 + error)
        self.state.belief = _blend(self.state.belief, proposed_belief, update_rate)

        chosen = action or self._policy_action(observation_vector, error)
        thought_steps = 0
        retrieved = 0
        emission = None
        if chosen == CognitiveAction.THINK:
            for _ in range(self.config.thought_steps):
                workspace_input = self.state.workspace + self.state.belief
                self.state.workspace = _tanh(_dot(self._workspace_core, workspace_input))
                thought_steps += 1
            self.state.last_prediction_error *= 0.7 ** thought_steps
        elif chosen == CognitiveAction.RETRIEVE:
            matches = self._retrieve(encoded)
            retrieved = len(matches)
            if matches:
                memory_vector = matches[0].value
                self.state.workspace = _blend(self.state.workspace, memory_vector, 0.6)
                self.state.belief = _blend(self.state.belief, memory_vector, 0.25)
        elif chosen == CognitiveAction.COMMIT:
            self._write_memory(encoded, self.state.belief, 1.0, "commitment")
            self.state.pending_commitment = None
        elif chosen == CognitiveAction.SPEAK:
            emission = _tanh(_dot(self._decoder, self.state.belief + self.state.workspace))

        memory_written = self._write_memory(
            encoded,
            self.state.belief,
            self.state.last_prediction_error,
            "surprise",
        )
        return StepResult(
            action=chosen,
            time=self.state.time,
            prediction_error=error,
            thought_steps=thought_steps,
            retrieved=retrieved,
            memory_written=memory_written,
            emission=emission,
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.state.to_dict(), handle, indent=2)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as handle:
            self.state = PersistentState.from_dict(json.load(handle))


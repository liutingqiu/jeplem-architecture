"""Predeclared delayed-fact scale curve in real Luanti worlds.

This is deliberately a system-level memory comparison, not an architecture
claim.  Every learned baseline receives the same public state, vision, and
reward-event history.  The fact-memory controller additionally receives only
the compact fact derived from that history by ``VerifiedObjectFactMemory``.
Increasing delay inserts fresh, action-free Luanti observations after the
reward has been hidden by recentering; it therefore tests retention beyond a
finite history window without duplicating Python-side observations.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_behavior import Evaluation, train_policy
from experiments.train_luanti_polarity_memory import _prerequisite, _reward
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    DIRECTIONS,
    DIRECTION_TO_INDEX,
    VISION_FEATURE_DIM,
    CausalControlWorldModel,
    LuantiLiveClient,
    NoMemoryRewardVisionObjectPolicy,
    ObjectTaskState,
    PolarityFactMemoryCurrentVisionObjectPolicy,
    VerifiedFactObjectPlanner,
    VerifiedObjectFactMemory,
    object_polarity_fact_memory_visual_frame_features,
    object_reward_visual_frame_features,
)


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")
HISTORY = 16
MAX_ACTIONS = 40
PUBLIC_INPUT_DIM = 23 + VISION_FEATURE_DIM


class PublicEventGRUPolicy(nn.Module):
    """Finite public-event history baseline with the same raw evidence."""

    def __init__(self, *, history_length: int = HISTORY, hidden_dim: int = 48) -> None:
        super().__init__()
        self.history_length = history_length
        self.encoder = nn.GRU(PUBLIC_INPUT_DIM, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, len(DIRECTIONS))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        encoded, _ = self.encoder(observations)
        logits = self.head(encoded[:, -1])
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]]) -> str:
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        device = next(self.parameters()).device
        batch = torch.tensor(
            [object_reward_visual_frame_features(state, vision) for state, vision in context],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        logits, _ = self(batch)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class PublicEventTransformerPolicy(nn.Module):
    """Finite-context Transformer baseline over exactly the same public events."""

    def __init__(self, *, history_length: int = HISTORY, model_dim: int = 48) -> None:
        super().__init__()
        self.history_length = history_length
        self.input_projection = nn.Linear(PUBLIC_INPUT_DIM, model_dim)
        self.position = nn.Parameter(torch.zeros(1, history_length, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(model_dim, len(DIRECTIONS))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        if observations.shape[1] > self.history_length:
            raise ValueError("event sequence exceeds Transformer context budget")
        encoded = self.encoder(self.input_projection(observations) + self.position[:, : observations.shape[1]])
        logits = self.head(encoded[:, -1])
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]]) -> str:
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        device = next(self.parameters()).device
        batch = torch.tensor(
            [object_reward_visual_frame_features(state, vision) for state, vision in context],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        logits, _ = self(batch)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


@dataclass
class Trajectory:
    seed: int
    states: list[ObjectTaskState] = field(default_factory=list)
    visions: list[Mapping[str, Any]] = field(default_factory=list)
    labels: list[str | None] = field(default_factory=list)
    key_facts: list[bool] = field(default_factory=list)
    charge_modes: list[str | None] = field(default_factory=list)


@dataclass
class ScalePoint:
    delay_observations: int
    training_samples: int
    fact_memory: Evaluation
    no_memory: Evaluation
    recurrent: Evaluation
    transformer: Evaluation
    teacher: Evaluation
    fact_memory_revisions: int
    parameter_counts: dict[str, int]


@dataclass
class Report:
    protocol: str
    device: str
    history_length: int
    train_seeds: list[int]
    heldout_seeds: list[int]
    scale_points: list[ScalePoint]
    accepted: bool
    artifact: str


def _append(trajectory: Trajectory, state: ObjectTaskState, vision: Mapping[str, Any], facts: VerifiedObjectFactMemory) -> None:
    trajectory.states.append(state)
    trajectory.visions.append(dict(vision))
    trajectory.labels.append(None)
    trajectory.key_facts.append(facts.key_verified)
    trajectory.charge_modes.append(facts.charge_mode)


def _is_polarity_only_world(state: ObjectTaskState) -> bool:
    """Accept the declared two-object course using public initial state only."""
    return not state.candidate_positions


def _select_polarity_only_seeds(
    client: LuantiLiveClient,
    *,
    seed_start: int,
    count: int,
    stride: int,
    episode_prefix: str,
) -> list[int]:
    """Screen task shape before data collection without inspecting rewards or seeds.

    The delayed-polarity model represents only a verified key/decoy mode.  Some
    generated worlds instead expose a public four-candidate layout, which is a
    distinct task with a different fact interface.  Those worlds are excluded
    for every policy before training/evaluation; their candidate choice is not
    inferred, and no reward or hidden state participates in selection.
    """
    if (
        isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0
        or isinstance(count, bool) or not isinstance(count, int) or count < 1
        or isinstance(stride, bool) or not isinstance(stride, int) or stride < 1
        or not isinstance(episode_prefix, str) or not episode_prefix
    ):
        raise ValueError("invalid public polarity seed-selection parameters")
    selected: list[int] = []
    candidate, attempts = seed_start, 0
    max_attempts = count * 16
    while len(selected) < count and attempts < max_attempts:
        state, _vision = client.begin_object_episode_with_vision(
            f"{episode_prefix}-screen-{attempts}", candidate,
        )
        if _is_polarity_only_world(state):
            selected.append(candidate)
        candidate += stride
        attempts += 1
    if len(selected) != count:
        raise RuntimeError("could not find enough public two-object polarity worlds")
    return selected


def _collect(client: LuantiLiveClient, seed: int, *, delay_observations: int) -> Trajectory:
    state, vision = client.begin_object_episode_with_vision(f"delay-teacher-{delay_observations}-{seed}", seed)
    trajectory = Trajectory(seed)
    planner, causal, facts = VerifiedFactObjectPlanner(), CausalControlWorldModel(), VerifiedObjectFactMemory()
    _append(trajectory, state, vision, facts)
    for _ in range(MAX_ACTIONS):
        actual = planner.choose_actual(
            state,
            verified_key=facts.key_verified,
            verified_charge=facts.charge_verified,
            charge_mode=facts.charge_mode,
        )
        if actual is None:
            break
        trajectory.labels[-1] = actual
        requested = causal.request_for_actual(actual)
        before = state
        interaction, state, vision = client.step_object_with_vision(requested)
        if state is None or vision is None or interaction.outcome.get("status") != "accepted":
            break
        causal.observe(before.base, requested, state.base)
        facts.observe_reward(interaction.reward)
        _append(trajectory, state, vision, facts)
        if _prerequisite(_reward(interaction)):
            recenter, state, vision = client.recenter_object_with_vision()
            if state is None or vision is None or recenter.outcome.get("status") != "accepted":
                break
            _append(trajectory, state, vision, facts)
            for _ in range(delay_observations):
                observation, state, vision = client.observe_object_with_vision()
                if state is None or vision is None or observation.outcome.get("status") != "accepted":
                    break
                _append(trajectory, state, vision, facts)
        if state.completed:
            break
    if not state.completed:
        raise RuntimeError(f"delay teacher failed seed {seed} at delay {delay_observations}")
    return trajectory


def _pad(values: Sequence[Any]) -> list[Any]:
    prefix = list(values[-HISTORY:])
    return [prefix[0]] * (HISTORY - len(prefix)) + prefix


def _tensors(trajectories: Sequence[Trajectory]) -> tuple[Tensor, Tensor, Tensor]:
    fact_windows: list[list[tuple[float, ...]]] = []
    public_windows: list[list[tuple[float, ...]]] = []
    labels: list[int] = []
    for trajectory in trajectories:
        for index, direction in enumerate(trajectory.labels):
            if direction is None:
                continue
            states = _pad(trajectory.states[: index + 1])
            visions = _pad(trajectory.visions[: index + 1])
            keys = _pad(trajectory.key_facts[: index + 1])
            modes = _pad(trajectory.charge_modes[: index + 1])
            fact_windows.append([
                object_polarity_fact_memory_visual_frame_features(state, vision, verified_key=key, charge_mode=mode)
                for state, vision, key, mode in zip(states, visions, keys, modes)
            ])
            public_windows.append([object_reward_visual_frame_features(state, vision) for state, vision in zip(states, visions)])
            labels.append(DIRECTION_TO_INDEX[direction])
    return (
        torch.tensor(fact_windows, dtype=torch.float32),
        torch.tensor(public_windows, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def _observe_delay(client: LuantiLiveClient, history: list[tuple[ObjectTaskState, Mapping[str, Any]]], count: int) -> tuple[ObjectTaskState, Mapping[str, Any]]:
    state, vision = history[-1]
    for _ in range(count):
        interaction, state, vision = client.observe_object_with_vision()
        if state is None or vision is None or interaction.outcome.get("status") != "accepted":
            raise RuntimeError("Luanti delayed observation was not accepted")
        history.append((state, dict(vision)))
    return state, vision


def _evaluate(
    client: LuantiLiveClient,
    *,
    kind: str,
    seed_values: Sequence[int],
    delay_observations: int,
    fact_memory: PolarityFactMemoryCurrentVisionObjectPolicy,
    no_memory: NoMemoryRewardVisionObjectPolicy,
    recurrent: PublicEventGRUPolicy,
    transformer: PublicEventTransformerPolicy,
) -> tuple[Evaluation, int]:
    successes, revisions, steps = 0, 0, []
    planner = VerifiedFactObjectPlanner()
    for index, seed in enumerate(seed_values):
        state, vision = client.begin_object_episode_with_vision(f"delay-{delay_observations}-{kind}-{index}", seed)
        history = [(state, dict(vision))]
        causal, facts = CausalControlWorldModel(), VerifiedObjectFactMemory()
        for step in range(MAX_ACTIONS):
            if kind == "teacher":
                actual = planner.choose_actual(state, verified_key=facts.key_verified, verified_charge=facts.charge_verified, charge_mode=facts.charge_mode)
            elif kind == "fact_memory":
                actual = fact_memory.choose(history[-HISTORY:], verified_key=facts.key_verified, charge_mode=facts.charge_mode)
            elif kind == "no_memory":
                actual = no_memory.choose(history[-HISTORY:])
            elif kind == "recurrent":
                actual = recurrent.choose(history[-HISTORY:])
            elif kind == "transformer":
                actual = transformer.choose(history[-HISTORY:])
            else:
                raise ValueError(f"unsupported delay evaluation policy {kind}")
            requested = causal.request_for_actual(actual)
            before = state
            interaction, state, vision = client.step_object_with_vision(requested)
            if state is None or vision is None or interaction.outcome.get("status") != "accepted":
                break
            causal.observe(before.base, requested, state.base)
            if kind in {"teacher", "fact_memory"}:
                facts.observe_reward(interaction.reward)
            history.append((state, dict(vision)))
            if _prerequisite(_reward(interaction)):
                recenter, state, vision = client.recenter_object_with_vision()
                if state is None or vision is None or recenter.outcome.get("status") != "accepted":
                    break
                history.append((state, dict(vision)))
                state, vision = _observe_delay(client, history, delay_observations)
            if state.completed:
                successes += 1
                steps.append(step + 1)
                break
        else:
            steps.append(MAX_ACTIONS)
        revisions += causal.revision_count
    return Evaluation(successes, len(seed_values), sum(steps) / len(seed_values)), revisions


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run(
    *,
    delays: Sequence[int] = (0, 16, 32),
    device: str | torch.device | None = None,
    bridge_root: Path = BRIDGE_ROOT,
    artifact_name: str = "luanti-polarity-delay-scale-report.json",
    train_seed_start: int = 4_100_000,
    heldout_seed_start: int = 4_000_000,
    model_seed: int = 3_000,
) -> Report:
    if not delays or any(isinstance(delay, bool) or not isinstance(delay, int) or delay < 0 for delay in delays):
        raise ValueError("delays must be non-empty non-negative integers")
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise ValueError("artifact_name must be a filename")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (train_seed_start, heldout_seed_start, model_seed)):
        raise ValueError("seed starts must be non-negative integers")
    resolved = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    # Course membership is established from the public initial state before
    # any trajectory action/reward.  It prevents a multi-choice task from
    # silently entering a two-choice polarity comparison.
    train_seeds = _select_polarity_only_seeds(
        client,
        seed_start=train_seed_start,
        count=16,
        stride=1,
        episode_prefix="delay-train",
    )
    heldout_seeds = _select_polarity_only_seeds(
        client,
        seed_start=heldout_seed_start,
        count=8,
        stride=31,
        episode_prefix="delay-heldout",
    )
    points: list[ScalePoint] = []
    for delay in delays:
        trajectories = [_collect(client, seed, delay_observations=delay) for seed in train_seeds]
        fact_data, public_data, labels = _tensors(trajectories)
        torch.manual_seed(model_seed + delay)
        fact_memory = PolarityFactMemoryCurrentVisionObjectPolicy().to(resolved)
        no_memory = NoMemoryRewardVisionObjectPolicy().to(resolved)
        recurrent = PublicEventGRUPolicy().to(resolved)
        transformer = PublicEventTransformerPolicy().to(resolved)
        train_policy(fact_memory, fact_data, labels, device=resolved, steps=1_600, seed=model_seed + 100 + delay)
        train_policy(no_memory, public_data, labels, device=resolved, steps=1_600, seed=model_seed + 200 + delay)
        train_policy(recurrent, public_data, labels, device=resolved, steps=1_600, seed=model_seed + 300 + delay)
        train_policy(transformer, public_data, labels, device=resolved, steps=1_600, seed=model_seed + 400 + delay)
        for model in (fact_memory, no_memory, recurrent, transformer):
            model.eval()
        evaluations = {}
        revisions = {}
        for kind in ("fact_memory", "no_memory", "recurrent", "transformer", "teacher"):
            evaluations[kind], revisions[kind] = _evaluate(
                client,
                kind=kind,
                seed_values=heldout_seeds,
                delay_observations=delay,
                fact_memory=fact_memory,
                no_memory=no_memory,
                recurrent=recurrent,
                transformer=transformer,
            )
        points.append(ScalePoint(
            delay,
            len(labels),
            evaluations["fact_memory"],
            evaluations["no_memory"],
            evaluations["recurrent"],
            evaluations["transformer"],
            evaluations["teacher"],
            revisions["fact_memory"],
            {
                "fact_memory": _parameter_count(fact_memory),
                "no_memory": _parameter_count(no_memory),
                "recurrent": _parameter_count(recurrent),
                "transformer": _parameter_count(transformer),
            },
        ))
    highest = points[-1]
    accepted = (
        all(point.fact_memory.success_rate >= 0.75 and point.teacher.success_rate == 1.0 for point in points)
        and highest.fact_memory.success_rate >= highest.no_memory.success_rate + 0.25
        and highest.fact_memory.success_rate >= highest.recurrent.success_rate + 0.25
        and highest.fact_memory.success_rate >= highest.transformer.success_rate + 0.25
    )
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    artifact = artifacts / artifact_name
    report = Report(
        "polarity_delay_v1: real Luanti hidden-reward observations; raw public reward/state/vision history for all baselines; verified fact only for fact-memory controller; teacher-labelled training; not an end-to-end architecture claim",
        str(resolved), HISTORY, train_seeds, heldout_seeds, points, accepted, str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 16, 32])
    parser.add_argument("--artifact-name", default="luanti-polarity-delay-scale-report.json")
    parser.add_argument("--train-seed-start", type=int, default=4_100_000)
    parser.add_argument("--heldout-seed-start", type=int, default=4_000_000)
    parser.add_argument("--model-seed", type=int, default=3_000)
    args = parser.parse_args()
    outcome = run(
        delays=args.delays, artifact_name=args.artifact_name,
        train_seed_start=args.train_seed_start, heldout_seed_start=args.heldout_seed_start,
        model_seed=args.model_seed,
    )
    print(json.dumps(asdict(outcome), indent=2))
    # A sub-context run is a smoke/reference point, not a complete curve.  It
    # must write its measurement but should not be reported as a failed gate.
    if max(args.delays) >= HISTORY and not outcome.accepted:
        raise SystemExit("delay scale curve did not meet its predeclared evidence gate")

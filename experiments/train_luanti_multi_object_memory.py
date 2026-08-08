"""Matched four-candidate delayed-choice evaluation in real Luanti worlds.

This is a bounded fact-memory-system study.  It is teacher-labelled and does
not claim an end-to-end PLWM architecture advantage.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_behavior import Evaluation, train_policy
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    DIRECTIONS,
    DIRECTION_TO_INDEX,
    VISION_FEATURE_DIM,
    CausalControlWorldModel,
    LuantiLiveClient,
    MultiChoiceFactMemoryCurrentVisionObjectPolicy,
    ObjectTaskState,
    VerifiedFactObjectPlanner,
    VerifiedObjectFactMemory,
    object_multi_choice_fact_memory_visual_frame_features,
    object_multi_choice_public_visual_frame_features,
)


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")
HISTORY = 16
DELAY_OBSERVATIONS = 16
MAX_ACTIONS = 48
PUBLIC_INPUT_DIM = 23 + 8 + VISION_FEATURE_DIM


class CurrentPublicPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(PUBLIC_INPUT_DIM, 64), nn.Tanh(), nn.Linear(64, len(DIRECTIONS)))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.network(observations[:, -1])
        return logits, torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]]) -> str:
        state, vision = history[-1]
        device = next(self.parameters()).device
        logits, _ = self(torch.tensor(object_multi_choice_public_visual_frame_features(state, vision), dtype=torch.float32, device=device).view(1, 1, -1))
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class PublicGRUPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.GRU(PUBLIC_INPUT_DIM, 48, batch_first=True)
        self.head = nn.Linear(48, len(DIRECTIONS))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        encoded, _ = self.encoder(observations)
        return self.head(encoded[:, -1]), torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]]) -> str:
        context = _pad(history)
        device = next(self.parameters()).device
        values = [object_multi_choice_public_visual_frame_features(state, vision) for state, vision in context]
        logits, _ = self(torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0))
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class PublicTransformerPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Linear(PUBLIC_INPUT_DIM, 48)
        self.position = nn.Parameter(torch.zeros(1, HISTORY, 48))
        layer = nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=96, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(48, len(DIRECTIONS))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(self.project(observations) + self.position[:, : observations.shape[1]])
        return self.head(encoded[:, -1]), torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]]) -> str:
        context = _pad(history)
        device = next(self.parameters()).device
        values = [object_multi_choice_public_visual_frame_features(state, vision) for state, vision in context]
        logits, _ = self(torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0))
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


@dataclass
class Trajectory:
    states: list[ObjectTaskState] = field(default_factory=list)
    visions: list[Mapping[str, Any]] = field(default_factory=list)
    labels: list[str | None] = field(default_factory=list)
    keys: list[bool] = field(default_factory=list)
    choices: list[int | None] = field(default_factory=list)


@dataclass
class Report:
    device: str
    training_samples: int
    delay_observations: int
    train_seeds: list[int]
    heldout_seeds: list[int]
    fact_memory: Evaluation
    no_memory: Evaluation
    recurrent: Evaluation
    transformer: Evaluation
    teacher: Evaluation
    fact_memory_revisions: int
    parameter_counts: dict[str, int]
    accepted: bool
    artifact: str


def _choice_for_curriculum(seed: int) -> int:
    return ((seed + 97) * 48271 % 2_147_483_647 // 65_536) % 4 + 1


def _balanced_train_seeds() -> list[int]:
    selected, counts, candidate = [], {1: 0, 2: 0, 3: 0, 4: 0}, 5_100_000
    while min(counts.values()) < 4:
        choice = _choice_for_curriculum(candidate)
        if counts[choice] < 4:
            selected.append(candidate)
            counts[choice] += 1
        candidate += 1
    return selected


def _balanced_heldout_seeds() -> list[int]:
    selected, counts, candidate = [], {1: 0, 2: 0, 3: 0, 4: 0}, 5_000_000
    while min(counts.values()) < 2:
        choice = _choice_for_curriculum(candidate)
        if counts[choice] < 2:
            selected.append(candidate)
            counts[choice] += 1
        candidate += 1
    return selected


def _append(trajectory: Trajectory, state: ObjectTaskState, vision: Mapping[str, Any], facts: VerifiedObjectFactMemory) -> None:
    trajectory.states.append(state); trajectory.visions.append(dict(vision)); trajectory.labels.append(None)
    trajectory.keys.append(facts.key_verified); trajectory.choices.append(facts.charge_choice)


def _pad(values: Sequence[Any]) -> list[Any]:
    prefix = list(values[-HISTORY:])
    return [prefix[0]] * (HISTORY - len(prefix)) + prefix


def _collect(client: LuantiLiveClient, seed: int) -> Trajectory:
    state, vision = client.begin_object_episode_with_vision(f"multi-teacher-{seed}", seed)
    trajectory, planner, causal, facts = Trajectory(), VerifiedFactObjectPlanner(), CausalControlWorldModel(), VerifiedObjectFactMemory()
    _append(trajectory, state, vision, facts)
    for _ in range(MAX_ACTIONS):
        actual = planner.choose_actual(state, verified_key=facts.key_verified, verified_charge=facts.charge_verified, charge_choice=facts.charge_choice)
        trajectory.labels[-1] = actual
        requested, before = causal.request_for_actual(actual), state
        interaction, state, vision = client.step_object_with_vision(requested)
        if state is None or vision is None or interaction.outcome.get("status") != "accepted" or state.base.failed:
            break
        causal.observe(before.base, requested, state.base)
        was_charged = facts.charge_verified
        facts.observe_reward(interaction.reward)
        _append(trajectory, state, vision, facts)
        if facts.charge_verified and not was_charged:
            recenter, state, vision = client.recenter_object_with_vision()
            if state is None or vision is None or recenter.outcome.get("status") != "accepted":
                break
            _append(trajectory, state, vision, facts)
            for _ in range(DELAY_OBSERVATIONS):
                observed, state, vision = client.observe_object_with_vision()
                if state is None or vision is None or observed.outcome.get("status") != "accepted":
                    break
                _append(trajectory, state, vision, facts)
        if state.completed:
            return trajectory
    raise RuntimeError(f"multi-object teacher failed seed {seed}")


def _tensors(trajectories: Sequence[Trajectory]) -> tuple[Tensor, Tensor, Tensor]:
    facts, public, labels = [], [], []
    for trajectory in trajectories:
        for index, direction in enumerate(trajectory.labels):
            if direction is None:
                continue
            states, visions, keys, choices = (_pad(trajectory.states[:index + 1]), _pad(trajectory.visions[:index + 1]), _pad(trajectory.keys[:index + 1]), _pad(trajectory.choices[:index + 1]))
            facts.append([object_multi_choice_fact_memory_visual_frame_features(state, vision, verified_key=key, charge_choice=choice) for state, vision, key, choice in zip(states, visions, keys, choices)])
            public.append([object_multi_choice_public_visual_frame_features(state, vision) for state, vision in zip(states, visions)])
            labels.append(DIRECTION_TO_INDEX[direction])
    return torch.tensor(facts, dtype=torch.float32), torch.tensor(public, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def _evaluate(client: LuantiLiveClient, kind: str, seeds: Sequence[int], fact: MultiChoiceFactMemoryCurrentVisionObjectPolicy, current: CurrentPublicPolicy, recurrent: PublicGRUPolicy, transformer: PublicTransformerPolicy) -> tuple[Evaluation, int]:
    successes, revisions, steps = 0, 0, []
    planner = VerifiedFactObjectPlanner()
    for index, seed in enumerate(seeds):
        state, vision = client.begin_object_episode_with_vision(f"multi-{kind}-{index}", seed)
        history, causal, facts = [(state, dict(vision))], CausalControlWorldModel(), VerifiedObjectFactMemory()
        for step in range(MAX_ACTIONS):
            if kind == "teacher": actual = planner.choose_actual(state, verified_key=facts.key_verified, verified_charge=facts.charge_verified, charge_choice=facts.charge_choice)
            elif kind == "fact_memory": actual = fact.choose(history, verified_key=facts.key_verified, charge_choice=facts.charge_choice)
            elif kind == "no_memory": actual = current.choose(history)
            elif kind == "recurrent": actual = recurrent.choose(history)
            else: actual = transformer.choose(history)
            requested, before = causal.request_for_actual(actual), state
            interaction, state, vision = client.step_object_with_vision(requested)
            if state is None or vision is None or interaction.outcome.get("status") != "accepted" or state.base.failed:
                break
            causal.observe(before.base, requested, state.base)
            was_charged = facts.charge_verified
            if kind in {"teacher", "fact_memory"}: facts.observe_reward(interaction.reward)
            history.append((state, dict(vision)))
            if facts.charge_verified and not was_charged:
                recenter, state, vision = client.recenter_object_with_vision()
                if state is None or vision is None or recenter.outcome.get("status") != "accepted": break
                history.append((state, dict(vision)))
                for _ in range(DELAY_OBSERVATIONS):
                    observed, state, vision = client.observe_object_with_vision()
                    if state is None or vision is None or observed.outcome.get("status") != "accepted": break
                    history.append((state, dict(vision)))
            if state.completed:
                successes += 1; steps.append(step + 1); break
        else:
            steps.append(MAX_ACTIONS)
        if len(steps) <= index: steps.append(MAX_ACTIONS)
        revisions += causal.revision_count
    return Evaluation(successes, len(seeds), sum(steps) / len(seeds)), revisions


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> Report:
    resolved, client = resolve_device(device), LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    train_seeds, heldout_seeds = _balanced_train_seeds(), _balanced_heldout_seeds()
    fact_data, public_data, labels = _tensors([_collect(client, seed) for seed in train_seeds])
    torch.manual_seed(5_000)
    fact, current, recurrent, transformer = (MultiChoiceFactMemoryCurrentVisionObjectPolicy().to(resolved), CurrentPublicPolicy().to(resolved), PublicGRUPolicy().to(resolved), PublicTransformerPolicy().to(resolved))
    for model, data, seed in ((fact, fact_data, 5_001), (current, public_data, 5_002), (recurrent, public_data, 5_003), (transformer, public_data, 5_004)):
        train_policy(model, data, labels, device=resolved, steps=1_800, seed=seed); model.eval()
    evaluations = {}; revisions = {}
    for kind in ("fact_memory", "no_memory", "recurrent", "transformer", "teacher"):
        evaluations[kind], revisions[kind] = _evaluate(client, kind, heldout_seeds, fact, current, recurrent, transformer)
    artifacts = ROOT / "artifacts"; artifacts.mkdir(exist_ok=True)
    artifact = artifacts / "luanti-multi-object-memory-report.json"
    checkpoint = artifacts / "luanti-multi-object-memory-policy.pt"
    torch.save({"state_dict": fact.state_dict(), "train_seeds": train_seeds}, checkpoint)
    counts = {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in {"fact_memory": fact, "no_memory": current, "recurrent": recurrent, "transformer": transformer}.items()}
    report = Report(str(resolved), len(labels), DELAY_OBSERVATIONS, train_seeds, heldout_seeds, evaluations["fact_memory"], evaluations["no_memory"], evaluations["recurrent"], evaluations["transformer"], evaluations["teacher"], revisions["fact_memory"], counts, False, str(artifact))
    report.accepted = report.fact_memory.success_rate >= .75 and report.teacher.success_rate == 1.0 and report.fact_memory.success_rate >= report.no_memory.success_rate + .25 and report.fact_memory.success_rate >= report.recurrent.success_rate + .25 and report.fact_memory.success_rate >= report.transformer.success_rate + .25
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run(); print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted: raise SystemExit("multi-object memory evaluation did not meet its evidence gate")

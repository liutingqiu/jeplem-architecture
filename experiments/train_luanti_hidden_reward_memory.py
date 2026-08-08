"""Hidden-fact Luanti curriculum: retain a key reward through later control work."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_behavior import Evaluation, _object_state_dict, train_policy
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    CausalControlWorldModel,
    DIRECTION_TO_INDEX,
    DIRECTIONS,
    LuantiLiveClient,
    NoMemoryRewardVisionObjectPolicy,
    ObjectTaskState,
    FactMemoryVisionObjectPLWM,
    FactMemoryCurrentVisionObjectPolicy,
    VerifiedObjectFactMemory,
    object_fact_memory_visual_frame_features,
)


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")
HISTORY = 16
MAX_STEPS = 28


@dataclass
class MemoryTrajectory:
    episode_id: str
    task_seed: int
    states: list[dict[str, Any]] = field(default_factory=list)
    visions: list[dict[str, Any]] = field(default_factory=list)
    actual_labels: list[str | None] = field(default_factory=list)
    verified_keys: list[bool] = field(default_factory=list)


@dataclass
class HiddenMemoryReport:
    device: str
    training_samples: int
    learned: Evaluation
    no_memory: Evaluation
    random: Evaluation
    teacher: Evaluation
    learned_revisions: int
    accepted: bool
    checkpoint: str
    artifact: str


def _reward_value(interaction) -> float:
    value = (interaction.reward or {}).get("value")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _teacher_direction(state: ObjectTaskState, *, has_key: bool) -> str | None:
    # The label generator sees the same causal reward event as the learner,
    # not the hidden Lua collection flag omitted from the observation.
    return replace(state, key_collected=has_key).oracle_actual_direction()


def _collect(client: LuantiLiveClient, *, seed: int, episode_id: str) -> MemoryTrajectory:
    state, vision = client.begin_object_episode_with_vision(episode_id, seed)
    trajectory = MemoryTrajectory(episode_id, seed, [_object_state_dict(state)], [dict(vision)], [None], [False])
    mechanism = CausalControlWorldModel()
    has_key = False
    for step in range(MAX_STEPS):
        actual = _teacher_direction(state, has_key=has_key)
        if actual is None:
            break
        trajectory.actual_labels[-1] = actual
        requested = mechanism.request_for_actual(actual)
        before = state
        interaction, next_state, next_vision = client.step_object_with_vision(requested)
        if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
            break
        mechanism.observe(before.base, requested, next_state.base)
        collected_now = not has_key and _reward_value(interaction) >= 0.19
        has_key = has_key or collected_now
        state, vision = next_state, next_vision
        trajectory.states.append(_object_state_dict(state))
        trajectory.visions.append(dict(vision))
        trajectory.actual_labels.append(None)
        trajectory.verified_keys.append(has_key)
        if step == 0 or collected_now:
            recenter, recentered_state, recentered_vision = client.recenter_object_with_vision()
            if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                break
            state, vision = recentered_state, recentered_vision
            trajectory.states.append(_object_state_dict(state))
            trajectory.visions.append(dict(vision))
            trajectory.actual_labels.append(None)
            trajectory.verified_keys.append(has_key)
        if state.completed:
            break
    return trajectory


def _tensors(trajectories: Sequence[MemoryTrajectory]) -> tuple[torch.Tensor, torch.Tensor]:
    windows: list[list[tuple[float, ...]]] = []
    labels: list[int] = []
    for trajectory in trajectories:
        states = [ObjectTaskState.from_payload(payload) for payload in trajectory.states]
        for index, direction in enumerate(trajectory.actual_labels):
            if direction is None:
                continue
            start = max(0, index - HISTORY + 1)
            state_prefix = states[start : index + 1]
            vision_prefix = trajectory.visions[start : index + 1]
            verified_prefix = trajectory.verified_keys[start : index + 1]
            while len(state_prefix) < HISTORY:
                state_prefix.insert(0, state_prefix[0])
                vision_prefix.insert(0, vision_prefix[0])
                verified_prefix.insert(0, verified_prefix[0])
            windows.append([
                object_fact_memory_visual_frame_features(item, frame, verified_key=verified)
                for item, frame, verified in zip(state_prefix, vision_prefix, verified_prefix)
            ])
            labels.append(DIRECTION_TO_INDEX[direction])
    if not labels:
        raise RuntimeError("hidden-memory curriculum produced no labels")
    return torch.tensor(windows, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def _read_trajectories(path: Path) -> list[MemoryTrajectory]:
    records: list[MemoryTrajectory] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(MemoryTrajectory(**payload))
    if not records:
        raise RuntimeError("hidden-memory rollout artifact was empty")
    return records


def _evaluate(client: LuantiLiveClient, *, chooser, seeds: Sequence[int], prefix: str, teacher: bool = False, use_fact_memory: bool = False) -> tuple[Evaluation, int]:
    successes, revisions = 0, 0
    steps_taken: list[int] = []
    for index, seed in enumerate(seeds):
        state, vision = client.begin_object_episode_with_vision(f"{prefix}-{index}", seed)
        history = [(state, dict(vision))]
        mechanism = CausalControlWorldModel()
        has_key = False
        facts = VerifiedObjectFactMemory()
        for step in range(MAX_STEPS):
            actual = _teacher_direction(state, has_key=has_key) if teacher else chooser(history, facts.key_verified)
            requested = mechanism.request_for_actual(actual or "north")
            before = state
            interaction, next_state, next_vision = client.step_object_with_vision(requested)
            if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
                break
            mechanism.observe(before.base, requested, next_state.base)
            collected_now = not has_key and _reward_value(interaction) >= 0.19
            has_key = has_key or collected_now
            if use_fact_memory:
                facts.observe_reward(interaction.reward)
            state, vision = next_state, dict(next_vision)
            history.append((state, vision))
            if step == 0 or collected_now:
                recenter, recentered_state, recentered_vision = client.recenter_object_with_vision()
                if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                    break
                state, vision = recentered_state, dict(recentered_vision)
                history.append((state, vision))
            if state.completed:
                successes += 1
                steps_taken.append(step + 1)
                break
        else:
            steps_taken.append(MAX_STEPS)
        revisions += mechanism.revision_count
    return Evaluation(successes, len(seeds), sum(steps_taken) / len(seeds)), revisions


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT, reuse_training_rollouts: bool = True) -> HiddenMemoryReport:
    resolved = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    train_seeds = [2_700_000 + index * 19 for index in range(32)]
    heldout_seeds = [2_800_000 + index * 31 for index in range(8)]
    artifact_dir = ROOT / "artifacts"
    rollout_path = artifact_dir / "luanti-hidden-reward-memory-rollouts.jsonl"
    trajectories = (
        _read_trajectories(rollout_path)
        if reuse_training_rollouts and rollout_path.exists()
        else [_collect(client, seed=seed, episode_id=f"hidden-memory-teacher-{seed}") for seed in train_seeds]
    )
    observations, labels = _tensors(trajectories)
    model = FactMemoryCurrentVisionObjectPolicy().to(resolved)
    baseline = NoMemoryRewardVisionObjectPolicy().to(resolved)
    train_policy(model, observations, labels, device=resolved, steps=1_200, seed=911)
    train_policy(baseline, observations, labels, device=resolved, steps=1_200, seed=912)
    model.eval()
    baseline.eval()
    learned, revisions = _evaluate(client, chooser=lambda history, verified: model.choose(history[-HISTORY:], verified_key=verified), seeds=heldout_seeds, prefix="hidden-memory-learned", use_fact_memory=True)
    no_memory, _ = _evaluate(client, chooser=lambda history, _verified: baseline.choose(history), seeds=heldout_seeds, prefix="hidden-memory-no-memory")
    rng = random.Random(913)
    random_result, _ = _evaluate(client, chooser=lambda _history, _verified: rng.choice(DIRECTIONS), seeds=heldout_seeds, prefix="hidden-memory-random")
    teacher_result, teacher_revisions = _evaluate(client, chooser=lambda _history, _verified: "north", seeds=heldout_seeds, prefix="hidden-memory-teacher", teacher=True)
    artifact_dir.mkdir(exist_ok=True)
    checkpoint = artifact_dir / "luanti-hidden-reward-memory-policy.pt"
    torch.save({"state_dict": model.state_dict(), "history_length": HISTORY, "training_seeds": train_seeds}, checkpoint)
    if not rollout_path.exists() or not reuse_training_rollouts:
        rollout_path.write_text("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in trajectories), encoding="utf-8")
    artifact = artifact_dir / "luanti-hidden-reward-memory-report.json"
    report = HiddenMemoryReport(
        device=str(resolved), training_samples=len(labels), learned=learned, no_memory=no_memory,
        random=random_result, teacher=teacher_result, learned_revisions=revisions,
        accepted=(
            learned.success_rate >= 0.75
            and learned.success_rate >= no_memory.success_rate + 0.25
            and learned.success_rate >= random_result.success_rate + 0.5
            and teacher_result.success_rate == 1.0
            and revisions == len(heldout_seeds)
            and teacher_revisions == len(heldout_seeds)
        ),
        checkpoint=str(checkpoint), artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("hidden reward memory curriculum did not meet its evidence gate")

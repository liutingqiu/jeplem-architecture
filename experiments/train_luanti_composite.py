"""Train object/vision policy for the high-seed Luanti rule-switch course."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_behavior import Evaluation, ExecutedAction, _executed_action, _object_state_dict, train_policy
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    CausalControlWorldModel,
    DIRECTIONS,
    LuantiLiveClient,
    NoMemoryVisionObjectPolicy,
    ObjectTaskState,
    VisionObjectGoalDirectedPLWM,
    padded_object_visual_histories,
)


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")
HISTORY = 4
MAX_STEPS = 28


@dataclass
class CompositeTrajectory:
    episode_id: str
    task_seed: int
    states: list[dict[str, Any]] = field(default_factory=list)
    visions: list[dict[str, Any]] = field(default_factory=list)
    actions: list[ExecutedAction] = field(default_factory=list)


@dataclass
class CompositeReport:
    device: str
    training_samples: int
    learned: Evaluation
    no_memory: Evaluation
    random: Evaluation
    teacher: Evaluation
    control_revisions: int
    accepted: bool
    checkpoint: str
    artifact: str


def _collect_teacher(client: LuantiLiveClient, *, task_seed: int, episode_id: str) -> CompositeTrajectory:
    state, vision = client.begin_object_episode_with_vision(episode_id, task_seed)
    trajectory = CompositeTrajectory(episode_id, task_seed, [_object_state_dict(state)], [dict(vision)])
    mechanism = CausalControlWorldModel()
    for step in range(MAX_STEPS):
        actual = state.oracle_actual_direction()
        if actual is None:
            break
        requested = mechanism.request_for_actual(actual)
        before = state
        interaction, next_state, next_vision = client.step_object_with_vision(requested)
        trajectory.actions.append(_executed_action(requested, interaction))
        if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
            break
        mechanism.observe(before.base, requested, next_state.base)
        state, vision = next_state, next_vision
        trajectory.states.append(_object_state_dict(state))
        trajectory.visions.append(dict(vision))
        if step == 0:
            recenter, recentered_state, recentered_vision = client.recenter_object_with_vision()
            trajectory.actions.append(_executed_action("recenter", recenter))
            if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                break
            state, vision = recentered_state, recentered_vision
            trajectory.states.append(_object_state_dict(state))
            trajectory.visions.append(dict(vision))
        if state.completed:
            break
    return trajectory


def _tensors(trajectories: Sequence[CompositeTrajectory]) -> tuple[torch.Tensor, torch.Tensor]:
    windows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for trajectory in trajectories:
        states = [ObjectTaskState.from_payload(payload) for payload in trajectory.states]
        skips = frozenset(index for index, action in enumerate(trajectory.actions) if action.direction == "recenter")
        rows, target = padded_object_visual_histories(
            states, trajectory.visions, history=HISTORY, skip_label_indices=skips, actual_direction_labels=True
        )
        if len(target):
            windows.append(rows)
            labels.append(target)
    if not labels:
        raise RuntimeError("composite course produced no training labels")
    return torch.cat(windows), torch.cat(labels)


def _read_trajectories(path: Path) -> list[CompositeTrajectory]:
    records: list[CompositeTrajectory] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records.append(
                CompositeTrajectory(
                    episode_id=payload["episode_id"],
                    task_seed=int(payload["task_seed"]),
                    states=payload["states"],
                    visions=payload["visions"],
                    actions=[ExecutedAction(**action) for action in payload["actions"]],
                )
            )
    if not records:
        raise RuntimeError("composite rollout artifact was empty")
    return records


def _evaluate(
    client: LuantiLiveClient,
    *,
    chooser,
    seeds: Sequence[int],
    prefix: str,
) -> tuple[Evaluation, int]:
    successes, revisions = 0, 0
    steps_taken: list[int] = []
    for index, seed in enumerate(seeds):
        state, vision = client.begin_object_episode_with_vision(f"{prefix}-{index}", seed)
        history = [(state, dict(vision))]
        mechanism = CausalControlWorldModel()
        for step in range(MAX_STEPS):
            actual = chooser(history)
            requested = mechanism.request_for_actual(actual)
            before = state
            interaction, next_state, next_vision = client.step_object_with_vision(requested)
            if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
                break
            mechanism.observe(before.base, requested, next_state.base)
            state, vision = next_state, dict(next_vision)
            history.append((state, vision))
            if step == 0:
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


def run(
    *,
    device: str | torch.device | None = None,
    bridge_root: Path = BRIDGE_ROOT,
    training_steps: int = 1_600,
    reuse_training_rollouts: bool = True,
) -> CompositeReport:
    resolved = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    training_seeds = [2_500_000 + index * 19 for index in range(24)]
    heldout_seeds = [2_600_000 + index * 31 for index in range(8)]
    artifact_dir = ROOT / "artifacts"
    rollout_path = artifact_dir / "luanti-composite-rollouts.jsonl"
    trajectories = (
        _read_trajectories(rollout_path)
        if reuse_training_rollouts and rollout_path.exists()
        else [_collect_teacher(client, task_seed=seed, episode_id=f"composite-teacher-{seed}") for seed in training_seeds]
    )
    observations, labels = _tensors(trajectories)
    model = VisionObjectGoalDirectedPLWM(history_length=HISTORY).to(resolved)
    baseline = NoMemoryVisionObjectPolicy().to(resolved)
    train_policy(model, observations, labels, device=resolved, steps=training_steps, seed=801)
    train_policy(baseline, observations, labels, device=resolved, steps=training_steps, seed=802)
    model.eval()
    baseline.eval()
    learned, revisions = _evaluate(client, chooser=lambda history: model.choose(history[-HISTORY:]), seeds=heldout_seeds, prefix="composite-learned")
    no_memory, _ = _evaluate(client, chooser=lambda history: baseline.choose(history), seeds=heldout_seeds, prefix="composite-no-memory")
    rng = random.Random(803)
    random_policy, _ = _evaluate(client, chooser=lambda _history: rng.choice(DIRECTIONS), seeds=heldout_seeds, prefix="composite-random")
    teacher, teacher_revisions = _evaluate(
        client,
        chooser=lambda history: history[-1][0].oracle_actual_direction() or "north",
        seeds=heldout_seeds,
        prefix="composite-teacher",
    )
    artifact_dir.mkdir(exist_ok=True)
    checkpoint = artifact_dir / "luanti-composite-vision-object-policy.pt"
    torch.save({"state_dict": model.state_dict(), "history_length": HISTORY, "training_seeds": training_seeds}, checkpoint)
    if not rollout_path.exists() or not reuse_training_rollouts:
        rollout_path.write_text("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in trajectories), encoding="utf-8")
    artifact = artifact_dir / "luanti-composite-training-report.json"
    report = CompositeReport(
        device=str(resolved),
        training_samples=len(labels),
        learned=learned,
        no_memory=no_memory,
        random=random_policy,
        teacher=teacher,
        control_revisions=revisions,
        accepted=(
            learned.success_rate >= 0.75
            and learned.success_rate >= no_memory.success_rate + 0.25
            and learned.success_rate >= random_policy.success_rate + 0.5
            and teacher.success_rate == 1.0
            and revisions == len(heldout_seeds)
            and teacher_revisions == len(heldout_seeds)
        ),
        checkpoint=str(checkpoint),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("composite visual object course did not meet its evidence gate")

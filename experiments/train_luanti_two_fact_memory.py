"""Learned-controller audit for the real Luanti two-fact course."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_behavior import Evaluation, _object_state_dict, train_policy
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    CausalControlWorldModel,
    DIRECTION_TO_INDEX,
    DIRECTIONS,
    GRUObjectPolicy,
    LuantiLiveClient,
    NoMemoryRewardVisionObjectPolicy,
    ObjectTaskState,
    TwoFactMemoryCurrentVisionObjectPolicy,
    VerifiedFactObjectPlanner,
    VerifiedObjectFactMemory,
    object_reward_visual_frame_features,
    object_two_fact_memory_visual_frame_features,
)


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")
HISTORY = 16
# The demonstrated teacher route takes 33 actions. Forty leaves recovery room
# while keeping serial, provenance-preserving real-world baseline evaluation
# within the local execution budget.
MAX_STEPS = 40


@dataclass
class TwoFactTrajectory:
    episode_id: str
    task_seed: int
    states: list[dict[str, Any]] = field(default_factory=list)
    visions: list[dict[str, Any]] = field(default_factory=list)
    actual_labels: list[str | None] = field(default_factory=list)
    verified_keys: list[bool] = field(default_factory=list)
    verified_charges: list[bool] = field(default_factory=list)


@dataclass
class TwoFactLearningReport:
    device: str
    training_samples: int
    learned: Evaluation
    no_memory: Evaluation
    recurrent: Evaluation
    random: Evaluation
    teacher: Evaluation
    learned_revisions: int
    accepted: bool
    checkpoint: str
    artifact: str


def _reward_value(interaction: Any) -> float:
    value = (interaction.reward or {}).get("value")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _is_prerequisite(reward: float) -> bool:
    return reward in (0.15, 0.2)


def _collect(client: LuantiLiveClient, *, seed: int, episode_id: str) -> TwoFactTrajectory:
    state, vision = client.begin_object_episode_with_vision(episode_id, seed)
    trajectory = TwoFactTrajectory(
        episode_id,
        seed,
        [_object_state_dict(state)],
        [dict(vision)],
        [None],
        [False],
        [False],
    )
    planner = VerifiedFactObjectPlanner()
    mechanism = CausalControlWorldModel()
    facts = VerifiedObjectFactMemory()
    for _step in range(MAX_STEPS):
        actual = planner.choose_actual(state, verified_key=facts.key_verified, verified_charge=facts.charge_verified)
        if actual is None:
            break
        trajectory.actual_labels[-1] = actual
        requested = mechanism.request_for_actual(actual)
        before = state
        interaction, next_state, next_vision = client.step_object_with_vision(requested)
        if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
            break
        mechanism.observe(before.base, requested, next_state.base)
        facts.observe_reward(interaction.reward)
        reward = _reward_value(interaction)
        state, vision = next_state, dict(next_vision)
        trajectory.states.append(_object_state_dict(state))
        trajectory.visions.append(vision)
        trajectory.actual_labels.append(None)
        trajectory.verified_keys.append(facts.key_verified)
        trajectory.verified_charges.append(facts.charge_verified)
        if _is_prerequisite(reward):
            recenter, recentered_state, recentered_vision = client.recenter_object_with_vision()
            if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                break
            state, vision = recentered_state, dict(recentered_vision)
            trajectory.states.append(_object_state_dict(state))
            trajectory.visions.append(vision)
            trajectory.actual_labels.append(None)
            trajectory.verified_keys.append(facts.key_verified)
            trajectory.verified_charges.append(facts.charge_verified)
        if state.completed:
            break
    if not state.completed:
        raise RuntimeError(f"teacher failed two-fact seed {seed}")
    return trajectory


def _padded(prefix: Sequence[Any]) -> list[Any]:
    items = list(prefix[-HISTORY:])
    return [items[0]] * (HISTORY - len(items)) + items


def _tensors(trajectories: Sequence[TwoFactTrajectory]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    two_fact_windows: list[list[tuple[float, ...]]] = []
    current_windows: list[list[tuple[float, ...]]] = []
    recurrent_windows: list[list[tuple[float, ...]]] = []
    labels: list[int] = []
    for trajectory in trajectories:
        states = [ObjectTaskState.from_payload(payload) for payload in trajectory.states]
        for index, direction in enumerate(trajectory.actual_labels):
            if direction is None:
                continue
            states_prefix = _padded(states[: index + 1])
            vision_prefix = _padded(trajectory.visions[: index + 1])
            key_prefix = _padded(trajectory.verified_keys[: index + 1])
            charge_prefix = _padded(trajectory.verified_charges[: index + 1])
            two_fact_windows.append([
                object_two_fact_memory_visual_frame_features(
                    item, frame, verified_key=key, verified_charge=charge
                )
                for item, frame, key, charge in zip(states_prefix, vision_prefix, key_prefix, charge_prefix)
            ])
            current_windows.append([
                object_reward_visual_frame_features(item, frame) for item, frame in zip(states_prefix, vision_prefix)
            ])
            recurrent_windows.append([item.features() for item in states_prefix])
            labels.append(DIRECTION_TO_INDEX[direction])
    if not labels:
        raise RuntimeError("two-fact collection produced no labels")
    return (
        torch.tensor(two_fact_windows, dtype=torch.float32),
        torch.tensor(current_windows, dtype=torch.float32),
        torch.tensor(recurrent_windows, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def _evaluate(
    client: LuantiLiveClient,
    *,
    chooser: Callable[[list[tuple[ObjectTaskState, dict[str, Any]]], VerifiedObjectFactMemory], str],
    seeds: Sequence[int],
    prefix: str,
    retain_facts: bool,
    teacher: bool = False,
) -> tuple[Evaluation, int]:
    successes, revisions = 0, 0
    steps_taken: list[int] = []
    planner = VerifiedFactObjectPlanner()
    for index, seed in enumerate(seeds):
        state, vision = client.begin_object_episode_with_vision(f"{prefix}-{index}", seed)
        history = [(state, dict(vision))]
        mechanism = CausalControlWorldModel()
        facts = VerifiedObjectFactMemory()
        for step in range(MAX_STEPS):
            actual = (
                planner.choose_actual(state, verified_key=facts.key_verified, verified_charge=facts.charge_verified)
                if teacher
                else chooser(history, facts)
            )
            requested = mechanism.request_for_actual(actual or "north")
            before = state
            interaction, next_state, next_vision = client.step_object_with_vision(requested)
            if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
                break
            mechanism.observe(before.base, requested, next_state.base)
            if retain_facts:
                facts.observe_reward(interaction.reward)
            reward = _reward_value(interaction)
            state, vision = next_state, dict(next_vision)
            history.append((state, vision))
            if _is_prerequisite(reward):
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


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> TwoFactLearningReport:
    resolved = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    train_seeds = [3_100_000 + index * 19 for index in range(16)]
    heldout_seeds = [3_000_000 + index * 31 for index in range(8)]
    trajectories = [_collect(client, seed=seed, episode_id=f"two-fact-teacher-{seed}") for seed in train_seeds]
    fact_observations, current_observations, recurrent_observations, labels = _tensors(trajectories)
    learned_model = TwoFactMemoryCurrentVisionObjectPolicy().to(resolved)
    no_memory_model = NoMemoryRewardVisionObjectPolicy().to(resolved)
    recurrent_model = GRUObjectPolicy(history_length=HISTORY).to(resolved)
    train_policy(learned_model, fact_observations, labels, device=resolved, steps=1_600, seed=1_001)
    train_policy(no_memory_model, current_observations, labels, device=resolved, steps=1_600, seed=1_002)
    train_policy(recurrent_model, recurrent_observations, labels, device=resolved, steps=1_600, seed=1_003)
    learned_model.eval()
    no_memory_model.eval()
    recurrent_model.eval()
    learned, revisions = _evaluate(
        client,
        chooser=lambda history, facts: learned_model.choose(
            history[-HISTORY:], verified_key=facts.key_verified, verified_charge=facts.charge_verified
        ),
        seeds=heldout_seeds,
        prefix="two-fact-learned",
        retain_facts=True,
    )
    no_memory, _ = _evaluate(
        client,
        chooser=lambda history, _facts: no_memory_model.choose(history),
        seeds=heldout_seeds,
        prefix="two-fact-no-memory",
        retain_facts=False,
    )
    recurrent, _ = _evaluate(
        client,
        chooser=lambda history, _facts: recurrent_model.choose([state for state, _vision in history[-HISTORY:]]),
        seeds=heldout_seeds,
        prefix="two-fact-recurrent",
        retain_facts=False,
    )
    rng = random.Random(1_004)
    random_result, _ = _evaluate(
        client,
        chooser=lambda _history, _facts: rng.choice(DIRECTIONS),
        seeds=heldout_seeds,
        prefix="two-fact-random",
        retain_facts=False,
    )
    teacher_result, teacher_revisions = _evaluate(
        client,
        chooser=lambda _history, _facts: "north",
        seeds=heldout_seeds,
        prefix="two-fact-teacher",
        retain_facts=True,
        teacher=True,
    )
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    checkpoint = artifact_dir / "luanti-two-fact-memory-policy.pt"
    torch.save({"state_dict": learned_model.state_dict(), "training_seeds": train_seeds}, checkpoint)
    rollouts = artifact_dir / "luanti-two-fact-memory-rollouts.jsonl"
    rollouts.write_text("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in trajectories), encoding="utf-8")
    artifact = artifact_dir / "luanti-two-fact-memory-report.json"
    report = TwoFactLearningReport(
        device=str(resolved),
        training_samples=len(labels),
        learned=learned,
        no_memory=no_memory,
        recurrent=recurrent,
        random=random_result,
        teacher=teacher_result,
        learned_revisions=revisions,
        accepted=(
            learned.success_rate >= 0.75
            and learned.success_rate >= no_memory.success_rate + 0.25
            and learned.success_rate >= random_result.success_rate + 0.5
            and teacher_result.success_rate == 1.0
            and revisions == len(heldout_seeds)
            and teacher_revisions == len(heldout_seeds)
        ),
        checkpoint=str(checkpoint),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("two-fact learned controller did not meet its evidence gate")

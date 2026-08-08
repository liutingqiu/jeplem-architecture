"""Learn whether a controller can use a reward-verified charge polarity."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_behavior import Evaluation, _object_state_dict, train_policy
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    CausalControlWorldModel, DIRECTION_TO_INDEX, DIRECTIONS, GRUObjectPolicy,
    LuantiLiveClient, NoMemoryRewardVisionObjectPolicy, ObjectTaskState, TransformerObjectPolicy,
    PolarityFactMemoryCurrentVisionObjectPolicy, VerifiedFactObjectPlanner,
    VerifiedObjectFactMemory, object_polarity_fact_memory_visual_frame_features,
    object_reward_visual_frame_features,
)


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")
HISTORY = 16
MAX_STEPS = 40


def _charge_mode_for_curriculum(seed: int) -> str:
    """Balance generated training worlds without exposing this value to models."""
    value = ((seed + 83) * 48271) % 2_147_483_647
    return "key" if (value // 65_536) % 2 == 0 else "decoy"


def _balanced_train_seeds(seed_start: int = 4_100_000) -> list[int]:
    if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
        raise ValueError("seed_start must be a non-negative integer")
    selected: list[int] = []
    counts = {"key": 0, "decoy": 0}
    candidate = seed_start
    while min(counts.values()) < 8:
        mode = _charge_mode_for_curriculum(candidate)
        if counts[mode] < 8:
            selected.append(candidate)
            counts[mode] += 1
        candidate += 1
    return selected


@dataclass
class Trajectory:
    seed: int
    states: list[dict[str, Any]] = field(default_factory=list)
    visions: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str | None] = field(default_factory=list)
    key_facts: list[bool] = field(default_factory=list)
    charge_modes: list[str | None] = field(default_factory=list)


@dataclass
class Report:
    device: str
    training_samples: int
    learned: Evaluation
    no_memory: Evaluation
    recurrent: Evaluation
    transformer: Evaluation
    random: Evaluation
    teacher: Evaluation
    learned_revisions: int
    accepted: bool
    checkpoint: str
    artifact: str


def _reward(interaction: Any) -> float:
    value = (interaction.reward or {}).get("value")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _prerequisite(value: float) -> bool:
    return value in (0.15, 0.16, 0.2)


def _collect(client: LuantiLiveClient, seed: int) -> Trajectory:
    state, vision = client.begin_object_episode_with_vision(f"polarity-teacher-{seed}", seed)
    result = Trajectory(seed, [_object_state_dict(state)], [dict(vision)], [None], [False], [None])
    planner, causal, facts = VerifiedFactObjectPlanner(), CausalControlWorldModel(), VerifiedObjectFactMemory()
    for _ in range(MAX_STEPS):
        actual = planner.choose_actual(state, verified_key=facts.key_verified, verified_charge=facts.charge_verified, charge_mode=facts.charge_mode)
        if actual is None:
            break
        result.labels[-1] = actual
        requested = causal.request_for_actual(actual)
        before = state
        interaction, state, vision = client.step_object_with_vision(requested)
        if state is None or vision is None or interaction.outcome.get("status") != "accepted":
            break
        causal.observe(before.base, requested, state.base)
        facts.observe_reward(interaction.reward)
        result.states.append(_object_state_dict(state))
        result.visions.append(dict(vision))
        result.labels.append(None)
        result.key_facts.append(facts.key_verified)
        result.charge_modes.append(facts.charge_mode)
        if _prerequisite(_reward(interaction)):
            recenter, state, vision = client.recenter_object_with_vision()
            if state is None or vision is None or recenter.outcome.get("status") != "accepted":
                break
            result.states.append(_object_state_dict(state))
            result.visions.append(dict(vision))
            result.labels.append(None)
            result.key_facts.append(facts.key_verified)
            result.charge_modes.append(facts.charge_mode)
        if state.completed:
            break
    if not state.completed:
        raise RuntimeError(f"polarity teacher failed seed {seed}")
    return result


def _pad(values: Sequence[Any]) -> list[Any]:
    prefix = list(values[-HISTORY:])
    return [prefix[0]] * (HISTORY - len(prefix)) + prefix


def _tensors(trajectories: Sequence[Trajectory]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fact_windows: list[list[tuple[float, ...]]] = []
    current_windows: list[list[tuple[float, ...]]] = []
    recurrent_windows: list[list[tuple[float, ...]]] = []
    labels: list[int] = []
    for trajectory in trajectories:
        states = [ObjectTaskState.from_payload(item) for item in trajectory.states]
        for index, direction in enumerate(trajectory.labels):
            if direction is None:
                continue
            state_prefix, vision_prefix = _pad(states[:index + 1]), _pad(trajectory.visions[:index + 1])
            key_prefix, mode_prefix = _pad(trajectory.key_facts[:index + 1]), _pad(trajectory.charge_modes[:index + 1])
            fact_windows.append([
                object_polarity_fact_memory_visual_frame_features(item, frame, verified_key=key, charge_mode=mode)
                for item, frame, key, mode in zip(state_prefix, vision_prefix, key_prefix, mode_prefix)
            ])
            current_windows.append([object_reward_visual_frame_features(item, frame) for item, frame in zip(state_prefix, vision_prefix)])
            recurrent_windows.append([item.features() for item in state_prefix])
            labels.append(DIRECTION_TO_INDEX[direction])
    return (
        torch.tensor(fact_windows, dtype=torch.float32), torch.tensor(current_windows, dtype=torch.float32),
        torch.tensor(recurrent_windows, dtype=torch.float32), torch.tensor(labels, dtype=torch.long),
    )


def _evaluate(client: LuantiLiveClient, *, kind: str, seed_values: Sequence[int], learned: PolarityFactMemoryCurrentVisionObjectPolicy, current: NoMemoryRewardVisionObjectPolicy, recurrent: GRUObjectPolicy, transformer: TransformerObjectPolicy) -> tuple[Evaluation, int]:
    successes, revisions, steps = 0, 0, []
    planner, rng = VerifiedFactObjectPlanner(), random.Random(2_003)
    for index, seed in enumerate(seed_values):
        state, vision = client.begin_object_episode_with_vision(f"polarity-{kind}-{index}", seed)
        history = [(state, dict(vision))]
        causal, facts = CausalControlWorldModel(), VerifiedObjectFactMemory()
        for step in range(MAX_STEPS):
            if kind == "teacher":
                actual = planner.choose_actual(state, verified_key=facts.key_verified, verified_charge=facts.charge_verified, charge_mode=facts.charge_mode)
            elif kind == "learned":
                actual = learned.choose(history[-HISTORY:], verified_key=facts.key_verified, charge_mode=facts.charge_mode)
            elif kind == "current":
                actual = current.choose(history)
            elif kind == "recurrent":
                actual = recurrent.choose([item for item, _ in history[-HISTORY:]])
            elif kind == "transformer":
                actual = transformer.choose([item for item, _ in history[-HISTORY:]])
            else:
                actual = rng.choice(DIRECTIONS)
            requested = causal.request_for_actual(actual or "north")
            before = state
            interaction, state, vision = client.step_object_with_vision(requested)
            if state is None or vision is None or interaction.outcome.get("status") != "accepted":
                break
            causal.observe(before.base, requested, state.base)
            if kind in {"teacher", "learned"}:
                facts.observe_reward(interaction.reward)
            history.append((state, dict(vision)))
            if _prerequisite(_reward(interaction)):
                recenter, state, vision = client.recenter_object_with_vision()
                if state is None or vision is None or recenter.outcome.get("status") != "accepted":
                    break
                history.append((state, dict(vision)))
            if state.completed:
                successes += 1
                steps.append(step + 1)
                break
        else:
            steps.append(MAX_STEPS)
        revisions += causal.revision_count
    return Evaluation(successes, len(seed_values), sum(steps) / len(seed_values)), revisions


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> Report:
    resolved = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    train_seeds = _balanced_train_seeds()
    heldout_seeds = [4_000_000 + index * 31 for index in range(8)]
    trajectories = [_collect(client, seed) for seed in train_seeds]
    fact_data, current_data, recurrent_data, labels = _tensors(trajectories)
    torch.manual_seed(2_000)
    learned = PolarityFactMemoryCurrentVisionObjectPolicy().to(resolved)
    current = NoMemoryRewardVisionObjectPolicy().to(resolved)
    recurrent = GRUObjectPolicy(history_length=HISTORY).to(resolved)
    transformer = TransformerObjectPolicy(history_length=HISTORY).to(resolved)
    train_policy(learned, fact_data, labels, device=resolved, steps=1_600, seed=2_001)
    train_policy(current, current_data, labels, device=resolved, steps=1_600, seed=2_002)
    train_policy(recurrent, recurrent_data, labels, device=resolved, steps=1_600, seed=2_003)
    train_policy(transformer, recurrent_data, labels, device=resolved, steps=1_600, seed=2_004)
    learned.eval(); current.eval(); recurrent.eval()
    transformer.eval()
    learned_eval, revisions = _evaluate(client, kind="learned", seed_values=heldout_seeds, learned=learned, current=current, recurrent=recurrent, transformer=transformer)
    current_eval, _ = _evaluate(client, kind="current", seed_values=heldout_seeds, learned=learned, current=current, recurrent=recurrent, transformer=transformer)
    recurrent_eval, _ = _evaluate(client, kind="recurrent", seed_values=heldout_seeds, learned=learned, current=current, recurrent=recurrent, transformer=transformer)
    transformer_eval, _ = _evaluate(client, kind="transformer", seed_values=heldout_seeds, learned=learned, current=current, recurrent=recurrent, transformer=transformer)
    random_eval, _ = _evaluate(client, kind="random", seed_values=heldout_seeds, learned=learned, current=current, recurrent=recurrent, transformer=transformer)
    teacher_eval, teacher_revisions = _evaluate(client, kind="teacher", seed_values=heldout_seeds, learned=learned, current=current, recurrent=recurrent, transformer=transformer)
    artifacts = ROOT / "artifacts"; artifacts.mkdir(exist_ok=True)
    checkpoint = artifacts / "luanti-polarity-memory-policy.pt"
    torch.save({"state_dict": learned.state_dict(), "training_seeds": train_seeds}, checkpoint)
    (artifacts / "luanti-polarity-memory-rollouts.jsonl").write_text("".join(json.dumps(asdict(item)) + "\n" for item in trajectories), encoding="utf-8")
    artifact = artifacts / "luanti-polarity-memory-report.json"
    report = Report(str(resolved), len(labels), learned_eval, current_eval, recurrent_eval, transformer_eval, random_eval, teacher_eval, revisions, False, str(checkpoint), str(artifact))
    report.accepted = (
        report.learned.success_rate >= 0.75 and report.learned.success_rate >= report.no_memory.success_rate + 0.25
        and report.learned.success_rate >= report.random.success_rate + 0.5 and report.teacher.success_rate == 1.0
        and report.learned_revisions == len(heldout_seeds) and teacher_revisions == len(heldout_seeds)
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("polarity learned controller did not meet its evidence gate")

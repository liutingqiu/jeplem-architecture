"""Predeclared visual-noise evaluation for the four-candidate fact contract.

All learned controllers receive exactly the same state/reward stream and the
same deterministic corrupted visual frame per episode.  The fact-memory
controller alone gets a compact, provenance-bound choice fact derived from the
same public reward event.  This evaluates interface robustness, not an
end-to-end PLWM architecture advantage or visual understanding claim.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.luanti_degradation_audit import DegradedVisualClient, NoiseSchedule
from experiments.train_luanti_behavior import Evaluation, train_policy
from experiments.train_luanti_multi_object_memory import (
    BRIDGE_ROOT,
    DELAY_OBSERVATIONS,
    MAX_ACTIONS,
    CurrentPublicPolicy,
    PublicGRUPolicy,
    PublicTransformerPolicy,
    _balanced_heldout_seeds,
    _balanced_train_seeds,
    _collect,
    _tensors,
)
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    CausalControlWorldModel,
    LuantiLiveClient,
    MultiChoiceFactMemoryCurrentVisionObjectPolicy,
    VerifiedFactObjectPlanner,
    VerifiedObjectFactMemory,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NoiseCondition:
    name: str
    cell_flip_rate: float
    noisy_steps: int
    stale_replay_steps: int

    def schedule(self) -> NoiseSchedule:
        return NoiseSchedule(self.cell_flip_rate, self.noisy_steps, self.stale_replay_steps)


@dataclass
class ConditionResult:
    condition: NoiseCondition
    fact_memory: Evaluation
    no_memory: Evaluation
    recurrent: Evaluation
    transformer: Evaluation
    teacher: Evaluation
    fact_memory_revisions: int


@dataclass
class Report:
    protocol: str
    device: str
    training_samples: int
    delay_observations: int
    train_seeds: list[int]
    heldout_seeds: list[int]
    conditions: list[ConditionResult]
    parameter_counts: dict[str, int]
    accepted: bool
    artifact: str


CONDITIONS = (
    NoiseCondition("clean", 0.0, 0, 0),
    # Existing Luanti degradation cadence: corrupted early frames plus two stale replays.
    NoiseCondition("moderate_noise_and_replay", 0.16, 6, 2),
    # Every visual frame exposed to the policy is independently corrupted.
    NoiseCondition("persistent_severe_noise", 0.35, MAX_ACTIONS + DELAY_OBSERVATIONS + 4, 0),
)


def _evaluate(
    client: LuantiLiveClient,
    *,
    kind: str,
    seeds: Sequence[int],
    condition: NoiseCondition,
    fact: MultiChoiceFactMemoryCurrentVisionObjectPolicy,
    current: CurrentPublicPolicy,
    recurrent: PublicGRUPolicy,
    transformer: PublicTransformerPolicy,
) -> tuple[Evaluation, int]:
    successes, revisions, steps = 0, 0, []
    planner = VerifiedFactObjectPlanner()
    for index, seed in enumerate(seeds):
        # Every policy kind receives the same seeded presentation for a given world.
        noisy = DegradedVisualClient(client, condition.schedule(), seed=81_000 + index)
        state, vision = noisy.begin_object_episode_with_vision(f"multi-noise-{condition.name}-{kind}-{index}", seed)
        history, causal, facts = [(state, dict(vision))], CausalControlWorldModel(), VerifiedObjectFactMemory()
        for step in range(MAX_ACTIONS):
            if kind == "teacher":
                actual = planner.choose_actual(state, verified_key=facts.key_verified, verified_charge=facts.charge_verified, charge_choice=facts.charge_choice)
            elif kind == "fact_memory":
                actual = fact.choose(history, verified_key=facts.key_verified, charge_choice=facts.charge_choice)
            elif kind == "no_memory":
                actual = current.choose(history)
            elif kind == "recurrent":
                actual = recurrent.choose(history)
            elif kind == "transformer":
                actual = transformer.choose(history)
            else:
                raise ValueError(f"unsupported policy {kind}")
            requested, before = causal.request_for_actual(actual), state
            interaction, state, vision = noisy.step_object_with_vision(requested)
            if state is None or vision is None or interaction.outcome.get("status") != "accepted" or state.base.failed:
                break
            causal.observe(before.base, requested, state.base)
            was_charged = facts.charge_verified
            # Rewards are public to every controller; only this compact verified
            # representation is additional input to fact-memory and teacher.
            if kind in {"teacher", "fact_memory"}:
                facts.observe_reward(interaction.reward)
            history.append((state, dict(vision)))
            if facts.charge_verified and not was_charged:
                recenter, state, vision = noisy.recenter_object_with_vision()
                if state is None or vision is None or recenter.outcome.get("status") != "accepted":
                    break
                history.append((state, dict(vision)))
                for _ in range(DELAY_OBSERVATIONS):
                    observed, state, vision = noisy.observe_object_with_vision()
                    if state is None or vision is None or observed.outcome.get("status") != "accepted":
                        break
                    history.append((state, dict(vision)))
            if state.completed:
                successes += 1
                steps.append(step + 1)
                break
        else:
            steps.append(MAX_ACTIONS)
        if len(steps) <= index:
            steps.append(MAX_ACTIONS)
        revisions += causal.revision_count
    return Evaluation(successes, len(seeds), sum(steps) / len(seeds)), revisions


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> Report:
    resolved = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    train_seeds, heldout_seeds = _balanced_train_seeds(), _balanced_heldout_seeds()
    fact_data, public_data, labels = _tensors([_collect(client, seed) for seed in train_seeds])
    torch.manual_seed(6_000)
    fact = MultiChoiceFactMemoryCurrentVisionObjectPolicy().to(resolved)
    current = CurrentPublicPolicy().to(resolved)
    recurrent = PublicGRUPolicy().to(resolved)
    transformer = PublicTransformerPolicy().to(resolved)
    for model, data, seed in ((fact, fact_data, 6_001), (current, public_data, 6_002), (recurrent, public_data, 6_003), (transformer, public_data, 6_004)):
        train_policy(model, data, labels, device=resolved, steps=1_800, seed=seed)
        model.eval()
    results = []
    for condition in CONDITIONS:
        evaluations, revisions = {}, {}
        for kind in ("fact_memory", "no_memory", "recurrent", "transformer", "teacher"):
            evaluations[kind], revisions[kind] = _evaluate(
                client, kind=kind, seeds=heldout_seeds, condition=condition,
                fact=fact, current=current, recurrent=recurrent, transformer=transformer,
            )
        results.append(ConditionResult(
            condition, evaluations["fact_memory"], evaluations["no_memory"], evaluations["recurrent"],
            evaluations["transformer"], evaluations["teacher"], revisions["fact_memory"],
        ))
    accepted = all(
        item.fact_memory.success_rate >= 0.75
        and item.teacher.success_rate == 1.0
        and item.fact_memory.success_rate >= item.no_memory.success_rate + 0.25
        and item.fact_memory.success_rate >= item.recurrent.success_rate + 0.25
        and item.fact_memory.success_rate >= item.transformer.success_rate + 0.25
        for item in results
    )
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    artifact = artifacts / "luanti-multi-object-noise-report.json"
    counts = {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in {
        "fact_memory": fact, "no_memory": current, "recurrent": recurrent, "transformer": transformer,
    }.items()}
    report = Report(
        "multi_object_noise_v1: clean-trained, teacher-labelled four-candidate contract; same seeded corrupted vision and public state/reward history for all learned baselines; reward-derived verified choice fact only for fact-memory; not an end-to-end or perception architecture claim",
        str(resolved), len(labels), DELAY_OBSERVATIONS, train_seeds, heldout_seeds, results, counts, accepted, str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("multi-object noise evaluation did not meet its evidence gate")

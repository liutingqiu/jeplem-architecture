"""Auditable temporal visual-evidence recovery for the four-candidate task.

This is a perception-system experiment.  It keeps the reward-derived choice
fact and controller fixed, and tests whether noisy or replayed visual evidence
can be repaired without giving the controller a clean frame.  It is not an
end-to-end PLWM architecture comparison.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.luanti_degradation_audit import DegradedVisualClient, NoiseSchedule
from experiments.train_luanti_multi_object_memory import (
    BRIDGE_ROOT, DELAY_OBSERVATIONS, MAX_ACTIONS, _balanced_heldout_seeds,
    _balanced_train_seeds, _collect,
)
from plwm.device import resolve_device
from plwm.evidence import EvidenceFreshnessGate, VersionedEvidence
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    CausalControlWorldModel, LuantiLiveClient, MultiChoiceFactMemoryCurrentVisionObjectPolicy,
    ObjectTaskState, VerifiedObjectFactMemory, VISION_TILES, VISION_WIDTH,
)
from plwm.vision_denoiser import TemporalVisionDenoiser, align_previous_vision, corrupt_vision, denoise_frame, vision_tensor


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Condition:
    name: str
    rate: float
    noisy_steps: int
    stale_steps: int

    def schedule(self) -> NoiseSchedule:
        return NoiseSchedule(self.rate, self.noisy_steps, self.stale_steps)


@dataclass
class Evaluation:
    successes: int
    episodes: int
    mean_steps: float
    rejected_replays: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


@dataclass
class ConditionResult:
    condition: Condition
    raw: Evaluation
    evidence_filtered: Evaluation


@dataclass
class Report:
    protocol: str
    device: str
    training_pairs: int
    train_seeds: list[int]
    heldout_seeds: list[int]
    conditions: list[ConditionResult]
    accepted: bool
    artifact: str


CONDITIONS = (
    Condition("clean", 0.0, 0, 0),
    Condition("moderate_noise_and_replay", 0.16, 6, 2),
    Condition("persistent_severe_noise", 0.35, MAX_ACTIONS + DELAY_OBSERVATIONS + 4, 0),
)


def _blank_vision(reference: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(reference)
    cells = ["air"] * (VISION_WIDTH * VISION_WIDTH)
    cells[(VISION_WIDTH // 2) * VISION_WIDTH + VISION_WIDTH // 2] = "agent"
    payload["cells"] = cells
    return payload


class VisualEvidenceFilter:
    """Freshness gate plus denoising, with no raw-frame recovery path."""

    def __init__(self, world_id: str, denoiser: TemporalVisionDenoiser) -> None:
        self.gate = EvidenceFreshnessGate(world_id)
        self.denoiser = denoiser
        self.world_id = world_id
        self.sequence_tick = 0
        self.last_accepted_tick = -1
        self.previous_state: ObjectTaskState | None = None
        self.previous_vision: Mapping[str, Any] | None = None

    def begin(self, state: ObjectTaskState, vision: Mapping[str, Any]) -> Mapping[str, Any]:
        self.gate.accept(VersionedEvidence(self.world_id, 0, 0, "vision", vision))
        self.last_accepted_tick = 0
        self.previous_state, self.previous_vision = state, dict(vision)
        return vision

    def observe(self, state: ObjectTaskState, vision: Mapping[str, Any], *, replayed: bool) -> Mapping[str, Any]:
        if self.previous_state is None or self.previous_vision is None:
            raise RuntimeError("visual evidence filter was not initialized")
        self.sequence_tick += 1
        source_tick = self.last_accepted_tick if replayed else self.sequence_tick
        decision = self.gate.accept(VersionedEvidence(self.world_id, 0, source_tick, "vision", vision))
        # Replayed evidence is quarantined.  The denoiser receives an agent-only
        # prediction slot and the prior accepted presentation, never raw Luanti I/O.
        if decision.accepted:
            # A fresh sensor observation is authoritative.  The prior run
            # established that replacing it with a reconstruction regresses
            # even clean control, so it must not be silently overwritten.
            recovered = vision
        else:
            recovered = denoise_frame(
                self.denoiser, self.previous_state.base, self.previous_vision,
                state.base, _blank_vision(vision),
            )
        self.previous_state, self.previous_vision = state, recovered
        if decision.accepted:
            self.last_accepted_tick = source_tick
        return recovered


def _training_pairs(client: LuantiLiveClient) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current, aligned, masks, targets = [], [], [], []
    for trajectory in (_collect(client, seed) for seed in _balanced_train_seeds()):
        for index in range(len(trajectory.states) - 1):
            before, after = trajectory.states[index], trajectory.states[index + 1]
            prior = vision_tensor(trajectory.visions[index])
            fresh = vision_tensor(trajectory.visions[index + 1])
            moved, mask = align_previous_vision(before.base, prior, after.base)
            current.append(fresh); aligned.append(moved); masks.append(mask); targets.append(fresh.argmax(dim=0))
    return torch.stack(current), torch.stack(aligned), torch.stack(masks), torch.stack(targets)


def _train_denoiser(
    client: LuantiLiveClient, device: torch.device
) -> tuple[TemporalVisionDenoiser, int]:
    current, aligned, masks, targets = _training_pairs(client)
    model = TemporalVisionDenoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 2.0, 4.0, 3.0, 3.0], device=device))
    generator = torch.Generator(device="cpu").manual_seed(7_113)
    for _ in range(1_400):
        indices = torch.randint(len(targets), (min(64, len(targets)),), generator=generator)
        rate = (0.08, 0.16, 0.35)[int(torch.randint(3, (1,), generator=generator).item())]
        noisy_current = torch.stack([corrupt_vision(frame, flip_rate=rate, generator=generator) for frame in current[indices]])
        noisy_aligned = torch.stack([corrupt_vision(frame, flip_rate=rate, generator=generator) for frame in aligned[indices]])
        loss = criterion(model(noisy_current.to(device), noisy_aligned.to(device), masks[indices].to(device)), targets[indices].to(device))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval()
    return model, len(targets)


def _load_policy(device: torch.device) -> MultiChoiceFactMemoryCurrentVisionObjectPolicy:
    payload = torch.load(ROOT / "artifacts" / "luanti-multi-object-memory-policy.pt", map_location=device, weights_only=True)
    model = MultiChoiceFactMemoryCurrentVisionObjectPolicy().to(device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model


def _evaluate(
    client: LuantiLiveClient, policy: MultiChoiceFactMemoryCurrentVisionObjectPolicy, denoiser: TemporalVisionDenoiser,
    condition: Condition, seeds: Sequence[int], *, filtered: bool,
) -> Evaluation:
    successes, steps, rejected = 0, [], 0
    for index, seed in enumerate(seeds):
        noisy = DegradedVisualClient(client, condition.schedule(), seed=97_000 + index)
        world_id = f"evidence-{condition.name}-{filtered}-{index}"
        state, vision = noisy.begin_object_episode_with_vision(world_id, seed)
        evidence = VisualEvidenceFilter(world_id, denoiser) if filtered else None
        if evidence is not None:
            vision = evidence.begin(state, vision)
        history, facts, causal = [(state, dict(vision))], VerifiedObjectFactMemory(), CausalControlWorldModel()
        for step in range(MAX_ACTIONS):
            actual = policy.choose(history, verified_key=facts.key_verified, charge_choice=facts.charge_choice)
            requested, before = causal.request_for_actual(actual), state
            interaction, state, vision = noisy.step_object_with_vision(requested)
            if state is None or vision is None or interaction.outcome.get("status") != "accepted" or state.base.failed:
                break
            causal.observe(before.base, requested, state.base)
            was_charged = facts.charge_verified
            facts.observe_reward(interaction.reward)
            if evidence is not None:
                before_rejected = evidence.gate.rejected_stale
                vision = evidence.observe(state, vision, replayed=noisy.last_is_replay)
                rejected += evidence.gate.rejected_stale - before_rejected
            history.append((state, dict(vision)))
            if facts.charge_verified and not was_charged:
                # The transition that verified the charge schedules one real
                # re-centering and delay block, exactly as in the base contract.
                recenter, state, vision = noisy.recenter_object_with_vision()
                if state is None or vision is None or recenter.outcome.get("status") != "accepted": break
                if evidence is not None: vision = evidence.observe(state, vision, replayed=noisy.last_is_replay)
                history.append((state, dict(vision)))
                for _ in range(DELAY_OBSERVATIONS):
                    observed, state, vision = noisy.observe_object_with_vision()
                    if state is None or vision is None or observed.outcome.get("status") != "accepted": break
                    if evidence is not None: vision = evidence.observe(state, vision, replayed=noisy.last_is_replay)
                    history.append((state, dict(vision)))
            if state.completed:
                successes += 1; steps.append(step + 1); break
        else: steps.append(MAX_ACTIONS)
        if len(steps) <= index: steps.append(MAX_ACTIONS)
    return Evaluation(successes, len(seeds), sum(steps) / len(seeds), rejected)


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> Report:
    resolved, client = resolve_device(device), LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    denoiser, pairs = _train_denoiser(client, resolved)
    policy, heldout = _load_policy(resolved), _balanced_heldout_seeds()
    results = [ConditionResult(condition, _evaluate(client, policy, denoiser, condition, heldout, filtered=False), _evaluate(client, policy, denoiser, condition, heldout, filtered=True)) for condition in CONDITIONS]
    moderate = next(item for item in results if item.condition.name == "moderate_noise_and_replay")
    report = Report(
        "object_visual_evidence_v1: fresh frames remain authoritative; only source-tick-stale presentations are rejected and predicted from the prior accepted presentation plus observed motion, never raw clean vision; fixed fact controller; not an architecture claim",
        str(resolved), pairs, _balanced_train_seeds(), heldout, results,
        results[0].evidence_filtered.successes >= results[0].raw.successes
        and moderate.evidence_filtered.successes >= moderate.raw.successes
        and moderate.evidence_filtered.rejected_replays >= len(heldout) * CONDITIONS[1].stale_steps,
        str(ROOT / "artifacts" / "luanti-multi-object-visual-evidence-report.json"),
    )
    Path(report.artifact).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run(); print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted: raise SystemExit("visual evidence recovery did not meet its gate")

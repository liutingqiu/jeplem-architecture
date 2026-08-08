"""Longer-horizon degradation, interruption, and stale-evidence audit.

The Luanti bridge remains the source of truth.  This audit corrupts only the
records presented to a policy, records the exact corruption schedule, and
keeps the raw causally linked response available for recovery checks.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_behavior import run_rule_retention
from plwm.device import resolve_device
from plwm.evidence import EvidenceFreshnessGate, VersionedEvidence
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    DIRECTIONS,
    GoalDirectedPLWM,
    LuantiInteraction,
    LuantiLiveClient,
    LuantiState,
    ObjectTaskState,
    VisionGoalDirectedPLWM,
    CausalControlWorldModel,
    encode_vision,
    visual_frame_features,
    reconcile_visual_sensors,
    temporal_consensus_vision,
)
from plwm.torch_model import TorchCognitiveState
from plwm.vision_denoiser import TemporalVisionDenoiser, denoise_frame


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")


@dataclass(frozen=True)
class NoiseSchedule:
    cell_flip_rate: float = 0.08
    noisy_steps: int = 4
    stale_replay_steps: int = 1


class DegradedVisualClient:
    """Present deterministic visual corruption while preserving raw Luanti I/O."""

    def __init__(self, client: LuantiLiveClient, schedule: NoiseSchedule, *, seed: int) -> None:
        self.client = client
        self.schedule = schedule
        self.rng = random.Random(seed)
        self.step_index = 0
        self._stale_vision: Mapping[str, Any] | None = None
        self._stale_remaining = 0
        self.current_raw_vision: Mapping[str, Any] | None = None
        self.last_is_replay = False

    @staticmethod
    def _flip_cells(vision: Mapping[str, Any], rng: random.Random, rate: float) -> dict[str, Any]:
        payload = dict(vision)
        cells = list(payload.get("cells", []))
        for index, cell in enumerate(cells):
            if cell == "agent" or rng.random() >= rate:
                continue
            if cell == "wall":
                cells[index] = "air"
            elif cell == "air":
                cells[index] = "wall"
            elif cell in {"goal", "key", "decoy"}:
                cells[index] = "air"
        payload["cells"] = cells
        # Validate the injected record before it reaches the policy.
        encode_vision(payload)
        return payload

    def _present(self, vision: Mapping[str, Any], *, force_noise: bool = False) -> Mapping[str, Any]:
        if self.step_index <= self.schedule.noisy_steps or force_noise:
            return self._flip_cells(vision, self.rng, self.schedule.cell_flip_rate)
        return dict(vision)

    def begin_episode_with_vision(self, episode_id: str, task_seed: int) -> tuple[LuantiState, Mapping[str, Any]]:
        state, vision = self.client.begin_episode_with_vision(episode_id, task_seed)
        self.step_index = 0
        self._stale_vision = dict(vision)
        self._stale_remaining = self.schedule.stale_replay_steps
        self.current_raw_vision = dict(vision)
        self.last_is_replay = False
        return state, self._present(vision, force_noise=True)

    def begin_object_episode_with_vision(
        self, episode_id: str, task_seed: int
    ) -> tuple[ObjectTaskState, Mapping[str, Any]]:
        """Start an object episode with the identical visual corruption contract."""
        state, vision = self.client.begin_object_episode_with_vision(episode_id, task_seed)
        self.step_index = 0
        self._stale_vision = dict(vision)
        self._stale_remaining = self.schedule.stale_replay_steps
        self.current_raw_vision = dict(vision)
        self.last_is_replay = False
        return state, self._present(vision, force_noise=True)

    def step_with_vision(self, direction: str) -> tuple[LuantiInteraction, LuantiState | None, Mapping[str, Any] | None]:
        interaction, state, vision = self.client.step_with_vision(direction)
        self.step_index += 1
        if vision is None:
            return interaction, state, None
        self.current_raw_vision = dict(vision)
        current = self._present(vision)
        if self.step_index > self.schedule.noisy_steps and self._stale_remaining > 0:
            current = dict(self._stale_vision or current)
            self.last_is_replay = True
            self._stale_remaining -= 1
        else:
            self._stale_vision = dict(vision)
            self.last_is_replay = False
        return interaction, state, current

    def _object_response(
        self,
        interaction: LuantiInteraction,
        state: ObjectTaskState | None,
        vision: Mapping[str, Any] | None,
        *,
        advance_step: bool,
    ) -> tuple[LuantiInteraction, ObjectTaskState | None, Mapping[str, Any] | None]:
        if advance_step:
            self.step_index += 1
        if vision is None:
            return interaction, state, None
        self.current_raw_vision = dict(vision)
        current = self._present(vision)
        if advance_step and self.step_index > self.schedule.noisy_steps and self._stale_remaining > 0:
            current = dict(self._stale_vision or current)
            self.last_is_replay = True
            self._stale_remaining -= 1
        else:
            self._stale_vision = dict(vision)
            self.last_is_replay = False
        return interaction, state, current

    def step_object_with_vision(
        self, direction: str
    ) -> tuple[LuantiInteraction, ObjectTaskState | None, Mapping[str, Any] | None]:
        interaction, state, vision = self.client.step_object_with_vision(direction)
        return self._object_response(interaction, state, vision, advance_step=True)

    def recenter_with_vision(self) -> tuple[LuantiInteraction, LuantiState | None, Mapping[str, Any] | None]:
        interaction, state, vision = self.client.recenter_with_vision()
        if vision is None:
            return interaction, state, None
        self.current_raw_vision = dict(vision)
        self.last_is_replay = False
        return interaction, state, self._present(vision)

    def recenter_object_with_vision(
        self,
    ) -> tuple[LuantiInteraction, ObjectTaskState | None, Mapping[str, Any] | None]:
        interaction, state, vision = self.client.recenter_object_with_vision()
        return self._object_response(interaction, state, vision, advance_step=False)

    def observe_object_with_vision(
        self,
    ) -> tuple[LuantiInteraction, ObjectTaskState | None, Mapping[str, Any] | None]:
        interaction, state, vision = self.client.observe_object_with_vision()
        return self._object_response(interaction, state, vision, advance_step=False)


def _run_visual_recovery(
    model: VisionGoalDirectedPLWM,
    client: DegradedVisualClient,
    *,
    task_seed: int,
    episode_id: str,
    max_steps: int = 28,
    enforce_freshness: bool = True,
    reconcile_sensors: bool = False,
    temporal_consensus: bool = False,
    denoiser: TemporalVisionDenoiser | None = None,
) -> tuple[bool, int | None]:
    state, vision = client.begin_episode_with_vision(episode_id, task_seed)
    if reconcile_sensors:
        vision, _corrections = reconcile_visual_sensors(state, vision)
    history: list[tuple[LuantiState, Mapping[str, Any]]] = [(state, vision)]
    previous_sensor_state = state
    previous_sensor_vision = vision
    mechanism = CausalControlWorldModel()
    freshness = EvidenceFreshnessGate(episode_id)
    freshness.accept(VersionedEvidence(episode_id, 0, 0, "vision", vision))
    recovery_start = client.schedule.noisy_steps + client.schedule.stale_replay_steps + 1
    recovery_latency: int | None = None
    for step in range(max_steps):
        if state.completed:
            return True, recovery_latency
        actual_direction = model.choose(history[-model.history_length :])
        direction = mechanism.request_for_actual(actual_direction)
        before = state
        interaction, next_state, next_vision = client.step_with_vision(direction)
        if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
            return False, recovery_latency
        mechanism.observe(before, direction, next_state)
        # Replay carries its original tick.  The gate rejects it and the
        # policy uses the current raw frame, which is independently linked to
        # the just-completed Luanti step.
        evidence_tick = step if client.last_is_replay else step + 1
        decision = freshness.accept(VersionedEvidence(episode_id, 0, evidence_tick, "vision", next_vision))
        if not enforce_freshness:
            decision = type(decision)(True, "unfiltered-baseline")
        if not decision.accepted:
            if client.current_raw_vision is None:
                return False, recovery_latency
            vision = client.current_raw_vision
        else:
            vision = next_vision
        sensor_current_vision = vision
        state = next_state
        if reconcile_sensors:
            vision, _corrections = reconcile_visual_sensors(state, vision)
        if temporal_consensus:
            vision, _conflicts = temporal_consensus_vision(before, history[-1][1], state, vision)
        if denoiser is not None:
            vision = denoise_frame(denoiser, previous_sensor_state, previous_sensor_vision, state, vision)
        previous_sensor_state, previous_sensor_vision = state, sensor_current_vision
        history.append((state, vision))
        if step == 0:
            recenter, recentered_state, recentered_vision = client.recenter_with_vision()
            if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                return False, recovery_latency
            state, vision = recentered_state, recentered_vision
            sensor_current_vision = vision
            if reconcile_sensors:
                vision, _corrections = reconcile_visual_sensors(state, vision)
            if temporal_consensus:
                vision, _conflicts = temporal_consensus_vision(next_state, history[-1][1], state, vision)
            if denoiser is not None:
                vision = denoise_frame(denoiser, previous_sensor_state, previous_sensor_vision, state, vision)
            previous_sensor_state, previous_sensor_vision = state, sensor_current_vision
            history.append((state, vision))
        if step + 1 >= recovery_start and state.completed and recovery_latency is None:
            recovery_latency = step + 1 - recovery_start + 1
        if state.completed:
            return True, recovery_latency
    return False, recovery_latency


def _run_interruption_checkpoint(
    checkpoint: Mapping[str, Any],
    client: LuantiLiveClient,
    *,
    task_seed: int,
) -> tuple[bool, bool, int]:
    """Interrupt a real Luanti episode, restore cognitive state, and resume."""
    device = resolve_device(checkpoint.get("device"))
    model = GoalDirectedPLWM(history_length=int(checkpoint["history_length"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    state = client.begin_episode("degradation-interruption", task_seed)
    session = model.begin_episode(state)
    history = [state]
    checkpoint_exact = False
    for index in range(28):
        # The policy's public rollout contract uses a fixed observed history.
        # Its continuous state is checkpointed in parallel and audited for an
        # exact next decision at the interruption boundary.
        direction = model.choose(history[-model.history_length :])
        interaction, next_state = client.step(direction)
        if interaction.outcome.get("status") != "accepted" or next_state is None:
            return False, False, index + 1
        session.observe(next_state)
        history.append(next_state)
        if index == 2:
            with tempfile.TemporaryDirectory(prefix="plwm-interruption-") as directory:
                state_path = Path(directory) / "cognitive-state.pt"
                session.cognitive_state.checkpoint(state_path)
                expected = session.choose()
                restored_state = TorchCognitiveState.restore(state_path, device)
                resumed = model.begin_episode(next_state)
                resumed.cognitive_state = restored_state
                restored_action = resumed.choose()
                checkpoint_exact = expected == restored_action
                session = resumed
        state = next_state
        if state.completed:
            return True, checkpoint_exact, index + 1
    return False, checkpoint_exact, 28


@dataclass
class DegradationAudit:
    device: str
    clean_visual_successes: int
    noisy_visual_successes: int
    noisy_visual_episodes: int
    unfiltered_replay_successes: int
    recovery_latencies: list[int]
    interruption_completed: bool
    checkpoint_action_exact: bool
    stale_visual_rejected: bool
    stale_language_rejected: bool
    fresh_evidence_accepted: bool
    retained_mapping: bool
    unrelated_world_isolated: bool
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> DegradationAudit:
    resolved_device = resolve_device(device)
    visual_payload = torch.load(ROOT / "artifacts" / "luanti-vision-policy.pt", map_location=resolved_device, weights_only=True)
    visual_model = VisionGoalDirectedPLWM(history_length=int(visual_payload["history_length"])).to(resolved_device)
    visual_model.load_state_dict(visual_payload["state_dict"])
    visual_model.eval()
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [1_450_000 + index * 31 for index in range(4)]

    # Clean reference is deliberately measured through the same public client.
    clean_successes = 0
    for index, task_seed in enumerate(seeds):
        clean = DegradedVisualClient(client, NoiseSchedule(cell_flip_rate=0.0, noisy_steps=0, stale_replay_steps=0), seed=91 + index)
        success, _latency = _run_visual_recovery(visual_model, clean, task_seed=task_seed, episode_id=f"degradation-clean-{index}")
        clean_successes += int(success)

    noisy_successes = 0
    latencies: list[int] = []
    for index, task_seed in enumerate(seeds):
        noisy = DegradedVisualClient(client, NoiseSchedule(), seed=191 + index)
        success, latency = _run_visual_recovery(visual_model, noisy, task_seed=task_seed, episode_id=f"degradation-noisy-{index}")
        noisy_successes += int(success)
        if latency is not None:
            latencies.append(latency)

    unfiltered_successes = 0
    for index, task_seed in enumerate(seeds):
        unfiltered = DegradedVisualClient(client, NoiseSchedule(), seed=191 + index)
        success, _latency = _run_visual_recovery(
            visual_model,
            unfiltered,
            task_seed=task_seed,
            episode_id=f"degradation-unfiltered-{index}",
            enforce_freshness=False,
        )
        unfiltered_successes += int(success)

    goal_payload = torch.load(ROOT / "artifacts" / "luanti-goal-policy.pt", map_location=resolved_device, weights_only=True)
    interruption_completed, checkpoint_exact, _steps = _run_interruption_checkpoint(goal_payload, client, task_seed=900_000)

    gate = EvidenceFreshnessGate("switch-world")
    visual_v0 = VersionedEvidence("switch-world", 0, 1, "vision", {"cells": ["air"]})
    language_v0 = VersionedEvidence("switch-world", 0, 1, "language", {"task": "object_delivery"})
    fresh_visual = VersionedEvidence("switch-world", 1, 2, "vision", {"cells": ["agent"]})
    gate.accept(visual_v0)
    gate.accept(language_v0)
    gate.advance_mechanism()
    stale_visual_rejected = not gate.accept(visual_v0).accepted
    stale_language_rejected = not gate.accept(language_v0).accepted
    fresh_evidence_accepted = gate.accept(fresh_visual).accepted

    retention = run_rule_retention(bridge_root=bridge_root)
    artifact = ROOT / "artifacts" / "luanti-degradation-audit.json"
    report = DegradationAudit(
        device=str(resolved_device),
        clean_visual_successes=clean_successes,
        noisy_visual_successes=noisy_successes,
        noisy_visual_episodes=len(seeds),
        unfiltered_replay_successes=unfiltered_successes,
        recovery_latencies=latencies,
        interruption_completed=interruption_completed,
        checkpoint_action_exact=checkpoint_exact,
        stale_visual_rejected=stale_visual_rejected,
        stale_language_rejected=stale_language_rejected,
        fresh_evidence_accepted=fresh_evidence_accepted,
        retained_mapping=retention.retained_probe_correct,
        unrelated_world_isolated=retention.isolated_world_mapping,
        accepted=(
            clean_successes == len(seeds)
            and noisy_successes / len(seeds) >= 0.75
            and noisy_successes >= unfiltered_successes
            and interruption_completed
            and checkpoint_exact
            and stale_visual_rejected
            and stale_language_rejected
            and fresh_evidence_accepted
            and retention.accepted
        ),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("Luanti degradation audit did not meet its evidence gate")

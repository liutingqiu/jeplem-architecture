"""Composition audit: language, verified fact, switch, stale vision, restart."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.evidence import EvidenceFreshnessGate, VersionedEvidence
from plwm.language import parse_luanti_language_goal
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import CausalControlWorldModel, LuantiLiveClient, VerifiedFactObjectPlanner, VerifiedObjectFactMemory


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")


@dataclass
class CompositionEpisode:
    seed: int
    completed: bool
    fact_verified: bool
    control_revisions: int
    stale_visual_rejected: bool
    fresh_visual_accepted: bool
    checkpoint_restored: bool
    actions: int


@dataclass
class CompositionReport:
    language_schema_accepted: bool
    fact_memory_successes: int
    no_fact_successes: int
    episodes: int
    all_revisions: bool
    all_stale_rejected: bool
    all_checkpoint_restored: bool
    accepted: bool
    artifact: str


def _restore(causal: CausalControlWorldModel, facts: VerifiedObjectFactMemory) -> tuple[CausalControlWorldModel, VerifiedObjectFactMemory]:
    with tempfile.TemporaryDirectory(prefix="plwm-composite-") as directory:
        path = Path(directory) / "runtime.json"
        path.write_text(json.dumps({"causal": causal.snapshot(), "facts": facts.snapshot()}), encoding="utf-8")
        payload = json.loads(path.read_text(encoding="utf-8"))
    return CausalControlWorldModel.restore(payload["causal"]), VerifiedObjectFactMemory.restore(payload["facts"])


def _run_episode(client: LuantiLiveClient, *, seed: int, use_facts: bool) -> CompositionEpisode:
    world_id = f"fact-composition-{seed}"
    state, vision = client.begin_object_episode_with_vision(world_id, seed)
    planner = VerifiedFactObjectPlanner()
    causal = CausalControlWorldModel()
    facts = VerifiedObjectFactMemory()
    gate = EvidenceFreshnessGate(world_id)
    gate.accept(VersionedEvidence(world_id, 0, 0, "vision", vision))
    previous_vision, previous_tick = dict(vision), 0
    event_tick = 0
    stale_visual_rejected = False
    fresh_visual_accepted = True
    checkpoint_restored = False
    actions = 0
    restarted = False
    revision_seen = 0
    for step in range(28):
        # Replay exactly one pre-switch visual item after the rule revision.
        if revision_seen and not stale_visual_rejected:
            stale = VersionedEvidence(world_id, gate.mechanism_version - 1, previous_tick, "vision", previous_vision)
            stale_visual_rejected = not gate.accept(stale).accepted
        actual = planner.choose_actual(state, verified_key=facts.key_verified if use_facts else False)
        if actual is None:
            break
        requested = causal.request_for_actual(actual)
        before = state
        interaction, next_state, next_vision = client.step_object_with_vision(requested)
        if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
            break
        causal.observe(before.base, requested, next_state.base)
        if causal.revision_count > revision_seen:
            revision_seen = causal.revision_count
            gate.advance_mechanism()
        facts.observe_reward(interaction.reward)
        state, vision = next_state, dict(next_vision)
        event_tick += 1
        evidence = VersionedEvidence(world_id, gate.mechanism_version, event_tick, "vision", vision)
        fresh_visual_accepted = fresh_visual_accepted and gate.accept(evidence).accepted
        previous_vision, previous_tick = vision, event_tick
        actions += 1
        # The environment clears all current key hints here. Restore both
        # fact and causal state from disk immediately after that interruption.
        if (interaction.reward or {}).get("value") == 0.2:
            recenter, recentered_state, recentered_vision = client.recenter_object_with_vision()
            if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                break
            state, vision = recentered_state, dict(recentered_vision)
            causal, facts = _restore(causal, facts)
            checkpoint_restored = facts.key_verified == use_facts
            event_tick += 1
            fresh = VersionedEvidence(world_id, gate.mechanism_version, event_tick, "vision", vision)
            fresh_visual_accepted = fresh_visual_accepted and gate.accept(fresh).accepted
            previous_vision, previous_tick = vision, event_tick
            restarted = True
        if state.completed:
            break
    return CompositionEpisode(
        seed=seed,
        completed=state.completed,
        fact_verified=facts.key_verified,
        control_revisions=causal.revision_count,
        stale_visual_rejected=stale_visual_rejected,
        fresh_visual_accepted=fresh_visual_accepted,
        checkpoint_restored=checkpoint_restored and restarted,
        actions=actions,
    )


def run(*, bridge_root: Path = BRIDGE_ROOT) -> CompositionReport:
    goal = parse_luanti_language_goal('{"task":"object_delivery","target":"goal","require_key":true,"avoid_decoy":true}')
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [2_800_000 + index * 31 for index in range(8)]
    with_facts = [_run_episode(client, seed=seed, use_facts=True) for seed in seeds]
    without_facts = [_run_episode(client, seed=seed, use_facts=False) for seed in seeds]
    artifact = ROOT / "artifacts" / "luanti-fact-memory-composition-report.json"
    report = CompositionReport(
        language_schema_accepted=goal.require_key and goal.avoid_decoy,
        fact_memory_successes=sum(item.completed for item in with_facts),
        no_fact_successes=sum(item.completed for item in without_facts),
        episodes=len(seeds),
        all_revisions=all(item.control_revisions == 1 for item in with_facts),
        all_stale_rejected=all(item.stale_visual_rejected and item.fresh_visual_accepted for item in with_facts),
        all_checkpoint_restored=all(item.checkpoint_restored and item.fact_verified for item in with_facts),
        accepted=False,
        artifact=str(artifact),
    )
    report.accepted = (
        report.language_schema_accepted
        and report.fact_memory_successes == report.episodes
        and report.no_fact_successes <= report.episodes // 2
        and report.all_revisions
        and report.all_stale_rejected
        and report.all_checkpoint_restored
    )
    artifact.write_text(json.dumps({**asdict(report), "with_facts": [asdict(item) for item in with_facts], "without_facts": [asdict(item) for item in without_facts]}, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("fact-memory composition audit did not meet its evidence gate")

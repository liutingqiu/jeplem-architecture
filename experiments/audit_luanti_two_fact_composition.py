"""Audit a two-fact delayed-consequence Luanti curriculum on real episodes."""

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
HIDDEN_FIELDS = ("key_collected", "decoy_collected", "charge_collected")


@dataclass
class TwoFactEpisode:
    seed: int
    completed: bool
    charge_verified: bool
    key_verified: bool
    prerequisite_rewards: list[float]
    control_revisions: int
    facts_hidden_after_prerequisites: bool
    stale_visual_rejected: bool
    fresh_visual_accepted: bool
    checkpoint_restored: bool
    fact_provenance_complete: bool
    actions: int


@dataclass
class TwoFactCompositionReport:
    language_schema_accepted: bool
    fact_memory_successes: int
    no_fact_successes: int
    episodes: int
    all_prerequisite_sequences: bool
    all_revisions: bool
    all_facts_hidden: bool
    all_stale_rejected: bool
    all_checkpoint_restored: bool
    all_fact_provenance_complete: bool
    accepted: bool
    artifact: str


def _restore(
    causal: CausalControlWorldModel, facts: VerifiedObjectFactMemory
) -> tuple[CausalControlWorldModel, VerifiedObjectFactMemory]:
    """Exercise an on-disk boundary rather than a shallow in-memory copy."""
    with tempfile.TemporaryDirectory(prefix="plwm-two-fact-") as directory:
        path = Path(directory) / "runtime.json"
        path.write_text(json.dumps({"causal": causal.snapshot(), "facts": facts.snapshot()}), encoding="utf-8")
        payload = json.loads(path.read_text(encoding="utf-8"))
    return CausalControlWorldModel.restore(payload["causal"]), VerifiedObjectFactMemory.restore(payload["facts"])


def _hidden_after_recenter(payload: dict[str, object] | None) -> bool:
    if payload is None:
        return False
    return (
        all(payload.get(name) is False for name in HIDDEN_FIELDS)
        and payload.get("last_reward") == 0
        and payload.get("last_direction") is None
        and payload.get("last_blocked") is False
    )


def _run_episode(client: LuantiLiveClient, *, seed: int, use_facts: bool) -> TwoFactEpisode:
    world_id = f"two-fact-composition-{seed}"
    state, vision = client.begin_object_episode_with_vision(world_id, seed)
    planner = VerifiedFactObjectPlanner()
    causal = CausalControlWorldModel()
    facts = VerifiedObjectFactMemory()
    gate = EvidenceFreshnessGate(world_id)
    gate.accept(VersionedEvidence(world_id, 0, 0, "vision", vision))
    previous_vision, previous_tick = dict(vision), 0
    event_tick = 0
    revision_seen = 0
    stale_visual_rejected = False
    fresh_visual_accepted = True
    facts_hidden = True
    checkpoint_restored = True
    fact_provenance_complete = True
    prerequisite_rewards: list[float] = []
    actions = 0

    for _step in range(64):
        if revision_seen and not stale_visual_rejected:
            stale = VersionedEvidence(world_id, gate.mechanism_version - 1, previous_tick, "vision", previous_vision)
            stale_visual_rejected = not gate.accept(stale).accepted
        actual = planner.choose_actual(
            state,
            verified_key=facts.key_verified if use_facts else False,
            verified_charge=facts.charge_verified if use_facts else False,
        )
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
        if use_facts:
            facts.observe_reward(interaction.reward)
        state, vision = next_state, dict(next_vision)
        event_tick += 1
        fresh_visual_accepted = fresh_visual_accepted and gate.accept(
            VersionedEvidence(world_id, gate.mechanism_version, event_tick, "vision", vision)
        ).accepted
        previous_vision, previous_tick = vision, event_tick
        actions += 1

        reward = float((interaction.reward or {}).get("value", 0.0))
        if reward in (0.15, 0.2):
            prerequisite_rewards.append(reward)
            raw = dict(interaction.observation or {})
            facts_hidden = facts_hidden and all(raw.get(name) is False for name in HIDDEN_FIELDS)
            recenter, recentered_state, recentered_vision = client.recenter_object_with_vision()
            if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                checkpoint_restored = False
                break
            facts_hidden = facts_hidden and _hidden_after_recenter(dict(recenter.observation or {}))
            causal, facts = _restore(causal, facts)
            expected_facts = int(facts.charge_verified) + int(facts.key_verified)
            fact_provenance_complete = fact_provenance_complete and len(facts.provenance) == expected_facts and all(
                item.episode_id == world_id
                and item.source == "luanti-reward"
                and item.tick >= 0
                and item.request_id
                and item.fact_schema_version == 1
                and item.confidence == 1.0
                for item in facts.provenance
            )
            if reward == 0.15:
                checkpoint_restored = checkpoint_restored and facts.charge_verified == use_facts and not facts.key_verified
            else:
                checkpoint_restored = checkpoint_restored and facts.charge_verified == use_facts and facts.key_verified == use_facts
            state, vision = recentered_state, dict(recentered_vision)
            event_tick += 1
            fresh_visual_accepted = fresh_visual_accepted and gate.accept(
                VersionedEvidence(world_id, gate.mechanism_version, event_tick, "vision", vision)
            ).accepted
            previous_vision, previous_tick = vision, event_tick
        if state.completed:
            break

    return TwoFactEpisode(
        seed=seed,
        completed=state.completed,
        charge_verified=facts.charge_verified,
        key_verified=facts.key_verified,
        prerequisite_rewards=prerequisite_rewards,
        control_revisions=causal.revision_count,
        facts_hidden_after_prerequisites=facts_hidden,
        stale_visual_rejected=stale_visual_rejected,
        fresh_visual_accepted=fresh_visual_accepted,
        checkpoint_restored=checkpoint_restored,
        fact_provenance_complete=fact_provenance_complete,
        actions=actions,
    )


def run(*, bridge_root: Path = BRIDGE_ROOT) -> TwoFactCompositionReport:
    language_goal = parse_luanti_language_goal(
        '{"task":"charged_object_delivery","target":"goal","require_key":true,"avoid_decoy":true,"require_charge":true}'
    )
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [3_000_000 + index * 31 for index in range(8)]
    with_facts = [_run_episode(client, seed=seed, use_facts=True) for seed in seeds]
    without_facts = [_run_episode(client, seed=seed, use_facts=False) for seed in seeds]
    artifact = ROOT / "artifacts" / "luanti-two-fact-composition-report.json"
    report = TwoFactCompositionReport(
        language_schema_accepted=(
            language_goal.task == "charged_object_delivery"
            and language_goal.require_charge
            and language_goal.require_key
            and language_goal.avoid_decoy
        ),
        fact_memory_successes=sum(item.completed for item in with_facts),
        no_fact_successes=sum(item.completed for item in without_facts),
        episodes=len(seeds),
        all_prerequisite_sequences=all(item.prerequisite_rewards == [0.15, 0.2] for item in with_facts),
        all_revisions=all(item.control_revisions == 1 for item in with_facts),
        all_facts_hidden=all(item.facts_hidden_after_prerequisites for item in with_facts),
        all_stale_rejected=all(item.stale_visual_rejected and item.fresh_visual_accepted for item in with_facts),
        all_checkpoint_restored=all(
            item.checkpoint_restored and item.charge_verified and item.key_verified for item in with_facts
        ),
        all_fact_provenance_complete=all(item.fact_provenance_complete for item in with_facts),
        accepted=False,
        artifact=str(artifact),
    )
    report.accepted = (
        report.language_schema_accepted
        and report.fact_memory_successes == report.episodes
        and report.no_fact_successes <= report.episodes // 2
        and report.all_prerequisite_sequences
        and report.all_revisions
        and report.all_facts_hidden
        and report.all_stale_rejected
        and report.all_checkpoint_restored
        and report.all_fact_provenance_complete
    )
    artifact.write_text(
        json.dumps(
            {
                **asdict(report),
                "with_facts": [asdict(item) for item in with_facts],
                "without_facts": [asdict(item) for item in without_facts],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("two-fact composition audit did not meet its evidence gate")

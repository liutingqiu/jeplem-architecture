"""Real-world audit for a charge-reward-selected object mechanism."""

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
class PolarityEpisode:
    seed: int
    completed: bool
    charge_mode: str | None
    positive_rewards: list[float]
    control_revisions: int
    facts_hidden: bool
    stale_visual_rejected: bool
    fresh_visual_accepted: bool
    checkpoint_restored: bool
    actions: int


@dataclass
class PolarityReport:
    language_schema_accepted: bool
    fact_memory_successes: int
    no_fact_successes: int
    episodes: int
    both_charge_modes_seen: bool
    all_sequences: bool
    all_revisions: bool
    all_facts_hidden: bool
    all_stale_rejected: bool
    all_checkpoint_restored: bool
    accepted: bool
    artifact: str


def _restore(causal: CausalControlWorldModel, facts: VerifiedObjectFactMemory) -> tuple[CausalControlWorldModel, VerifiedObjectFactMemory]:
    with tempfile.TemporaryDirectory(prefix="plwm-polarity-") as directory:
        path = Path(directory) / "runtime.json"
        path.write_text(json.dumps({"causal": causal.snapshot(), "facts": facts.snapshot()}), encoding="utf-8")
        payload = json.loads(path.read_text(encoding="utf-8"))
    return CausalControlWorldModel.restore(payload["causal"]), VerifiedObjectFactMemory.restore(payload["facts"])


def _hidden(payload: dict[str, object] | None, *, after_recenter: bool) -> bool:
    if payload is None or not all(payload.get(field) is False for field in HIDDEN_FIELDS):
        return False
    if after_recenter:
        return payload.get("last_reward") == 0 and payload.get("last_direction") is None and payload.get("last_blocked") is False
    return True


def _run_episode(client: LuantiLiveClient, *, seed: int, use_facts: bool) -> PolarityEpisode:
    world_id = f"polarity-composition-{seed}"
    state, vision = client.begin_object_episode_with_vision(world_id, seed)
    planner, causal, facts = VerifiedFactObjectPlanner(), CausalControlWorldModel(), VerifiedObjectFactMemory()
    gate = EvidenceFreshnessGate(world_id)
    gate.accept(VersionedEvidence(world_id, 0, 0, "vision", vision))
    previous_vision, previous_tick = dict(vision), 0
    event_tick = 0
    revision_seen = 0
    stale_visual_rejected = False
    fresh_visual_accepted = True
    facts_hidden = True
    checkpoint_restored = True
    positive_rewards: list[float] = []
    actions = 0
    for _step in range(64):
        if revision_seen and not stale_visual_rejected:
            stale_visual_rejected = not gate.accept(
                VersionedEvidence(world_id, gate.mechanism_version - 1, previous_tick, "vision", previous_vision)
            ).accepted
        actual = planner.choose_actual(
            state,
            verified_key=facts.key_verified if use_facts else False,
            verified_charge=facts.charge_verified if use_facts else False,
            charge_mode=facts.charge_mode if use_facts else None,
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
        if reward > 0:
            positive_rewards.append(reward)
        if reward in (0.15, 0.16, 0.2):
            facts_hidden = facts_hidden and _hidden(dict(interaction.observation or {}), after_recenter=False)
            recenter, recentered_state, recentered_vision = client.recenter_object_with_vision()
            if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                checkpoint_restored = False
                break
            facts_hidden = facts_hidden and _hidden(dict(recenter.observation or {}), after_recenter=True)
            causal, facts = _restore(causal, facts)
            if reward in (0.15, 0.16):
                checkpoint_restored = checkpoint_restored and facts.charge_verified and facts.charge_mode in {"key", "decoy"}
            else:
                checkpoint_restored = checkpoint_restored and facts.key_verified and facts.charge_verified
            state, vision = recentered_state, dict(recentered_vision)
            event_tick += 1
            fresh_visual_accepted = fresh_visual_accepted and gate.accept(
                VersionedEvidence(world_id, gate.mechanism_version, event_tick, "vision", vision)
            ).accepted
            previous_vision, previous_tick = vision, event_tick
        if state.completed:
            break
    return PolarityEpisode(
        seed=seed,
        completed=state.completed,
        charge_mode=facts.charge_mode,
        positive_rewards=positive_rewards,
        control_revisions=causal.revision_count,
        facts_hidden=facts_hidden,
        stale_visual_rejected=stale_visual_rejected,
        fresh_visual_accepted=fresh_visual_accepted,
        checkpoint_restored=checkpoint_restored,
        actions=actions,
    )


def run(*, bridge_root: Path = BRIDGE_ROOT) -> PolarityReport:
    goal = parse_luanti_language_goal(
        '{"task":"polarity_object_delivery","target":"goal","require_key":true,"require_charge":true,"select_charged_object":true}'
    )
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [4_000_000 + index * 31 for index in range(8)]
    with_facts = [_run_episode(client, seed=seed, use_facts=True) for seed in seeds]
    without_facts = [_run_episode(client, seed=seed, use_facts=False) for seed in seeds]
    artifact = ROOT / "artifacts" / "luanti-polarity-composition-report.json"
    report = PolarityReport(
        language_schema_accepted=(goal.task == "polarity_object_delivery" and goal.require_charge and goal.select_charged_object),
        fact_memory_successes=sum(item.completed for item in with_facts),
        no_fact_successes=sum(item.completed for item in without_facts),
        episodes=len(seeds),
        both_charge_modes_seen={item.charge_mode for item in with_facts} == {"key", "decoy"},
        all_sequences=all(item.positive_rewards in ([0.15, 0.2, 1.0], [0.16, 0.2, 1.0]) for item in with_facts),
        all_revisions=all(item.control_revisions == 1 for item in with_facts),
        all_facts_hidden=all(item.facts_hidden for item in with_facts),
        all_stale_rejected=all(item.stale_visual_rejected and item.fresh_visual_accepted for item in with_facts),
        all_checkpoint_restored=all(item.checkpoint_restored and item.charge_mode is not None for item in with_facts),
        accepted=False,
        artifact=str(artifact),
    )
    report.accepted = (
        report.language_schema_accepted
        and report.fact_memory_successes == report.episodes
        and report.no_fact_successes <= report.episodes // 2
        and report.both_charge_modes_seen
        and report.all_sequences
        and report.all_revisions
        and report.all_facts_hidden
        and report.all_stale_rejected
        and report.all_checkpoint_restored
    )
    artifact.write_text(json.dumps({**asdict(report), "with_facts": [asdict(item) for item in with_facts], "without_facts": [asdict(item) for item in without_facts]}, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("polarity composition audit did not meet its evidence gate")

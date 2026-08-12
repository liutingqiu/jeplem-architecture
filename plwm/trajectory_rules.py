"""Unified trajectory meta-rule layer (2026-08-12).

Integrates the causality (family 2) and interaction (family 4) relations into
ONE engine-facing module, so the architecture treats both as the same kind of
thing: TRAJECTORY-LEVEL meta-rules extracted from an episode of experience.

Both operate on a Luanti episode (a sequence of states + actions) and use the
same RuleLedger for compile/reuse/revise:

  causality   : learn action->(dx,dz) causal map (world movement rules);
                the mapping CHANGES per world (normal vs controls-inverted) ->
                a new causal signature = a world change, detected and compiled.
  interaction : learn agent-goal distance -> outcome rule (success conditions);
                world-consistent (same rule across all episodes).

This module exposes one `TrajectoryRelation` interface (fit/apply over an
episode) and a `learn_trajectory_rules(episode, ledger)` driver that runs BOTH
families on an episode and updates the shared ledger.  The engine-level
integration makes the two families one pipeline instead of two scripts.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.rule_ledger import RuleLedger


# --- trajectory-relation interface -------------------------------------------

@dataclass(frozen=True)
class TrajectoryRelation:
    family: str                 # "causality" | "interaction"
    name: str
    fit: Callable               # (episode) -> rule | None
    signature: Callable         # (rule) -> str  (the cache key)
    serialize: Callable         # (rule) -> dict (JSON-safe)
    verify: Callable            # (rule, episode) -> bool

    @property
    def key(self) -> str:
        return f"{self.family}/{self.name}"


# --- causality (family 2) ----------------------------------------------------

@dataclass(frozen=True)
class CausalMap:
    mapping: dict[str, tuple[int, int]]


def _causal_fit(episode: dict) -> CausalMap | None:
    states, actions = episode["states"], episode["actions"]
    mapping: dict[str, tuple[int, int]] = {}
    consistent = True
    for i in range(len(states) - 1):
        act = actions[i].get("direction") if i < len(actions) else None
        if act is None or act == "recenter":
            continue
        dx = states[i + 1]["x"] - states[i]["x"]
        dz = states[i + 1]["z"] - states[i]["z"]
        blocked = (dx == 0 and dz == 0) and bool(states[i].get("last_blocked"))
        if blocked:
            continue
        if act in mapping:
            if mapping[act] != (dx, dz):
                consistent = False
                break
        else:
            mapping[act] = (dx, dz)
    if not consistent or not mapping:
        return None
    return CausalMap(mapping)


def _causal_signature(rule: CausalMap) -> str:
    return "|".join(f"{a}->{d}" for a, d in sorted(rule.mapping.items()))


def _causal_serialize(rule: CausalMap) -> dict:
    return {"kind": "causal", "mapping": {a: list(d) for a, d in rule.mapping.items()}}


def _causal_verify(rule: CausalMap, episode: dict) -> bool:
    states, actions = episode["states"], episode["actions"]
    for i in range(len(states) - 1):
        act = actions[i].get("direction") if i < len(actions) else None
        if act is None or act == "recenter" or act not in rule.mapping:
            continue
        dx = states[i + 1]["x"] - states[i]["x"]
        dz = states[i + 1]["z"] - states[i]["z"]
        blocked = (dx == 0 and dz == 0) and bool(states[i].get("last_blocked"))
        if blocked:
            continue
        if rule.mapping[act] != (dx, dz):
            return False
    return True


# --- interaction (family 4) --------------------------------------------------

@dataclass(frozen=True)
class InteractionRule:
    success_radius: int
    fail_radius: int


def _interaction_fit(episode: dict) -> InteractionRule | None:
    states, actions = episode["states"], episode["actions"]
    successes, failures = [], []
    for i, a in enumerate(actions):
        if i >= len(states) or a.get("reward") is None:
            continue
        st = states[i]
        dist = abs(st["goal_x"] - st["x"]) + abs(st["goal_z"] - st["z"])
        if a["reward"] > 0:
            successes.append(dist)
        elif a["reward"] < 0:
            failures.append(dist)
    if not successes or not failures:
        return None
    succ_max = max(successes)
    fail_min = min(failures)
    if succ_max < fail_min:
        return InteractionRule(success_radius=succ_max, fail_radius=fail_min)
    return None


def _interaction_signature(rule: InteractionRule) -> str:
    return f"succ<={rule.success_radius}|fail>={rule.fail_radius}"


def _interaction_serialize(rule: InteractionRule) -> dict:
    return {"kind": "interaction",
            "interaction": {"success_radius": rule.success_radius,
                            "fail_radius": rule.fail_radius}}


def _interaction_verify(rule: InteractionRule, episode: dict) -> bool:
    states, actions = episode["states"], episode["actions"]
    for i, a in enumerate(actions):
        if i >= len(states) or a.get("reward") is None:
            continue
        st = states[i]
        dist = abs(st["goal_x"] - st["x"]) + abs(st["goal_z"] - st["z"])
        pred = "success" if dist <= rule.success_radius else "fail"
        actual = "success" if a["reward"] > 0 else "fail"
        if pred != actual:
            return False
    return True


# --- unified driver ----------------------------------------------------------

def trajectory_relations() -> list[TrajectoryRelation]:
    return [
        TrajectoryRelation("causality", "cause_effect",
                           _causal_fit, _causal_signature,
                           _causal_serialize, _causal_verify),
        TrajectoryRelation("interaction", "interact",
                           _interaction_fit, _interaction_signature,
                           _interaction_serialize, _interaction_verify),
    ]


def learn_trajectory_rules(episode: dict, ledger: RuleLedger) -> dict:
    """Run all trajectory relations on one episode; update the shared ledger.
    Returns telemetry for this episode."""
    result = {"compiled": 0, "reused": 0, "gaps": 0, "world_changes": 0}
    for rel in trajectory_relations():
        rule = rel.fit(episode)
        if rule is None:
            result["gaps"] += 1
            continue
        sig = rel.signature(rule)
        cached = ledger.lookup(sig, lambda stored, _r=rel: _verify_stored(_r, stored, episode))
        if cached is not None:
            result["reused"] += 1
            continue
        serialized = rel.serialize(rule)
        was_stale = ledger.is_stale(sig)
        ledger.compile(sig, serialized, rel.key)
        if was_stale:
            result["world_changes"] += 1
        else:
            result["compiled"] += 1
    return result


def _verify_stored(rel: TrajectoryRelation, stored, episode: dict) -> bool:
    """Rebuild a rule from its serialized form and verify against the episode."""
    raw = stored.rule
    if rel.family == "causality":
        mapping = {a: tuple(d) for a, d in raw.get("mapping", {}).items()}
        rule = CausalMap(mapping)
        return _causal_verify(rule, episode)
    if rel.family == "interaction":
        r = raw.get("interaction", {})
        rule = InteractionRule(r.get("success_radius", 0), r.get("fail_radius", 0))
        return _interaction_verify(rule, episode)
    return False

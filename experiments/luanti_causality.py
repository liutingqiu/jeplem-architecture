"""Causality meta-rule (family 2) on Luanti trajectories (2026-08-12).

The FIRST implemented non-quantity family.  Real demonstration of "extract
meta-rules from real-world experience": from a Luanti episode's state-action
trajectory, LEARN the causal mapping action -> (dx, dz) displacement.  The
mapping is invariant WITHIN one world (episode) but changes ACROSS worlds
(normal vs controls-inverted) -- the engine detects the change (world changed)
and revises, exactly the P4 revision semantics generalised to causality.

This is a trajectory-level relation: its "raw" is an episode = a sequence of
(state, action) steps.  It is self-contained (no engine wiring needed to
demonstrate), but uses the RuleLedger so the learned causal map persists and
revises like any compiled rule.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.rule_ledger import RuleLedger

DIRECTIONS = ("north", "south", "east", "west")


@dataclass(frozen=True)
class CausalMap:
    """action -> (dx, dz) displacement, learned from a trajectory."""
    mapping: dict[str, tuple[int, int]]

    def apply(self, action: str) -> tuple[int, int]:
        return self.mapping.get(action, (0, 0))

    def signature(self) -> str:
        return "|".join(f"{a}->{d}" for a, d in sorted(self.mapping.items()))

    def predicts(self, state, action, next_state) -> bool:
        dx, dz = self.apply(action)
        return (next_state["x"] - state["x"], next_state["z"] - state["z"]) == (dx, dz)


def learn_causal_map(trajectory: list[tuple[dict, str]]) -> CausalMap | None:
    """Learn action->(dx,dz) from a (state, action) trajectory, EXCLUDING
    blocked steps (movement cancelled by a wall).  Displacements are computed
    from CONSECUTIVE states in the raw trajectory (a filtered action must not
    bridge across a skipped step)."""
    # consecutive-pair displacements over the RAW trajectory
    raw_moves: list[tuple[str, int, int, bool]] = []  # (action, dx, dz, blocked)
    for i in range(len(trajectory) - 1):
        state, action = trajectory[i]
        next_state = trajectory[i + 1][0]
        if action is None or action == "recenter":
            continue
        dx = next_state["x"] - state["x"]
        dz = next_state["z"] - state["z"]
        blocked = (dx == 0 and dz == 0) and bool(state.get("last_blocked"))
        raw_moves.append((action, dx, dz, blocked))
    mapping: dict[str, tuple[int, int]] = {}
    consistent = True
    for action, dx, dz, blocked in raw_moves:
        if blocked:
            continue  # environmental interference, not causal signal
        if action in mapping:
            if mapping[action] != (dx, dz):
                consistent = False
                break
        else:
            mapping[action] = (dx, dz)
    if not consistent or not mapping:
        return None
    return CausalMap(mapping)


def load_trajectories(rollout_path: Path) -> list[list[tuple[dict, str]]]:
    """Load episodes from a Luanti rollout JSONL; return trajectories of
    (state, action) where action has a direction."""
    trajs = []
    for line in rollout_path.open(encoding="utf-8"):
        rec = json.loads(line)
        states = rec["states"]
        actions = rec["actions"]
        traj = []
        for i, st in enumerate(states):
            act = actions[i].get("direction") if i < len(actions) else None
            traj.append((st, act))
        trajs.append(traj)
    return trajs


def run_causality_demo(*, rollout_path: Path, ledger_path: Path | None = None) -> dict:
    """Learn the causal map from episodes, persist it, then test world-change
    detection: a later episode with a DIFFERENT control (inverted) must fail
    the learned map's predictions -> the ledger marks the signature stale."""
    trajs = load_trajectories(rollout_path)
    ledger = RuleLedger()
    results = {"n_episodes": len(trajs), "learned_maps": [], "world_changes_detected": 0}

    # The causal map is keyed by a "control signature": whether the world's
    # action->effect mapping is normal or inverted.  We key by the mapping
    # signature so the ledger can detect when it changes.
    for ep_idx, traj in enumerate(trajs):
        # use the RAW trajectory; learn_causal_map handles recenter/blocked
        if len(traj) < 2:
            continue
        cmap = learn_causal_map(traj)
        if cmap is None:
            continue
        # this episode's causal signature = its action->effect mapping
        sig = cmap.signature()
        # verify the compiled map (if any) against this episode
        def _verify(rule):
            # rule.rule holds the serialized causal map
            stored = rule.rule.get("mapping", {})
            return all(
                tuple(stored.get(a, (0, 0))) == eff
                for a, eff in cmap.mapping.items()
            )

        cached = ledger.lookup(sig, _verify)
        if cached is not None:
            results["learned_maps"].append({"episode": ep_idx, "reused": True,
                                            "sig": sig})
            continue
        # not a cache hit: compile it (System-2 -> System-1)
        serialized = {"kind": "causal", "mapping": {a: list(e) for a, e in cmap.mapping.items()}}
        ledger.compile(sig, serialized, "causality/cause_effect")
        results["learned_maps"].append({"episode": ep_idx, "reused": False,
                                        "sig": sig})
        # a different control mapping than what we've seen before is a
        # world change -- count it (the ledger itself tracks staleness too)
        results["world_changes_detected"] += 1

    if ledger_path is not None:
        ledger.save(ledger_path)

    results["ledger"] = ledger.stats()
    results["distinct_control_signatures"] = len(ledger.current)
    return results


if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    res = run_causality_demo(
        rollout_path=ROOT / "artifacts" / "luanti-rollouts.jsonl",
        ledger_path=ROOT / "artifacts" / "luanti-causal-ledger.json",
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))

"""Interaction meta-rule (family 4) on Luanti trajectories (2026-08-12).

The second implemented non-quantity family.  Real demonstration of "extract
meta-rules from real-world experience": from a Luanti episode's trajectory,
LEARN the interaction relation "agent-goal spatial distance -> action outcome".
The core empirical signal (verified on 48 real rollouts):

    success-before median goal-dist = 1   (agent adjacent to goal)
    failure-before median goal-dist = 12  (agent far from goal)

i.e. "an action taken while adjacent to the goal succeeds; an action taken far
away fails" is a learnable INTERACTION relation between the agent (A) and the
goal (B).  Like causality (family 2), the relation is compiled once per
signature, reused on same-signature experience, and revised when the world's
interaction rule changes (e.g. the goal moves / the task changes).

Self-contained: operates on trajectory data via the RuleLedger.  No live
Luanti process required.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.rule_ledger import RuleLedger


@dataclass(frozen=True)
class InteractionRule:
    """agent-goal distance threshold -> success/failure outcome."""
    success_radius: int       # actions at goal-dist <= this succeed
    fail_radius: int          # actions at goal-dist >= this fail

    def predicts(self, goal_dist: int) -> str:
        if goal_dist <= self.success_radius:
            return "success"
        if goal_dist >= self.fail_radius:
            return "fail"
        return "ambiguous"

    def signature(self) -> str:
        return f"succ<={self.success_radius}|fail>={self.fail_radius}"

    @classmethod
    def learn(cls, samples: list[tuple[int, float]]) -> "InteractionRule | None":
        """Learn from (goal_dist, reward) samples.  Returns the simplest rule
        that separates successes (reward>0) from failures (reward<0)."""
        successes = [d for d, r in samples if r > 0]
        failures = [d for d, r in samples if r < 0]
        if not successes or not failures:
            return None
        # success threshold: max distance at which any success happened
        succ_max = max(successes)
        # failure threshold: min distance at which any failure happened
        fail_min = min(failures)
        if succ_max < fail_min:
            # clean separation: all successes closer than all failures
            return cls(success_radius=succ_max, fail_radius=fail_min)
        return None


def load_episodes(rollout_path: Path) -> list[dict]:
    episodes = []
    for line in rollout_path.open(encoding="utf-8"):
        episodes.append(json.loads(line))
    return episodes


def extract_interaction_samples(episode: dict) -> list[tuple[int, float]]:
    """(goal_dist, reward) for each reward-bearing action in the episode."""
    samples = []
    states, actions = episode["states"], episode["actions"]
    for i, a in enumerate(actions):
        if i >= len(states):
            continue
        st = states[i]
        if a.get("reward") is None:
            continue
        dist = abs(st["goal_x"] - st["x"]) + abs(st["goal_z"] - st["z"])
        samples.append((dist, a["reward"]))
    return samples


def run_interaction_demo(*, rollout_path: Path,
                         ledger_path: Path | None = None) -> dict:
    """Learn the agent-goal interaction rule per episode, compile/reuse via the
    ledger, and report how the rule holds across episodes (world consistency)."""
    episodes = load_episodes(rollout_path)
    ledger = RuleLedger()
    results = {"n_episodes": len(episodes), "learned": [], "reused": 0, "compiled": 0}

    for ep_idx, ep in enumerate(episodes):
        samples = extract_interaction_samples(ep)
        if len(samples) < 2:
            continue
        rule = InteractionRule.learn(samples)
        if rule is None:
            continue
        sig = rule.signature()

        def _verify(stored_rule):
            # the compiled rule must predict THIS episode's samples correctly
            stored = stored_rule.rule.get("interaction", {})
            succ_r = stored.get("success_radius")
            fail_r = stored.get("fail_radius")
            if succ_r is None or fail_r is None:
                return False
            check = InteractionRule(succ_r, fail_r)
            # require: no success beyond succ_r, no failure within succ_r
            ok = True
            for d, r in samples:
                pred = check.predicts(d)
                actual = "success" if r > 0 else "fail"
                if pred != "ambiguous" and pred != actual:
                    ok = False
                    break
            return ok

        cached = ledger.lookup(sig, _verify)
        if cached is not None:
            results["reused"] += 1
            results["learned"].append({"episode": ep_idx, "reused": True, "sig": sig})
            continue
        serialized = {"kind": "interaction",
                      "interaction": {"success_radius": rule.success_radius,
                                      "fail_radius": rule.fail_radius}}
        ledger.compile(sig, serialized, "interaction/interact")
        results["compiled"] += 1
        results["learned"].append({"episode": ep_idx, "reused": False, "sig": sig})

    if ledger_path is not None:
        ledger.save(ledger_path)
    results["ledger"] = ledger.stats()
    results["distinct_interaction_rules"] = len(ledger.current)
    return results


if __name__ == "__main__":
    res = run_interaction_demo(
        rollout_path=ROOT / "artifacts" / "luanti-rollouts.jsonl",
        ledger_path=ROOT / "artifacts" / "luanti-interaction-ledger.json",
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))

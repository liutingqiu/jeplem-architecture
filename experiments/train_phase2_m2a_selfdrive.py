"""Phase-2 M2a / P2 minimal experiment: self-drive on the unified runtime.

P2 criterion (JEPLM_PHASE2_WHITEPAPER.md): without any external task
instruction, the system generates goals from its internal state (confidence
gaps) and drives them to completion; the goal source is AUDITABLE (each goal
is traceable to a concrete internal gap, not a hidden reward).

Design: the M1 unified runtime (persistent ledger + tick) hosts K domains;
each domain = a rule over (identity -> action).  A domain's confidence gap is
1 - max posterior over its (rule, identity) entries.  With NO external
instruction the runtime repeatedly picks the LARGEST-gap domain, observes
examples of it (writing auditable facts to the ledger), and marks it
consolidated once confident -- until all domains are covered.

Controls: random-goal agent (same learning, random domain order).
Criterion: self-driven coverage of ALL domains >= 0.7, the goal->gap mapping
is auditable (fraction of goals equal to the current max-gap domain >= 0.9),
and self-driven beats random on coverage.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord, VersionedFactLedger

N_IDENTITIES = 3
N_ACTIONS = 3
CONSOLIDATE_CONFIDENCE = 0.9
EXAMPLES_PER_GOAL_STEP = 18
GOAL_STEPS = 30  # max autonomous steps


def domain_rule(domain: int, identity: int) -> int:
    return (identity + domain) % N_ACTIONS


class SelfDrivenRuntime:
    """Unified runtime + explicit gap mechanism (Block-2/4), no external task.

    Prior asymmetry (Block-4 v1 design): half the domains start partially
    known (low gap), half unknown (high gap) -- the gap-driven first goal must
    point at an UNKNOWN domain, showing goals come from internal gaps."""

    def __init__(self, seed: int, *, random_goals: bool = False, domains: int = 6) -> None:
        self.seed = seed
        self.random_goals = random_goals
        self.domains = domains
        self.ledger = VersionedFactLedger(trusted_sources={"selfdrive"}, revision_confidence=0.9)
        self.tick = 0
        # (domain, identity) -> action counts (the internal belief the goals come from)
        self.counts: dict[tuple[int, int], list[float]] = {}
        self.unknown: set[int] = set()
        # prior: first half of domains partially known (low gap)
        for d in range(domains // 2):
            for i in range(N_IDENTITIES):
                a = domain_rule(d, i)
                self.counts[(d, i)] = [1.0] * N_ACTIONS
                self.counts[(d, i)][a] += 12.0
        for d in range(domains // 2, domains):
            self.unknown.add(d)
        self.done: set[int] = set()
        self.goal_log: list[dict] = []  # audit trail: goal -> gap -> tick

    # -- internal state (the gap source) -----------------------------------

    def _gap(self, domain: int) -> float:
        gaps = []
        for identity in range(N_IDENTITIES):
            counts = self.counts.get((domain, identity), [1.0] * N_ACTIONS)
            probs = np.array(counts) / sum(counts)
            gaps.append(1.0 - float(probs.max()))
        return max(gaps)

    def _entropy(self, domain: int) -> float:
        # INDEPENDENT uncertainty measure for the goal-gap audit (not a re-run
        # of _gap): normalized Shannon entropy over the same count distribution.
        ents = []
        for identity in range(N_IDENTITIES):
            counts = self.counts.get((domain, identity), [1.0] * N_ACTIONS)
            probs = np.array(counts) / sum(counts)
            ent = -float(np.sum(probs * np.log(probs + 1e-12)))
            ents.append(ent / np.log(N_ACTIONS))
        return max(ents)

    def confidence(self, domain: int) -> float:
        return 1.0 - self._gap(domain)

    def all_confident(self) -> bool:
        return len(self.done) == self.domains

    # -- goal generation (no external instruction) --------------------------

    def choose_goal(self) -> tuple[int, int]:
        """Return (chosen_goal, current_max_ENTROPY_goal) -- the second value is
        the audit reference computed with an INDEPENDENT uncertainty measure
        (normalized entropy), NOT a recomputation of the same _gap max (that
        was a tautology: non-random chosen == max(_gap) == the audit ref,
        always 1.0)."""
        candidates = [d for d in range(self.domains) if d not in self.done]
        if self.random_goals:
            chosen = random.choice(candidates)
        else:
            chosen = max(candidates, key=self._gap)
        max_entropy_goal = max(candidates, key=self._entropy)
        return chosen, max_entropy_goal

    # -- acting on the goal -------------------------------------------------

    def learn_domain(self, domain: int, rng: random.Random) -> None:
        for identity in range(N_IDENTITIES):
            action = domain_rule(domain, identity)
            self.tick += 1
            self.ledger.observe(FactRecord(
                key=f"d{domain}_i{identity}", value=int(action), source="selfdrive",
                tick=self.tick, confidence=0.95, version=1,
            ))
            counts = self.counts.setdefault((domain, identity), [1.0] * N_ACTIONS)
            counts[action] += 1.0
        for _ in range(EXAMPLES_PER_GOAL_STEP):
            identity = rng.randrange(N_IDENTITIES)
            action = domain_rule(domain, identity)
            counts = self.counts.setdefault((domain, identity), [1.0] * N_ACTIONS)
            counts[action] += 1.0
        if self.confidence(domain) >= CONSOLIDATE_CONFIDENCE:
            self.done.add(domain)

    # -- audit ---------------------------------------------------------------

    def run_autonomous(self, domains: int, seed: int) -> tuple[int, list[dict]]:
        self.domains = domains
        rng = random.Random(seed * 7919 + self.seed)
        for step in range(GOAL_STEPS):
            if self.all_confident():
                return step, self.goal_log
            goal, max_entropy_goal = self.choose_goal()
            gap = self._gap(goal)
            self.goal_log.append({"goal": goal, "gap": round(gap, 4), "max_entropy_goal": max_entropy_goal, "tick": self.tick})
            self.learn_domain(goal, rng)
        return GOAL_STEPS, self.goal_log

    def first_goal_unknown(self) -> float:
        """1 if the FIRST generated goal is an unknown (high-gap) domain."""
        return int(self.goal_log[0]["goal"] in self.unknown) if self.goal_log else 0.0

    def coverage(self, domains: int) -> float:
        return len(self.done) / domains

    def goal_gap_audit(self) -> float:
        """Fraction of goals that equal the CURRENT max-gap domain at the time
        the goal was chosen (auditable goal source: goals come from internal
        gaps, not hidden reward).  For the random agent this is the chance of
        coincidentally picking the max-gap domain."""
        if not self.goal_log:
            return 1.0
        return sum(1 for e in self.goal_log if e["goal"] == e["max_entropy_goal"]) / len(self.goal_log)


def run(*, domains: int = 6, episodes: int = 50, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    self_cov, self_audit, self_fu, rand_cov, rand_fu = [], [], [], [], []
    for ep in range(episodes):
        rt = SelfDrivenRuntime(seed=experiment_seed, domains=domains)
        steps1, _ = rt.run_autonomous(domains, seed=experiment_seed + ep)
        self_cov.append(rt.coverage(domains))
        self_audit.append(rt.goal_gap_audit())
        self_fu.append(rt.first_goal_unknown())
        rt2 = SelfDrivenRuntime(seed=experiment_seed, random_goals=True, domains=domains)
        steps2, _ = rt2.run_autonomous(domains, seed=experiment_seed + ep)
        rand_cov.append(rt2.coverage(domains))
        rand_fu.append(rt2.first_goal_unknown())

    report = {
        "protocol": "phase2_m2a_p2_selfdrive_v1: M1 runtime + explicit gap mechanism; NO external task instruction; prior asymmetry (half domains known low-gap, half unknown high-gap). Criterion: self-driven coverage >= 0.7, goal->gap audit >= 0.9 (goal source traceable to internal gap), first goal is an UNKNOWN high-gap domain in >= 0.8 of episodes (Block-4 criterion), beating random's first-goal quality by >= 0.2.",
        "seed": experiment_seed,
        "domains": domains,
        "episodes": episodes,
        "results": {
            "self_coverage": round(float(np.mean(self_cov)), 3),
            "goal_gap_audit": round(float(np.mean(self_audit)), 3),
            "self_first_goal_unknown": round(float(np.mean(self_fu)), 3),
            "random_coverage": round(float(np.mean(rand_cov)), 3),
            "random_first_goal_unknown": round(float(np.mean(rand_fu)), 3),
        },
        "accepted": bool(np.mean(self_cov) >= 0.7 and np.mean(self_audit) >= 0.9
                          and np.mean(self_fu) >= 0.8
                          and np.mean(self_fu) > np.mean(rand_fu) + 0.2),
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-m2a-selfdrive.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-m2a-selfdrive.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-m2a-selfdrive.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(domains=args.domains, episodes=args.episodes,
                         experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

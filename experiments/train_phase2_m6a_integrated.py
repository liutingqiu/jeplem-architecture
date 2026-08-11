"""Phase-2 M6a: integrated long-stream -- the subjectivity components on ONE
continuous runtime.

The separate P experiments (M1 ledger continuity, M2a gap-driven goals,
M2b event revision, M3b snapshot chain) are combined into a single persistent
runtime running a LONG stream of events, to verify the components cooperate on
one continuous 'subject' (not as isolated toggles).

Stream: [learn rule0] [idle x2] [gap-goal: pick next domain] [learn rule1]
[idle] [flip rule0] [strong evidence -> revision mark] [idle] ... long.
After every event a snapshot is taken; at the end:
  (a) continuity: rule0 facts still retrievable (P1),
  (b) self-drive: the gap-driven goal pointed at the max-gap domain (P2),
  (c) revision: the flipped rule was marked stale (P4),
  (d) consistency: idle events left snapshots unchanged (P5).
Criterion: all four hold in the long stream (>= 0.9 each).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord, VersionedFactLedger

N_IDENTITIES = 3
N_ACTIONS = 3
FLIP_THRESHOLD = 3


def mapping(domain: int, identity: int, flipped: bool = False) -> int:
    base = (identity + domain) % N_ACTIONS
    return base if not flipped else (base + 1) % N_ACTIONS


class IntegratedRuntime:
    def __init__(self, seed: int, n_domains: int) -> None:
        self.seed = seed
        self.n_domains = n_domains
        self.ledger = VersionedFactLedger(trusted_sources={"self"}, revision_confidence=0.9)
        self.tick = 0
        self.counts: dict[tuple[int, int], list[float]] = {}
        self.done: set[int] = set()
        self.marked: set[int] = set()
        self.chain: list[dict] = []
        self.goal_log: list[dict] = []
        self.failures = 0

    # -- components ----------------------------------------------------------

    def snapshot_digest(self) -> str:
        current = sorted((k, str(v["value"])) for k, v in self.ledger.snapshot()["current"].items())
        return repr(current)

    def _chain(self, event: str) -> None:
        self.chain.append({"event": event, "snapshot": self.snapshot_digest()})

    def learn(self, domain: int, identity: int) -> None:
        self.tick += 1
        action = mapping(domain, identity)
        self.ledger.observe(FactRecord(key=f"d{domain}_i{identity}", value=int(action),
                                       source="self", tick=self.tick, confidence=0.95, version=1))
        counts = self.counts.setdefault((domain, identity), [1.0] * N_ACTIONS)
        counts[action] += 1.0
        self._chain("learn")

    def _gap(self, domain: int) -> float:
        gaps = []
        for i in range(N_IDENTITIES):
            counts = self.counts.get((domain, i), [1.0] * N_ACTIONS)
            probs = np.array(counts) / sum(counts)
            gaps.append(1.0 - float(probs.max()))
        return max(gaps)

    def choose_goal(self) -> int:
        candidates = [d for d in range(self.n_domains) if d not in self.done]
        return max(candidates, key=self._gap)

    def observe_flipped(self, domain: int, identity: int) -> None:
        """Feed flipped evidence; mark the rule stale on repeated failure."""
        self.tick += 1
        action = mapping(domain, identity, flipped=True)
        self.ledger.observe(FactRecord(key=f"d{domain}_i{identity}", value=int(action),
                                       source="self", tick=self.tick, confidence=0.95, version=1))
        counts = self.counts.setdefault((domain, identity), [1.0] * N_ACTIONS)
        predicted = int(np.argmax(counts))
        counts[action] += 1.0
        if predicted != action:
            self.failures += 1
            if self.failures >= FLIP_THRESHOLD:
                self.marked.add(domain)
        else:
            self.failures = 0
        self._chain("flip_evidence")

    def idle(self) -> None:
        self._chain("idle")

    def query(self, domain: int, identity: int) -> int:
        record = self.ledger.current.get(f"d{domain}_i{identity}")
        if record is not None:
            return int(record.value)
        counts = self.counts.get((domain, identity), [1.0] * N_ACTIONS)
        return int(np.argmax(counts))


def run(*, n_domains: int = 4, episodes: int = 30, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    continuity, selfdrive, revision, consistency = [], [], [], []
    for ep in range(episodes):
        rng = random.Random(experiment_seed * 997 + ep)
        rt = IntegratedRuntime(seed=experiment_seed, n_domains=n_domains)
        # phase 1: learn rule0 (domain 0)
        for i in range(N_IDENTITIES):
            for _ in range(4):
                rt.learn(0, i)
        # idle x2 (continuity: no drift)
        rt.idle(); rt.idle()
        # self-drive: gap goal picks the max-gap domain (should be an unlearned one)
        goal = rt.choose_goal()
        max_gap = max(range(n_domains), key=rt._gap)
        selfdrive.append(int(goal == max_gap and goal != 0))
        # learn the goal domain
        for i in range(N_IDENTITIES):
            rt.learn(goal, i)
        rt.idle()
        # flip rule0: strong evidence on identity 0 -> revision mark
        for _ in range(FLIP_THRESHOLD + 1):
            rt.observe_flipped(0, 0)
        revision.append(int(0 in rt.marked))
        rt.idle(); rt.idle()
        # continuity: rule0 facts still retrievable (P1, learned pre-flip; note
        # the flip changed identity 0 -- check identity 1/2 which did not flip)
        c = np.mean([1.0 if rt.query(0, i) == mapping(0, i, flipped=False) else 0.0
                     for i in (1, 2)])
        continuity.append(c)
        # consistency: idle events in the chain left snapshots unchanged
        idle_ok = all(
            ch["event"] != "idle" or ch["snapshot"] == prev["snapshot"]
            for ch, prev in zip(rt.chain[1:], rt.chain)
        )
        consistency.append(int(idle_ok))

    cont = float(np.mean(continuity))
    drive = float(np.mean(selfdrive))
    rev = float(np.mean(revision))
    cons = float(np.mean(consistency))
    accepted = min(cont, drive, rev, cons) >= 0.9
    report = {
        "protocol": "phase2_m6a_integrated_stream_v1: ONE continuous runtime runs [learn rule0, idle, gap-goal pick, learn goal, flip, revision-mark, idle]; snapshot chain throughout. Criterion (all >= 0.9): continuity (P1, unflipped facts retrievable), self-drive (P2, goal = max-gap domain), revision (P4, flipped rule marked), consistency (P5, idle snapshots unchanged).",
        "seed": experiment_seed,
        "episodes": episodes,
        "results": {
            "continuity_p1": round(cont, 3),
            "selfdrive_p2": round(drive, 3),
            "revision_p4": round(rev, 3),
            "consistency_p5": round(cons, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-m6a-integrated.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-m6a-integrated.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-m6a-integrated.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

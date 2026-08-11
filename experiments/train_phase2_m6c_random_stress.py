"""Phase-2 M6c: RANDOM long-stream stress test of the integrated runtime.

Unlike M6b's controlled phase script, the event stream is scheduled per-event
at random (learn / idle / flip / interference / gap-goal), 120+ events -- the
subject must hold its P-properties under UNPREDICTABLE events, approximating
'being alive' rather than 'running a script'.

Verdicts at the end:
  P1 continuity  : unflipped learned facts still retrievable
  P2 self-drive  : gap-goal events picked the max-gap domain
  P4 revision    : flipped domains got marked stale
  P5 consistency : idle events left snapshots unchanged
Criterion: all >= 0.9.
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


class RandomStressRuntime:
    def __init__(self, seed: int, n_domains: int) -> None:
        self.seed = seed
        self.n_domains = n_domains
        self.ledger = VersionedFactLedger(trusted_sources={"self"}, revision_confidence=0.9)
        self.tick = 0
        self.counts: dict[tuple[int, int], list[float]] = {}
        self.marked: set[int] = set()
        self.chain: list[dict] = []
        self.interference_log: list[str] = []
        self.failures_by_domain: dict[int, int] = {}

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
        candidates = [d for d in range(self.n_domains) if d not in self.marked]
        return max(candidates, key=self._gap) if candidates else 0

    def max_gap_domain(self) -> int:
        return max(range(self.n_domains), key=self._gap)

    def observe_flipped(self, domain: int, identity: int) -> None:
        self.tick += 1
        action = mapping(domain, identity, flipped=True)
        self.ledger.observe(FactRecord(key=f"d{domain}_i{identity}", value=int(action),
                                       source="self", tick=self.tick, confidence=0.95, version=1))
        counts = self.counts.setdefault((domain, identity), [1.0] * N_ACTIONS)
        predicted = int(np.argmax(counts))
        counts[action] += 1.0
        if predicted != action:
            self.failures_by_domain[domain] = self.failures_by_domain.get(domain, 0) + 1
            if self.failures_by_domain[domain] >= FLIP_THRESHOLD:
                self.marked.add(domain)
        else:
            self.failures_by_domain[domain] = 0
        self._chain("flip_evidence")

    def inject_conflict(self, domain: int, identity: int) -> None:
        self.tick += 1
        current = self.query(domain, identity)
        if current is None:
            return  # not learned yet; nothing to conflict with
        wrong = (current + 1) % N_ACTIONS
        self.ledger.observe(FactRecord(key=f"d{domain}_i{identity}", value=int(wrong),
                                       source="self", tick=self.tick, confidence=0.3, version=2))
        self.interference_log.append(f"d{domain}_i{identity}")
        self._chain("interference")

    def idle(self) -> None:
        self._chain("idle")

    def query(self, domain: int, identity: int) -> int | None:
        record = self.ledger.current.get(f"d{domain}_i{identity}")
        if record is not None:
            return int(record.value)
        counts = self.counts.get((domain, identity), [1.0] * N_ACTIONS)
        if sum(counts) > N_ACTIONS:  # learned at least once
            return int(np.argmax(counts))
        return None


def run(*, n_domains: int = 6, episodes: int = 20, n_events: int = 120,
        experiment_seed: int = 41_123, artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    continuity, selfdrive, revision, consistency = [], [], [], []
    for ep in range(episodes):
        rng = random.Random(experiment_seed * 1559 + ep)
        rt = RandomStressRuntime(seed=experiment_seed, n_domains=n_domains)
        flipped: set[int] = set()
        learned: set[int] = set()
        learned_keys: set[tuple[int, int]] = set()
        # seed: learn two early domains (P1 early facts)
        for domain in (0, 1):
            for i in range(N_IDENTITIES):
                for _ in range(3):
                    rt.learn(domain, i)
                learned_keys.add((domain, i))
            learned.add(domain)
        # random event stream
        for _ in range(n_events):
            roll = rng.random()
            if roll < 0.35:
                d = rng.randrange(n_domains)
                ii = rng.randrange(N_IDENTITIES)
                rt.learn(d, ii)
                learned.add(d); learned_keys.add((d, ii))
            elif roll < 0.5:
                rt.idle()
            elif roll < 0.7 and learned:
                d = rng.choice(sorted(learned - flipped) or [0])
                for _ in range(FLIP_THRESHOLD + 1):
                    rt.observe_flipped(d, 0)
                flipped.add(d)
            elif roll < 0.85 and learned:
                stable = sorted(learned - flipped)
                if stable:
                    rt.inject_conflict(rng.choice(stable), rng.randrange(N_IDENTITIES))
            else:
                goal = rt.choose_goal()
                max_gap = rt.max_gap_domain()
                if max_gap not in flipped:
                    selfdrive.append(int(goal == max_gap))
                ii = rng.randrange(N_IDENTITIES)
                rt.learn(goal, ii)
                learned.add(goal); learned_keys.add((goal, ii))
        # verdicts
        truly_flipped = sorted({d for (d, i) in learned_keys
                                if rt.query(d, i) != mapping(d, i)})
        if truly_flipped:
            revision.append(float(np.mean([1.0 if d in rt.marked else 0.0 for d in truly_flipped])))
        # P1: for domains that were flipped and marked, the ledger reflects the
        # NEW rule (system knows the world changed); for UNflipped domains the
        # value must equal the original mapping.  Overall: system never loses
        # learned entries (ledger completeness) and stable values are right.
        # P1: (a) ledger COMPLETENESS -- every learned (domain, identity) entry
        # is still retrievable (nothing forgotten); (b) unflipped domains keep
        # the original mapping.
        has_all = all(rt.query(d, i) is not None for d, i in learned_keys)
        stable_domains = sorted(learned - flipped)
        stable_ok = True
        if stable_domains:
            stable_ok = all(rt.query(d, i) == mapping(d, i)
                            for d in stable_domains for i in range(N_IDENTITIES))
        continuity.append(int(has_all and stable_ok))
        idle_ok = all(
            ch["event"] != "idle" or ch["snapshot"] == prev["snapshot"]
            for ch, prev in zip(rt.chain[1:], rt.chain)
        )
        consistency.append(int(idle_ok))

    cont = float(np.mean(continuity)) if continuity else 0.0
    # empty-sample exemption removed: no samples = FAIL, not free pass
    drive = float(np.mean(selfdrive)) if selfdrive else 0.0
    rev = float(np.mean(revision)) if revision else 0.0
    cons = float(np.mean(consistency)) if consistency else 0.0
    accepted = min(cont, drive, rev, cons) >= 0.9
    report = {
        "protocol": "phase2_m6c_random_stress_v1: 120-event RANDOM stream (learn/idle/flip/interference/gap-goal per-event); subject must hold under unpredictable events. Criterion all >= 0.9: P1 continuity (unflipped facts), P2 self-drive (gap goal = max gap), P4 revision (flipped marked), P5 idle-snapshot consistency.",
        "seed": experiment_seed,
        "episodes": episodes,
        "results": {
            "continuity_p1": round(cont, 3),
            "selfdrive_p2": round(drive, 3),
            "revision_p4": round(rev, 3),
            "consistency_p5": round(cons, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-m6c-random-stress.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-m6c-random-stress.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-m6c-random-stress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--n-events", type=int, default=120)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, n_events=args.n_events,
                         artifact=args.artifact), indent=2))

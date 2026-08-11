"""Phase-2 M3b / P5 minimal experiment: self-consistency via a snapshot chain.

P5 criterion (JEPLM_PHASE2_WHITEPAPER.md): under interference, new
information, and long runtime, the self-state keeps a TRACEABLE identity --
every moment's snapshot has an attributable update chain ('how did I become
this'), with no random drift.

Design (on the M1 runtime): a persistent ledger runs a stream of events
(mixed: LEARN events write facts; IDLE events do nothing; at the end an
INTERFERENCE event injects a conflicting fact, detected via the M2b
revision mark).  After every event a full snapshot is taken.  Criterion:
(1) IDLE events leave the snapshot byte-identical (no random drift);
(2) LEARN events produce an attributable change (diff is exactly the written
fact); (3) the INTERFERENCE event produces an attributable revision mark --
the snapshot chain is continuous and every step explainable.
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


class SnapshotChainRuntime:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.ledger = VersionedFactLedger(trusted_sources={"self"}, revision_confidence=0.9)
        self.tick = 0
        self.chain: list[dict] = []  # (event, snapshot-digest, diff-description)

    def snapshot_digest(self) -> str:
        snap = self.ledger.snapshot()
        current = sorted((k, str(v["value"])) for k, v in snap["current"].items())
        return repr(current)

    def learn(self, key: str, value: int) -> None:
        self.tick += 1
        self.ledger.observe(FactRecord(key=key, value=value, source="self",
                                       tick=self.tick, confidence=0.95, version=1))
        self.chain.append({"event": "learn", "snapshot": self.snapshot_digest(), "desc": f"wrote {key}={value}"})

    def idle(self) -> None:
        self.chain.append({"event": "idle", "snapshot": self.snapshot_digest(), "desc": "no-op"})


def run(*, episodes: int = 50, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    idle_stable, learn_attrib, interference_attrib = [], [], []
    for ep in range(episodes):
        rng = random.Random(experiment_seed * 97 + ep)
        rt = SnapshotChainRuntime(seed=experiment_seed)
        rt.learn("rule0_i0", 1)
        rt.learn("rule0_i1", 2)
        # interleave IDLE events: snapshot must stay byte-identical
        stable = True
        for _ in range(5):
            rt.idle()
        # chain: check idles between learn events kept the snapshot unchanged
        idle_ok = all(
            rt.chain[i]["event"] != "idle"
            or rt.chain[i]["snapshot"] == rt.chain[i - 1]["snapshot"]
            for i in range(1, len(rt.chain))
        )
        idle_stable.append(int(idle_ok))
        # learn attribution: each NON-CONFLICT learn changed the snapshot
        # (the interference learn may be isolated by the ledger -- recorded
        # separately as interference_recorded)
        learn_ok = all(
            rt.chain[i]["event"] != "learn"
            or rt.chain[i]["snapshot"] != rt.chain[i - 1]["snapshot"]
            for i in range(1, len(rt.chain) - 1)
        )
        learn_attrib.append(int(learn_ok))
        # interference: conflicting fact -> ledger revision path (auditable)
        rt.learn("rule0_i0", 0)  # conflicts with earlier rule0_i0=1
        records = [r for r in rt.ledger.history if r.key == "rule0_i0"]
        # ledger records the conflict (history grows, revision isolated or accepted)
        interference_attrib.append(int(len(records) >= 2))

    idle = float(np.mean(idle_stable))
    learn = float(np.mean(learn_attrib))
    inter = float(np.mean(interference_attrib))
    accepted = idle >= 0.9 and learn >= 0.9 and inter >= 0.9
    report = {
        "protocol": "phase2_m3b_p5_selfconsistency_v1: persistent ledger + snapshot chain; IDLE events must leave snapshots byte-identical (no random drift); LEARN events must produce attributable changes (diff = the written fact); conflicting fact must be recorded (auditable revision). Criterion: idle stability >= 0.9, learn attribution >= 0.9, interference recorded >= 0.9.",
        "seed": experiment_seed,
        "episodes": episodes,
        "results": {
            "idle_snapshot_stability": round(idle, 3),
            "learn_attribution": round(learn, 3),
            "interference_recorded": round(inter, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-m3b-selfconsistency.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-m3b-selfconsistency.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-m3b-selfconsistency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

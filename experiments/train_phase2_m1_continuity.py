"""Phase-2 M1 / P1 minimal experiment: continuity of a unified runtime.

P1 criterion (JEPLM_PHASE2_WHITEPAPER.md): the system keeps the SAME knowledge
across tasks (no reset, no catastrophic forgetting) and any moment's full state
is exportable (auditable snapshot).  M1 = a unified runtime MVP built on
VersionedFactLedger (auditable, snapshot/restore) + a monotonic tick (event
time), running a THREE-TASK stream:

  Task A (rules 1-2) -> Task B (rules 3-4) -> Task A' (query rules 1-2 again)

The runtime is NOT reset between tasks.  Controls: (1) a RESET variant (new
ledger per task -- the 'forgetful' control), (2) snapshot/restore round-trip
after Task B.  Criterion: continuous variant answers A' >= 0.9, snapshot
restore also >= 0.9, and continuous beats reset by >= 0.3.
"""
from __future__ import annotations

import argparse
import json
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
CONFIDENCE = 0.95


def rule_mapping(rule: int, identity: int) -> int:
    return (identity + rule) % N_ACTIONS


def _rules_for(task: str, seed: int) -> list[int]:
    rng = random.Random(seed)
    if task == "A":
        return [0, 1]
    if task == "B":
        return [2, 3]
    raise ValueError(task)


class UnifiedRuntime:
    """M1 MVP: persistent auditable knowledge + monotonic event time + snapshot."""

    def __init__(self, *, reset_per_task: bool = False, seed: int = 0) -> None:
        self.ledger = VersionedFactLedger(trusted_sources={"consolidation"}, revision_confidence=0.9)
        self.reset_per_task = reset_per_task
        self.seed = seed
        self.tick = 0
        self._task_events = 0

    def observe_rule(self, rule: int, identity: int, action: int) -> None:
        self.tick += 1
        record = FactRecord(
            key=f"rule{rule}_{identity}", value=int(action), source="consolidation",
            tick=self.tick, confidence=CONFIDENCE, version=1,
        )
        self.ledger.observe(record)

    def query(self, rule: int, identity: int) -> int | None:
        record = self.ledger.current.get(f"rule{rule}_{identity}")
        return record.value if record is not None else None

    def run_task(self, task: str) -> None:
        if self.reset_per_task:
            self.ledger = VersionedFactLedger(trusted_sources={"consolidation"}, revision_confidence=0.9)
            self.tick = 0
        rng = random.Random(self.seed * 1000 + self._task_events)
        self._task_events += 1
        for rule in _rules_for(task, self.seed):
            for identity in range(N_IDENTITIES):
                self.observe_rule(rule, identity, rule_mapping(rule, identity))
            # a few repeated observations to strengthen (consolidation-like)
            for _ in range(3):
                identity = rng.randrange(N_IDENTITIES)
                self.observe_rule(rule, identity, rule_mapping(rule, identity))

    def snapshot(self) -> dict:
        return {"ledger": self.ledger.snapshot(), "tick": self.tick}

    @classmethod
    def restore(cls, snap: dict) -> "UnifiedRuntime":
        runtime = cls(reset_per_task=False, seed=0)
        runtime.ledger = VersionedFactLedger.restore(snap["ledger"])
        runtime.tick = snap["tick"]
        return runtime

    def evaluate_task(self, task: str) -> float:
        correct = total = 0
        for rule in _rules_for(task, self.seed):
            for identity in range(N_IDENTITIES):
                value = self.query(rule, identity)
                total += 1
                correct += int(value == rule_mapping(rule, identity))
        return correct / total if total else 0.0


def run(*, experiment_seed: int = 41_123, artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    # continuous variant: no reset between tasks
    runtime = UnifiedRuntime(reset_per_task=False, seed=experiment_seed)
    runtime.run_task("A")
    runtime.run_task("B")
    snap = runtime.snapshot()
    after_b = runtime.evaluate_task("A")
    # snapshot round-trip: restore from the Task-B snapshot, then query A again
    restored = UnifiedRuntime.restore(snap)
    after_restore = restored.evaluate_task("A")

    # reset control: fresh ledger per task (forgetful)
    reset = UnifiedRuntime(reset_per_task=True, seed=experiment_seed)
    reset.run_task("A")
    reset.run_task("B")
    reset_a = reset.evaluate_task("A")

    accepted = after_b >= 0.9 and after_restore >= 0.9 and (after_b - reset_a) >= 0.3
    report = {
        "protocol": "phase2_m1_p1_continuity_v1: three-task stream A(rule1-2)->B(rule3-4)->A'(query rule1-2); unified runtime = persistent VersionedFactLedger + monotonic tick, NOT reset between tasks; controls: reset-per-task ledger, snapshot/restore round-trip after B. Criterion: A' accuracy >= 0.9 (continuous), >= 0.9 (after snapshot restore), and continuous - reset >= 0.3.",
        "seed": experiment_seed,
        "results": {
            "continuous_after_B": after_b,
            "after_snapshot_restore": after_restore,
            "reset_control": reset_a,
            "snapshot_has_A": int(all(f"rule{r}_{i}" in snap["ledger"]["current"] for r in [0, 1] for i in range(N_IDENTITIES))),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-m1-continuity.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-m1-continuity.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-m1-continuity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

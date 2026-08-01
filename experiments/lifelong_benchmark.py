"""Lifelong memory benchmark: non-stationarity, interference, and provenance."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.lifelong import LastWriteMemory, VersionedFactMemory


@dataclass
class LifelongResult:
    stable_retention: float
    changed_fact_revision: bool
    stale_burst_resistance: bool
    provenance_complete: bool
    consolidation_count: int
    last_write_stable_retention: float
    last_write_changed_fact_revision: bool


def _seed(memory) -> dict[int, int]:
    facts = {0: 1, 1: 0, 2: 2, 3: 1}
    for key, value in facts.items():
        memory.ingest(key, value, confidence=0.99, source="verified-bootstrap", verified=True)
    return facts


def run() -> LifelongResult:
    persistent = VersionedFactMemory()
    naive = LastWriteMemory()
    facts = _seed(persistent)
    _seed(naive)

    # A fact changes. The new observation is verified; later stale low-confidence
    # reports should not undo it or touch unrelated keys.
    persistent.ingest(0, 2, confidence=0.99, source="verified-update", verified=True)
    naive.ingest(0, 2, confidence=0.99, source="verified-update", verified=True)
    for index in range(40):
        stale_value = 1 if index % 2 == 0 else 0
        persistent.ingest(0, stale_value, confidence=0.55, source="noisy-burst", verified=False)
        naive.ingest(0, stale_value, confidence=0.55, source="noisy-burst", verified=False)
    for _ in range(3):
        persistent.ingest(3, facts[3], confidence=0.83, source="sleep-replay", verified=False)
    persistent.sleep(replay_limit=64)
    naive.sleep()

    stable_keys = (1, 2, 3)
    stable_retention = sum(persistent.retrieve(key) == facts[key] for key in stable_keys) / len(stable_keys)
    naive_stable_retention = sum(naive.retrieve(key) == facts[key] for key in stable_keys) / len(stable_keys)
    provenance = persistent.provenance(0)
    return LifelongResult(
        stable_retention=stable_retention,
        changed_fact_revision=persistent.retrieve(0) == 2,
        stale_burst_resistance=persistent.retrieve(0) == 2,
        provenance_complete=any(item.source == "verified-update" and item.verified for item in provenance)
        and any(item.source == "noisy-burst" and not item.verified for item in provenance),
        consolidation_count=persistent.consolidation_count,
        last_write_stable_retention=naive_stable_retention,
        last_write_changed_fact_revision=naive.retrieve(0) == 2,
    )


if __name__ == "__main__":
    print(asdict(run()))

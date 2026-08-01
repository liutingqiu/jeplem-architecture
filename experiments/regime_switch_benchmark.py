"""Causal-rule change benchmark with cross-world interference controls."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.regime import LastObservationRegime, RegimeMemory


@dataclass
class RegimeSwitchResult:
    revised_action_correct: bool
    stale_reports_ignored: bool
    unrelated_world_retained: bool
    naive_stale_action_correct: bool


def run() -> RegimeSwitchResult:
    persistent, naive = RegimeMemory(), LastObservationRegime()
    for memory in (persistent, naive):
        memory.ingest("alpha", 1, confidence=0.99, verified=True, source="calibration")
        memory.ingest("beta", -1, confidence=0.99, verified=True, source="calibration")
        memory.ingest("alpha", -1, confidence=0.99, verified=True, source="intervention")
        for value in (1, -1, 1, 1):
            memory.ingest("alpha", value, confidence=0.52, verified=False, source="stale-telemetry")
    return RegimeSwitchResult(
        persistent.plan_action("alpha", -1) == 1,
        persistent.regime("alpha") == -1,
        persistent.plan_action("beta", -1) == 1,
        naive.plan_action("alpha", -1) == 1,
    )


if __name__ == "__main__":
    print(asdict(run()))

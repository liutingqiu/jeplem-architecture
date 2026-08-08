"""Contract audit for source-bound, conflict-isolated fact revision."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord, VersionedFactLedger


@dataclass
class Report:
    protocol: str
    accepted_initial: bool
    weak_conflict_isolated: bool
    unrelated_retained: bool
    strong_revision_applied: bool
    stale_conflict_isolated: bool
    restored_complete: bool
    accepted: bool
    artifact: str


def run(*, artifact: Path | None = None) -> Report:
    ledger = VersionedFactLedger(trusted_sources={"luanti-reward", "causal-audit"})
    accepted_initial = ledger.observe(FactRecord("door_mode", "open", "luanti-reward", 10, 1.0, 1, "a")) == "accepted"
    ledger.observe(FactRecord("key_color", "green", "luanti-reward", 11, 1.0, 1, "b"))
    weak_conflict_isolated = ledger.observe(FactRecord("door_mode", "closed", "vision-noise", 12, 0.4, 1, "c")) == "isolated-conflict"
    unrelated_retained = ledger.current["key_color"].value == "green"
    strong_revision_applied = ledger.observe(FactRecord("door_mode", "closed", "causal-audit", 13, 1.0, 1, "d")) == "revised"
    stale_conflict_isolated = ledger.observe(FactRecord("door_mode", "open", "luanti-reward", 12, 1.0, 1, "e")) == "isolated-stale"
    restored = VersionedFactLedger.restore(ledger.snapshot())
    restored_complete = restored.snapshot() == ledger.snapshot()
    target = artifact or ROOT / "artifacts" / "versioned-fact-ledger-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "versioned_fact_ledger_v1: strong source/time/confidence evidence may revise durable facts; weak, stale, or cross-source conflicts are isolated with their provenance and never silently overwrite unrelated facts.",
        accepted_initial, weak_conflict_isolated, unrelated_retained, strong_revision_applied,
        stale_conflict_isolated, restored_complete,
        all((accepted_initial, weak_conflict_isolated, unrelated_retained, strong_revision_applied, stale_conflict_isolated, restored_complete)),
        str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("versioned fact ledger did not meet its audit contract")

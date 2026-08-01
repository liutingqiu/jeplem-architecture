"""Verify persistent commitment selection through interruption and evidence."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.commitments import Commitment, CommitmentLedger, CommitmentStatus


@dataclass
class Report:
    selected_initially: str | None
    selected_after_interruption: str | None
    selected_after_completion: str | None
    completed_status: str
    cancelled_status: str
    evidence_recorded: bool


def run() -> Report:
    ledger = CommitmentLedger()
    ledger.add(Commitment("ship", target_state=2, priority=0.9, deadline=5))
    ledger.add(Commitment("research", target_state=1, priority=0.6))
    initial = ledger.select([0.9, 0.0, 0.0])
    ledger.tick()
    ledger.tick()  # unrelated interruption, neither commitment disappears
    after_interruption = ledger.select([0.1, 0.2, 0.0])
    ledger.reconcile([0.05, 0.01, 0.94])
    after_completion = ledger.select([0.05, 0.01, 0.94])
    ledger.cancel("research", "superseded by external evidence")
    ship = next(item for item in ledger.commitments if item.id == "ship")
    research = next(item for item in ledger.commitments if item.id == "research")
    return Report(
        selected_initially=initial.id if initial else None,
        selected_after_interruption=after_interruption.id if after_interruption else None,
        selected_after_completion=after_completion.id if after_completion else None,
        completed_status=ship.status.value,
        cancelled_status=research.status.value,
        evidence_recorded=bool(ship.evidence) and bool(research.evidence),
    )


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not (
        outcome.selected_initially == "ship"
        and outcome.selected_after_interruption == "ship"
        and outcome.selected_after_completion == "research"
        and outcome.completed_status == CommitmentStatus.COMPLETED.value
        and outcome.cancelled_status == CommitmentStatus.CANCELLED.value
        and outcome.evidence_recorded
    ):
        raise SystemExit("commitment recovery gate failed")

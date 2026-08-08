"""Verify Luanti reward envelopes enter the versioned fact ledger safely."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord, LuantiRewardFactLedger, VersionedFactLedger


@dataclass
class Report:
    protocol: str
    charge_accepted: bool
    key_accepted: bool
    source_bound: bool
    weak_conflict_isolated: bool
    restored_complete: bool
    accepted: bool
    artifact: str


def run(*, artifact: Path | None = None) -> Report:
    adapter = LuantiRewardFactLedger()
    charge = {"value": 0.15, "_plwm_provenance": {"source": "luanti-reward", "tick": 41, "request_id": "bridge-charge"}}
    key = {"value": 0.2, "_plwm_provenance": {"source": "luanti-reward", "tick": 58, "request_id": "bridge-key"}}
    charge_accepted = adapter.observe_reward(charge) == "accepted"
    key_accepted = adapter.observe_reward(key) == "accepted"
    source_bound = (
        adapter.ledger.current["charge_verified"].request_id == "bridge-charge"
        and adapter.ledger.current["key_verified"].request_id == "bridge-key"
        and adapter.ledger.current["key_verified"].source == "luanti-reward"
    )
    weak_conflict_isolated = adapter.ledger.observe(
        FactRecord("key_verified", False, "vision-noise", 60, 0.3, 1, "weak-conflict")
    ) == "isolated-conflict"
    restored_complete = VersionedFactLedger.restore(adapter.ledger.snapshot()).snapshot() == adapter.ledger.snapshot()
    target = artifact or ROOT / "artifacts" / "luanti-reward-ledger-contract-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "luanti_reward_ledger_contract_v1: only existing Luanti reward envelopes with source/tick/request provenance become bounded prerequisite facts; a weak non-reward conflict remains isolated.",
        charge_accepted, key_accepted, source_bound, weak_conflict_isolated, restored_complete,
        all((charge_accepted, key_accepted, source_bound, weak_conflict_isolated, restored_complete)), str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("Luanti reward ledger contract did not meet its gate")

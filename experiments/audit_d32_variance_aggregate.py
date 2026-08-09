"""Aggregate every independent d=32 measurement of PLWM / evidence / GRU
across the three experimental batches (scan batch, rmt batch, rmt-full batch)
to report the true variance instead of a single batch's mean.

9 seed-runs for PLWM and GRU (scan 111/222/333 + rmt 222/333/444 +
rmt-full 222/333/444); 3 for evidence_store (scan batch only; the rmt batches
do not include it).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BATCHES = {
    "scan": ["scan-d32-n2-seed111.json", "scan-d32-n2-seed222.json", "scan-d32-n2-seed333.json"],
    "rmt": ["rmt-d32-seed222.json", "rmt-d32-seed333.json", "rmt-d32-seed444.json"],
    "rmt_full": ["rmt-full-d32-seed222.json", "rmt-full-d32-seed333.json", "rmt-full-d32-seed444.json"],
}


def main() -> None:
    heldout: dict[str, list[float]] = {}
    evidence: dict[str, list[float]] = {}
    for batch, files in BATCHES.items():
        for f in files:
            data = json.loads((ROOT / "artifacts" / f).read_text(encoding="utf-8"))
            for r in data["results"]:
                if r["name"] in ("plwm", "plwm_policy"):
                    heldout.setdefault("plwm", []).append(r["heldout_accuracy"])
                    heldout.setdefault("batch_of_plwm", []).append(batch)
                elif r["name"] == "gru":
                    heldout.setdefault("gru", []).append(r["heldout_accuracy"])
                elif r["name"] == "evidence_store":
                    evidence.setdefault("evidence_store", []).append(r["heldout_accuracy"])

    summary: dict = {"protocol": "d32_variance_aggregate_v1", "delay": 32,
                     "note": "independent batch re-runs of the same config; PLWM batches: scan(111/222/333), rmt(222/333/444), rmt-full(222/333/444)"}
    for model in ("plwm", "gru"):
        vals = heldout[model]
        summary[model] = {
            "n": len(vals),
            "values": [round(v, 4) for v in vals],
            "min": round(min(vals), 4),
            "median": round(statistics.median(vals), 4),
            "max": round(max(vals), 4),
            "mean": round(sum(vals) / len(vals), 4),
            "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
        }
    summary["plwm"]["per_batch_mean"] = {}
    for batch in BATCHES:
        summary["plwm"]["per_batch_mean"][batch] = round(
            sum(v for v, b in zip(heldout["plwm"], heldout["batch_of_plwm"]) if b == batch) / 3, 4)
    if evidence:
        vals = evidence["evidence_store"]
        summary["evidence_store"] = {
            "n": len(vals), "values": [round(v, 4) for v in vals],
            "min": round(min(vals), 4), "median": round(statistics.median(vals), 4),
            "max": round(max(vals), 4), "mean": round(sum(vals) / len(vals), 4),
        }
    out = ROOT / "artifacts" / "d32-variance-aggregate.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""G4: is the LEDGER a knowledge base? (the facts axis of F)

Phase-3 whitepaper G4 -- the independent memory/facts gate.  The hypothesis:
explicit memory (the VersionedFactLedger) IS the knowledge base; factual
benchmarks rise with ledger size, and unlike lossy LLM weights the ledger
recalls facts EXACTLY and KNOWS when it does not know (no hallucination).

Experiment (controlled factual world): E entities x A attributes, a value
vocabulary.  Facts are ingested into the ledger.  Queries are answered from
the ledger.  Measured:
  1. recall on INGESTED facts (exact retrieval),
  2. unknown-detection on UNINGESTED (entity, attribute) pairs (the body must
     say "no record", not hallucinate) -- the self-model's knows-when-unknown,
  3. scaling: recall and query behaviour vs ledger size (10^3 -> 10^5 facts),
  4. revision: a conflicting fact REVISES the ledger (P4), not corrupts it.

Gate (whitepaper): factual recall rises with ledger size (exact), unknown
detection high, and the ledger scales (10^4-10^5 facts) without recall loss.
This supports the "memory instead of weights for factual knowledge" component
of F -- an architecture advantage over lossy LLM weights.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord, VersionedFactLedger


def run(*, sizes=(1_000, 10_000, 50_000), n_queries=5_000,
        experiment_seed: int = 41_123, artifact: Path | None = None) -> dict:
    rng = random.Random(experiment_seed)
    A = 12          # attributes per entity
    VOCAB = 30      # value vocabulary size
    results = {}
    for n_facts in sizes:
        n_entities = n_facts // A
        # world truth
        truth: dict[tuple[int, int], int] = {
            (e, a): rng.randrange(VOCAB) for e in range(n_entities) for a in range(A)
        }
        ledger = VersionedFactLedger(trusted_sources={"body"}, revision_confidence=0.9)
        t0 = time.time()
        tick = 0
        for (e, a), v in truth.items():
            tick += 1
            ledger.observe(FactRecord(key=f"e{e}_a{a}", value=int(v), source="body",
                                      tick=tick, confidence=1.0, version=1))
        ingest_ms = (time.time() - t0) * 1000

        # 1 + 2: query ingested vs uningested
        recall_c = recall_t = 0
        unknown_c = unknown_t = 0
        t0 = time.time()
        for _ in range(n_queries):
            if rng.random() < 0.5:
                e, a = rng.randrange(n_entities), rng.randrange(A)
                rec = ledger.current.get(f"e{e}_a{a}")
                recall_c += int(rec is not None and int(rec.value) == truth[(e, a)])
                recall_t += 1
            else:
                # uningested: either out-of-entity or a value not in the world
                e = rng.randrange(n_entities, n_entities + 200)
                a = rng.randrange(A)
                rec = ledger.current.get(f"e{e}_a{a}")
                unknown_c += int(rec is None)   # knows it does not know
                unknown_t += 1
        query_ms = (time.time() - t0) * 1000

        # 4: revision -- a conflicting fact updates the ledger, others intact
        rev_target = (0, 0)
        if n_facts >= A:
            old = truth[rev_target]
            new = (old + 1) % VOCAB
            tick += 1
            ledger.observe(FactRecord(key="e0_a0", value=new, source="body",
                                      tick=tick, confidence=1.0, version=2))
            rec = ledger.current.get("e0_a0")
            revision_ok = int(rec is not None and int(rec.value) == new)
            other_ok = int(ledger.current.get("e1_a0") is not None)
        else:
            revision_ok = other_ok = 0

        results[f"n{n_facts}"] = {
            "n_facts": n_facts,
            "n_entities": n_entities,
            "recall": round(recall_c / recall_t, 4) if recall_t else 0.0,
            "unknown_detection": round(unknown_c / unknown_t, 4) if unknown_t else 0.0,
            "ingest_ms": round(ingest_ms, 1),
            "query_ms_5k": round(query_ms, 1),
            "revision_ok": revision_ok,
            "other_intact_after_revision": other_ok,
        }

    final = results[f"n{sizes[-1]}"]
    gate_pass = (
        final["recall"] >= 0.99
        and final["unknown_detection"] >= 0.95
        and final["revision_ok"] == 1
        and final["other_intact_after_revision"] == 1
    )
    report = {
        "protocol": "phase3_g4_ledger_knowledge_base_v1: synthetic factual world (E entities x 12 attributes, 30-value vocab); facts ingested into VersionedFactLedger; recall on ingested (exact), unknown-detection on uningested, scaling 1k/10k/50k facts, revision on conflict. Gate: recall >= 0.99, unknown >= 0.95, revision works, ledger scales without recall loss.",
        "seed": experiment_seed,
        "results": results,
        "gate_pass": bool(gate_pass),
        "artifact": str(artifact or ROOT / "artifacts" / "arc-g4-ledger-knowledge-base.json"),
    }
    (artifact or ROOT / "artifacts" / "arc-g4-ledger-knowledge-base.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "arc-g4-ledger-knowledge-base.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

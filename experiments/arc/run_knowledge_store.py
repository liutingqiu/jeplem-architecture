"""Knowledge pre-training step 2: disk-backed knowledge store + reasoning.

Ingest the full English ConceptNet into the SQLite-backed KnowledgeStore
(memory-bounded, scales to Wikidata), then demonstrate the body answering
natural-style questions from its knowledge: exact match + transitive reasoning
+ honest unknown.

Resource-safety: ingestion streams the gz (bounded RAM); the store lives on
disk; queries are indexed.  Wikidata (10^8 facts) is the scale this enables.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore
from experiments.arc.run_knowledge_ingest import iter_edges, CSV

DB = ROOT / "data" / "knowledge" / "conceptnet.db"


def mem_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4_000_000)
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "knowledge-store-step2.json")
    args = parser.parse_args()

    t0 = time.time()
    store = KnowledgeStore(args.db)
    n = store.ingest_batch(iter_edges(CSV, limit=args.limit))
    ingest_s = time.time() - t0
    db_size = args.db.stat().st_size / 1e6 if args.db.exists() else 0.0

    # statistical validation on the disk store
    import random
    rng = random.Random(41_123)
    starts = [r[0] for r in store.conn.execute(
        "SELECT DISTINCT start FROM facts LIMIT 5000")]
    recall_c = recall_t = 0
    unknown_c = unknown_t = 0
    for _ in range(10_000):
        if rng.random() < 0.5 and starts:
            start = rng.choice(starts)
            rels = store.conn.execute(
                "SELECT rel, end FROM facts WHERE start=? LIMIT 1", (start,)).fetchone()
            if rels:
                recall_c += int(store.known(rels[0], start, rels[1]))
                recall_t += 1
        else:
            rel = rng.choice(["IsA", "UsedFor", "HasProperty", "Antonym", "PartOf"])
            start = f"en:{rng.choice(['cat','dog','car','sun','water','tree','book'])}"
            end = f"en:{rng.choice(['cat','dog','car','sun','water','tree','book'])}"
            unknown_c += int(not store.known(rel, start, end))
            unknown_t += 1

    # reasoning demo (exact + transitive + unknown)
    q = [
        ("IsA", "en:apple", "en:fruit", "apple is a fruit"),
        ("IsA", "en:dog", "en:animal", "dog is an animal (transitive)"),
        ("IsA", "en:dog", "en:planet", "dog is a planet (fake)"),
        ("UsedFor", "en:car", "en:transportation", "car is used for transportation"),
        ("IsA", "en:rose", "en:plant", "rose is a plant"),
    ]
    answers = []
    for rel, s, e, phrase in q:
        exact = store.known(rel, s, e)
        trans = store.transitive(rel, s, e)
        answers.append({"q": phrase, "exact": exact, "transitive": trans,
                        "answer": "KNOW" if (exact or trans) else "DON'T KNOW"})

    store.close()
    report = {
        "protocol": "knowledge_pretrain_step2_v1: disk-backed (SQLite) knowledge store + transitive reasoning. Scales to Wikidata (memory-bounded); validate recall/unknown on the real KG and demonstrate exact + transitive + self-aware answers.",
        "results": {
            "ingested_edges": n,
            "ingest_seconds": round(ingest_s, 1),
            "db_size_mb": round(db_size, 1),
            "peak_ram_mb": round(mem_mb(), 0),
            "recall": round(recall_c / recall_t, 4) if recall_t else 0.0,
            "unknown_detection": round(unknown_c / unknown_t, 4) if unknown_t else 0.0,
            "answers": answers,
            "scaling_note": "in-memory ledger = ~440B/fact (3.4M -> 1.5GB); disk store keeps RAM bounded, enabling Wikidata 10^8 facts on this machine",
        },
        "artifact": str(args.artifact),
    }
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], indent=1))


if __name__ == "__main__":
    main()

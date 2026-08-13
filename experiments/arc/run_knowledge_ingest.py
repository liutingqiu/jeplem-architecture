"""Knowledge pre-training, step 1: ingest a REAL knowledge graph (ConceptNet
5.7) into the body's VersionedFactLedger and validate the knowledge-base
behaviour on real data (G4 was synthetic; this is the real-data check).

ConceptNet CSV columns (tab-separated): assertion, relation, start, end, weight,
[source, surfaceText].  We keep English concepts (/c/en/...) and store each
edge as an EXISTENCE fact in the ledger:
  key = "REL|<start>|<end>"  ->  value = 1   (the fact "REL(start,end)" is true)
Querying the ledger for a key answers "do I know this fact?" with exact recall;
a missing key answers "I do not know" (the self-aware unknown-detection).

Validation on real data:
  - recall: sample ingested edges -> all present in the ledger
  - unknown-detection: sample unseen (relation, start, end) triples -> absent
  - revision (P4): a single-valued fact "capitalOf(France)" ingested as Paris,
    then updated to Lyon with a higher tick -> the ledger revises, not corrupts
  - scaling note: memory/time for the ingested subset vs the full ~3M edges
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord, VersionedFactLedger

CSV = ROOT / "data" / "knowledge" / "conceptnet-assertions-5.7.0.csv.gz"


def parse_uri(uri: str) -> str | None:
    """/c/en/apple/n -> 'en:apple' (None if not a concept or not English)."""
    parts = uri.split("/")
    # /c/<lang>/<word>/...
    if len(parts) >= 4 and parts[1] == "c" and parts[2] == "en":
        word = parts[3].replace("_", " ")
        return f"en:{word}"
    return None


def iter_edges(path: Path, limit: int):
    """Yield (relation, start, end) for English ConceptNet edges, up to limit."""
    n = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            rel = cols[1].strip("/").split("/")[-1] if "/" in cols[1] else cols[1].strip("/")
            start = parse_uri(cols[2])
            end = parse_uri(cols[3])
            if start is None or end is None or not rel:
                continue
            yield rel, start, end
            n += 1
            if n >= limit:
                return


def run(*, limit: int = 500_000, n_query: int = 20_000,
        experiment_seed: int = 41_123, artifact: Path | None = None) -> dict:
    rng = random.Random(experiment_seed)
    ledger = VersionedFactLedger(trusted_sources={"knowledge"}, revision_confidence=0.9)

    t0 = time.time()
    tick = 0
    ingested_keys: set[str] = set()
    n_total = 0
    for rel, start, end in iter_edges(CSV, limit):
        tick += 1
        key = f"{rel}|{start}|{end}"
        ledger.observe(FactRecord(key=key, value=1, source="knowledge",
                                  tick=tick, confidence=1.0, version=1))
        ingested_keys.add(key)
        n_total += 1
    ingest_s = time.time() - t0

    # 1 + 2: recall on ingested, unknown-detection on unseen
    keys = sorted(ingested_keys)
    recall_c = recall_t = 0
    unknown_c = unknown_t = 0
    for _ in range(n_query):
        if rng.random() < 0.5:
            key = rng.choice(keys)
            recall_c += int(key in ledger.current)
            recall_t += 1
        else:
            rel = rng.choice(["IsA", "UsedFor", "HasProperty", "Antonym", "PartOf"])
            start = f"en:{rng.choice(['cat', 'dog', 'car', 'sun', 'water', 'tree', 'book', 'door', 'apple', 'house'])}"
            end = f"en:{rng.choice(['cat', 'dog', 'car', 'sun', 'water', 'tree', 'book', 'door', 'apple', 'house'])}"
            key = f"{rel}|{start}|{end}"
            unknown_c += int(key not in ledger.current)
            unknown_t += 1

    # 3: revision (P4) on a single-valued fact
    rev_key = "capitalOf|en:france"
    ledger.observe(FactRecord(key=rev_key, value="Paris", source="knowledge",
                              tick=tick + 1, confidence=1.0, version=1))
    ledger.observe(FactRecord(key=rev_key, value="Lyon", source="knowledge",
                              tick=tick + 2, confidence=1.0, version=2))
    rec = ledger.current.get(rev_key)
    revision_ok = int(rec is not None and rec.value == "Lyon")

    report = {
        "protocol": "knowledge_pretrain_step1_v1: ingest ConceptNet 5.7 (real KG) into VersionedFactLedger as existence facts; validate recall (ingested), unknown-detection (unseen), revision (P4). G4 was synthetic -- this is the real-data knowledge-base check.",
        "seed": experiment_seed,
        "results": {
            "ingested_edges": n_total,
            "unique_keys": len(ingested_keys),
            "ingest_seconds": round(ingest_s, 1),
            "recall": round(recall_c / recall_t, 4) if recall_t else 0.0,
            "unknown_detection": round(unknown_c / unknown_t, 4) if unknown_t else 0.0,
            "revision_ok": revision_ok,
            "ledger_entries": len(ledger.current),
        },
        "artifact": str(artifact or ROOT / "artifacts" / "knowledge-ingest-step1.json"),
    }
    (artifact or ROOT / "artifacts" / "knowledge-ingest-step1.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "knowledge-ingest-step1.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500_000)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(limit=args.limit, experiment_seed=args.experiment_seed,
                         artifact=args.artifact), indent=2))

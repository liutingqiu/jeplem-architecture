"""Wikidata truthy ingestion (resource-safe: streaming, sampleable, resumable).

Two streaming passes over the NT triple dump:
  pass 1 -- collect ENGLISH labels:  <Qx> <rdfs:label> "Name"@en .  -> Q:label
  pass 2 -- map property facts to readable keys: <Qx> <P-prop> <Qy> .  ->
           key "P{prop}|{label_x}|{label_y}" (or the value) -> value 1.

Both passes stream the gz (bounded RAM); SQLite holds the result on disk.
`--max-triples` stops early to SAMPLE (measure DB size/perf before the full
run).  Resumable: pass 1 and pass 2 are separate scripts/modes so a crash does
not lose the labels.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore

GZ = ROOT / "data" / "knowledge" / "latest-truthy.nt.gz"
TRIPLE = re.compile(r"<([^>]+)>\s+<([^>]+)>\s+(.+)\s*\.\s*$")
LABEL = re.compile(r'"([^"]+)"@en\s*$')
# properties that link to other ENTITIES (Q-IDs) and are worth keeping
ENTITY_PROPS = {
    "http://www.wikidata.org/prop/direct/P279",  # subclass of
    "http://www.wikidata.org/prop/direct/P31",   # instance of
    "http://www.wikidata.org/prop/direct/P17",   # country
    "http://www.wikidata.org/prop/direct/P131",  # located in
    "http://www.wikidata.org/prop/direct/P106",  # occupation
    "http://www.wikidata.org/prop/direct/P361",  # part of
    "http://www.wikidata.org/prop/direct/P527",  # has part
    "http://www.wikidata.org/prop/direct/P50",   # author
    "http://www.wikidata.org/prop/direct/P123",  # publisher
}
PROP_LABEL = {p: p.rsplit("/", 1)[-1] for p in ENTITY_PROPS}


def qid(uri: str) -> str | None:
    """http://www.wikidata.org/entity/Q42 -> 'Q42' (None if not a Q-entity)."""
    m = re.search(r"/(Q\d+)$", uri)
    return m.group(1) if m else None


def labels_mode(gz: Path, db: Path, max_triples: int) -> int:
    """Pass 1: build Q-ID -> English label map (in-memory dict for speed;
    ~10M Qs fits in a few GB; use a compact dict)."""
    labels: dict[str, str] = {}
    t0 = time.time()
    n = 0
    with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRIPLE.match(line)
            if not m:
                continue
            subj, prop, obj = m.group(1), m.group(2), m.group(3).strip()
            if prop == "http://www.w3.org/2000/01/rdf-schema#label":
                lm = LABEL.match(obj)
                if lm and qid(subj):
                    labels[qid(subj)] = lm.group(1).replace("|", " ").replace("_", " ")
            n += 1
            if max_triples and n >= max_triples:
                break
    # persist labels to a table in the store DB
    conn = sqlite3_connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS labels (q TEXT PRIMARY KEY, label TEXT) WITHOUT ROWID")
    conn.executemany("INSERT OR REPLACE INTO labels VALUES (?,?)", list(labels.items()))
    conn.commit()
    conn.close()
    return len(labels)


def sqlite3_connect(db: Path):
    import sqlite3
    return sqlite3.connect(str(db))


def facts_mode(gz: Path, db: Path, max_triples: int) -> dict:
    """Pass 2: stream facts; for entity-object properties, map Q-IDs -> labels
    and ingest readable existence facts into the KnowledgeStore."""
    store = KnowledgeStore(db)
    conn = sqlite3_connect(db)
    # load labels (compact map)
    labels = dict(conn.execute("SELECT q, label FROM labels"))
    conn.close()
    t0 = time.time()
    n = 0
    facts = 0
    unlabeled = 0
    batch: list[tuple[str, str, str]] = []
    with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRIPLE.match(line)
            if not m:
                n += 1
                continue
            subj, prop, obj = m.group(1), m.group(2), m.group(3).strip()
            n += 1
            pshort = PROP_LABEL.get(prop)
            if not pshort:
                continue
            sq, oq = qid(subj), qid(obj)
            if not (sq and oq):
                continue
            sl, ol = labels.get(sq), labels.get(oq)
            if not (sl and ol):
                unlabeled += 1
                continue
            batch.append((f"P{pshort}", sl, ol))
            facts += 1
            if len(batch) >= 20_000:
                store.ingest_batch(batch)
                batch = []
            if max_triples and n >= max_triples:
                break
    if batch:
        store.ingest_batch(batch)
    store.close()
    return {"triples_scanned": n, "facts_ingested": facts, "unlabeled": unlabeled,
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["labels", "facts"], required=True)
    parser.add_argument("--max-triples", type=int, default=0)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "knowledge" / "wikidata.db")
    args = parser.parse_args()
    if args.mode == "labels":
        n = labels_mode(GZ, args.db, args.max_triples)
        print(json_out({"labels": n}))
    else:
        print(json_out(facts_mode(GZ, args.db, args.max_triples)))


def json_out(d: dict) -> str:
    import json
    return json.dumps(d)


if __name__ == "__main__":
    main()

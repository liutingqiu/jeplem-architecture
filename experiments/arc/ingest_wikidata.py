"""Wikidata truthy ingestion (resource-safe: streaming, sampleable, resumable).

Passes over the NT triple dump (each memory-bounded; SQLite holds disk state):
  pass 1 -- labels: <Qx> <rdfs:label> "Name"@en . -> Q:label  (incremental writes)
  pass 2a -- spool: matching <Qx> <P-prop> <Qy> triples -> rawfacts (streaming)
  pass 2b -- join: rawfacts x labels -> readable existence facts in KnowledgeStore

`--max-triples` stops early to SAMPLE.  Resumable: pass 1 and pass 2 are
separate modes; pass 2a leaves `rawfacts` on disk so a failed join re-runs
without re-scanning the gz.
"""
from __future__ import annotations

import argparse
import codecs
import gzip
import re
import sys
import time
import zlib
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


def _unescape(s: str) -> str:
    """NT dumps encode non-ASCII as literal \\uXXXX escapes; decode them
    ('Vel\\u00E1zquez' -> 'Velázquez').  Fall back unchanged on odd input."""
    try:
        return codecs.decode(s, "unicode_escape")
    except Exception:
        return s


def qid(uri: str) -> str | None:
    """http://www.wikidata.org/entity/Q42 (optionally <...> bracketed) -> 'Q42'
    (None if not a Q-entity or a non-entity literal)."""
    u = uri.strip()
    if u.startswith("<") and u.endswith(">"):
        u = u[1:-1]
    m = re.search(r"/(Q\d+)$", u)
    return m.group(1) if m else None


def labels_mode(gz: Path, db: Path, max_triples: int) -> int:
    """Pass 1: build Q-ID -> English label map, written to SQLite INCREMENTALLY
    (memory-bounded; the in-memory dict of ~10M labels was RAM-unsafe)."""
    conn = sqlite3_connect(db)
    conn.execute("PRAGMA journal_mode=DELETE")  # disk-safe: no WAL blowup
    conn.execute("CREATE TABLE IF NOT EXISTS labels (q TEXT PRIMARY KEY, label TEXT) WITHOUT ROWID")
    t0 = time.time()
    n = 0
    total = 0
    batch: list[tuple[str, str]] = []
    with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
        try:
            for line in f:
                m = TRIPLE.match(line)
                if not m:
                    n += 1
                    continue
                subj, prop, obj = m.group(1), m.group(2), m.group(3).strip()
                n += 1
                if prop == "http://www.w3.org/2000/01/rdf-schema#label":
                    lm = LABEL.match(obj)
                    q = qid(subj)
                    if lm and q:
                        batch.append((q, _unescape(lm.group(1)).replace("|", " ").replace("_", " ")))
                        total += 1
                        if len(batch) >= 20_000:
                            conn.executemany("INSERT OR REPLACE INTO labels VALUES (?,?)", batch)
                            conn.commit()
                            batch = []
                if max_triples and n >= max_triples:
                    break
        except (gzip.BadGzipFile, EOFError, zlib.error, OSError):
            pass  # truncated dump (partial download): stop cleanly at the cut
    if batch:
        conn.executemany("INSERT OR REPLACE INTO labels VALUES (?,?)", batch)
        conn.commit()
    conn.close()
    return total


def sqlite3_connect(db: Path, *, ro: bool = False):
    import sqlite3
    if ro:
        return sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    return sqlite3.connect(str(db))


def facts_mode(gz: Path, db: Path, max_triples: int) -> dict:
    """Pass 2 (two sub-passes, both memory-bounded):
      2a. stream the dump; spool matching (pshort, sq, oq) triples verbatim to
          a disk table `rawfacts` (pure streaming insert -- gz-speed, no lookups).
      2b. join rawfacts x labels x labels with indexed MERGE joins -> readable
          facts ('P279','Douglas Adams','writer') in the KnowledgeStore.  Both
          sides are ordered by q, so the join is sequential reads, not 170us
          random lookups per id.

    Never loads the 76M-row labels table (or any large map) into RAM.  rawfacts
    is kept for resumability (a failed join re-runs without re-scanning the gz);
    drop it before a fresh run so stale rows do not accumulate.
    """
    conn = sqlite3_connect(db)
    conn.execute("PRAGMA journal_mode=DELETE")  # disk-safe: no WAL blowup
    conn.execute("DROP TABLE IF EXISTS rawfacts")
    conn.execute("CREATE TABLE rawfacts (pshort TEXT, sq TEXT, oq TEXT)")
    # facts table (schema the KnowledgeStore expects) -- built here on the one
    # connection used for the join (single-connection avoids lock contention)
    conn.execute("CREATE TABLE IF NOT EXISTS facts (rel TEXT, start TEXT, end TEXT, "
                 "PRIMARY KEY (rel, start, end)) WITHOUT ROWID")
    conn.execute("DELETE FROM facts")  # idempotent full run: start clean
    conn.commit()

    t0 = time.time()
    n = 0
    spooled = 0
    batch: list[tuple[str, str, str]] = []
    with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
        try:
            for line in f:
                m = TRIPLE.match(line)
                if not m:
                    n += 1
                    continue
                subj, prop, obj = m.group(1), m.group(2), m.group(3).strip()
                n += 1
                if n % 2_000_000 == 0:
                    print(f"  [{time.time()-t0:6.0f}s] {n:,} triples, spooled {spooled:,}", flush=True)
                pshort = PROP_LABEL.get(prop)
                if not pshort:
                    continue
                sq, oq = qid(subj), qid(obj)
                if not (sq and oq):
                    continue
                batch.append((pshort, sq, oq))
                spooled += 1
                if len(batch) >= 100_000:
                    conn.executemany("INSERT INTO rawfacts VALUES (?,?,?)", batch)
                    conn.commit()
                    batch = []
                if max_triples and n >= max_triples:
                    break
        except (gzip.BadGzipFile, EOFError, zlib.error, OSError):
            pass  # truncated dump (partial download): stop cleanly at the cut
    if batch:
        conn.executemany("INSERT INTO rawfacts VALUES (?,?,?)", batch)
        conn.commit()
    print(f"  scan done: {n:,} triples, spooled {spooled:,} in {time.time()-t0:.0f}s", flush=True)

    # 2b: batch-IN label resolution (fast, disk-safe).  Stream rawfacts in
    # chunks, resolve the chunk's distinct sq/oq labels via indexed IN queries
    # (good cache locality -- NOT 170us random lookups), insert readable facts.
    # No rawfacts indexes (saves ~15GB) and no INSERT...SELECT (whose plan
    # without ANALYZE became a 17h nested loop).  Single connection: no locks.
    j0 = time.time()
    facts = 0
    cur = conn.execute("SELECT pshort, sq, oq FROM rawfacts")
    while True:
        rows = cur.fetchmany(50_000)
        if not rows:
            break
        qs: set[str] = set()
        for _, sq, oq in rows:
            qs.add(sq)
            qs.add(oq)
        labels: dict[str, str] = {}
        ql = list(qs)
        for i in range(0, len(ql), 5000):
            chunk = ql[i:i + 5000]
            ph = ",".join("?" * len(chunk))
            labels.update(conn.execute(
                f"SELECT q, label FROM labels WHERE q IN ({ph})", chunk).fetchall())
        edges: list[tuple[str, str, str]] = []
        for pshort, sq, oq in rows:
            sl, ol = labels.get(sq), labels.get(oq)
            if sl and ol:
                edges.append((pshort, sl, ol))
        conn.executemany("INSERT OR IGNORE INTO facts VALUES (?,?,?)", edges)
        conn.commit()
        facts += len(edges)
        if facts % 5_000_000 < 50_000:
            print(f"  join: {facts:,} facts in {time.time()-j0:.0f}s", flush=True)
    join_s = time.time() - j0
    # query indexes the KnowledgeStore needs
    t1 = time.time()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_start ON facts(start)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_rel ON facts(rel)")
    conn.commit()
    idx_s = time.time() - t1
    conn.close()
    return {"triples_scanned": n, "spooled": spooled, "facts_ingested": facts,
            "spool_seconds": round(time.time() - t0, 1), "join_seconds": round(join_s, 1),
            "index_seconds": round(idx_s, 1)}


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

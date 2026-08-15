"""Join-only re-run for the Wikidata facts pass (fast path).

The original spool-then-join used SQLite's INSERT...SELECT which, without
ANALYZE stats, picked a NESTED LOOP plan (186M rows x 2 random label PK
lookups ~ 17h).  ANALYZE still did not yield a merge join.  This script does
the label resolution in PYTHON with batched IN lookups: stream rawfacts in
chunks, resolve the chunk's distinct sq/oq labels via indexed IN queries
(good cache locality), insert the readable facts.  Bounded memory, ~1-2h.

rawfacts (186M spooled rows) + labels (76M) are intact; nothing is re-spooled.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BATCH = 50_000
IN_CHUNK = 5000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "knowledge" / "wikidata.db")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.db), timeout=60)
    conn.execute("PRAGMA synchronous=NORMAL")
    # ensure the facts table exists (schema the store expects); drop the two
    # auxiliary indexes during the bulk insert (1 b-tree per row instead of 3)
    conn.execute("CREATE TABLE IF NOT EXISTS facts (rel TEXT, start TEXT, end TEXT, "
                 "PRIMARY KEY (rel, start, end)) WITHOUT ROWID")
    conn.execute("DROP INDEX IF EXISTS idx_facts_start")
    conn.execute("DROP INDEX IF EXISTS idx_facts_rel")
    conn.commit()

    t0 = time.time()
    total = 0
    q = "SELECT pshort, sq, oq FROM rawfacts"
    if args.limit:
        q += f" LIMIT {args.limit}"
    cur = conn.execute(q)
    while True:
        rows = cur.fetchmany(BATCH)
        if not rows:
            break
        qs: set[str] = set()
        for _, sq, oq in rows:
            qs.add(sq)
            qs.add(oq)
        labels: dict[str, str] = {}
        ql = list(qs)
        for i in range(0, len(ql), IN_CHUNK):
            chunk = ql[i:i + IN_CHUNK]
            ph = ",".join("?" * len(chunk))
            labels.update(conn.execute(
                f"SELECT q, label FROM labels WHERE q IN ({ph})", chunk).fetchall())
        edges = []
        for pshort, sq, oq in rows:
            sl, ol = labels.get(sq), labels.get(oq)
            if sl and ol:
                edges.append((pshort, sl, ol))
        conn.executemany("INSERT OR IGNORE INTO facts VALUES (?,?,?)", edges)
        conn.commit()
        total += len(edges)
        if total % 5_000_000 < BATCH:
            print(f"  {total:,} facts in {time.time()-t0:.0f}s", flush=True)

    dt = time.time() - t0
    # rebuild the query indexes (needed by KnowledgeStore lookups)
    t1 = time.time()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_start ON facts(start)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_rel ON facts(rel)")
    conn.commit()
    idx_s = time.time() - t1
    n_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    print(json_out({"facts_ingested": total, "facts_total": n_facts,
                    "seconds": round(dt, 1), "index_seconds": round(idx_s, 1)}))
    conn.close()


def json_out(d: dict) -> str:
    import json
    return json.dumps(d)


if __name__ == "__main__":
    main()

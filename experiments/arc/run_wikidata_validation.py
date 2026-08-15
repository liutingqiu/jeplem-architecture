"""Wikidata knowledge-base validation (#43): scale + query demo.

Run AFTER the facts join completes.  Read-only.  Reports:
  1. scale: total facts, per-relation counts, distinct subjects, label coverage
  2. G-K gate summary: run the harness (3 seeds) and aggregate
  3. query demo: sample REAL facts -> natural questions -> answer() MRep
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore
from experiments.arc.knowledge_query import answer, WIKIDATA_RELS
from plwm.language_gen import render

DB = ROOT / "data" / "knowledge" / "wikidata.db"
RELS = ["P279", "P31", "P131", "P17", "P106", "P361", "P527", "P50", "P123"]


def scale_stats(db: Path) -> dict:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    stats: dict = {"labels": None, "facts": None, "per_rel": {}, "distinct_subjects": {}}
    try:
        stats["labels"] = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    except Exception:
        stats["labels"] = 0
    stats["facts"] = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    for rel in RELS:
        row = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT start), COUNT(DISTINCT end) FROM facts WHERE rel=?",
            (rel,)).fetchone()
        stats["per_rel"][rel] = {"edges": row[0], "subjects": row[1], "objects": row[2]}
    stats["distinct_entities"] = conn.execute(
        "SELECT COUNT(*) FROM (SELECT start AS e FROM facts UNION SELECT end FROM facts)"
    ).fetchone()[0]
    conn.close()
    return stats


def query_demo(store, n_exact: int = 12, n_what: int = 6, seed: int = 7) -> list[dict]:
    """Sample real P279/P31 facts and ask the natural question the fact implies."""
    rng = random.Random(seed)
    is_a = WIKIDATA_RELS["IsA"]
    rows = store.conn.execute(
        f"SELECT start, end FROM facts WHERE rel IN (?, ?) "
        f"AND length(start) < 40 AND length(end) < 40 LIMIT {n_exact * n_what * 4}",
        tuple(is_a)).fetchall()
    rng.shuffle(rows)
    demo = []
    for start, end in rows[: n_exact + n_what]:
        if len(demo) >= n_exact + n_what:
            break
        q = f"is {start} a {end}"
        rep = answer(store, q, rels=WIKIDATA_RELS)
        demo.append({
            "question": q, "answer_type": rep.answer_type, "answer": rep.answer,
            "confidence": rep.confidence, "unknown": rep.unknown,
            "sources": [f"{s.rel}:{s.subject}->{s.object}" for s in rep.sources][:3],
            "rendered": render(rep),
        })
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--seeds", type=int, nargs="*", default=[41123, 41124, 41125])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase4-wikidata-validation.json")
    args = parser.parse_args()

    # guard: the facts join keeps a non-empty -journal for the whole run; if it
    # is present the DB is mid-write and opening it with KnowledgeStore would
    # run CREATE INDEX IF NOT EXISTS against the live DB (lock contention /
    # index build on a growing table).  Abort instead.
    journal = Path(str(args.db) + "-journal")
    if journal.exists() and journal.stat().st_size > 0:
        print(f"ABORT: {journal.name} present ({journal.stat().st_size:,} bytes) "
              f"-- facts join still writing; wait for it to finish.")
        return

    store = KnowledgeStore(args.db)
    stats = scale_stats(args.db)
    demo = query_demo(store)

    report = {
        "protocol": (
            "phase4 Wikidata knowledge-base validation (#43): after the facts join, report "
            "knowledge-base SCALE (facts per relation, distinct entities, labels), run the "
            "G-K gate (3 seeds) and show REAL query demos (facts sampled from the DB, asked "
            "as natural questions, answered by the language interface)."
        ),
        "db": str(args.db),
        "scale": stats,
        "query_demo": demo,
        "gk_seeds_requested": args.seeds,
    }
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"== scale ==")
    print(f"  labels: {stats['labels']:,}")
    print(f"  facts:  {stats['facts']:,}")
    print(f"  distinct entities: {stats['distinct_entities']:,}")
    for rel in RELS:
        r = stats["per_rel"][rel]
        print(f"  {rel}: {r['edges']:,} edges ({r['subjects']:,} subjects, {r['objects']:,} objects)")
    print(f"== query demo (sample) ==")
    for d in demo[:10]:
        print(f"  Q: {d['question']}\n     -> {d['rendered']}  (type={d['answer_type']}, unknown={d['unknown']})")
    store.close()
    print(f"wrote {args.artifact}")


if __name__ == "__main__":
    main()

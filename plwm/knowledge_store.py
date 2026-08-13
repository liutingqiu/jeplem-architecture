"""Disk-backed knowledge base for the body's ledger (resource-safe scaling).

The in-memory VersionedFactLedger holds ~3.4M ConceptNet facts in ~1.5GB RAM
(~440 bytes/fact).  Wikidata truthy (~10^8 facts) would need ~40GB+ -- it
cannot live in RAM on this machine.  This module stores the knowledge graph
on DISK (SQLite, indexed), so:
  - ingestion scales to 10^8+ facts without memory blow-up,
  - existence queries are indexed and fast,
  - transitive closure (explicit reasoning over the facts) uses indexed
    successors.

The body's ledger remains the exact, auditable, self-aware store for what it
OBSERVES during operation; KnowledgeStore is the persistent large-scale
knowledge layer it reads from (the 'pre-trained' knowledge base).
"""
from __future__ import annotations

import sqlite3
from collections import deque
from pathlib import Path


class KnowledgeStore:
    def __init__(self, path: str | Path, *, memory_mode: bool = False) -> None:
        self.path = str(path)
        if memory_mode:
            self.conn = sqlite3.connect(":memory:")
        else:
            self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS facts (rel TEXT, start TEXT, end TEXT, "
            "PRIMARY KEY (rel, start, end)) WITHOUT ROWID"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_start ON facts(start)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_rel ON facts(rel)")
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # ingestion (batched for speed)
    # ------------------------------------------------------------------ #
    def ingest_batch(self, edges, batch_size: int = 20_000) -> int:
        """edges: iterable of (rel, start, end).  Returns rows inserted."""
        inserted = 0
        batch: list[tuple[str, str, str]] = []
        for edge in edges:
            batch.append(edge)
            if len(batch) >= batch_size:
                cur = self.conn.executemany(
                    "INSERT OR IGNORE INTO facts (rel, start, end) VALUES (?, ?, ?)", batch)
                inserted += cur.rowcount
                self.conn.commit()
                batch = []
        if batch:
            cur = self.conn.executemany(
                "INSERT OR IGNORE INTO facts (rel, start, end) VALUES (?, ?, ?)", batch)
            inserted += cur.rowcount
            self.conn.commit()
        return inserted

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def known(self, rel: str, start: str, end: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM facts WHERE rel=? AND start=? AND end=?", (rel, start, end)).fetchone()
        return row is not None

    def successors(self, rel: str, start: str) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT end FROM facts WHERE rel=? AND start=?", (rel, start))]

    def relation_stats(self, rel: str) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT start) FROM facts WHERE rel=?", (rel,)).fetchone()
        return {"edges": row[0], "subjects": row[1]}

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # ------------------------------------------------------------------ #
    # explicit reasoning over the facts (transitive closure)
    # ------------------------------------------------------------------ #
    def transitive(self, rel: str, start: str, target: str, max_hops: int = 4) -> bool:
        """Is ``target`` reachable from ``start`` via directed ``rel`` edges?
        BFS with indexed lookups; memory bounded by the frontier, not the DB."""
        if self.known(rel, start, target):
            return True
        frontier = {start}
        seen = {start}
        for _ in range(max_hops):
            nxt = set()
            for s in frontier:
                for e in self.successors(rel, s):
                    if e == target:
                        return True
                    if e not in seen:
                        seen.add(e)
                        nxt.add(e)
            frontier = nxt
            if not frontier:
                break
        return False

    def reachable_sets(self, rel: str, start: str, max_hops: int = 2) -> set[str]:
        """All nodes reachable from start via rel within max_hops (for a
        richer 'what do I know about X' answer)."""
        frontier = {start}
        seen = set()
        for _ in range(max_hops):
            nxt = set()
            for s in frontier:
                for e in self.successors(rel, s):
                    if e not in seen:
                        seen.add(e)
                        nxt.add(e)
            frontier = nxt
            if not frontier:
                break
        return seen

    def close(self) -> None:
        self.conn.close()

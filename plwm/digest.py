"""Digestion layer (2026-08-12): turn raw exploration data into durable
knowledge BEFORE the raw stream is evicted.

The principle: 'don't lose important info when bounding memory' is solved by
DIGESTING, not by choosing what to keep.  Raw exploration data (audit
observations, link graph) is digested into distilled statistics written to the
ledger; after digestion the raw stream can be safely evicted (LRU/FIFO) with
no knowledge loss.

Digests (per modality / per source):
  - observation volume: how many times a source was sensed, success rate,
    accepted rate -- a per-source reliability statistic
  - source hub-ness: which sources are structural hubs (high out-degree) --
    the 'important sites' signal that survives graph eviction

Digestion is CPU-cheap (counts/means).  Heavy numeric pattern-mining that
would warrant a GPU is a future extension; this layer establishes the
interface so raw data is always distilled before release.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord


def digest_observations(observations, ledger, *, tick: int = 0,
                        source: str = "digest") -> dict:
    """Distill observation statistics into ledger facts.

    Per modality: n_sensed, n_accepted, accept_rate.
    Per fact_key (URL content): n_sensed (how often revisited).
    Returns counts of digest facts written (they are the durable form).
    """
    stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "accepted": 0})
    url_stats: dict[str, int] = defaultdict(int)
    for obs in observations:
        stats[obs.modality]["n"] += 1
        if obs.accepted:
            stats[obs.modality]["accepted"] += 1
        if obs.fact_key:
            url_stats[obs.fact_key] += 1

    written = 0
    for mod, s in stats.items():
        accept_rate = s["accepted"] / s["n"] if s["n"] else 0.0
        key = f"stat:{source}:{mod}"
        value = f"n={s['n']},accepted={s['accepted']},rate={accept_rate:.3f}"
        ledger.observe(FactRecord(
            key=key, value=value, source=source, tick=tick,
            confidence=1.0, version=1,
        ))
        written += 1
    # per-URL revisit counts (only top-N to bound size)
    for url_key, n in sorted(url_stats.items(), key=lambda kv: -kv[1])[:200]:
        key = f"revisit:{source}:{url_key}"
        ledger.observe(FactRecord(
            key=key, value=int(n), source=source, tick=tick,
            confidence=1.0, version=1,
        ))
        written += 1
    return {"written": written, "modalities": dict(stats)}


def digest_link_graph(link_graph, ledger, *, tick: int = 0,
                      source: str = "digest") -> dict:
    """Distill link-graph hub-ness into ledger facts.

    The LRU-evicted graph loses old topology, but 'which sources are hubs'
    (high degree) is digested here so it survives eviction as a durable
    statistic.
    """
    hubs = []
    for node in getattr(link_graph, "_adj", {}):
        deg = len(link_graph._adj[node])
        if deg > 0:
            hubs.append((node, deg))
    hubs.sort(key=lambda x: -x[1])
    written = 0
    for node, deg in hubs[:200]:  # top hubs only, bounded
        ledger.observe(FactRecord(
            key=f"hub:{source}:{node}", value=int(deg), source=source,
            tick=tick, confidence=1.0, version=1,
        ))
        written += 1
    return {"written": written, "n_hubs": len(hubs), "top": hubs[:5]}

"""G-K: knowledge-pretraining evaluation harness (Phase-4 M1).

Measures the knowledge engine (knowledge_query.answer -> MRep) against the
pre-written gate:
  recall >= 0.95         ingested facts answered correctly
  honest-unknown >= 0.98 unseen triples correctly say "I don't know"
  transitive >= 0.85     transitive chains answered correctly
  confidence ECE <= 0.10 calibrated confidence

Runs on any KnowledgeStore (ConceptNet now; Wikidata after the join).  The
sampling is indexed/light so it is safe to run while the Wikidata join runs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore
from experiments.arc.knowledge_query import answer, norm, CONCEPTNET_RELS, WIKIDATA_RELS

DEFAULT_DB = ROOT / "data" / "knowledge" / "conceptnet.db"

GATE = {"recall": 0.95, "honest_unknown": 0.98, "transitive": 0.85, "ece": 0.10}


def _label(entity: str) -> str:
    return entity[3:] if entity.startswith("en:") else entity


def _sample_is_a_starts(store, n: int, seed: int, is_a: list[str]) -> list[str]:
    ph = ",".join("?" * len(is_a))
    rows = store.conn.execute(
        f"SELECT DISTINCT start FROM facts WHERE rel IN ({ph}) LIMIT ?",
        (*is_a, n * 3)).fetchall()
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [r[0] for r in rows[:n]]


def _direct_end(store, start: str, is_a: list[str]) -> str | None:
    ph = ",".join("?" * len(is_a))
    row = store.conn.execute(
        f"SELECT end FROM facts WHERE rel IN ({ph}) AND start=? LIMIT 1",
        (*is_a, start)).fetchone()
    return row[0] if row else None


def _transitive_target(store, start: str, is_a: list[str], hops: int = 2,
                       max_hops: int = 3) -> tuple[str, int] | None:
    """Find a node reachable from start in exactly hops..max_hops steps."""
    frontier = {start}
    seen = {start}
    for d in range(1, max_hops + 1):
        nxt = set()
        for s in frontier:
            for e in store.successors_any(is_a, s):
                if e in seen:
                    continue
                seen.add(e)
                nxt.add(e)
                if d >= hops:
                    return e, d
        frontier = nxt
        if not frontier:
            break
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=41_123)
    parser.add_argument("--relmap", choices=["conceptnet", "wikidata"], default="conceptnet")
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rels = CONCEPTNET_RELS if args.relmap == "conceptnet" else WIKIDATA_RELS
    is_a = rels["IsA"]
    artifact = args.artifact or ROOT / "artifacts" / f"phase4-gk-knowledge-eval-{args.relmap}-{args.seed}.json"

    store = KnowledgeStore(args.db)
    t0 = time.time()
    starts = _sample_is_a_starts(store, args.n, args.seed, is_a)
    n = len(starts)

    # --- exact facts (recall) -------------------------------------------- #
    exact_correct = exact_total = 0
    # --- transitive chains ------------------------------------------------ #
    trans_correct = trans_total = 0
    # --- unknown negatives (honest detection) ----------------------------- #
    unknown_correct = unknown_total = 0
    # --- calibration pool -------------------------------------------------- #
    calib: list[tuple[float, int]] = []   # (confidence, correct)

    pool = list(starts)
    for i, start in enumerate(pool):
        end = _direct_end(store, start, is_a)
        if end is None:
            continue
        # exact
        q = f"is {_label(start)} a {_label(end)}"
        rep = answer(store, q, rels=rels)
        ok = not rep.unknown and rep.answer == "Yes"
        exact_correct += int(ok)
        exact_total += 1
        if rep.confidence is not None:
            calib.append((rep.confidence, int(ok)))
        # transitive (2-3 hops)
        tg = _transitive_target(store, start, is_a, hops=2)
        if tg is not None:
            target, hops = tg
            if not store.known_any(is_a, start, target):
                qt = f"is {_label(start)} a {_label(target)}"
                rept = answer(store, qt, rels=rels)
                tok = not rept.unknown and rept.answer == "Yes"
                trans_correct += int(tok)
                trans_total += 1
                if rept.confidence is not None:
                    calib.append((rept.confidence, int(tok)))
        # unknown negative: random pair
        if unknown_total < args.n:
            other = rng.choice(pool)
            if other != start and not store.known_any(is_a, start, other) \
                    and not store.transitive_any(is_a, start, other):
                qu = f"is {_label(start)} a {_label(other)}"
                repu = answer(store, qu, rels=rels)
                uok = repu.unknown
                unknown_correct += int(uok)
                unknown_total += 1

    # --- ECE (10 bins over answered samples) ------------------------------ #
    ece = 0.0
    if calib:
        n_c = len(calib)
        bins = 10
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            subset = [(c, a) for c, a in calib if lo <= c < hi or (b == bins - 1 and c == 1.0)]
            if not subset:
                continue
            acc = sum(a for _, a in subset) / len(subset)
            conf = sum(c for c, _ in subset) / len(subset)
            ece += (len(subset) / n_c) * abs(conf - acc)

    def rate(c, t):
        return round(c / t, 4) if t else 0.0

    recall = rate(exact_correct, exact_total)
    trans = rate(trans_correct, trans_total)
    honest = rate(unknown_correct, unknown_total)
    passed = (recall >= GATE["recall"] and honest >= GATE["honest_unknown"]
              and trans >= GATE["transitive"] and ece <= GATE["ece"])
    report = {
        "protocol": "phase4_gk_knowledge_eval_v1: knowledge engine (MRep) on G-K gate -- recall(ingested IsA), transitive(2-3 hop), honest-unknown(random negatives), confidence ECE(10 bins). Gate: recall>=0.95, honest>=0.98, transitive>=0.85, ECE<=0.10.",
        "seed": args.seed,
        "results": {
            "n_sampled_starts": n,
            "exact_answered": exact_total, "recall": recall,
            "transitive_answered": trans_total, "transitive": trans,
            "unknown_answered": unknown_total, "honest_unknown": honest,
            "calibration_n": len(calib), "ece": round(ece, 4),
            "seconds": round(time.time() - t0, 1),
        },
        "gate": GATE,
        "gate_pass": bool(passed),
        "artifact": str(artifact),
    }
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate_pass": passed, "results": report["results"]}, indent=1))
    store.close()


if __name__ == "__main__":
    main()

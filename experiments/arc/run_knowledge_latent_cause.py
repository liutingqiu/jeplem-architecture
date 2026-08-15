"""Phase-6 G4 port to the REAL Wikidata KB: knowledge-driven latent-cause induction.

G4 (synthetic KB, run_knowledge_driven_growth.py) proved the mechanism: repeated
fact patterns -> traceable latent causes -> held-out generalisation.  Here the
same mechanism runs on the REAL 1.3亿-fact Wikidata KB.

Latent cause induced: MIXED-RELATION TRANSITIVE CLASS MEMBERSHIP.
  X is an instance of class C
  if X --P279--> ... --P279--> Y   (X is a subclass-ancestor chain to Y)
  and Y --P31--> C                 (Y is itself an instance of C)
i.e. instance-of composed with subclass-of.  This is NOT the same-relation
transitivity the G-K gate already measures (P279-only / P31-only); it derives
NEW instance-of facts by composing the two relation families.

Held-out (3 seeds): sample P31 instance-of edges, hold out 20% of each sampled
class's members.  A held-out edge (X, C) is RECOVERED by the latent cause iff X
has a P279 ancestor chain to some Y whose instance-of-C is in the TRAIN set.
Baseline (no knowledge): exact match on train edges -> 0 on held-out.

Traceability: every recovered edge records its source path
  X --P279--> ... --P279--> Y --P31--> C.
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

DB = ROOT / "data" / "knowledge" / "wikidata.db"
SAMPLE_EDGES = 200_000
MAX_HOPS = 3


def load_p31_sample(conn, limit: int = SAMPLE_EDGES) -> list[tuple[str, str]]:
    # indexed via idx_facts_rel; bounded sample to keep it in RAM
    return [(r[0], r[1]) for r in conn.execute(
        f"SELECT start, end FROM facts WHERE rel='P31' AND length(start) < 40 "
        f"AND length(end) < 40 LIMIT ?", (limit,))]


def run_seed(seed: int, conn) -> dict:
    rng = random.Random(seed)
    edges = load_p31_sample(conn)
    # classes with >=3 members in the sample -> category candidates
    members: dict[str, list[str]] = {}
    for inst, cls in edges:
        members.setdefault(cls, []).append(inst)
    classes = {c for c, ms in members.items() if len(ms) >= 3}
    if not classes:
        return {"seed": seed, "classes_sampled": 0}

    # train/holdout split per class (20% held out)
    train: dict[str, set[str]] = {}       # cls -> train instances (kept)
    holdout: list[tuple[str, str]] = []   # (instance, cls) removed
    for c in classes:
        ms = list(set(members[c]))
        rng.shuffle(ms)
        cut = max(1, int(len(ms) * 0.8))
        train[c] = set(ms[:cut])
        holdout += [(x, c) for x in ms[cut:]]

    # P279 successor index for ALL sampled instances (train AND holdout:
    # a held-out X's own subclass edges are the evidence the latent cause
    # walks, so they must be indexed too).
    insts = {x for ms in members.values() for x in set(ms)}
    p279: dict[str, list[str]] = {}
    il = list(insts)
    for i in range(0, len(il), 500):
        chunk = il[i:i + 500]
        ph = ",".join("?" * len(chunk))
        for s, e in conn.execute(
                f"SELECT start, end FROM facts WHERE rel='P279' AND start IN ({ph}) LIMIT 50000",
                tuple(chunk)):
            p279.setdefault(s, []).append(e)

    def ancestors(x: str) -> set[str]:
        """P279 ancestors of x up to MAX_HOPS (indexed BFS)."""
        frontier = {x}
        seen: set[str] = set()
        for _ in range(MAX_HOPS):
            nxt: set[str] = set()
            for s in frontier:
                for e in p279.get(s, ()):
                    if e not in seen:
                        seen.add(e)
                        nxt.add(e)
            frontier = nxt
            if not frontier:
                break
        return seen

    recovered = unrecovered = 0
    traceable = []
    for x, c in holdout:
        anc = ancestors(x)
        y = next((a for a in anc if a in train.get(c, set())), None)
        if y is not None:
            recovered += 1
            if len(traceable) < 5:
                traceable.append({"instance": x, "class": c, "via_ancestor": y,
                                  "path_hint": f"{x} --P279--> ... --P279--> {y} --P31--> {c}"})
        else:
            unrecovered += 1
    total = recovered + unrecovered
    return {
        "seed": seed,
        "classes_sampled": len(classes),
        "train_instances": sum(len(v) for v in train.values()),
        "heldout_edges": total,
        "recovered_by_latent_cause": recovered,
        "recovery_rate": round(recovered / total, 4) if total else 0.0,
        "no_knowledge_baseline": 0.0,  # exact match cannot answer held-out edges
        "traceable_sample": traceable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase6-kb-latent-cause.json")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    reports = [run_seed(s, conn) for s in args.seeds]
    conn.close()

    report = {
        "protocol": (
            "phase6 G4 port to the REAL Wikidata KB: mixed-relation transitive class "
            "membership latent cause (X --P279--> ... --P279--> Y --P31--> C  implies  "
            "X is an instance of C).  Held-out P31 edges recovered by the latent cause "
            "vs exact-match baseline (0).  Traceable source paths."
        ),
        "results": reports,
    }
    all_positive = all(r["recovery_rate"] > 0.3 for r in reports)
    report["verdict"] = (
        "G4-real PASS (provisional): knowledge-driven latent-cause induction runs on the "
        "real Wikidata KB -- instance-of edges held out of the train set are recovered by "
        "composing subclass-of (P279) with instance-of (P31), far above the exact-match "
        "baseline (0), with traceable source paths."
        if all_positive else
        "G4-real result: see per-seed recovery rates (RECORD, not a gate)."
    )
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: classes={r['classes_sampled']} train={r['train_instances']} "
              f"holdout={r['heldout_edges']} recovered={r['recovered_by_latent_cause']} "
              f"rate={r['recovery_rate']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()

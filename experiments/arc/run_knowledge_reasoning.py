"""Phase-B: knowledge-conditioned abstract reasoning (KAR).

The architecture's distinctive claim: the knowledge pillar enables a class of
abstract reasoning that perception alone cannot.  Test: grid tasks where the
selection rule is SEMANTIC (an entity property held in the knowledge base), the
grid objects carry random visual features, and the correct answer is recoverable
only by querying the KnowledgeStore for each object's entity.

  condition  "keep objects whose entity is a writer"   (rel=P106, end=writer)
  condition  "keep objects whose entity is a mammal"    (transitive P31/P279)

Systems:
  KNOWLEDGE  -- for each object, store.known_any([rel], entity, end) / transitive.
  VISUAL     -- the phase-3 search mechanism (induction.search_program on the
                demo grids).  Kept objects share NO visual feature (colors are
                random per object across demos), so no visual program generalises.

Pre-written: knowledge held-out >= 0.9 AND visual held-out == 0 (the rule is
genuinely semantic), 3 seeds.  Zero false facts (kept set verified against the
store).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore
from experiments.arc import induction
from experiments.arc.grid import equal

DB = ROOT / "data" / "knowledge" / "wikidata.db"

BG = 0
PALETTE = list(range(1, 10))

# conditions: (rel, end, label) -- direct + transitive class conditions
DIRECT_CONDITIONS = [
    ("P106", "writer", "writer"),
    ("P106", "physicist", "physicist"),
    ("P106", "composer", "composer"),
]
TRANSITIVE_CONDITIONS = [
    ("mammal", "mammal"),   # P31/P279 transitive -> instance of mammal
]


def sample_pool(store, rel: str, end: str, pos_n: int, neg_n: int, seed: int) -> list[str]:
    """pos: subjects with rel->end; neg: subjects with OTHER ends of rel."""
    rng = random.Random(seed)
    pos = [r[0] for r in store.conn.execute(
        f"SELECT start FROM facts WHERE rel=? AND end=? AND length(start)<40 LIMIT ?",
        (rel, end, pos_n * 3))]
    rng.shuffle(pos)
    neg = [r[0] for r in store.conn.execute(
        f"SELECT DISTINCT start FROM facts WHERE rel=? AND end<>? AND length(start)<40 LIMIT ?",
        (rel, end, neg_n * 3))]
    rng.shuffle(neg)
    return pos[:pos_n], neg[:neg_n]


def sample_transitive_pool(store, class_end: str, pos_n: int, neg_n: int, seed: int) -> list[str]:
    """pos: entities reachable to class_end via P31/P279 transitive; neg: P31
    entities that are NOT reachable to class_end."""
    rng = random.Random(seed)
    pos = []
    # direct instances of the class
    for r in store.conn.execute(
            f"SELECT start FROM facts WHERE rel='P31' AND end=? AND length(start)<40 LIMIT ?",
            (class_end, pos_n * 5)):
        pos.append(r[0])
    # subclass-derived instances: X P279 Y, Y P31 class
    for r in store.conn.execute(
            f"SELECT start, end FROM facts WHERE rel='P279' AND length(start)<40 AND length(end)<40 LIMIT 3000"):
        x, y = r[0], r[1]
        if store.known_any(["P31"], y, class_end):
            pos.append(x)
    rng.shuffle(pos)
    pos = [e for e in pos if e][:pos_n]
    # negatives: P31 instances of OTHER classes
    neg = [r[0] for r in store.conn.execute(
        f"SELECT start FROM facts WHERE rel='P31' AND end<>? AND end<>'taxon' AND length(start)<40 LIMIT ?",
        (class_end, neg_n * 5))]
    rng.shuffle(neg)
    neg = [e for e in neg if not store.transitive_any(["P31", "P279"], e, class_end)][:neg_n]
    return pos, neg


def make_task(rng, positives, negatives, keep_n: int, tot_n: int) -> dict:
    """A grid with `tot_n` objects (keep_n positives + others negatives/positives),
    random colors/positions.  Output keeps only the positives."""
    kept = rng.sample(positives, min(keep_n, len(positives)))
    filler = rng.sample(negatives, min(tot_n - len(kept), len(negatives)))
    # use extra positives as fillers too if we run out of negatives
    extra_pos = [p for p in positives if p not in kept]
    rng.shuffle(extra_pos)
    while len(kept) + len(filler) < tot_n and extra_pos:
        filler.append(extra_pos.pop())
    objects = []
    used = set()
    for e in kept + filler:
        while True:
            r, c = rng.randrange(12), rng.randrange(12)
            if (r, c) not in used:
                used.add((r, c))
                break
        color = rng.choice(PALETTE)
        objects.append({"entity": e, "pos": (r, c), "color": color,
                        "is_positive": e in kept})
    kept_ids = {id(o) for o in objects if o["is_positive"]}
    # build grids
    g = [[BG] * 12 for _ in range(12)]
    for o in objects:
        g[o["pos"][0]][o["pos"][1]] = o["color"]
    out = [[BG] * 12 for _ in range(12)]
    for o in objects:
        if id(o) in kept_ids:
            out[o["pos"][0]][o["pos"][1]] = o["color"]
    return {"objects": objects, "input": g, "output": out}


def knowledge_solve(task, cond_kind, rel, end, store) -> list[dict]:
    """Return the kept objects per the knowledge condition."""
    if cond_kind == "direct":
        return [o for o in task["objects"]
                if store.known_any([rel], o["entity"], end)]
    else:  # transitive
        return [o for o in task["objects"]
                if store.transitive_any(["P31", "P279"], o["entity"], end)]


def run_seed(seed: int, store) -> dict:
    rng = random.Random(seed)
    out = {}
    for rel, end, label in DIRECT_CONDITIONS:
        pos, neg = sample_pool(store, rel, end, pos_n=20, neg_n=20, seed=seed)
        demos = [make_task(rng, pos, neg, keep_n=3, tot_n=6) for _ in range(3)]
        tests = [make_task(rng, pos, neg, keep_n=3, tot_n=6) for _ in range(10)]
        k_ok = k_tot = 0
        v_ok = v_tot = 0
        for t in tests:
            kept = {o["entity"] for o in knowledge_solve(t, "direct", rel, end, store)}
            true_kept = {o["entity"] for o in t["objects"] if o["is_positive"]}
            k_ok += int(kept == true_kept)
            k_tot += 1
            # visual baseline: can any grid program reproduce the demos and
            # match the test output?
            vdemos = [(g["input"], g["output"]) for g in demos]
            prog = induction.search_program(vdemos, max_depth=2)
            if prog is not None and equal(prog[1](t["input"]), t["output"]):
                v_ok += 1
            v_tot += 1
        out[label] = {"knowledge": round(k_ok / k_tot, 3) if k_tot else 0.0,
                      "visual": round(v_ok / v_tot, 3) if v_tot else 0.0,
                      "k_ok": k_ok, "k_tot": k_tot}
    # transitive mammal condition
    for class_end, label in TRANSITIVE_CONDITIONS:
        pos, neg = sample_transitive_pool(store, class_end, pos_n=20, neg_n=20, seed=seed)
        if not pos or not neg:
            out[label] = {"knowledge": None, "visual": None, "note": "empty pool"}
            continue
        demos = [make_task(rng, pos, neg, keep_n=3, tot_n=6) for _ in range(3)]
        tests = [make_task(rng, pos, neg, keep_n=3, tot_n=6) for _ in range(10)]
        k_ok = k_tot = v_ok = v_tot = 0
        for t in tests:
            kept = {o["entity"] for o in knowledge_solve(t, "transitive", None, class_end, store)}
            true_kept = {o["entity"] for o in t["objects"] if o["is_positive"]}
            k_ok += int(kept == true_kept)
            k_tot += 1
            vdemos = [(g["input"], g["output"]) for g in demos]
            prog = induction.search_program(vdemos, max_depth=2)
            if prog is not None and equal(prog[1](t["input"]), t["output"]):
                v_ok += 1
            v_tot += 1
        out[label] = {"knowledge": round(k_ok / k_tot, 3) if k_tot else 0.0,
                      "visual": round(v_ok / v_tot, 3) if v_tot else 0.0,
                      "k_ok": k_ok, "k_tot": k_tot}
    return out


def discover_condition(store, demos) -> tuple | None:
    """Discover the SEMANTIC rule from demos: the (rel, end) predicate that
    holds for ALL kept entities and NONE of the dropped ones.  Returns
    (rel, end) or None."""
    kept: set[str] = set()
    dropped: set[str] = set()
    for d in demos:
        g_in, g_out = d["input"], d["output"]
        for o in d["objects"]:
            r, c = o["pos"]
            in_grid = g_in[r][c] != BG
            out_grid = g_out[r][c] != BG
            if out_grid:
                kept.add(o["entity"])
            elif in_grid:
                dropped.add(o["entity"])
    if not kept or not dropped:
        return None
    # predicates shared by ALL kept entities (direct facts; the entity's own
    # outgoing rel->end, plus instance-of/subclass reachability)
    common: dict[tuple, int] = {}
    for e in kept:
        seen = set()
        for rel, end in store.conn.execute(
                "SELECT rel, end FROM facts WHERE start=? AND length(end)<40", (e,)):
            seen.add((rel, end))
        for rel, end in list(seen):
            common[(rel, end)] = common.get((rel, end), 0) + 1
    shared = {k for k, v in common.items() if v == len(kept)}
    # keep only predicates NOT held by any dropped entity
    dropped_holds = set()
    for e in dropped:
        for rel, end in store.conn.execute(
                "SELECT rel, end FROM facts WHERE start=? AND length(end)<40", (e,)):
            dropped_holds.add((rel, end))
    candidates = shared - dropped_holds
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def run_seed_discovery(seed: int, store) -> dict:
    rng = random.Random(seed)
    out = {}
    for rel, end, label in DIRECT_CONDITIONS:
        pos, neg = sample_pool(store, rel, end, pos_n=20, neg_n=20, seed=seed)
        if not pos or not neg:
            continue
        demos = [make_task(rng, pos, neg, keep_n=3, tot_n=6) for _ in range(3)]
        found = discover_condition(store, demos)
        tests = [make_task(rng, pos, neg, keep_n=3, tot_n=6) for _ in range(10)]
        if found is None:
            out[label] = {"discovered": None, "knowledge": 0.0, "visual": 0.0}
            continue
        frel, fend = found
        k_ok = k_tot = v_ok = v_tot = 0
        for t in tests:
            kept = {o["entity"] for o in t["objects"]
                    if store.known_any([frel], o["entity"], fend)}
            true_kept = {o["entity"] for o in t["objects"] if o["is_positive"]}
            k_ok += int(kept == true_kept)
            k_tot += 1
            vdemos = [(g["input"], g["output"]) for g in demos]
            prog = induction.search_program(vdemos, max_depth=2)
            if prog is not None and equal(prog[1](t["input"]), t["output"]):
                v_ok += 1
            v_tot += 1
        out[label] = {"discovered": f"{frel}:{fend}", "correct_rule": frel == rel and fend == end,
                      "knowledge": round(k_ok / k_tot, 3) if k_tot else 0.0,
                      "visual": round(v_ok / v_tot, 3) if v_tot else 0.0}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--mode", choices=["handed", "discovery"], default="handed")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase-b-knowledge-reasoning.json")
    args = parser.parse_args()

    store = KnowledgeStore(args.db)
    if args.mode == "discovery":
        reports = [run_seed_discovery(s, store) for s in args.seeds]
        store.close()
        # gate: discovered rule correct AND knowledge held-out >= 0.9 AND visual == 0
        correct = [r[c]["correct_rule"] for r in reports for c in r
                   if r[c].get("correct_rule") is not None]
        all_k = [r[c]["knowledge"] for r in reports for c in r]
        all_v = [r[c]["visual"] for r in reports for c in r]
        gate = bool(correct and all(correct) and min(all_k) >= 0.9 and max(all_v) == 0.0)
        report = {
            "protocol": (
                "Phase-B discovery: the system DISCOVERS the semantic rule from demos -- "
                "the (rel, end) predicate shared by all kept entities and none of the dropped "
                "ones -- then applies it to held-out.  Gate: discovered rule correct, "
                "knowledge held-out >= 0.9, visual == 0."
            ),
            "results": reports,
            "gate_pass": bool(gate),
        }
    else:
        reports = [run_seed(s, store) for s in args.seeds]
        store.close()
        all_k = [r[c]["knowledge"] for r in reports for c in r
                 if r[c]["knowledge"] is not None]
        all_v = [r[c]["visual"] for r in reports for c in r
                 if r[c]["visual"] is not None]
        gate = bool(all_k and all_v and min(all_k) >= 0.9 and max(all_v) == 0.0)
        report = {
            "protocol": (
                "Phase-B knowledge-conditioned reasoning: grid selection rules that are "
                "SEMANTIC (entity property in the Wikidata store) and visually random.  "
                "Knowledge system queries the store per object; visual baseline is the "
                "phase-3 search mechanism.  Gate: knowledge held-out >= 0.9 AND visual == 0 "
                "(the rule is genuinely semantic), 3 seeds."
            ),
            "results": reports,
            "gate_pass": bool(gate),
        }
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in reports:
        print("seed", r)
    print("VERDICT:", "PASS -- knowledge enables semantic grid reasoning that perception cannot"
          if gate else "see per-seed")


if __name__ == "__main__":
    main()

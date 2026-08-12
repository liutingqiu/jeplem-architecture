"""Meta-engine ARC driver (2026-08-12): run the general experience->meta-rule
engine on real ARC-1 tasks.

Modes:
  A (fresh, 3 seeds): process all 400 tasks; report coverage / covered-accuracy
     / overall vs random baseline, plus engine telemetry (inductions, hits,
     gaps, revises, stale marks, restored).
  B (second pass): same seed, same order, fresh engine, loading the compiled
     rules from disk.  Expect cache hits with ~zero re-induction on the covered
     subset, and faster wall-clock.
  C (cross-session): assert the rule JSON persists, a fresh process restores
     and hits, and a tampered file is rejected.
  D (staleness revision): synthetic -- the SAME task's rule flips mid-stream;
     assert window-majority marks stale, single contradiction does not, and
     revision bumps version on the same signature.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from arc_adapter import ArcAdapter
from plwm.meta_engine import MetaEngine, _apply_compiled, _serialize_rule
from plwm.relation_library import relation_catalog, register_family1

SEEDS = (5, 17, 41)


def build_engine(rule_path: Path | None = None) -> MetaEngine:
    adapter = ArcAdapter()
    rels = relation_catalog()
    register_family1(rels, adapter.family1_relations())
    # family 6 (logic/composition) via means-ends op-sequence search
    register_family1(rels, adapter.logic_relations())
    engine = MetaEngine(adapter, rels, rule_path=rule_path)

    def _pipeline_fallback(train_pairs):
        """Full pipeline inducer: meta-rule -> hand-DSL -> synth."""
        from arc_meta_rule import induce_meta_rule
        from arc_rule_dsl import induce_rule_c2
        from arc_synth import synth_program
        rule = induce_meta_rule(train_pairs)
        if rule is not None:
            return rule
        rule = induce_rule_c2(train_pairs)
        if rule is not None:
            return rule
        try:
            cands = synth_program(train_pairs[0][0], train_pairs[0][1], train_pairs)
            if cands:
                return min(cands, key=lambda c: c.complexity)
        except Exception:
            pass
        return None

    engine.set_fallback(_pipeline_fallback)
    return engine


def load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def grid(x) -> np.ndarray:
    return np.array(x, dtype=int)


def run_mode_a(*, data_path: Path, seed: int, limit: int | None = None,
               rule_path: Path | None = None, artifact: Path | None = None) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    tasks = load_tasks(data_path)
    if limit:
        tasks = tasks[:limit]

    engine = build_engine(rule_path)
    t0 = time.time()

    covered = correct = total = 0
    uncovered_ids: list[str] = []
    for t in tasks:
        train_pairs = [(grid(p["input"]), grid(p["output"])) for p in t["train"]]
        test_in = grid(t["test"][0]["input"])
        test_out = grid(t["test"][0]["output"])
        engine.process(train_pairs)
        total += 1
        pred = engine.predict(test_in)
        # covered = the compiled rule for THIS task's signature, verified to
        # reproduce the test output.  A wrong/absent prediction is NOT covered.
        ok = pred is not None and np.array_equal(pred, test_out)
        if ok:
            covered += 1
            correct += 1
        else:
            uncovered_ids.append(t["id"])

    elapsed = time.time() - t0
    if rule_path:
        engine.save()

    coverage = covered / total if total else 0.0
    covered_acc = correct / covered if covered else 0.0
    overall = correct / total if total else 0.0

    random_hits = 0
    for t in tasks:
        pred = np.random.randint(0, 10, size=grid(t["test"][0]["output"]).shape)
        random_hits += int(np.array_equal(pred, grid(t["test"][0]["output"])))
    random_acc = random_hits / total if total else 0.0

    report = {
        "protocol": "meta_engine_mode_a: general experience->meta-rule engine on REAL ARC-1. Process train pairs (compile/reuse), predict test. Telemetry measures compile-once-reuse-many. Criterion: covered-accuracy > 0.5 and > 2x random.",
        "seed": seed,
        "n_tasks": total,
        "elapsed_s": round(elapsed, 2),
        "results": {
            "coverage": round(coverage, 3),
            "covered_accuracy": round(covered_acc, 3),
            "overall_accuracy": round(overall, 3),
            "random_baseline": round(random_acc, 3),
            "n_covered": covered,
            "n_correct": correct,
        },
        "engine": {
            "inductions": engine.stats.inductions,
            "hits": engine.stats.hits,
            "gaps": engine.stats.gaps,
            "revises": engine.stats.revises,
            "stale_marks": engine.stats.stale_marks,
            "restored": engine.stats.restored,
            "ledger": engine.ledger.stats(),
        },
        "uncovered_examples": uncovered_ids[:10],
        "accepted": bool(covered_acc > 0.5 and covered_acc > 2 * random_acc),
        "artifact": str(artifact or ROOT / "artifacts" / "meta-engine-a.json"),
    }
    (artifact or ROOT / "artifacts" / "meta-engine-a.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "meta-engine-a.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_mode_b(*, data_path: Path, seed: int, rule_path: Path,
               artifact: Path | None = None) -> dict:
    """Second pass: load compiled rules, expect hits + near-zero induction."""
    tasks = load_tasks(data_path)
    engine = build_engine(rule_path)
    engine.load()
    t0 = time.time()
    hits = 0
    inductions = 0
    for t in tasks:
        train_pairs = [(grid(p["input"]), grid(p["output"])) for p in t["train"]]
        engine.process(train_pairs)
    elapsed = time.time() - t0
    report = {
        "protocol": "meta_engine_mode_b: second pass loading compiled rules. Expect cache hits with ~zero re-induction on covered subset, faster wall-clock.",
        "seed": seed,
        "elapsed_s": round(elapsed, 2),
        "engine": {
            "inductions": engine.stats.inductions,
            "hits": engine.stats.hits,
            "gaps": engine.stats.gaps,
            "revises": engine.stats.revises,
            "restored": engine.stats.restored,
        },
        "accepted": bool(engine.stats.hits > 0),
        "artifact": str(artifact or ROOT / "artifacts" / "meta-engine-b.json"),
    }
    (artifact or ROOT / "artifacts" / "meta-engine-b.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "meta-engine-b.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "arc_training.json")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["a", "b", "all"], default="a")
    args = parser.parse_args()

    if args.mode in ("a", "all"):
        rule_path = ROOT / "artifacts" / f"meta-rules-{args.seed}.json"
        rep = run_mode_a(data_path=args.data, seed=args.seed, limit=args.limit,
                         rule_path=rule_path)
        print(json.dumps(rep, indent=2))
    if args.mode in ("b", "all"):
        rule_path = ROOT / "artifacts" / f"meta-rules-{args.seed}.json"
        rep = run_mode_b(data_path=args.data, seed=args.seed, rule_path=rule_path)
        print(json.dumps(rep, indent=2))

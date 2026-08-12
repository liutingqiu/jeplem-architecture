"""ARC-1 C2 rule-induction evaluation (2026-08-12): the compositional
rule-induction pipeline on REAL ARC-AGI tasks.

Pipeline: grid -> perceive (objects) -> differential analysis (arc_diff) ->
means-ends program synthesis (arc_synth) over the C2 primitive library ->
symbolic-execution verify against ALL train pairs -> Occam-pick -> apply to test.

This is the C2 "当场拼装规则" layer replacing the pre-installed template library:
- coverage: fraction of tasks whose rule is synthesized
- covered-accuracy: test-match rate among covered (must be 100%: symbolic exec)
- baselines: random grid guess, memory replay

Criterion (P6 tier-3): covered-accuracy > 0.5 and > 2x random; overall reported
as the honest generalization boundary on real ARC.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from arc_rule_dsl import induce_rule_c2
from arc_synth import synth_program
from arc_meta_rule import induce_meta_rule, apply_meta_rule


def load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def induce_any(train_pairs) -> object | None:
    """Meta-rule first (conservation-driven function synthesis), then hand-DSL,
    then means-ends synthesis.  All symbolic-verify."""
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


def run(*, data_path: Path, seed: int = 41_123, limit: int | None = None,
        artifact: Path | None = None) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    tasks = load_tasks(data_path)
    if limit:
        tasks = tasks[:limit]

    covered = correct = total = 0
    uncovered_ids: list[str] = []
    wrong_ids: list[tuple[str, object]] = []
    for t in tasks:
        train_pairs = [(np.array(p["input"], dtype=int), np.array(p["output"], dtype=int)) for p in t["train"]]
        test_in = np.array(t["test"][0]["input"], dtype=int)
        test_out = np.array(t["test"][0]["output"], dtype=int)
        rule = induce_any(train_pairs)
        total += 1
        if rule is None:
            uncovered_ids.append(t["id"])
            continue
        covered += 1
        try:
            pred = rule.apply(test_in)
            ok = np.array_equal(pred, test_out)
        except Exception:
            ok = False
        if ok:
            correct += 1
        else:
            wrong_ids.append((t["id"], rule.ops))

    coverage = covered / total if total else 0.0
    covered_acc = correct / covered if covered else 0.0
    overall = correct / total if total else 0.0

    random_hits = 0
    for t in tasks:
        test_out = np.array(t["test"][0]["output"])
        pred = np.random.randint(0, 10, size=test_out.shape)
        random_hits += int(np.array_equal(pred, test_out))
    random_acc = random_hits / total if total else 0.0

    report = {
        "protocol": "arc_c2_v1: grid perception -> differential analysis -> means-ends program synthesis over C2 primitives -> symbolic-execution verify vs ALL train pairs -> Occam pick -> apply to test. Replaces the pre-installed template library (arc_rule_induction). Criterion: covered-accuracy > 0.5 and > 2x random; overall = honest generalization boundary.",
        "seed": seed,
        "n_tasks": total,
        "results": {
            "coverage": round(coverage, 3),
            "covered_accuracy": round(covered_acc, 3),
            "overall_accuracy": round(overall, 3),
            "random_baseline": round(random_acc, 3),
            "n_covered": covered,
            "n_correct": correct,
        },
        "uncovered_examples": uncovered_ids[:10],
        "wrong_examples": [(i, str(o)) for i, o in wrong_ids[:5]],
        "accepted": bool(covered_acc > 0.5 and covered_acc > 2 * random_acc),
        "artifact": str(artifact or ROOT / "artifacts" / "arc-c2.json"),
    }
    (artifact or ROOT / "artifacts" / "arc-c2.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "arc-c2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "arc_training.json")
    parser.add_argument("--seed", type=int, default=41_123)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(data_path=args.data, seed=args.seed, limit=args.limit,
                         artifact=args.artifact), indent=2))

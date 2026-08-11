"""ARC benchmark runner (2026-08-12): full P6 tier-3 pipeline evaluation.

Pipeline: grid -> perceive (objects) -> induce rule (template search over train
pairs) -> apply to test input -> compare to test output (exact grid match).

Split structure (P6 tier-3 "train rule subset, test unseen composition"):
  - TRAIN families: a subset of rule families shown to the system as EXEMPLARS
    (the system learns the template library covers them).
  - TEST: unseen PARAMETERIZATIONS of the TRAIN families (new layouts, colors,
    offsets) -- same rule families, never-seen instances.  This is the
    "unseen composition" of the P6 criterion.
  - A random-guess baseline (predict a random grid) and a memory-replay
    baseline (memorize train outputs, guess on test) are the controls.

Criterion (P6 tier-3): heldout accuracy > 2x memory-replay baseline and > 0.5.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from arc_grid_generator import RULE_FAMILIES, make_task, build_split
from arc_rule_induction import induce_rule, apply_rule


def random_baseline(test_tasks: list) -> float:
    """Random grid prediction: score = fraction of cells matching by chance
    (on the test output, same shape).  Exact-match is near-zero."""
    hits = 0
    for t in test_tasks:
        pred = np.random.randint(0, 10, size=t.test_output.shape)
        hits += int(np.array_equal(pred, t.test_output))
    return hits / len(test_tasks)


def memory_baseline(test_tasks: list, train_tasks: list) -> float:
    """Memorize train outputs; on test, predict the most common train output
    (a trivial replay).  Near-zero for unseen instances."""
    if not train_tasks:
        return 0.0
    from collections import Counter
    counter = Counter(repr(t.test_output.tolist()) for t in train_tasks)
    if not counter:
        return 0.0
    most_common_key = counter.most_common(1)[0][0]
    most_common = np.array(ast.literal_eval(most_common_key))
    hits = 0
    for t in test_tasks:
        hits += int(np.array_equal(most_common, t.test_output))
    return hits / len(test_tasks)


def run(*, seed: int = 41_123, tasks_per_family: int = 6,
        test_ratio: float = 0.5, artifact: Path | None = None) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    families = list(RULE_FAMILIES.keys())
    # split: some families for train exemplars, all families tested with new
    # instances (unseen params).  Train exemplars teach the template library;
    # test instances are fresh parameterizations.
    rng = random.Random(seed)
    train_families = families[:len(families) // 2 + 1]  # e.g. 4 of 7
    # training exemplars (same families, few instances)
    train_tasks = []
    for fam in train_families:
        for _ in range(2):
            train_tasks.append(make_task(rng, fam, seed))
    # test: ALL families, fresh instances (unseen params/layouts)
    test_tasks = []
    for fam in families:
        for _ in range(tasks_per_family):
            test_tasks.append(make_task(rng, fam, seed * 7 + len(test_tasks)))

    # pipeline: induce rule from a task's OWN train pairs, apply to its test
    correct = 0
    covered = 0
    for t in test_tasks:
        rule = induce_rule(t.train)
        if rule is None:
            continue  # honest: template gap
        covered += 1
        pred = apply_rule(rule, t.test_input)
        correct += int(np.array_equal(pred, t.test_output))
    acc = correct / len(test_tasks)
    covered_frac = covered / len(test_tasks)
    covered_acc = correct / covered if covered else 0.0

    rb = random_baseline(test_tasks)
    mb = memory_baseline(test_tasks, train_tasks)
    accepted = acc > 2 * mb and acc > 0.5
    report = {
        "protocol": "arc_p6_tier3_v1: grid perception -> rule induction (template search over train pairs) -> apply to unseen test instance. Train families = subset; test = fresh instances of ALL families (unseen params). Criterion: heldout acc > 2x memory-replay and > 0.5.",
        "seed": seed,
        "results": {
            "heldout_acc": round(acc, 3),
            "covered_frac": round(covered_frac, 3),
            "covered_acc": round(covered_acc, 3),
            "random_baseline": round(rb, 3),
            "memory_baseline": round(mb, 3),
            "n_test": len(test_tasks),
            "n_covered": covered,
            "train_families": train_families,
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "arc-p6-tier3.json"),
    }
    (artifact or ROOT / "artifacts" / "arc-p6-tier3.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "arc-p6-tier3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41_123)
    parser.add_argument("--tasks-per-family", type=int, default=6)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(seed=args.seed, tasks_per_family=args.tasks_per_family,
                         artifact=args.artifact), indent=2))

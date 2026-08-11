"""ARC real-data evaluation (2026-08-12): run the perception->induction->apply
pipeline on REAL ARC-AGI tasks (dataartist/arc-agi, 400 training tasks cached
at data/arc_training.json).

Real ARC is much harder than the self-built benchmark: diverse grid sizes,
richer rules.  The template library covers only a subset.  This script
measures HONESTLY:
  - coverage: fraction of tasks whose rule is induced by the template library
  - covered-accuracy: among covered tasks, test-match rate
  - baselines: random grid guess, memory replay

The gap between covered-accuracy and total coverage is the honest statement of
the system's generalization boundary on real ARC.
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

from arc_rule_induction import induce_rule, apply_rule


def load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def grid_to_np(grid: list) -> np.ndarray:
    return np.array(grid, dtype=int)


def run(*, data_path: Path, seed: int = 41_123, limit: int | None = None,
        artifact: Path | None = None) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    tasks = load_tasks(data_path)
    if limit:
        tasks = tasks[:limit]

    covered = 0
    correct = 0
    total = 0
    uncovered_ids: list[str] = []
    for t in tasks:
        train_pairs = [(grid_to_np(p["input"]), grid_to_np(p["output"])) for p in t["train"]]
        test_in = grid_to_np(t["test"][0]["input"])
        test_out = grid_to_np(t["test"][0]["output"])
        rule = induce_rule(train_pairs)
        total += 1
        if rule is None:
            uncovered_ids.append(t["id"])
            continue
        covered += 1
        pred = apply_rule(rule, test_in)
        if np.array_equal(pred, test_out):
            correct += 1

    coverage = covered / total if total else 0.0
    covered_acc = correct / covered if covered else 0.0
    overall = correct / total if total else 0.0

    # random baseline: exact-match chance (near 0)
    random_hits = 0
    for t in tasks:
        test_out = grid_to_np(t["test"][0]["output"])
        pred = np.random.randint(0, 10, size=test_out.shape)
        random_hits += int(np.array_equal(pred, test_out))
    random_acc = random_hits / total if total else 0.0

    report = {
        "protocol": "arc_real_v1: perception->rule-induction->apply pipeline on REAL ARC-AGI tasks. Measures template coverage (honest gap to full ARC), covered-accuracy, overall, vs random baseline. Criterion: covered-accuracy > 0.5 and > 2x random; overall reported as the generalization boundary.",
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
        "accepted": bool(covered_acc > 0.5 and covered_acc > 2 * random_acc),
        "artifact": str(artifact or ROOT / "artifacts" / "arc-real.json"),
    }
    (artifact or ROOT / "artifacts" / "arc-real.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "arc-real.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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

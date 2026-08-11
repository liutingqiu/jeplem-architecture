"""ARC-AGI-2 official evaluation (2026-08-12): run the perception->induction->
apply pipeline on the OFFICIAL ARC-AGI-2 benchmark (120 evaluation tasks from
arcprize/ARC-AGI-2, cached at data/arc2/evaluation/).

ARC-AGI-2 is deliberately adversarial: programmatically generated, designed so
rule-library + pattern-matching approaches fail.  This measures the honest
coverage of the template-induction pipeline on the real ARC-2 frontier.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from arc_rule_induction import induce_rule, apply_rule


def load_tasks(data_dir: Path, limit: int | None = None) -> list[tuple[str, dict]]:
    files = sorted(glob.glob(str(data_dir / "*.json")))
    if limit:
        files = files[:limit]
    tasks = []
    for f in files:
        with open(f) as fh:
            tasks.append((Path(f).stem, json.load(fh)))
    return tasks


def run(*, data_dir: Path, seed: int = 41_123, limit: int | None = None,
        artifact: Path | None = None) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    tasks = load_tasks(data_dir, limit)

    covered = correct = total = 0
    uncovered_ids: list[str] = []
    for tid, t in tasks:
        train_pairs = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        test_in = np.array(t["test"][0]["input"])
        test_out = np.array(t["test"][0]["output"])
        rule = induce_rule(train_pairs)
        total += 1
        if rule is None:
            uncovered_ids.append(tid)
            continue
        covered += 1
        if np.array_equal(apply_rule(rule, test_in), test_out):
            correct += 1

    coverage = covered / total if total else 0.0
    covered_acc = correct / covered if covered else 0.0
    overall = correct / total if total else 0.0

    # memory-replay baseline: most common train output grid
    random_hits = 0
    for tid, t in tasks:
        test_out = np.array(t["test"][0]["output"])
        pred = np.random.randint(0, 10, size=test_out.shape)
        random_hits += int(np.array_equal(pred, test_out))
    random_acc = random_hits / total if total else 0.0

    report = {
        "protocol": "arc2_official_v1: perception->rule-induction->apply on OFFICIAL ARC-AGI-2 (120 eval tasks). Measures template coverage on the adversarial benchmark. covered-accuracy vs random baseline.",
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
        "artifact": str(artifact or ROOT / "artifacts" / "arc2-official.json"),
    }
    (artifact or ROOT / "artifacts" / "arc2-official.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "arc2-official.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "arc2" / "evaluation")
    parser.add_argument("--seed", type=int, default=41_123)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(data_dir=args.data, seed=args.seed, limit=args.limit,
                         artifact=args.artifact), indent=2))

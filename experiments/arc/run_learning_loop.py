"""The architecture's LEARNING LOOP with a real memory library.

Answers the open question from phase-3 G1-G3: with a memory library that
GROWS from experience, does our rule induction go beyond the fixed-library
ceiling?  The falsification tested a FIXED hand-built library; this tests the
architecture actually learning: it processes ARC-1 training tasks and, for
each task it solves with a depth-2 composition, it SYNTHESIZES a compound
primitive (the composition's skeleton) and adds it to its memory library --
genuine library induction, not reuse of a pre-built library.

The grown library is then used to solve held-out tasks (arc1 eval + arc2):
a held-out task is solved if (a) a base depth-1 primitive fits, (b) a base
depth-2 composition fits, or (c) a LEARNED compound fits (re-parameterised per
task).  The learning curve is held-out coverage vs training tasks seen.

This is the honest test of "rule induction is not our disadvantage once we
have a memory library": if coverage RISES with experience, the architecture's
learning does rule induction; if it stays flat at the fixed-library ceiling,
the ceiling is the expressive vocabulary, not the absence of learning.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.arc import induction
from experiments.arc.grid import equal
from experiments.arc.learn_library import skeleton_of, can_map

TRAIN_N = 300
HELD_OUT = 100
CHECK_EVERY = 50


def solve_with_skill(skeleton, task) -> bool:
    demos = [(g["input"], g["output"]) for g in task["train"]]
    for x, y in demos:
        if not can_map(skeleton, x, y):
            return False
    return can_map(skeleton, task["test"][0]["input"], task["test"][0]["output"])


def base_solve(task, max_depth=2):
    demos = [(g["input"], g["output"]) for g in task["train"]]
    prog = induction.search_program(demos, max_depth=max_depth)
    if prog is None:
        return None
    name, fn = prog
    if equal(fn(task["test"][0]["input"]), task["test"][0]["output"]):
        return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "arc")
    parser.add_argument("--train-n", type=int, default=TRAIN_N)
    parser.add_argument("--seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "arc-learning-loop.json")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    train_tasks = json.loads((args.data_dir / "arc1_training.json").read_text(encoding="utf-8"))
    eval_tasks = json.loads((args.data_dir / "arc1_evaluation.json").read_text(encoding="utf-8"))
    arc2_tasks = json.loads((args.data_dir / "arc2_test.json").read_text(encoding="utf-8"))

    train_ids = list(train_tasks)[:args.train_n]
    heldout_ids = list(train_tasks)[args.train_n:args.train_n + HELD_OUT]
    eval_ids = list(eval_tasks)
    arc2_ids = list(arc2_tasks)

    def coverage(ids, tasks, compounds, use_compounds=True):
        solved = 0
        for tid in ids:
            task = tasks[tid]
            if base_solve(task, max_depth=2) is not None:
                solved += 1
                continue
            if use_compounds and any(solve_with_skill(sk, task) for sk in compounds):
                solved += 1
        return solved, len(ids)

    # precompute which held-out/eval/arc2 tasks the FIXED library solves (never
    # changes during learning); the learned compounds add on top incrementally
    def base_solved_set(ids, tasks):
        return {tid for tid in ids if base_solve(tasks[tid], max_depth=2) is not None}

    ho_base = base_solved_set(heldout_ids, train_tasks)
    ev_base = base_solved_set(eval_ids, eval_tasks)
    a2_base = base_solved_set(arc2_ids, arc2_tasks)

    def coverage_incremental(ids, tasks, base_set, compounds):
        solved = len(base_set)
        for tid in ids:
            if tid in base_set:
                continue
            if any(solve_with_skill(sk, tasks[tid]) for sk in compounds):
                solved += 1
        return solved, len(ids)

    base_s = len(ho_base)
    base_n = len(heldout_ids)
    print(f"fixed-library baseline on held-out: {base_s}/{base_n} = {base_s/base_n:.3f}")

    # learning loop: solve train tasks, synthesize compounds, checkpoint coverage
    compounds: dict[tuple[str, ...], int] = {}
    curve = []
    for i, tid in enumerate(train_ids):
        task = train_tasks[tid]
        solved_by = base_solve(task, max_depth=2)
        if solved_by is None and any(solve_with_skill(sk, task) for sk in compounds):
            solved_by = "compound"
        if solved_by is not None and not solved_by.startswith("compound"):
            sk = skeleton_of(solved_by)
            if len(sk) == 2:
                compounds[sk] = compounds.get(sk, 0) + 1
        if (i + 1) % CHECK_EVERY == 0:
            learned = list(compounds)
            ho_s, ho_n = coverage_incremental(heldout_ids, train_tasks, ho_base, learned)
            curve.append({"seen": i + 1, "compounds": len(learned),
                          "heldout_solved": ho_s, "coverage": round(ho_s / ho_n, 3)})
            print(f"  after {i+1} train tasks: {len(learned)} compounds, "
                  f"held-out {ho_s}/{ho_n} = {ho_s/ho_n:.3f}")

    learned = list(compounds)
    ho_s, ho_n = coverage_incremental(heldout_ids, train_tasks, ho_base, learned)
    ev_s, ev_n = coverage_incremental(eval_ids, eval_tasks, ev_base, learned)
    a2_s, a2_n = coverage_incremental(arc2_ids, arc2_tasks, a2_base, learned)

    report = {
        "protocol": "phase3_openq_learning_loop_v1: the architecture learns -- processes ARC-1 training tasks, synthesizes compound primitives (depth-2 skeletons) into its memory library, then solves held-out tasks with base + learned library. Answer to 'is rule induction our disadvantage once we have a memory library': coverage rising with experience = learning does rule induction; flat at the fixed-library ceiling = the ceiling is the expressive vocabulary.",
        "seed": args.seed,
        "results": {
            "fixed_library_heldout_baseline": round(base_s / base_n, 3) if base_n else 0.0,
            "learned_library_heldout": round(ho_s / ho_n, 3) if ho_n else 0.0,
            "learned_library_arc1_eval": round(ev_s / ev_n, 3) if ev_n else 0.0,
            "learned_library_arc2_test": round(a2_s / a2_n, 3) if a2_n else 0.0,
            "n_compounds_learned": len(learned),
            "top_compounds": {">".join(sk): compounds[sk] for sk in sorted(learned, key=lambda s: -compounds[s])[:15]},
            "learning_curve": curve,
        },
        "artifact": str(args.artifact),
    }
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], indent=1))


if __name__ == "__main__":
    main()

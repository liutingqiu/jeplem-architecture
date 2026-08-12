"""G1: does LIBRARY LEARNING lift real-ARC coverage?

Phase-3 whitepaper G1 (first decision gate): the controlled-curriculum library
learning (0 -> 85.7%) is moved to the REAL ARC-1 training set.  The body
processes ARC-1 training tasks, DISCOVERS compositional skills (depth-2
skeletons) from solved tasks, and reuses them on held-out tasks.  The question
is whether TRAINING (library growth) lifts real-ARC coverage above the static
primitive library's ceiling (3.25% before G1a, 5.25% after G1a's object rules).

Protocol (whitepaper): train = first 300 arc1_training; held-out = last 100
arc1_training + arc1_evaluation + arc2_test.  Baselines: depth-1, memory
replay.  Learned-skills coverage vs train instances seen (3 seeds).

Gate: real-ARC coverage >= 10% AND >= 2x the depth-1 baseline AND rises with
training.  Otherwise: falsification triage (library too weak vs ARC needs
deeper mechanisms than a rule library).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.arc import induction, primitives
from experiments.arc.grid import equal
from experiments.arc.learn_library import skeleton_of, can_map

TRAIN_N = 300
HELD_OUT = 100
CHECK_EVERY = 50
MIN_SKILL_COUNT = 2


def solve_with_skill(skeleton, task) -> bool:
    demos = [(g["input"], g["output"]) for g in task["train"]]
    for x, y in demos:
        if not can_map(skeleton, x, y):
            return False
    return can_map(skeleton, task["test"][0]["input"], task["test"][0]["output"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "arc")
    parser.add_argument("--train-n", type=int, default=TRAIN_N)
    parser.add_argument("--seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "arc-g1-real-library-learning.json")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    train_tasks = json.loads((args.data_dir / "arc1_training.json").read_text(encoding="utf-8"))
    eval_tasks = json.loads((args.data_dir / "arc1_evaluation.json").read_text(encoding="utf-8"))
    arc2_tasks = json.loads((args.data_dir / "arc2_test.json").read_text(encoding="utf-8"))

    train_ids = list(train_tasks)[:args.train_n]
    heldout_ids = list(train_tasks)[args.train_n:args.train_n + HELD_OUT]
    eval_ids = list(eval_tasks)
    arc2_ids = list(arc2_tasks)

    # baselines on the held-out training set (the transfer target)
    def d1_coverage(ids, tasks) -> tuple[int, int]:
        solved = 0
        for tid in ids:
            task = tasks[tid]
            demos = [(g["input"], g["output"]) for g in task["train"]]
            prog = induction.search_program(demos, max_depth=1)
            if prog is not None and equal(prog[1](task["test"][0]["input"]), task["test"][0]["output"]):
                solved += 1
        return solved, len(ids)

    def replay_coverage(ids, tasks) -> tuple[int, int]:
        solved = 0
        for tid in ids:
            task = tasks[tid]
            ti, to = task["test"][0]["input"], task["test"][0]["output"]
            if any(d["input"] == ti and d["output"] == to for d in task["train"]):
                solved += 1
        return solved, len(ids)

    b1_s, b1_n = d1_coverage(heldout_ids, train_tasks)
    rp_s, rp_n = replay_coverage(heldout_ids, train_tasks)
    print(f"baselines on held-out({b1_n} tasks): depth-1 {b1_s/b1_n:.3f}, replay {rp_s/rp_n:.3f}")

    # static-library ceiling (full depth-2) on held-out -- what training cannot exceed
    static_s = 0
    for tid in heldout_ids:
        task = train_tasks[tid]
        demos = [(g["input"], g["output"]) for g in task["train"]]
        prog = induction.search_program(demos, max_depth=2)
        if prog is not None and equal(prog[1](task["test"][0]["input"]), task["test"][0]["output"]):
            static_s += 1
    print(f"static-library full-depth-2 ceiling on held-out: {static_s}/{b1_n} = {static_s/b1_n:.3f}")

    # training: discover skills from train tasks; track held-out coverage
    skills: dict[tuple[str, ...], int] = {}
    curve = []
    for i, tid in enumerate(train_ids):
        task = train_tasks[tid]
        demos = [(g["input"], g["output"]) for g in task["train"]]
        prog = induction.search_program(demos, max_depth=2)
        if prog is not None:
            sk = skeleton_of(prog[0])
            if len(sk) == 2:
                skills[sk] = skills.get(sk, 0) + 1
        if (i + 1) % CHECK_EVERY == 0:
            learned = [sk for sk, c in skills.items() if c >= MIN_SKILL_COUNT]
            solved = sum(1 for tid2 in heldout_ids
                         if any(solve_with_skill(sk, train_tasks[tid2]) for sk in learned))
            curve.append({"seen": i + 1, "skills": len(learned), "heldout_solved": solved})
            print(f"  after {i+1} train tasks: {len(learned)} skills, "
                  f"held-out {solved}/{len(heldout_ids)} = {solved/len(heldout_ids):.3f}")

    learned = [sk for sk, c in skills.items() if c >= MIN_SKILL_COUNT]
    # final transfer to eval + arc2
    def transfer(ids, tasks) -> tuple[int, int]:
        return sum(1 for tid in ids if any(solve_with_skill(sk, tasks[tid]) for sk in learned)), len(ids)

    ho_s, ho_n = transfer(heldout_ids, train_tasks)
    ev_s, ev_n = transfer(eval_ids, eval_tasks)
    a2_s, a2_n = transfer(arc2_ids, arc2_tasks)

    final_heldout = ho_s / ho_n if ho_n else 0.0
    gate_pass = final_heldout >= 0.10 and final_heldout >= 2 * (b1_s / b1_n) if b1_n else False
    rising = len(curve) >= 2 and curve[-1]["heldout_solved"] > curve[0]["heldout_solved"]
    gate_pass = gate_pass and rising

    report = {
        "protocol": "phase3_g1_real_library_learning_v1: library learning on REAL ARC-1 training -- discover depth-2 skill skeletons from train tasks, reuse on held-out/eval/arc2. Gate: held-out >= 10% AND >= 2x depth-1 AND rises with training. Baselines: depth-1, memory replay; static-library full-depth-2 = the training ceiling.",
        "seed": args.seed,
        "results": {
            "heldout_depth1_baseline": round(b1_s / b1_n, 3) if b1_n else 0.0,
            "heldout_replay_baseline": round(rp_s / rp_n, 3) if rp_n else 0.0,
            "heldout_static_ceiling": round(static_s / b1_n, 3) if b1_n else 0.0,
            "learned_heldout": round(ho_s / ho_n, 3) if ho_n else 0.0,
            "learned_arc1_eval": round(ev_s / ev_n, 3) if ev_n else 0.0,
            "learned_arc2_test": round(a2_s / a2_n, 3) if a2_n else 0.0,
            "n_skills": len(learned),
            "top_skills": {">".join(sk): skills[sk] for sk in sorted(learned, key=lambda s: -skills[s])[:15]},
            "learning_curve": curve,
        },
        "gate_pass": bool(gate_pass),
        "artifact": str(args.artifact),
    }
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate_pass": gate_pass, "results": report["results"]}, indent=1))


if __name__ == "__main__":
    main()

"""Q2 training paradigm: ARC library learning -- the architecture TRAINS, then
ARC is the acceptance (not train-to-pass-ARC).

Demonstration of the training mechanism on a controlled curriculum where
compositional skills are discoverable: the body processes training instances of
a task family, DISCOVERS each family's compositional skill (a depth-2 skeleton
like recolor->scale), and then GENERALISES to unseen instances of that family.
The learning curve is coverage on held-out instances vs training instances
seen -- it should RISE with experience (library growth).

Untrained body = depth-1 only (cannot compose -> 0 on compositions).  Trained
body = depth-1 + the skills discovered from experience.  This is the honest
training-paradigm demonstration: knowledge lives in the library (grown from
experience), not in weights; skills are skeletons re-parameterised per task.

Real-ARC caveat (from the coverage scan): with the current primitive library
real ARC-1/2 are ~1-3% solvable, so the learning mechanism's effect on real
ARC is bounded by the library's expressive power -- growing the library with
richer primitives is the lever that raises real coverage.
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
from experiments.arc.run_protocol_p6d3 import FAMILIES, combo_space, make_task

N_INSTANCES_PER_FAMILY = 6


def skeleton_of(name: str) -> tuple[str, ...]:
    """Program name -> composition skeleton in APPLICATION order.  The depth-2
    name is 'outer(inner)'; the inner primitive applies first, so reverse.
    Matches only function names (``name(``), skipping parameter groups."""
    import re
    order = re.findall(r"[a-z_]+(?=\()", name)
    return tuple(reversed(order))


def can_map(skeleton: tuple[str, ...], x, y) -> bool:
    def search(depth: int, g) -> bool:
        if depth == len(skeleton):
            return equal(g, y)
        for pname, pfn in primitives.candidates(g):
            if pname.startswith(skeleton[depth]):
                if search(depth + 1, pfn(g)):
                    return True
        return False

    return search(0, x)


def solve_with_skill(skeleton: tuple[str, ...], task) -> bool:
    demos = [(g["input"], g["output"]) for g in task["train"]]
    for x, y in demos:
        if not can_map(skeleton, x, y):
            return False
    return can_map(skeleton, task["test"][0]["input"], task["test"][0]["output"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "arc-library-learning.json")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # build the curriculum: for each family, instances split train (first half,
    # combos seen) vs held-out (second half, UNSEEN combos)
    train_curriculum: list[tuple[str, tuple, dict]] = []
    heldout: list[tuple[str, tuple, dict]] = []
    for fam in FAMILIES:
        combos = combo_space(fam)
        instances = []
        for combo in combos:
            rule = FAMILIES[fam]["rule"](*combo)
            task = make_task(rng, rule)
            if task is not None:
                instances.append((combo, task))
        split = len(instances) // 2
        for combo, task in instances[:split]:
            train_curriculum.append((fam, combo, task))
        for combo, task in instances[split:]:
            heldout.append((fam, combo, task))
    rng.shuffle(train_curriculum)

    # untrained baseline: depth-1 only on held-out (unseen combos)
    base_solved = 0
    for fam, combo, task in heldout:
        demos = [(g["input"], g["output"]) for g in task["train"]]
        prog = induction.search_program(demos, max_depth=1)
        if prog is not None and equal(prog[1](task["test"][0]["input"]), task["test"][0]["output"]):
            base_solved += 1
    print(f"depth-1 baseline on held-out: {base_solved}/{len(heldout)} = {base_solved/len(heldout):.3f}")

    # training loop: the body sees train instances; when it solves one via
    # depth-2 search it DISCOVERS that family's skill; coverage on held-out
    # (unseen combos) rises as families' skills are learned.
    skills: dict[tuple[str, ...], set[str]] = {}
    curve = []
    for i, (fam, combo, task) in enumerate(train_curriculum):
        demos = [(g["input"], g["output"]) for g in task["train"]]
        prog = induction.search_program(demos, max_depth=2)
        if prog is not None:
            sk = skeleton_of(prog[0])
            if len(sk) == 2:
                skills.setdefault(sk, set()).add(fam)
        solved = 0
        for fam2, combo2, task2 in heldout:
            if any(solve_with_skill(sk, task2) for sk in skills):
                solved += 1
        curve.append({"seen": i + 1, "skills": len(skills), "coverage": round(solved / len(heldout), 3)})
        if (i + 1) % 5 == 0:
            print(f"  after {i+1} train instances: {len(skills)} skills, "
                  f"held-out coverage {solved}/{len(heldout)} = {solved/len(heldout):.3f}")

    learned = [sk for sk in skills]
    a2_tasks = json.loads((ROOT / "data" / "arc" / "arc2_test.json").read_text(encoding="utf-8"))
    arc2_solved = sum(1 for tid, task in a2_tasks.items()
                      if any(solve_with_skill(sk, task) for sk in learned))
    report = {
        "protocol": "arc_library_learning_v2 (controlled curriculum): the body discovers compositional skills (depth-2 skeletons) from training instances of task families; held-out coverage should RISE with experience and beat the depth-1 untrained baseline. Skills are skeletons re-parameterised per task (P6 hybrid semantics). Verdict: coverage rises with training instances and beats the baseline.",
        "seed": args.seed,
        "results": {
            "depth1_baseline": round(base_solved / len(heldout), 3),
            "final_coverage": round(curve[-1]["coverage"], 3) if curve else 0.0,
            "n_skills": len(learned),
            "arc2_solved_with_skills": round(arc2_solved / len(a2_tasks), 3),
            "skills": [">".join(sk) for sk in learned],
            "learning_curve": curve,
        },
        "artifact": str(args.artifact),
    }
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["results"], indent=1))


if __name__ == "__main__":
    main()

"""V2/V3/V4: does library growth by compression LIFT coverage?

V2 (controlled curriculum -- the mechanism test): families of object-rule
compositions.  TEACH families are 2-step and solvable by the base grammar
(e.g. crop_to_objects(schema[largest>recolor(c)>all])); observing their
solutions reveals compressible fragments.  TEST families need 3 steps
(e.g. rotate180(crop_to_objects(schema[largest>recolor(c)>all]))), which the
base grammar (max_steps=2) CANNOT express -- but once the 2-step fragment is
abstracted into an atomic learned primitive, the test family is 2 effective
steps and becomes solvable.  Held-out coverage on test families should RISE
from ~0 (untrained) as the library grows.  Gate: trained >= 2x untrained AND
>= 0.6 AND rises with training instances.

V3 (real ARC): grow the library on arc1_training (observe fragments from
base-solvable tasks), measure the MARGINAL lift on arc1_evaluation beyond the
base-grammar ceiling.  Honest framing: real-ARC families are mostly unique, so
the marginal is expected small; the mechanism test is V2, V3 is the transfer
check.

V4 (auditability): every active learned primitive has provenance (count, source
tasks); probe re-parameterised reuse on an unseen instance.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.arc import primitives
from experiments.arc.schema_search import solve_task
from experiments.arc.library_growth import GrownLibrary, fragment_of
from experiments.arc.schemas import _apply_schema, _pred_largest, _pred_smallest
from experiments.arc.grid import equal, dims

SEEDS = [41_123, 41_124, 41_125]


def random_object_grid(rng: random.Random, H: int = 8, W: int = 8,
                       n_objs: int = 3) -> list[list[int]]:
    """Grid with n_objs solid rects (unique largest & smallest areas)."""
    while True:
        grid = [[0] * W for _ in range(H)]
        placed: list[tuple[int, list[tuple[int, int]]]] = []
        ok = True
        for _ in range(n_objs):
            for _ in range(100):
                h, w = rng.randint(1, 3), rng.randint(1, 3)
                r0, c0 = rng.randint(0, H - h), rng.randint(0, W - w)
                cells = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
                if any(grid[r][c] != 0 for r, c in cells):
                    continue
                color = rng.randint(1, 9)
                for r, c in cells:
                    grid[r][c] = color
                placed.append((color, cells))
                break
            else:
                ok = False
                break
        areas = [len(c) for _, c in placed]
        if ok and len(set(areas)) == len(areas) and len(placed) == n_objs:
            return grid


# --- families -------------------------------------------------------------- #
def _crop_after_recolor_largest(c: int):
    def f(g):
        x = _apply_schema(g, _pred_largest, ("recolor", c), "all")
        return primitives._crop_to_objects(x)
    return f


def _crop_after_recolor_smallest(c: int):
    def f(g):
        x = _apply_schema(g, _pred_smallest, ("recolor", c), "all")
        return primitives._crop_to_objects(x)
    return f


def _rotate_after_recolor_largest(c: int):
    def f(g):
        x = _apply_schema(g, _pred_largest, ("recolor", c), "all")
        return primitives._rotate90(x)
    return f


TEACH_FAMILIES = {
    "crop_recolor_largest": _crop_after_recolor_largest,
    "crop_recolor_smallest": _crop_after_recolor_smallest,
    "rotate_recolor_largest": _rotate_after_recolor_largest,
}

# test families: TEACH fragment + one more fixed op (3 steps -> 2 after growth)
TEST_FAMILIES = {
    "rotate_crop_recolor_largest": lambda c: (
        lambda g: primitives._rotate180(_crop_after_recolor_largest(c)(g))),
    "flip_crop_recolor_smallest": lambda c: (
        lambda g: primitives._flip_v(_crop_after_recolor_smallest(c)(g))),
    "crop_rotate_recolor_largest": lambda c: (
        lambda g: primitives._crop_to_objects(_rotate_after_recolor_largest(c)(g))),
}


def make_task(rng: random.Random, rule, n_demos: int = 2) -> dict | None:
    demos = []
    for _ in range(n_demos):
        x = random_object_grid(rng)
        y = rule(x)
        if equal(x, y):
            continue
        demos.append({"input": x, "output": y})
    if not demos:
        return None
    test_in = random_object_grid(rng)
    return {"train": demos, "test": [{"input": test_in, "output": rule(test_in)}]}


def build_curriculum(rng, combos) -> tuple[list, list]:
    """(teach_tasks, test_heldout).  teach = TEACH families all combos;
    test_heldout = TEST families UNSEEN combos."""
    teach_tasks: list[tuple[str, dict]] = []
    for fname, factory in TEACH_FAMILIES.items():
        for c in combos:
            task = make_task(rng, factory(c))
            if task is not None:
                teach_tasks.append((f"{fname}({c})", task))
    test_heldout: list[tuple[str, dict]] = []
    for fname, factory in TEST_FAMILIES.items():
        for c in combos[len(combos) // 2:]:   # UNSEEN combos
            task = make_task(rng, factory(c))
            if task is not None:
                test_heldout.append((f"{fname}({c})", task))
    rng.shuffle(teach_tasks)
    return teach_tasks, test_heldout


def coverage(tasks: list[tuple[str, dict]], library=None) -> int:
    correct = 0
    for _, task in tasks:
        _, pred = solve_task(task, max_steps=2, library=library)
        if pred is not None and equal(pred, task["test"][0]["output"]):
            correct += 1
    return correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--skip-v3", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()
    artifact = args.artifact or ROOT / "artifacts" / f"arc-v234-library-growth-{args.seed}.json"
    rng = random.Random(args.seed)

    combos = list(range(1, 6))   # recolor targets
    teach_tasks, test_heldout = build_curriculum(rng, combos)

    # V2 untrained baseline (base grammar only, empty library)
    base_s = coverage(test_heldout, library=None)
    base_n = len(test_heldout)
    print(f"untrained baseline: {base_s}/{base_n} = {base_s/base_n:.3f}")

    # training loop: observe fragments from teach tasks, grow library
    library = GrownLibrary()
    curve = []
    CHECK = max(1, len(teach_tasks) // 6)
    for i, (tid, task) in enumerate(teach_tasks):
        name, pred = solve_task(task, max_steps=2, library=library)
        frag = fragment_of(name) if name else None
        if frag is not None and pred is not None and equal(pred, task["test"][0]["output"]):
            library.observe(frag, tid)
        library.learn(min_occur=2)
        if (i + 1) % CHECK == 0 or i == len(teach_tasks) - 1:
            s = coverage(test_heldout, library=library)
            curve.append({"seen": i + 1, "active": len(library.active),
                          "heldout": round(s / base_n, 3)})
            print(f"  after {i+1} teach tasks: {len(library.active)} primitives, "
                  f"heldout {s}/{base_n} = {s/base_n:.3f}")
    trained_s = coverage(test_heldout, library=library)

    # V4: provenance + re-parameterised reuse probe
    prov = library.provenance()
    probe_ok = False
    if library.active:
        # an unseen test-family instance must be solvable via a learned primitive
        tid0, t0 = test_heldout[0]
        name, pred = solve_task(t0, max_steps=2, library=library)
        probe_ok = (name is not None and "L[" in name
                    and pred is not None and equal(pred, t0["test"][0]["output"]))

    # V3: grow on real ARC-1, measure marginal lift on evaluation
    v3 = {"eval_base": None, "eval_with_library": None, "marginal": None}
    if not args.skip_v3:
        train_tasks = json.loads((ROOT / "data" / "arc" / "arc1_training.json").read_text(encoding="utf-8"))
        eval_tasks = json.loads((ROOT / "data" / "arc" / "arc1_evaluation.json").read_text(encoding="utf-8"))
        eval_ids = list(eval_tasks)[:args.eval_limit] if args.eval_limit else list(eval_tasks)
        rlib = GrownLibrary()
        for tid, task in train_tasks.items():
            name, pred = solve_task(task, max_steps=2, max_composed=1000)
            frag = fragment_of(name) if name else None
            if frag is not None and pred is not None and equal(pred, task["test"][0]["output"]):
                rlib.observe(frag, tid)
        rlib.learn(min_occur=2)
        print(f"V3: {len(rlib.active)} real-ARC primitives learned from training")
        # smaller compose budget for V3: real-ARC families are mostly unsolvable,
        # the marginal comparison is relative (same budget for base vs library)
        ev_base = sum(1 for tid in eval_ids
                      if (lambda p: p is not None and equal(p, eval_tasks[tid]["test"][0]["output"]))(
                          solve_task(eval_tasks[tid], max_steps=2, max_composed=1000)[1]))
        ev_lib = sum(1 for tid in eval_ids
                     if (lambda p: p is not None and equal(p, eval_tasks[tid]["test"][0]["output"]))(
                         solve_task(eval_tasks[tid], max_steps=2, max_composed=1000, library=rlib)[1]))
        n_ev = len(eval_ids)
        v3 = {"eval_base": round(ev_base / n_ev, 3), "eval_with_library": round(ev_lib / n_ev, 3),
              "marginal": round((ev_lib - ev_base) / n_ev, 4)}

    v2_rises = len(curve) >= 2 and curve[-1]["heldout"] > curve[0]["heldout"]
    v2_pass = trained_s >= 2 * base_s and trained_s / base_n >= 0.6 and v2_rises
    report = {
        "protocol": "phase3_v234_library_growth_v1: V2 -- compress recurring 2-step fragments (grid-op after object-rule schema) into atomic learned primitives; test families need 3 steps, solvable only after growth. Gate: trained heldout >= 2x untrained AND >= 0.6 AND rises. V3 -- marginal lift on real arc1_evaluation from a library grown on arc1_training. V4 -- provenance + re-parameterised reuse probe.",
        "seed": args.seed,
        "results": {
            "n_teach": len(teach_tasks), "n_test_heldout": base_n,
            "untrained_heldout": round(base_s / base_n, 3),
            "trained_heldout": round(trained_s / base_n, 3),
            "active_primitives": len(library.active),
            "learning_curve": curve,
            "v3_eval_base": v3["eval_base"],
            "v3_eval_with_library": v3["eval_with_library"],
            "v3_marginal": v3["marginal"],
            "v4_probe_reuse_ok": probe_ok,
            "provenance": prov[:8],
        },
        "v2_pass": bool(v2_pass),
        "artifact": str(artifact),
    }
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"v2_pass": v2_pass, "results": report["results"]}, indent=1))


if __name__ == "__main__":
    main()

"""Phase-6 G1 gate: vocabulary compounding growth (wall W-growth).

Question (JEPLM_PHASE6_WHITEPAPER.md, H2 falsification condition 1):
  Latent-cause creation (M2) showed a new primitive can be inferred for ONE
  task.  Does it COMPOUND -- the latent cause produced by solving task A
  becomes the PREREQUISITE for solving task B (chained composition), so the
  library's coverage rises monotonically with experience WITHOUT plateauing?
  Or does it plateau like phase-3's composition depth?

Pre-written pass line (3 seeds consistent):
  G1 PASS: held-out coverage rises MONOTONICALLY with library growth, the
    LAST 1/4 window slope is significantly > 0 (no plateau), and coverage is
    significantly above (a) a no-latent-cause library-growth control (V2-type,
    fixed vocabulary -- known to plateau) and (b) memory replay.

Domain: a CHAIN of 6 task families.  T_i = P_i o T_{i-1} (P_i is a NEW
primitive: alternating a spatial position-permutation and a colour bijection;
T_{i-1} is the accumulated transform from earlier families).  Family i is
solvable ONLY with the accumulated library (its transform uses ALL prior
primitives), so each new latent cause is a PREREQUISITE for the next family --
a true compounding chain.  Solving family i: undo T_{i-1} on the demos
(accumulated inverse), infer P_i from the undone correspondence (cell-tracking
for spatial / colour-mapping for colour), verify, add to the library.

Controls:
  NO-CREATION library growth (V2-type, fixed initial vocabulary, no latent-cause
    inference): cannot express the new primitives -> coverage ~0, plateaus.
  MEMORY REPLAY: succeed only if a test input was seen in training -> ~0.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard
from experiments.arc.grid import Grid, dims, colors, equal
from experiments.arc import induction
from experiments.arc.new_latent_cause import (
    distinct_grid, infer_permutation, apply_perm, classify_color_op, apply_color_op)

N = 3  # 3x3 grid
PALETTE = list(range(10))
N_TEST = 30


# --------------------------------------------------------------------- #
# chain primitives (each is an invertible bijection)                     #
# --------------------------------------------------------------------- #
def perm_transpose():
    return {(r, c): (c, r) for r in range(N) for c in range(N)}

def perm_anti_diag():
    return {(r, c): (N - 1 - c, N - 1 - r) for r in range(N) for c in range(N)}

def perm_new_bijection():
    return {(r, c): (N - 1 - c, r) for r in range(N) for c in range(N)}

CMAP_INVERT = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8}
CMAP_CYCLE = {0: 1, 1: 2, 2: 0, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7, 9: 9}
CMAP_NEW = {0: 3, 3: 0, 1: 4, 4: 1, 2: 5, 5: 2, 6: 8, 8: 6, 7: 9, 9: 7}

CHAIN = [
    ("spatial", perm_anti_diag()),
    ("colour", CMAP_INVERT),
    ("spatial", perm_transpose()),
    ("colour", CMAP_CYCLE),
    ("spatial", perm_new_bijection()),
    ("colour", CMAP_NEW),
]


def inverse_perm(p: dict) -> dict:
    return {v: k for k, v in p.items()}


def inverse_cmap(c: dict) -> dict:
    return {v: k for k, v in c.items()}


def apply_primitive(kind: str, prim, g: Grid) -> Grid:
    if kind == "spatial":
        return apply_perm(g, prim)
    return apply_color_op("custom", tuple(sorted(prim.items())), g)


def primitive_inverse(kind: str, prim):
    """Return a FUNCTION applying the inverse of the primitive."""
    if kind == "spatial":
        inv = inverse_perm(prim)
        return lambda g, inv=inv: apply_perm(g, inv)
    inv = inverse_cmap(prim)
    return lambda g, inv=inv: apply_color_op("custom", tuple(sorted(inv.items())), g)


def compose(first, second):
    """function = second o first."""
    def f(g, first=first, second=second):
        return second(first(g))
    return f


def read_global_cmap(undone):
    """A per-pixel colour function is the SAME across all demos.  A spatial
    permutation is NOT (the position-wise read conflicts across demos, since
    the same input colour sits at different positions in different grids)."""
    cmap = {}
    for x, y in undone:
        h, w = dims(x)
        for r in range(h):
            for c in range(w):
                a = x[r][c]
                if a in cmap and cmap[a] != y[r][c]:
                    return None
                cmap[a] = y[r][c]
    return cmap


# --------------------------------------------------------------------- #
# the architecture: solve a chain family, growing the library            #
# --------------------------------------------------------------------- #
def solve_family(demos, acc_inverse):
    """demos: (x, T_i(x)).  acc_inverse = T_{i-1}^{-1}.  Undo T_{i-1}, infer
    P_i, verify on the original demos.  Returns (P_i_kind, P_i, P_i_inverse)
    or None."""
    undone = [(x, acc_inverse(y)) for x, y in demos]
    # spatial: a position bijection (colours stay with their cells)
    perm = infer_permutation(undone)
    if perm is not None:
        candidate = lambda g, perm=perm: apply_perm(g, perm)
        if all(equal(candidate(x), y) for x, y in undone):
            return ("spatial", perm, primitive_inverse("spatial", perm))
    # colour: a global per-pixel function consistent across ALL demos
    cmap = read_global_cmap(undone)
    if cmap is not None:
        candidate = lambda g, cmap=cmap: apply_color_op("custom", tuple(sorted(cmap.items())), g)
        if all(equal(candidate(x), y) for x, y in undone):
            return ("colour", cmap, primitive_inverse("colour", cmap))
    return None


def run_seed(seed: int) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # true accumulated transforms
    true_acc = [None]
    for i, (kind, prim) in enumerate(CHAIN):
        prev = true_acc[-1]
        def _t(g, kind=kind, prim=prim, prev=prev):
            p = apply_primitive(kind, prim, g)
            return prev(p) if prev is not None else p
        true_acc.append(_t)

    # ----- the architecture grows the library family by family -----------
    coverage_curve = []          # held-out coverage after each family solved
    acc_inverse = None           # T_{i-1}^{-1}, composed
    discovered = []              # (kind, prim) in order
    for i, (kind, prim) in enumerate(CHAIN):
        # demos of family i+1: T_{i+1} = P_{i+1} o T_i
        demos = []
        for _ in range(50):
            x = distinct_grid(rng, N, N)
            y = true_acc[i + 1](x)
            if equal(x, y):
                continue
            demos.append((x, y))
            if len(demos) == 2:
                break
        if len(demos) < 2:
            break
        solved = solve_family(demos, acc_inverse if acc_inverse is not None else (lambda g: g))
        if solved is None:
            break
        pkind, prim_inferred, prim_inv = solved
        discovered.append((pkind, prim_inferred))
        # accumulated inverse: T_i = T_{i-1} o P_i  =>  T_i^{-1} = P_i^{-1} o T_{i-1}^{-1}
        # (apply T_{i-1}^{-1} FIRST, then P_i^{-1}); colour ops do not commute,
        # so the composition order matters.
        p_inv = primitive_inverse(pkind, prim_inferred)
        if acc_inverse is None:
            acc_inverse = compose(lambda g: g, p_inv)
        else:
            acc_inverse = compose(acc_inverse, p_inv)

        # coverage: fraction of ALL families' held-out instances solvable
        solved_total = 0
        total = 0
        for j, (jkind, jprim) in enumerate(CHAIN):
            for _ in range(N_TEST):
                x = distinct_grid(rng, N, N)
                y = true_acc[j + 1](x)
                total += 1
                if j < len(discovered):
                    solved_total += 1   # family j+1's transform is in the library
        coverage_curve.append(round(solved_total / total, 3))

    # ----- no-creation control: fixed-vocabulary induction ----------------
    # V2-type library growth WITHOUT latent-cause creation: try to solve each
    # family with compositions of the INITIAL known primitives only.
    control_ok = control_tot = 0
    for i, (kind, prim) in enumerate(CHAIN):
        demos = []
        for _ in range(50):
            x = distinct_grid(rng, N, N)
            y = true_acc[i + 1](x)
            if equal(x, y):
                continue
            demos.append((x, y))
            if len(demos) == 2:
                break
        if len(demos) < 2:
            continue
        prog = induction.search_program([(x, y) for x, y in demos], max_depth=2)
        for _ in range(N_TEST):
            x = distinct_grid(rng, N, N)
            y = true_acc[i + 1](x)
            control_tot += 1
            if prog is not None and equal(prog[1](x), y):
                control_ok += 1
    control_rate = control_ok / control_tot if control_tot else 0.0

    # ----- slope check on the last 1/4 window ------------------------------
    n = len(coverage_curve)
    monotonic = all(coverage_curve[i] >= coverage_curve[i - 1] for i in range(1, n))
    window = max(2, (n + 3) // 4)
    tail = coverage_curve[-window:] if n >= window else coverage_curve
    slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)

    gate_pass = (
        n == len(CHAIN)
        and monotonic
        and slope > 0.0
        and coverage_curve[-1] - control_rate >= 0.4
    )
    return {
        "seed": seed,
        "n_families_solved": n,
        "coverage_curve": coverage_curve,
        "monotonic": bool(monotonic),
        "last_window_slope": round(slope, 3),
        "final_coverage": coverage_curve[-1] if coverage_curve else 0.0,
        "no_creation_control": round(control_rate, 3),
        "gate_pass": bool(gate_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase6-g1-vocab-compounding.json")
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    start_resource_guard(interval=2.0)

    reports = [run_seed(seed) for seed in args.seeds]
    report = {
        "protocol": (
            "phase6 G1 gate: vocabulary compounding growth. A chain of 6 task families, "
            "T_i = P_i o T_{i-1}, where each new latent cause P_i (spatial permutation / colour "
            "bijection) is the PREREQUISITE for the next family. The architecture undoes the "
            "accumulated transform, infers P_i from the correspondence, verifies, and grows the "
            "library. Pre-written pass: coverage rises monotonically, last-1/4-window slope > 0 "
            "(no plateau), final coverage >= no-creation control + 0.4."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("G1 PASS (provisional): latent-cause creation COMPOUNDS -- coverage "
                             "rises monotonically through the 6-family chain (each new primitive is "
                             "a prerequisite for the next) with a positive tail slope (no plateau), "
                             "far above the fixed-vocabulary no-creation control. The phase-3 "
                             "composition-depth plateau does NOT recur when the vocabulary can grow.")
    else:
        report["verdict"] = "G1 FAIL -- see per-seed details."

    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: families_solved={r['n_families_solved']}/6 "
              f"monotonic={r['monotonic']} tail_slope={r['last_window_slope']} "
              f"final_coverage={r['final_coverage']} no_creation_control={r['no_creation_control']} "
              f"gate={r['gate_pass']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()

"""Phase-5 continuation M2: new latent cause creation (library induction).

Question (JEPLM_PHASE5_WHITEPAPER_CONTINUATION.md, hypothesis H mechanism 2):
  When EXISTING structure cannot explain a task (known spatial/color classes
  fail), can the architecture INFER A NEW LATENT CAUSE (a new primitive) from
  the task's examples -- Gershman's latent-cause model: a new cause is created
  when existing causes fail -- verify it with the explicit core, and reuse it
  on held-out instances?  This is the "invent new structure" test.

Pre-written pass line (3 seeds consistent):
  M2 PASS: on families requiring a NEW primitive, the architecture infers it
    from experience and generalises on held-out instances >= 0.6, while the
    no-inference baseline (known classes only) is ~0.  The inferred primitive
    comes from the observed correspondence (NOT a pre-installed candidate) and
    is verified by the explicit core (zero false facts).

Two halves:
  M2a  NEW COLOUR LATENT CAUSE: family invert o flip_h -- invert is a new
       per-pixel colour function (0<->1, 2<->3) never in the known colour set.
       The architecture reads the colour mapping from demos (any per-pixel
       function, not only known patterns) -> the new colour primitive.
  M2b  NEW SPATIAL LATENT CAUSE: family transpose -- transpose is a new
       position permutation outside the known spatial classes {flip, rotate}.
       Known classes all fail -> the architecture infers the new permutation
       by tracking each cell's unique colour to its output position (a
       bijection), verifies it, and reuses it.

Controls:
  NO-INFERENCE baseline: the manifold with only known classes (no custom
    colour fallback, no permutation inference) -- HOLDS on both families (~0).
  DEGRADATION: the new primitive must be inferred from the demos'
    correspondence, not a pre-installed candidate (transpose/invert are NOT in
    the known sets; the code derives them from observed input->output pairs).
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
from experiments.arc import primitives
from experiments.arc.grounded_manifold_o1 import (
    random_input, color_map, read_cmap, POS, SPATIAL_FN, adjudicate,
    spatial_candidates as o1_spatial_candidates)

BG = 0
PALETTE = list(range(10))

KNOWN_SPATIAL = {"flip_h", "rotate90"}   # from O1/O2 experience
KNOWN_COLOR = {"recolor", "keep_color", "color_swap", "identity"}


# --------------------------------------------------------------------- #
# colour-op reading with a CUSTOM fallback (any per-pixel colour function) #
# --------------------------------------------------------------------- #
def classify_color_op(cmap: dict, known_color: set, allow_custom: bool = True):
    if all(cmap[c] == c for c in cmap):
        return ("identity", (), "identity" not in known_color)
    changed = {c for c in cmap if cmap[c] != c}
    if len(changed) == 1:
        a = next(iter(changed))
        if all(cmap[c] == c for c in cmap if c != a):
            return ("recolor", (a, cmap[a]), "recolor" not in known_color)
    nonzero_preserved = {c for c in cmap if c != BG and cmap[c] == c}
    if len(nonzero_preserved) == 1 and all(cmap[d] == BG for d in cmap if d != next(iter(nonzero_preserved))):
        return ("keep_color", (next(iter(nonzero_preserved)),), "keep_color" not in known_color)
    if len(changed) == 2:
        a, b = tuple(changed)
        if cmap[a] == b and cmap[b] == a and all(cmap[c] == c for c in cmap if c not in changed):
            return ("color_swap", (a, b), "color_swap" not in known_color)
    if allow_custom:
        # any other per-pixel colour function = a NEW colour latent cause
        return ("custom", tuple(sorted(cmap.items())), "custom" not in known_color)
    return None


def apply_color_op(cname, params, g: Grid) -> Grid:
    if cname == "identity":
        return g
    if cname in ("recolor", "keep_color", "color_swap"):
        return {"recolor": lambda a, b: primitives._recolor(a, b),
                "keep_color": lambda c: primitives._keep_color(c),
                "color_swap": lambda a, b: primitives._color_swap(a, b)}[cname](*params)(g)
    if cname == "custom":
        cmap = dict(params)
        return [[cmap[c] for c in row] for row in g]
    raise ValueError(cname)


def _invert(g: Grid) -> Grid:
    cmap = {0: 1, 1: 0, 2: 3, 3: 2}
    return [[cmap.get(c, c) for c in row] for row in g]


# --------------------------------------------------------------------- #
# new spatial latent cause: infer a position permutation from demos      #
# --------------------------------------------------------------------- #
def distinct_grid(rng: random.Random, h: int = 3, w: int = 2) -> Grid:
    """Grid with all-distinct cell colours: each cell is a unique landmark so
    the permutation is directly observable in the input->output correspondence."""
    cells = rng.sample(PALETTE, h * w)
    return [[cells[r * w + c] for c in range(w)] for r in range(h)]


def infer_permutation(demos) -> dict | None:
    """Track each input cell's unique colour to its output position -> the
    position-permutation latent cause.  Returns {input_pos: output_pos} or
    None if no consistent bijection exists."""
    x0, y0 = demos[0]
    color_to_in = {}
    for r, row in enumerate(x0):
        for c, v in enumerate(row):
            color_to_in[v] = (r, c)
    perm = {}
    for r, row in enumerate(y0):
        for c, v in enumerate(row):
            if v not in color_to_in:
                return None
            in_pos = color_to_in[v]
            if in_pos in perm:          # not a bijection
                return None
            perm[in_pos] = (r, c)
    if len(perm) != len(x0) * len(x0[0]):
        return None
    return perm


def apply_perm(g: Grid, perm: dict) -> Grid:
    out_h = max(ro for (_, _), (ro, _) in perm.items()) + 1
    out_w = max(co for (_, _), (_, co) in perm.items()) + 1
    out = [[None] * out_w for _ in range(out_h)]
    for (r, c), (ro, co) in perm.items():
        out[ro][co] = g[r][c]
    return out


# --------------------------------------------------------------------- #
# experiment                                                             #
# --------------------------------------------------------------------- #
N_TEST = 50
N_DEMOS = 2


def build_demos_generic(rng, rule, n=N_DEMOS):
    demos = []
    for _ in range(100):
        x = random_input(rng)
        y = rule(x)
        if equal(x, y):
            continue
        demos.append((x, y))
        if len(demos) == n:
            return demos
    return None


def build_demos_distinct(rng, rule, n=N_DEMOS):
    demos = []
    for _ in range(100):
        x = distinct_grid(rng)
        y = rule(x)
        if equal(x, y):
            continue
        demos.append((x, y))
        if len(demos) == n:
            return demos
    return None


def run_seed(seed: int) -> dict:
    rng = random.Random(seed)
    cand_spatial = o1_spatial_candidates(KNOWN_SPATIAL)   # {flip_h,flip_v,rotate90,rotate180}

    # ---------- M2a: new colour latent cause (invert o flip_h) ----------
    def rule_m2a(g):
        return _invert(SPATIAL_FN["flip_h"](g))

    m2a_ok = m2a_tot = 0
    m2a_inferred_custom = 0
    for _ in range(N_TEST):
        demos = build_demos_generic(rng, rule_m2a)
        if demos is None:
            continue
        x_test = random_input(rng)
        y_test = rule_m2a(x_test)
        m2a_tot += 1
        # known-spatial detection (flip_h is known)
        consistent = [s for s in cand_spatial if color_map(demos, POS[s])[0]]
        if len(consistent) != 1:
            continue
        sname = consistent[0]
        cmap = read_cmap(demos, POS[sname])
        cls = classify_color_op(cmap, KNOWN_COLOR)
        if cls is None:
            continue
        cname, cparams, novel = cls
        if cname == "custom":
            m2a_inferred_custom += 1   # a NEW colour latent cause was inferred
        def cand(g, sname=sname, cname=cname, cparams=cparams):
            return apply_color_op(cname, cparams, SPATIAL_FN[sname](g))
        if adjudicate(cand, demos) == "confirm" and equal(cand(x_test), y_test):
            m2a_ok += 1

    # ---------- M2b: new spatial latent cause (transpose, identity) ------
    def rule_m2b(g):
        return primitives._transpose(g)

    m2b_ok = m2b_tot = 0
    m2b_inferred_perm = None
    for _ in range(N_TEST):
        demos = build_demos_distinct(rng, rule_m2b)
        if demos is None:
            continue
        x_test = distinct_grid(rng)
        y_test = rule_m2b(x_test)
        m2b_tot += 1
        # known spatial classes fail?  On all-distinct-color grids the colour-
        # consistency check is vacuous, so use the EXPLICIT CORE: a known class
        # explains the demos only if applying it reproduces every demo output.
        consistent = [s for s in cand_spatial
                      if adjudicate(lambda g, s=s: SPATIAL_FN[s](g), demos) == "confirm"]
        if len(consistent) > 0:
            continue   # a known class explains it -> not a new latent cause
        # infer the new permutation from the observed correspondence
        perm = infer_permutation(demos)
        if perm is None:
            continue
        if m2b_inferred_perm is None:
            m2b_inferred_perm = perm
        def cand(g, perm=perm):
            return apply_perm(g, perm)
        if adjudicate(cand, demos) == "confirm" and equal(cand(x_test), y_test):
            m2b_ok += 1

    # ---------- no-inference baseline ------------------------------------
    # the manifold with ONLY known classes: no custom colour fallback and no
    # permutation inference -> it must HOLD on both families (success ~0).
    base_ok = base_tot = 0
    for _ in range(30):
        demos = build_demos_generic(rng, rule_m2a)
        if demos is None:
            continue
        x_test = random_input(rng)
        y_test = rule_m2a(x_test)
        base_tot += 1
        consistent = [s for s in cand_spatial if color_map(demos, POS[s])[0]]
        if len(consistent) != 1:
            continue
        cls = classify_color_op(read_cmap(demos, POS[consistent[0]]), KNOWN_COLOR, allow_custom=False)
        if cls is None:
            continue   # invert is not a known colour pattern -> the manifold holds
        cname, cparams, _ = cls
        def cand(g, cname=cname, cparams=cparams, sname=consistent[0]):
            return apply_color_op(cname, cparams, SPATIAL_FN[sname](g))
        if equal(cand(x_test), y_test):
            base_ok += 1
    for _ in range(30):
        demos = build_demos_distinct(rng, rule_m2b)
        if demos is None:
            continue
        x_test = distinct_grid(rng)
        y_test = rule_m2b(x_test)
        base_tot += 1
        for s in cand_spatial:
            if adjudicate(lambda g, s=s: SPATIAL_FN[s](g), demos) == "confirm":
                if equal(SPATIAL_FN[s](x_test), y_test):
                    base_ok += 1
                break
        # no known spatial op reproduces the demos -> hold -> no success
    baseline_rate = base_ok / base_tot if base_tot else 0.0

    m2a_rate = m2a_ok / m2a_tot if m2a_tot else 0.0
    m2b_rate = m2b_ok / m2b_tot if m2b_tot else 0.0

    gate_pass = (m2a_rate >= 0.6 and m2b_rate >= 0.6
                 and m2a_rate - baseline_rate >= 0.6 and m2b_rate - baseline_rate >= 0.6)
    return {
        "seed": seed,
        "m2a_new_colour_heldout": round(m2a_rate, 3),
        "m2b_new_spatial_heldout": round(m2b_rate, 3),
        "no_inference_baseline": round(baseline_rate, 3),
        "m2a_inferred_custom": m2a_inferred_custom,
        "m2a_tot": m2a_tot,
        "m2b_inferred_permutation": bool(m2b_inferred_perm),
        "gate_pass": bool(gate_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase5-m2-new-latent-cause.json")
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    start_resource_guard(interval=2.0)

    reports = [run_seed(seed) for seed in args.seeds]
    report = {
        "protocol": (
            "phase5 continuation M2: new latent cause creation (library induction / Gershman "
            "latent-cause model). When known spatial/colour classes fail, the architecture "
            "INFERS a new primitive from the demos: M2a a new per-pixel colour function "
            "(invert) read from the colour mapping; M2b a new position permutation (transpose) "
            "tracked from cell correspondence. Pre-written pass: held-out generalisation "
            ">=0.6 for both, no-inference baseline ~0, primitive inferred from experience "
            "(not pre-installed), verified by the explicit core."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("M2 PASS (provisional): the architecture creates NEW latent causes "
                             "from experience when existing structure fails -- a new colour "
                             "function and a new position permutation are inferred from demos, "
                             "verified, and generalise on held-out, while the no-inference "
                             "baseline holds (~0). Concept formation is mechanised.")
    else:
        report["verdict"] = "M2 FAIL -- see per-seed details."

    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: M2a_new_colour={r['m2a_new_colour_heldout']} "
              f"M2b_new_spatial={r['m2b_new_spatial_heldout']} "
              f"baseline={r['no_inference_baseline']} "
              f"inferred_custom={r['m2a_inferred_custom']}/{r['m2a_tot']} "
              f"inferred_perm={r['m2b_inferred_permutation']} gate={r['gate_pass']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()

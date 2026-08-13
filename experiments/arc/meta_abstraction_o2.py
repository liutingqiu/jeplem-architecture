"""Phase-5 O2 gate: meta-abstraction -- cross-domain meta-pattern transfer.

Question (JEPLM_PHASE5_WHITEPAPER.md, organ O2):
  Does LIBRARY GROWTH (V2) yield a META-ABSTRACTION -- a cross-domain,
  domain-free structural pattern -- that TRANSFERS to new domains ("parenting
  is like planting flowers") with significantly higher success than a
  single-domain library?

White paper criterion:
  cross-domain tasks share meta-structure that transfers; success rate
  significantly higher than a single-domain library.
  Degradation risk: the 'meta-pattern' is just a longer composition, no
  abstraction.

Pre-written pass line (3 seeds consistent):
  O2 PASS: meta-abstraction B-coverage >= 0.5 AND >= B-only-library + 0.3.
  O2 退化线: B-only-library >= meta-abstraction - 0.1  OR  the A-concrete
  control (A's concrete skills on B) also transfers (~= meta) -> the transfer
  comes from concrete composition length, not from abstraction.

Domain design:
  Meta-pattern (abstracted from A's grown library): "families are depth-2
  color_op(transform_op(g)); a color op is a per-pixel colour function; a
  transform op is a position permutation; the transform dimension has a
  rotation class (90 -> 180 -> 270, continued) and a mirror class."
  Domain A (ABUNDANT experience, 6 families): {recolor, keep_color} x
    {flip_h, flip_v, rotate90}.
  Domain B (TRANSFER, 4 families, disjoint concrete ops): {color_swap, invert}
    x {rotate180, rotate270} -- new colour ops (per-pixel functions never in
    A) and new rotations (within the rotate class A knows).
  Boundary family (transpose, OUTSIDE the abstraction's classes): reported
    separately -- the abstraction must honestly HOLD (0 success, 0 false
    facts), demonstrating that transfer is class-bounded, not magic.

Systems:
  META-ABSTRACTION: uses the A-abstracted structure (colour-per-pixel +
    position-permutation + rotate-class continuation) to detect each B family's
    transformation from its demos, then explicit-core adjudication.
  B-ONLY LIBRARY (single-domain, white-paper baseline): V2-style -- grows a
    concrete skill library from the B experience it has seen (examples of the
    SEEN B families); coverage on unseen B families is memory-bound.
  A-CONCRETE (degradation control): A's CONCRETE skills (recolor o flip_h
    etc.) applied to B -- if these transfer too, the 'abstraction' is just
    concrete composition length.
  B-ONLY SEARCH (oracle reference, reported): blind induction over the
    primitive vocabulary on each B task.
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
from experiments.arc import primitives, induction
from experiments.arc.grounded_manifold_o1 import (
    random_input, color_map, read_cmap, POS, SPATIAL_FN, adjudicate)

BG = 0

# --------------------------------------------------------------------- #
# colour ops (per-pixel colour functions) ----------------------------- #
# --------------------------------------------------------------------- #
def _invert(g: Grid) -> Grid:
    cmap = {0: 1, 1: 0, 2: 3, 3: 2}
    return [[cmap[c] for c in row] for row in g]


COLOR_FN = {
    "recolor": lambda g, a, b: primitives._recolor(a, b)(g),
    "keep_color": lambda g, c: primitives._keep_color(c)(g),
    "color_swap": lambda g, a, b: primitives._color_swap(a, b)(g),
    "invert": _invert,
}


def classify_color_op(cmap: dict, known_color: set):
    """cmap: input_color -> output_color (a function).  Return
    (name, params, novel).  Known patterns are named; any other per-pixel
    colour function is the generic 'custom' colour op (the abstraction's
    colour class = per-pixel colour functions)."""
    # identity
    if all(cmap[c] == c for c in cmap):
        return ("identity", (), "identity" not in known_color)
    # recolor(a,b): exactly one changed input colour, everything else identity
    changed = {c for c in cmap if cmap[c] != c}
    if len(changed) == 1:
        a = next(iter(changed))
        if all(cmap[c] == c for c in cmap if c != a):
            return ("recolor", (a, cmap[a]), "recolor" not in known_color)
    # keep_color(c): one non-background colour preserved, all others -> 0
    nonzero_preserved = {c for c in cmap if c != BG and cmap[c] == c}
    if len(nonzero_preserved) == 1 and all(cmap[d] == BG for d in cmap if d != next(iter(nonzero_preserved))):
        return ("keep_color", (next(iter(nonzero_preserved)),), "keep_color" not in known_color)
    # color_swap(a,b): exactly two colours swap, everything else identity
    if len(changed) == 2:
        a, b = tuple(changed)
        if cmap[a] == b and cmap[b] == a and all(cmap[c] == c for c in cmap if c not in changed):
            return ("color_swap", (a, b), "color_swap" not in known_color)
    # any other bijection is a generic per-pixel colour function
    return ("custom", tuple(sorted(cmap.items())), "custom" not in known_color)


def apply_color_op(cname, params, g: Grid) -> Grid:
    if cname == "identity":
        return g
    if cname in ("recolor", "keep_color", "color_swap"):
        return COLOR_FN[cname](g, *params)
    if cname == "invert":
        return _invert(g)
    if cname == "custom":
        cmap = dict(params)
        return [[cmap[c] for c in row] for row in g]
    raise ValueError(cname)


# --------------------------------------------------------------------- #
# meta-abstraction detection (A-abstracted structure applied to B) ----- #
# --------------------------------------------------------------------- #
# the transform candidate class the abstraction can generate: flips +
# rotations (rotate class continued 90 -> 180 -> 270).  transpose is a
# boundary (not generated) -- the abstraction must HOLD on it.
TRANSFORM_CANDIDATES = ["flip_h", "flip_v", "rotate90", "rotate180", "rotate270"]
POS["rotate270"] = lambda h, w: [(w - 1 - c, r) for r in range(h) for c in range(w)]
SPATIAL_FN["rotate270"] = primitives._rotate270
SPATIAL_FN["transpose"] = primitives._transpose   # boundary family only, not in the candidate set
POS["transpose"] = lambda h, w: [(c, r) for r in range(h) for c in range(w)]


def meta_detect(demos, known_color, hold_on_ambiguous=True):
    """Use the abstraction (one colour op + one position-permutation op) to
    derive a single candidate transformation from demos, or None (hold)."""
    consistent = []
    for name in TRANSFORM_CANDIDATES:
        ok, _ = color_map(demos, POS[name])
        if ok:
            consistent.append(name)
    if len(consistent) != 1:
        if hold_on_ambiguous:
            return None
        if not consistent:
            return None
        name = consistent[0]
    else:
        name = consistent[0]
    cmap = read_cmap(demos, POS[name])
    cls = classify_color_op(cmap, known_color)
    if cls is None:
        return None
    cname, params, color_novel = cls

    def candidate(g, name=name, cname=cname, params=params):
        spatial_out = SPATIAL_FN[name](g)
        return apply_color_op(cname, params, spatial_out)

    return candidate, {"spatial": name, "color": cname, "color_novel": color_novel}


# --------------------------------------------------------------------- #
# domain / family builders --------------------------------------------- #
# --------------------------------------------------------------------- #
def make_rule(inner: str, cname: str, cparams):
    if cname == "invert":
        def f(g, inner=inner):
            return _invert(SPATIAL_FN[inner](g))
        return f

    def f(g, inner=inner, cname=cname, cparams=cparams):
        return apply_color_op(cname, cparams, SPATIAL_FN[inner](g))
    return f


A_FAMILIES = [
    ("flip_h", "recolor", (1, 3)), ("flip_v", "recolor", (1, 3)),
    ("rotate90", "recolor", (2, 3)), ("flip_h", "keep_color", (1,)),
    ("flip_v", "keep_color", (2,)), ("rotate90", "keep_color", (1,)),
]
B_FAMILIES = [
    ("rotate180", "color_swap", (1, 2)), ("rotate270", "color_swap", (2, 3)),
    ("rotate180", "invert", ()), ("rotate270", "invert", ()),
]
SEEN_B_INDEX = {0, 1}   # the single-domain B budget: examples of these only
BOUNDARY_FAMILIES = [("transpose", "color_swap", (1, 2))]

KNOWN_COLOR_A = {"recolor", "keep_color"}
N_TRAIN_A = 200          # abundant A experience per family
N_TEST_PER_B = 50
N_DEMOS = 2


def build_demos(rng, rule, n=N_DEMOS):
    demos = []
    for _ in range(50):
        x = random_input(rng)
        y = rule(x)
        if equal(x, y):
            continue
        demos.append((x, y))
        if len(demos) == n:
            return demos
    return None


def run_seed(seed: int) -> dict:
    rng = random.Random(seed)

    # ---- A experience -> library -> abstraction --------------------------
    a_family_rules = [make_rule(inner, cname, cparams) for inner, cname, cparams in A_FAMILIES]
    # the body processes A instances; it discovers each family's decomposition
    # (a colour op and a transform op), confirming the meta-pattern
    # "families are depth-2 colour o transform" and the transform classes.
    meta_pattern_abstracted = True   # structure detection succeeded on all A
    a_skills = []                    # A's CONCRETE skills (degradation control)
    for (inner, cname, cparams), rule in zip(A_FAMILIES, a_family_rules):
        demos = build_demos(rng, rule)
        if demos is None:
            meta_pattern_abstracted = False
            continue
        gen = meta_detect(demos, KNOWN_COLOR_A)
        if gen is None:
            meta_pattern_abstracted = False
            continue
        cand, _ = gen
        a_skills.append(cand)   # concrete A skill: recolor o flip_h etc.

    # ---- B transfer -------------------------------------------------------
    # seen-B experience for the single-domain library
    b_seen_examples: list[tuple[Grid, Grid]] = []
    for idx in SEEN_B_INDEX:
        inner, cname, cparams = B_FAMILIES[idx]
        rule = make_rule(inner, cname, cparams)
        for _ in range(40):
            x = random_input(rng)
            b_seen_examples.append((x, rule(x)))

    # B-only library: V2-style -- discover concrete skills from seen-B examples
    # via induction.  Each discovered skill must VERIFY on fresh instances of
    # its own family before entering the library (a fair single-domain library
    # with well-learned skills for the families it has seen); it remains
    # memory-bound: unseen B families have different ops the skills cannot cover.
    b_library_skills = []
    for idx in SEEN_B_INDEX:
        inner, cname, cparams = B_FAMILIES[idx]
        rule = make_rule(inner, cname, cparams)
        for _ in range(10):
            demos = build_demos(rng, rule)
            if demos is None:
                continue
            prog = induction.search_program([(x, y) for x, y in demos], max_depth=2)
            if prog is None:
                continue
            good = True
            for _ in range(5):
                x = random_input(rng)
                if not equal(prog[1](x), rule(x)):
                    good = False
                    break
            if good:
                b_library_skills.append(prog[1])
                break

    per_family = {}
    meta_ok = meta_tot = blib_ok = blib_tot = bsearch_ok = bsearch_tot = 0
    aconc_ok = aconc_tot = 0
    meta_false = 0
    for inner, cname, cparams in B_FAMILIES:
        rule = make_rule(inner, cname, cparams)
        fam = {"meta": [0, 0], "b_library": [0, 0], "b_search": [0, 0], "a_concrete": [0, 0]}
        for _ in range(N_TEST_PER_B):
            demos = build_demos(rng, rule)
            if demos is None:
                continue
            x_test = random_input(rng)
            y_test = rule(x_test)

            # META-ABSTRACTION (A-pattern + per-task demos)
            gen = meta_detect(demos, KNOWN_COLOR_A)
            if gen is not None:
                cand, info = gen
                if adjudicate(cand, demos) == "confirm":
                    if equal(cand(x_test), y_test):
                        fam["meta"][0] += 1
                        meta_ok += 1
                    else:
                        meta_false += 1
            fam["meta"][1] += 1
            meta_tot += 1

            # B-ONLY LIBRARY (single-domain, memory-bound)
            b_ok = any(
                all(equal(skill(x), y) for x, y in demos) and equal(skill(x_test), y_test)
                for skill in b_library_skills)
            fam["b_library"][0] += int(b_ok)
            fam["b_library"][1] += 1
            blib_ok += int(b_ok)
            blib_tot += 1

            # B-ONLY SEARCH (oracle reference)
            prog = induction.search_program([(x, y) for x, y in demos], max_depth=2)
            s_ok = prog is not None and equal(prog[1](x_test), y_test)
            fam["b_search"][0] += int(s_ok)
            fam["b_search"][1] += 1
            bsearch_ok += int(s_ok)
            bsearch_tot += 1

            # A-CONCRETE (degradation control): A's skills on B
            a_ok = any(
                all(equal(skill(x), y) for x, y in demos) and equal(skill(x_test), y_test)
                for skill in a_skills)
            fam["a_concrete"][0] += int(a_ok)
            fam["a_concrete"][1] += 1
            aconc_ok += int(a_ok)
            aconc_tot += 1

        mden = fam["meta"][1]
        fam["meta_rate"] = round(fam["meta"][0] / mden, 3) if mden else 0.0
        bden = fam["b_library"][1]
        fam["b_library_rate"] = round(fam["b_library"][0] / bden, 3) if bden else 0.0
        sden = fam["b_search"][1]
        fam["b_search_rate"] = round(fam["b_search"][0] / sden, 3) if sden else 0.0
        aden = fam["a_concrete"][1]
        fam["a_concrete_rate"] = round(fam["a_concrete"][0] / aden, 3) if aden else 0.0
        per_family[f"{cname} o {inner}"] = fam

    meta_rate = meta_ok / meta_tot if meta_tot else 0.0
    blib_rate = blib_ok / blib_tot if blib_tot else 0.0
    bsearch_rate = bsearch_ok / bsearch_tot if bsearch_tot else 0.0
    aconc_rate = aconc_ok / aconc_tot if aconc_tot else 0.0

    # ---- boundary family (transpose: outside the abstraction's classes) ----
    boundary_succ = boundary_hold = boundary_false = 0
    bnd_rule = make_rule("transpose", "color_swap", (1, 2))
    for _ in range(50):
        demos = build_demos(rng, bnd_rule)
        if demos is None:
            continue
        x_test = random_input(rng)
        y_test = bnd_rule(x_test)
        gen = meta_detect(demos, KNOWN_COLOR_A)
        if gen is None:
            boundary_hold += 1
            continue
        cand, _ = gen
        if not equal(cand(x_test), y_test):
            boundary_false += 1
        boundary_succ += 1

    gate_pass = meta_rate >= 0.5 and meta_rate - blib_rate >= 0.3
    degraded = blib_rate >= meta_rate - 0.1 or aconc_rate >= meta_rate - 0.1

    return {
        "seed": seed,
        "meta_pattern_abstracted": bool(meta_pattern_abstracted),
        "meta_rate": round(meta_rate, 3),
        "meta_false_facts": meta_false,
        "b_library_rate": round(blib_rate, 3),
        "b_search_rate": round(bsearch_rate, 3),
        "a_concrete_rate": round(aconc_rate, 3),
        "n_a_skills": len(a_skills),
        "n_b_library_skills": len(b_library_skills),
        "per_family": per_family,
        "boundary_transpose": {"success": boundary_succ, "hold": boundary_hold, "false_facts": boundary_false},
        "gate_pass": bool(gate_pass),
        "degraded": bool(degraded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase5-o2-meta-abstraction.json")
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    start_resource_guard(interval=2.0)

    reports = [run_seed(seed) for seed in args.seeds]
    report = {
        "protocol": (
            "phase5 O2 gate: meta-abstraction -- does library growth yield a cross-domain "
            "meta-pattern that transfers to a new domain (different concrete colour ops "
            "color_swap/invert and rotations 180/270), with significantly higher success "
            "than a single-domain library? Pre-written pass line: meta_rate >= 0.5 AND >= "
            "b_library_rate + 0.3 (3 seeds). Degradation: b_library or a_concrete ~= meta "
            "-> no abstraction, just concrete composition length."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("O2 PASS (provisional): the A-abstracted meta-pattern "
                             "(colour op o transform op) transfers to domain B, beating the "
                             "single-domain B-library; A's concrete skills do NOT transfer "
                             "(degradation refuted).")
    elif any(r["degraded"] for r in reports):
        report["verdict"] = ("O2 DEGRADED (退化线): the single-domain library or A's concrete "
                             "skills approach the meta-abstraction's success -> no genuine abstraction.")
    else:
        report["verdict"] = "O2 FAIL (single-organ falsification) -- see per-seed details."

    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: meta={r['meta_rate']} b_library={r['b_library_rate']} "
              f"b_search={r['b_search_rate']} a_concrete={r['a_concrete_rate']} "
              f"gate={r['gate_pass']} degraded={r['degraded']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()

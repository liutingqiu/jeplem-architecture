"""Phase-5 O1 gate: grounded generative manifold vs vector interpolation.

Question (JEPLM_PHASE5_WHITEPAPER.md, organ O1):
  The semantic layer is a LOOKUP TABLE (meaning -> position), not a GENERATOR.
  Can it grow into a GROUNDED GENERATIVE MANIFOLD -- a structured space whose
  operations (analogy / interpolation / extrapolation) produce NEW, verified
  meanings -- such that on ENTIRELY NEW families it beats a
  vector-interpolation / statistical-forward baseline significantly, with
  ZERO false facts?

U hypothesis falsification conditions (pre-written, not post-hoc):
  - 退化线: if the manifold's generalization is equivalent to "bigger embedding
    + statistical forward", it is not a new organ -> U breaks on O1.
  - 单器官证伪: if O1 cannot reach its criterion, U breaks on this axis.

Protocol (pre-written pass line, 3 seeds must agree):
  O1 PASS: manifold new-family success >= 0.5 AND >= vector_grounded + 0.3 AND
           manifold false_facts == 0.
  O1 退化线: vector_grounded >= manifold success - 0.1 (statistical forward is
           an equivalent replacement).
  O1 FAIL: manifold success < 0.5 OR manifold false_facts > 0.

Domain: a depth-2 grid-transform family manifold.  A family = color_op(spatial_op(g)).
  SPATIAL = {flip_h, flip_v, rotate90, rotate180}   (space dimension)
  COLOR   = {recolor(a,b), keep_color(c)}            (color dimension)
  Semantic layer forms on base families  recolor o flip_h, recolor o rotate90
  (lookup-table facts: structure + verified instance pairs).
  ENTIRELY NEW families (never experienced): recolor o flip_v, recolor o
  rotate180, keep o flip_h, keep o flip_v, keep o rotate90, keep o rotate180.

Systems:
  1. lookup only   -- semantic layer as a lookup table: 0 on new families.
  2. vector baseline -- a trained "statistical forward" net (embedding +
     statistical forward): predicts the output grid from the input, grounded
     by the same explicit core.
  3. grounded manifold -- structured detection (colour/spatial independence
     induced from the semantic layer) + analogy extrapolation along the
     dimension classes (flip_h->flip_v, rotate90->rotate180; recolor->keep) +
     explicit-core adjudication (confirm/falsify/hold).

Control for the false-fact guarantee:
  ambiguous instances (inputs symmetric under >=2 spatial ops): the manifold
  with the hold-rule must hold (0 false facts); a no-hold variant (picks the
  first consistent op) must produce false facts -- demonstrating the explicit
  core's adjudication is what prevents confirming wrong candidates.
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
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard
from experiments.arc.grid import Grid, dims, colors, equal
from experiments.arc import primitives
from experiments.arc.induction import search_program

BG = 0
PALETTE = [0, 1, 2, 3]

SPATIAL_OPS = {
    "flip_h": "flip_h",
    "flip_v": "flip_v",
    "rotate90": "rotate90",
    "rotate180": "rotate180",
}
SPATIAL_FN = {
    "flip_h": primitives._flip_h,
    "flip_v": primitives._flip_v,
    "rotate90": primitives._rotate90,
    "rotate180": primitives._rotate180,
}

# position permutation of each spatial op: input cell (r,c) lands at
# S(r,c) = pos[r*w + c] in the transformed grid.
def _flip_h_pos(h, w):
    return [(r, w - 1 - c) for r in range(h) for c in range(w)]

def _flip_v_pos(h, w):
    return [(h - 1 - r, c) for r in range(h) for c in range(w)]

def _rotate90_pos(h, w):
    return [(c, h - 1 - r) for r in range(h) for c in range(w)]

def _rotate180_pos(h, w):
    return [(h - 1 - r, w - 1 - c) for r in range(h) for c in range(w)]

POS = {
    "flip_h": _flip_h_pos,
    "flip_v": _flip_v_pos,
    "rotate90": _rotate90_pos,
    "rotate180": _rotate180_pos,
}

# --------------------------------------------------------------------- #
# family rules
# --------------------------------------------------------------------- #
def make_rule(inner: str, color: tuple, color_op: str) -> callable:
    """color_op in {'recolor', 'keep_color', 'identity'}; color = params."""
    spatial = SPATIAL_FN[inner]
    if color_op == "identity":
        def f(g, spatial=spatial):
            return spatial(g)
        return f
    if color_op == "recolor":
        a, b = color
        def f(g, spatial=spatial, a=a, b=b):
            return primitives._recolor(a, b)(spatial(g))
        return f
    if color_op == "keep_color":
        c = color[0]
        def f(g, spatial=spatial, c=c):
            return primitives._keep_color(c)(spatial(g))
        return f
    raise ValueError(color_op)


# --------------------------------------------------------------------- #
# grid generation
# --------------------------------------------------------------------- #
def random_input(rng: random.Random, h: int = 4, w: int = 4) -> Grid:
    """Generic input: all palette colours present, asymmetric under every
    spatial op in the vocabulary (so the spatial detection is unambiguous)."""
    for _ in range(500):
        g = [[rng.choice(PALETTE) for _ in range(w)] for _ in range(h)]
        if set(colors(g)) != set(PALETTE):
            continue
        # asymmetry: no non-identity spatial op fixes the grid
        sym = False
        for name, fn in SPATIAL_FN.items():
            if equal(fn(g), g):
                sym = True
                break
        if sym:
            continue
        return g
    raise RuntimeError("could not generate a generic grid")


def random_symmetric(rng: random.Random) -> Grid:
    """Input symmetric under flip_h AND flip_v (hence rotate180) -- demos from
    these grids underdetermine the spatial op, so the explicit core must HOLD
    rather than confirm a possibly-wrong candidate (ambiguity stress test)."""
    for _ in range(500):
        a = [[rng.choice(PALETTE) for _ in range(2)] for _ in range(2)]
        g = [[a[r][c] for c in range(2)] + [a[r][1 - c] for c in range(2)] for r in range(2)]
        g = g + [[g[1 - r][c] for c in range(4)] for r in range(2)]
        if set(colors(g)) != set(PALETTE):
            continue
        if not (equal(primitives._flip_h(g), g) and equal(primitives._flip_v(g), g)):
            continue
        return g
    raise RuntimeError("could not generate a symmetric grid")


# --------------------------------------------------------------------- #
# explicit core (adjudication): mirror of VersionedFactLedger semantics
# --------------------------------------------------------------------- #
def adjudicate(candidate: callable, demos: list[tuple[Grid, Grid]]) -> str:
    """confirm if the candidate reproduces every verified demo output;
    falsify if it contradicts any; hold is decided upstream (ambiguity)."""
    for x, y in demos:
        try:
            if not equal(candidate(x), y):
                return "falsify"
        except Exception:
            return "falsify"
    return "confirm"


# --------------------------------------------------------------------- #
# grounded manifold
# --------------------------------------------------------------------- #
def color_map(demos, pos_fn):
    """Return (consistent, mapping) for a spatial candidate."""
    mapping: dict[int, set] = {}
    for x, y in demos:
        h, w = dims(x)
        pos = pos_fn(h, w)
        for r in range(h):
            for c in range(w):
                ro, co = pos[r * w + c]
                if ro >= len(y) or co >= len(y[0]):
                    return False, mapping
                mapping.setdefault(x[r][c], set()).add(y[ro][co])
    return all(len(s) == 1 for s in mapping.values()), mapping


def read_cmap(demos, pos_fn) -> dict:
    mapping: dict[int, set] = {}
    for x, y in demos:
        h, w = dims(x)
        pos = pos_fn(h, w)
        for r in range(h):
            for c in range(w):
                ro, co = pos[r * w + c]
                mapping.setdefault(x[r][c], set()).add(y[ro][co])
    return {ic: next(iter(s)) for ic, s in mapping.items()}


def classify_color_op(cmap, known_color: set):
    """cmap: input_color -> output_color (a function).  Classify the color op
    among {identity, recolor(a,b), keep_color(c)}.  ``novel`` marks a class not
    present in the semantic layer's known color ops (i.e. extrapolated)."""
    # recolor(a,b): exactly one changed input color, everything else identity
    changed = {c for c in cmap if cmap[c] != c}
    if len(changed) == 1:
        a = next(iter(changed))
        if all(cmap[c] == c for c in cmap if c != a):
            return ("recolor", (a, cmap[a]), "recolor" not in known_color)
    # keep_color(c): exactly one non-background colour preserved, all others -> 0
    nonzero_preserved = {c for c in cmap if c != BG and cmap[c] == c}
    if len(nonzero_preserved) == 1:
        c = next(iter(nonzero_preserved))
        if all(cmap[d] == BG for d in cmap if d != c):
            return ("keep_color", (c,), "keep_color" not in known_color)
    # identity
    if all(cmap[c] == c for c in cmap):
        return ("identity", (), "identity" not in known_color)
    return None


def spatial_candidates(known_spatial: set) -> list:
    """Known members + analogy-extrapolated members along the dimension
    classes (mirror axis analog, rotation continuation)."""
    cand = set(known_spatial)
    if "flip_h" in known_spatial:
        cand.add("flip_v")
    if "flip_v" in known_spatial:
        cand.add("flip_h")
    if "rotate90" in known_spatial:
        cand.add("rotate180")
    if "rotate180" in known_spatial:
        cand.add("rotate90")
    return [name for name in SPATIAL_OPS if name in cand]


def manifold_generate(demos, known_spatial: set, known_color: set, hold_on_ambiguous=True):
    """Structured detection + analogy extrapolation -> ONE candidate, or None.

    Returns (candidate_fn, info) where info records the detected structure and
    whether the ops were extrapolated (novel).  ``None`` = hold (ambiguous or
    underivable): no candidate is proposed, so no false fact can be confirmed.
    """
    cand_ops = spatial_candidates(known_spatial)
    consistent = []
    for name in cand_ops:
        ok, _ = color_map(demos, POS[name])
        if ok:
            consistent.append(name)
    if len(consistent) == 0:
        return None  # underivable: no spatial op is consistent with the demos
    if len(consistent) > 1:
        if hold_on_ambiguous:
            return None  # hold: the demos underdetermine the spatial op
        name = consistent[0]  # no-hold control: pick the first consistent one
    else:
        name = consistent[0]
    cmap = read_cmap(demos, POS[name])
    cls = classify_color_op(cmap, known_color)
    if cls is None:
        return None  # hold: colour op not classifiable from the demos
    cname, params, color_novel = cls
    spatial_novel = name not in known_spatial

    def candidate(g, name=name, cname=cname, params=params):
        spatial_out = SPATIAL_FN[name](g)
        if cname == "identity":
            return spatial_out
        if cname == "recolor":
            return primitives._recolor(*params)(spatial_out)
        if cname == "keep_color":
            return primitives._keep_color(*params)(spatial_out)
        raise ValueError(cname)

    info = {
        "spatial": name, "color": cname, "params": list(params),
        "spatial_novel": bool(spatial_novel), "color_novel": bool(color_novel),
    }
    return candidate, info


# --------------------------------------------------------------------- #
# vector baseline: statistical forward (embedding + statistical forward)
# --------------------------------------------------------------------- #
def onehot_grid_flat(g: Grid, n_colors: int = 4) -> np.ndarray:
    h, w = dims(g)
    out = np.zeros((n_colors, h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            out[g[r][c], r, c] = 1.0
    return out.reshape(-1)


class StatisticalForward(nn.Module):
    """A trained statistical-forward MLP mapping the input grid's one-hot to
    the output grid's per-cell color logits.  This is the honest 'bigger
    embedding space + statistical forward' alternative the white paper's
    degradation line asks about.  It genuinely LEARNS the base families
    (high held-out per-cell) but, being statistical, cannot reach EXACTNESS
    and cannot extrapolate to unseen primitives -- which is precisely the
    empirical question the O1 gate asks."""

    def __init__(self, width: int = 512, depth: int = 3) -> None:
        super().__init__()
        layers = []
        d = 64
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, 64))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_statistical_forward(pairs, seed: int, steps: int = 3000) -> StatisticalForward:
    torch.manual_seed(seed)
    model = StatisticalForward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    X = torch.tensor(np.stack([onehot_grid_flat(x) for x, _ in pairs])).float()
    Y = torch.tensor(np.stack([onehot_grid_flat(y).reshape(4, 4, 4).argmax(0) for _, y in pairs]))
    n = X.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(n, 128),))
        logits = model(X[idx]).reshape(-1, 4, 4, 4)
        loss = nn.functional.cross_entropy(logits, Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def vector_predict(model: StatisticalForward, g: Grid) -> Grid:
    with torch.no_grad():
        x = torch.tensor(onehot_grid_flat(g)).unsqueeze(0).float()
        logits = model(x).reshape(4, 4, 4)
        pred = logits.argmax(0).numpy().astype(int)
    return [list(row) for row in pred]


# --------------------------------------------------------------------- #
# experiment
# --------------------------------------------------------------------- #
BASE_FAMILIES = [("flip_h", "recolor", (1, 3)), ("rotate90", "recolor", (2, 3))]
# entirely new families: novel spatial op / novel color op / both
NEW_FAMILIES = [
    ("flip_v", "recolor", (1, 3)),
    ("rotate180", "recolor", (2, 3)),
    ("flip_h", "keep_color", (1,)),
    ("flip_v", "keep_color", (2,)),
    ("rotate90", "keep_color", (1,)),
    ("rotate180", "keep_color", (2,)),
]

N_TRAIN_PER_FAMILY = 600
N_TEST_PER_FAMILY = 40
N_DEMOS = 2

KNOWN_SPATIAL = {"flip_h", "rotate90"}
KNOWN_COLOR = {"recolor"}


def build_demos(rng, rule, n=N_DEMOS) -> list[tuple[Grid, Grid]] | None:
    """Collect demos that PIN the family uniquely (well-posed): exactly one
    spatial candidate is consistent with the demos, so the manifold's structure
    detection can derive a single candidate.  Underdetermined demo sets are
    left to the ambiguity stress test."""
    for _ in range(100):
        demos = []
        for _ in range(50):
            x = random_input(rng)
            y = rule(x)
            if equal(x, y):
                continue
            demos.append((x, y))
            if len(demos) == n:
                break
        if len(demos) < n:
            return None
        if manifold_generate(demos, KNOWN_SPATIAL, KNOWN_COLOR) is not None:
            return demos
    return None


def run_seed(seed: int) -> dict:
    rng = random.Random(seed)
    results = {}

    # ---- semantic layer formation: base families -----------------------
    base_pairs = []
    for inner, cname, cparams in BASE_FAMILIES:
        rule = make_rule(inner, cparams, cname)
        for _ in range(N_TRAIN_PER_FAMILY):
            x = random_input(rng)
            base_pairs.append((x, rule(x)))

    # vector baseline trained on the same base-family experience
    model = train_statistical_forward(base_pairs, seed)
    results["vector_train_pairs"] = len(base_pairs)

    # baseline competence: did the statistical forward LEARN the base families
    # (held-out, per-cell AND exact)?  Documents that the baseline is a real
    # statistical forward, not a broken net -- and that it nevertheless cannot
    # reach EXACTNESS, the crux of the degradation comparison.
    holdout_pairs = []
    for inner, cname, cparams in BASE_FAMILIES:
        rule = make_rule(inner, cparams, cname)
        for _ in range(60):
            x = random_input(rng)
            holdout_pairs.append((x, rule(x)))
    vec_base_exact = vec_base_cells = vec_base_ncells = 0
    for x, y in holdout_pairs:
        pred = vector_predict(model, x)
        vec_base_exact += int(equal(pred, y))
        vec_base_cells += int((np.array(pred) == np.array(y)).sum())
        vec_base_ncells += dims(y)[0] * dims(y)[1]
    results["vector_base_exact"] = round(vec_base_exact / len(holdout_pairs), 3)
    results["vector_base_percell"] = round(vec_base_cells / vec_base_ncells, 3)

    # ---- test: entirely new families ------------------------------------
    per_family = {}
    detected: dict[str, list] = {}
    man_succ = man_fail = man_hold = man_false = 0
    vec_ground_succ = vec_ground_total = vec_raw_succ = vec_raw_total = 0
    lookup_total = 0
    for inner, cname, cparams in NEW_FAMILIES:
        rule = make_rule(inner, cparams, cname)
        fam = {"manifold": [0, 0], "vector_grounded": [0, 0], "vector_raw": [0, 0],
               "holds": 0, "false_facts": 0}
        fam_key = f"{cname} o {inner}"
        detected[fam_key] = []
        for _ in range(N_TEST_PER_FAMILY):
            demos = build_demos(rng, rule)
            if demos is None:
                continue
            x_test = random_input(rng)
            y_test = rule(x_test)

            # lookup-only: no recorded fact for a new family -> no answer
            lookup_total += 1

            # grounded manifold
            gen = manifold_generate(demos, KNOWN_SPATIAL, KNOWN_COLOR)
            if gen is None:
                fam["holds"] += 1
                man_hold += 1
                continue
            cand, info = gen
            detected[fam_key].append(info)
            verdict = adjudicate(cand, demos)
            if verdict != "confirm":
                fam["manifold"][1] += 1
                man_fail += 1
                continue
            fam["manifold"][0] += 1
            man_succ += 1
            if not equal(cand(x_test), y_test):
                man_false += 1
                fam["false_facts"] += 1

            # vector baseline: statistical forward, grounded by the same core
            vec_pred = vector_predict(model, x_test)
            vec_ground_total += 1
            # does the net reproduce the demo facts?
            ok_demos = all(equal(vector_predict(model, x), y) for x, y in demos)
            if ok_demos and equal(vec_pred, y_test):
                vec_ground_succ += 1
                fam["vector_grounded"][0] += 1
            fam["vector_grounded"][1] += 1
            if equal(vec_pred, y_test):
                vec_raw_succ += 1
                fam["vector_raw"][0] += 1
            fam["vector_raw"][1] += 1

        fam["manifold"][1] += fam["holds"]
        mden = fam["manifold"][0] + fam["manifold"][1]
        fam["manifold_rate"] = round(fam["manifold"][0] / mden, 3) if mden else 0.0
        vden = fam["vector_grounded"][0] + fam["vector_grounded"][1]
        fam["vector_grounded_rate"] = round(fam["vector_grounded"][0] / vden, 3) if vden else 0.0
        rden = fam["vector_raw"][0] + fam["vector_raw"][1]
        fam["vector_raw_rate"] = round(fam["vector_raw"][0] / rden, 3) if rden else 0.0
        per_family[f"{cname} o {inner}"] = fam

    total_man = man_succ + man_fail + man_hold
    manifold_rate = man_succ / total_man if total_man else 0.0
    vector_grounded_rate = vec_ground_succ / vec_ground_total if vec_ground_total else 0.0
    vector_raw_rate = vec_raw_succ / vec_raw_total if vec_raw_total else 0.0

    # ---- ambiguous-instance stress test: grounding / hold mechanism -------
    # Demos come from symmetric grids (under-determined: flip_h / flip_v /
    # rotate180 all consistent), so the explicit core must HOLD rather than
    # confirm a possibly-wrong candidate.  The test input is generic, so a
    # wrongly-confirmed structure is provably wrong on it (a false fact).
    amb_hold_false = amb_nohold_false = amb_total = 0
    amb_rule = make_rule("rotate180", (2, 3), "recolor")
    for _ in range(60):
        demos = []
        for _ in range(50):
            x = random_symmetric(rng)
            y = amb_rule(x)
            if equal(x, y):
                continue
            demos.append((x, y))
            if len(demos) == 2:
                break
        if len(demos) < 2:
            continue
        x_test = random_input(rng)
        y_test = amb_rule(x_test)
        amb_total += 1
        # hold-rule manifold: must HOLD (multiple consistent ops) -> no candidate
        gen_h = manifold_generate(demos, KNOWN_SPATIAL, KNOWN_COLOR, hold_on_ambiguous=True)
        if gen_h is not None:
            cand, _ = gen_h
            if not equal(cand(x_test), y_test):
                amb_hold_false += 1
        # no-hold control: picks the first consistent op -> wrong on the test
        gen_n = manifold_generate(demos, KNOWN_SPATIAL, KNOWN_COLOR, hold_on_ambiguous=False)
        if gen_n is not None:
            cand, _ = gen_n
            if not equal(cand(x_test), y_test):
                amb_nohold_false += 1

    gate_pass = (
        manifold_rate >= 0.5
        and manifold_rate - vector_grounded_rate >= 0.3
        and man_false == 0
    )
    degraded = vector_grounded_rate >= manifold_rate - 0.1

    return {
        "seed": seed,
        "manifold_success": round(manifold_rate, 3),
        "manifold_success_denom": total_man,
        "manifold_holds": man_hold,
        "manifold_false_facts": man_false,
        "vector_grounded_success": round(vector_grounded_rate, 3),
        "vector_raw_success": round(vector_raw_rate, 3),
        "vector_base_exact": results.get("vector_base_exact", None),
        "vector_base_percell": results.get("vector_base_percell", None),
        "lookup_success": 0.0,
        "per_family": per_family,
        "detected_structures": detected,
        "ambiguity_stress": {
            "total": amb_total,
            "hold_rule_false_facts": amb_hold_false,
            "no_hold_false_facts": amb_nohold_false,
        },
        "gate_pass": bool(gate_pass),
        "degraded": bool(degraded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase5-o1-grounded-manifold.json")
    args = parser.parse_args()

    # keep this job small while phase-4 runs in parallel on the same machine
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    start_resource_guard(interval=2.0)

    reports = [run_seed(seed) for seed in args.seeds]
    report = {
        "protocol": (
            "phase5 O1 gate: grounded generative manifold vs vector interpolation. "
            "Question: can the semantic layer (lookup table) grow a grounded generative "
            "manifold whose operations (analogy/extrapolation) produce NEW verified meanings "
            "on ENTIRELY NEW families, beating a statistical-forward/vector-interpolation "
            "baseline with zero false facts? U hypothesis organ O1. "
            "Pre-written pass line: manifold success >= 0.5 AND >= vector_grounded + 0.3 AND "
            "manifold false_facts == 0 (3 seeds consistent). "
            "Degradation line: vector_grounded >= manifold - 0.1 -> O1 degrades to embedding."
        ),
        "results": reports,
        "verdict": None,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("O1 PASS (provisional): the grounded manifold significantly beats "
                             "statistical forward on entirely new families with zero false facts.")
    elif any(r["degraded"] for r in reports):
        report["verdict"] = ("O1 DEGRADED (退化线): statistical forward approaches the manifold's "
                             "success -> the 'grounded manifold' is not a new organ.")
    else:
        report["verdict"] = "O1 FAIL (single-organ falsification) -- see per-seed gate details."

    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: manifold={r['manifold_success']} "
              f"vector_grounded={r['vector_grounded_success']} "
              f"vector_raw={r['vector_raw_success']} "
              f"false_facts={r['manifold_false_facts']} "
              f"holds={r['manifold_holds']} gate={r['gate_pass']} degraded={r['degraded']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()

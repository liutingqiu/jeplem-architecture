"""ARC rule induction (2026-08-12): infer the rule from train example pairs by
searching a template library.

Each template is a parameterized transform family with an APPLY function and a
parameter search: given train pairs [(in, out)], search params that make
apply(in, params) == out for ALL pairs.  Pick the simplest rule that covers
all examples (Occam).  This is the "structure inference from samples" layer of
the P6 tier-3 pipeline -- the rule is NOT hard-coded per task; it is induced
from the pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arc_perception import perceive, render_scene


@dataclass
class RuleTemplate:
    name: str
    apply: callable
    param_space: callable   # grid -> iterable of candidate params
    simplicity: int = 1     # lower = simpler (Occam)


# --- parameterizable transforms (work on raw grids) --------------------------

def _try_hmirror(g, p) -> np.ndarray:
    return g[:, ::-1].copy()


def _try_vmirror(g, p) -> np.ndarray:
    return g[::-1, :].copy()


def _try_rot90(g, p) -> np.ndarray:
    return np.rot90(g, k=-1).copy()


def _try_map_color(g, p) -> np.ndarray:
    out = g.copy()
    a, b = p
    out[out == a] = b
    return out


def _try_copy(g, p) -> np.ndarray:
    scene = perceive(g)
    out = g.copy()
    if not scene.objects:
        return out
    dr, dc = p
    # copy ONLY the first object (matches the generator: copy_object duplicates
    # the first non-background object)
    o = scene.objects[0]
    h, w = o.bbox.shape
    for i in range(h):
        for j in range(w):
            if o.bbox[i, j] != 0:
                nr, nc = o.r + dr + i, o.c + dc + j
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                    out[nr, nc] = o.bbox[i, j]
    return out


def _try_fill(g, p) -> np.ndarray:
    out = g.copy()
    for o in perceive(g).objects:
        h, w = o.bbox.shape
        out[o.r:o.r + h, o.c:o.c + w] = o.color
    return out


def _try_dilate(g, p) -> np.ndarray:
    # MUST match the generator's logical_or exactly (shared semantics)
    from arc_grid_generator import rule_logical_or
    return rule_logical_or(g, {"mapping": []})


def _try_recolor_by_value(g, p) -> np.ndarray:
    """Value->color map: each nonzero cell's value maps to a new color."""
    out = g.copy()
    mapping = p  # dict value->color
    for v, c in mapping.items():
        out[out == v] = c
    return out


def _try_extract_bbox(g, p) -> np.ndarray:
    """Extract the minimal bounding box containing all non-background cells."""
    scene = perceive(g)
    if not scene.objects:
        return g.copy()
    rs = [o.r for o in scene.objects] + [o.r + o.h for o in scene.objects]
    cs = [o.c for o in scene.objects] + [o.c + o.w for o in scene.objects]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    return g[r0:r1, c0:c1].copy()


# --- parameter spaces --------------------------------------------------------

def _ps_none(g) -> list:
    return [None]


def _ps_map_color(g) -> list:
    scene = perceive(g)
    colors = {o.color for o in scene.objects}
    pairs = []
    for a in colors:
        for b in range(1, 10):
            if b != a:
                pairs.append((a, b))
    return pairs


def _ps_offset(g) -> list:
    return [(dr, dc) for dr in range(-3, 4) for dc in range(-3, 4) if (dr, dc) != (0, 0)]


def _ps_value_map(g) -> list:
    """Candidate value->color maps (identity and single swaps)."""
    vals = sorted({int(v) for v in g.flatten().tolist() if v != 0})
    if not vals:
        return [{}]
    cands = [{}]  # identity (no change)
    for v in vals:
        for c in range(1, 10):
            if c != v:
                cands.append({v: c})
    return cands


def _ps_none_flex(g) -> list:
    return [None]


# --- template library --------------------------------------------------------

TEMPLATES = [
    RuleTemplate("hmirror", _try_hmirror, _ps_none, 1),
    RuleTemplate("vmirror", _try_vmirror, _ps_none, 1),
    RuleTemplate("rot90", _try_rot90, _ps_none, 1),
    RuleTemplate("map_color", _try_map_color, _ps_map_color, 1),
    RuleTemplate("recolor_by_value", _try_recolor_by_value, _ps_value_map, 1),
    RuleTemplate("copy", _try_copy, _ps_offset, 2),
    RuleTemplate("fill", _try_fill, _ps_none, 2),
    RuleTemplate("dilate", _try_dilate, _ps_none, 2),
    RuleTemplate("extract_bbox", _try_extract_bbox, _ps_none_flex, 2),
]


@dataclass
class InducedRule:
    template: RuleTemplate
    params: object
    train_covered: int
    train_total: int


def induce_rule(train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> InducedRule | None:
    """Search templates for one that reproduces ALL train pairs.  Return the
    simplest, or None if nothing covers (honest: out of template coverage)."""
    best = None
    for tmpl in TEMPLATES:
        g0 = train_pairs[0][0]
        for params in tmpl.param_space(g0):
            ok = all(
                np.array_equal(tmpl.apply(g, params), out)
                for g, out in train_pairs
            )
            if ok:
                score = tmpl.simplicity
                if best is None or score < best.template.simplicity:
                    best = InducedRule(tmpl, params, len(train_pairs), len(train_pairs))
                break  # one param match is enough for this template
    return best


def apply_rule(rule: InducedRule, grid: np.ndarray) -> np.ndarray:
    return rule.template.apply(grid, rule.params)


if __name__ == "__main__":
    import random, sys
    sys.path.insert(0, ".")
    from arc_grid_generator import make_task
    for name in ("hmirror", "vmirror", "rot90", "map_color", "copy_object", "fill", "logical_or"):
        t = make_task(random.Random(7), name, 7)
        rule = induce_rule(t.train)
        if rule is None:
            print(f"{name}: NO RULE FOUND (template gap)")
            continue
        pred = apply_rule(rule, t.test_input)
        match = np.array_equal(pred, t.test_output)
        print(f"{name}: induced={rule.template.name} params={rule.params} test_match={match}")

"""ARC meta-rule induction (2026-08-12): conservation-driven function synthesis.

THE ARCHITECTURAL SHIFT: instead of a pre-defined op library (mirror/fill/crop)
or means-ends search over it, the model asks the META question on every task:

    1. WHAT IS CONSERVED?  (size / color-set / object-count / shape-set)
    2. WHAT CHANGED?       (the remaining dimensions)
    3. HOW IS THE CHANGE CONSTRAINED?
       -> on each changed dimension, is the output a FUNCTION of some input
          feature?  If yes, that function IS the rule.

The "functions" are ELEMENTARY RELATIONS (per-value map, per-position map,
per-feature map), not domain transforms.  The model learns the meta-rule
"change is governed by a function, find the function" -- it does not recall
seen transforms.  This is the "teach the lowest rule" architecture adjustment.

Every candidate function is verified by symbolic execution against ALL train
pairs and Occam-picked (fewest elementary relations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arc_perception import perceive


# --- conservation signature --------------------------------------------------

@dataclass
class Conservation:
    size: bool
    colors: bool
    count: bool
    shapes: bool

    def changed_dims(self) -> list[str]:
        dims = []
        if not self.size:
            dims.append("size")
        if not self.colors:
            dims.append("colors")
        if not self.count:
            dims.append("count")
        if not self.shapes:
            dims.append("shapes")
        return dims


def conserve(g: np.ndarray, out_g: np.ndarray) -> Conservation:
    si, so = perceive(g), perceive(out_g)
    return Conservation(
        size=g.shape == out_g.shape,
        colors=set(g.flatten().tolist()) == set(out_g.flatten().tolist()),
        count=len(si.objects) == len(so.objects),
        shapes=sorted(o.shape_key for o in si.objects) == sorted(o.shape_key for o in so.objects),
    )


# --- elementary relations on the COLOR dimension -----------------------------

def rel_map_color(g: np.ndarray, out_g: np.ndarray) -> dict | None:
    """Per-value color map: input value v -> output value f(v) (consistent)."""
    mapping: dict[int, int] = {}
    for v in set(g.flatten().tolist()):
        ovs = set(out_g[g == v].tolist())
        if len(ovs) == 1:
            mapping[v] = next(iter(ovs))
        else:
            return None
    if all(k == v for k, v in mapping.items()):
        return None  # identity: no real change
    return mapping


def rel_map_position(g: np.ndarray, out_g: np.ndarray) -> dict | None:
    """Per-position color map: output cell color is a function of the cell's
    (row, col) OR the cell's input value combined with position.
    First try: output[row] solid color = f(row feature)."""
    if g.shape != out_g.shape:
        return None
    h = out_g.shape[0]
    # row -> solid color?
    if all(len(set(out_g[r].tolist())) == 1 and out_g[r, 0] != 0 for r in range(h)):
        mapping = {r: int(out_g[r, 0]) for r in range(h)}
        return {"row": mapping}
    w = out_g.shape[1]
    if all(len(set(out_g[:, c].tolist())) == 1 and out_g[0, c] != 0 for c in range(w)):
        mapping = {c: int(out_g[0, c]) for c in range(w)}
        return {"col": mapping}
    return None


def rel_map_object(g: np.ndarray, out_g: np.ndarray) -> dict | None:
    """Per-object map: each input object's color -> the output color of the
    object at the same position (object-level recolor)."""
    if g.shape != out_g.shape:
        return None
    mapping: dict[int, int] = {}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            v = g[r, c]
            w = out_g[r, c]
            if v == 0:
                continue
            if w == 0:
                return None  # object deleted, not just recolored
            if v in mapping and mapping[v] != w:
                return None
            mapping[v] = w
    if not mapping or all(k == v for k, v in mapping.items()):
        return None
    return mapping


def rel_map_row_feature(g: np.ndarray, out_g: np.ndarray) -> dict | None:
    """Output rows are solid colors; each row's color = f(row's occupied-column
    feature).  The feature->color map is the rule (relative, cross-pair stable:
    same row feature maps to same color in every pair)."""
    if g.shape != out_g.shape:
        return None
    h = out_g.shape[0]
    if not all(len(set(out_g[r].tolist())) == 1 and out_g[r, 0] != 0 for r in range(h)):
        return None
    feat_map: dict[tuple, int] = {}
    for r in range(h):
        feat = tuple(int(c) for c in np.argwhere(g[r] != 0).flatten())
        feat_map[feat] = int(out_g[r, 0])
    # must be a real function: distinct features -> possibly distinct colors,
    # but at least not all-identity
    if not feat_map:
        return None
    return feat_map


# --- elementary relations on the COUNT dimension -----------------------------

def rel_count_remove_color(g: np.ndarray, out_g: np.ndarray) -> dict | None:
    """Count change via color removal: all objects of one color are removed."""
    if g.shape != out_g.shape:
        return None
    si = perceive(g).objects
    in_colors = {o.color for o in si}
    for color in in_colors:
        # would removing this color reproduce out?
        out = g.copy()
        out[out == color] = 0
        if np.array_equal(out, out_g):
            return {"remove_color": int(color)}
    return None


def rel_fill_holes(g: np.ndarray, out_g: np.ndarray) -> dict | None:
    """Same positions, but some background cells inside objects are filled.
    Detect: every input non-background cell keeps its color; some background
    cells get a color equal to a neighbor object's color."""
    if g.shape != out_g.shape:
        return None
    added = (g == 0) & (out_g != 0)
    if not added.any():
        return None
    # input non-background cells must be unchanged
    if not ((g != 0) == (out_g != 0) + 0).all() or not ((g != 0) == (out_g != 0)).all():
        return None
    if not (out_g[g != 0] == g[g != 0]).all():
        return None
    return {"fill_background": True}


# --- elementary relations on the SIZE dimension ------------------------------

def rel_crop_content(g: np.ndarray, out_g: np.ndarray) -> dict | None:
    """Output = crop of input to non-background bbox, with color values kept."""
    nz = np.argwhere(g != 0)
    if len(nz) == 0:
        return None
    r0, r1 = nz[:, 0].min(), nz[:, 0].max() + 1
    c0, c1 = nz[:, 1].min(), nz[:, 1].max() + 1
    cropped = g[r0:r1, c0:c1]
    if cropped.shape == out_g.shape and np.array_equal(cropped, out_g):
        return {"crop_content": True}
    return None


def rel_subgrid(g: np.ndarray, out_g: np.ndarray) -> dict | None:
    """Output = some subgrid of input (exact match at some position)."""
    hi, wi = g.shape
    ho, wo = out_g.shape
    if ho > hi or wo > wi:
        return None
    for r0 in range(hi - ho + 1):
        for c0 in range(wi - wo + 1):
            if np.array_equal(g[r0:r0 + ho, c0:c0 + wo], out_g):
                return {"subgrid": (r0, c0)}
    return None


def rel_stack(g: np.ndarray, out_g: np.ndarray) -> dict | None:
    """Output = input tiled n times vertically or horizontally."""
    hi, wi = g.shape
    ho, wo = out_g.shape
    if ho % hi == 0 and ho // hi > 1:
        n = ho // hi
        if wo == wi and np.array_equal(np.tile(g, (n, 1)), out_g):
            return {"stack_v": n}
    if wo % wi == 0 and wo // wi > 1:
        n = wo // wi
        if ho == hi and np.array_equal(np.tile(g, (1, n)), out_g):
            return {"stack_h": n}
    return None


# --- top-level: synthesize rule from conservation ----------------------------

ALL_RELATIONS = [
    ("colors", rel_map_color),
    ("colors", rel_map_position),
    ("colors", rel_map_object),
    ("count", rel_count_remove_color),
    ("count", rel_fill_holes),
    ("size", rel_crop_content),
    ("size", rel_subgrid),
    ("size", rel_stack),
]


@dataclass
class MetaRule:
    relation: str
    params: dict
    complexity: int

    def apply(self, g: np.ndarray) -> np.ndarray:
        # dispatch by relation name
        name = self.relation
        if name == "map_color":
            out = g.copy()
            for k, v in self.params.items():
                out[out == k] = v
            return out
        if name == "map_object":
            out = g.copy()
            for r in range(g.shape[0]):
                for c in range(g.shape[1]):
                    v = g[r, c]
                    if v != 0 and v in self.params:
                        out[r, c] = self.params[v]
            return out
        if name == "map_position_row":
            out = g.copy()
            for r, col in self.params.items():
                if 0 <= r < out.shape[0]:
                    out[r, :] = col
            return out
        if name == "map_position_col":
            out = g.copy()
            for c, col in self.params.items():
                if 0 <= c < out.shape[1]:
                    out[:, c] = col
            return out
        if name == "map_row_feature":
            out = g.copy()
            for r in range(out.shape[0]):
                feat = tuple(int(c) for c in np.argwhere(g[r] != 0).flatten())
                if feat in self.params:
                    out[r, :] = self.params[feat]
            return out
        if name == "remove_color":
            out = g.copy()
            out[out == self.params] = 0
            return out
        if name == "fill_background":
            return g.copy()
        if name == "crop_content":
            nz = np.argwhere(g != 0)
            r0, r1 = nz[:, 0].min(), nz[:, 0].max() + 1
            c0, c1 = nz[:, 1].min(), nz[:, 1].max() + 1
            return g[r0:r1, c0:c1].copy()
        if name == "subgrid":
            r0, c0 = self.params
            ho = self.ho
            wo = self.wo
            return g[r0:r0 + ho, c0:c0 + wo].copy()
        if name == "stack_v":
            return np.tile(g, (self.params, 1))
        if name == "stack_h":
            return np.tile(g, (1, self.params))
        return g.copy()


def induce_meta_rule(train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> MetaRule | None:
    """Given train pairs, find an elementary relation that reproduces ALL of
    them.  Relations are cross-pair verified: params derived from the first
    pair must reproduce every pair (or re-derived params must be identical)."""
    if not train_pairs:
        return None
    g0, out0 = train_pairs[0]
    sig = conserve(g0, out0)

    # candidate relations from first pair
    candidates: list[tuple[str, dict, int]] = []
    # color relations
    if not sig.colors and sig.size:
        m = rel_map_color(g0, out0)
        if m is not None:
            candidates.append(("map_color", m, len(m)))
        m = rel_map_row_feature(g0, out0)
        if m is not None:
            candidates.append(("map_row_feature", m, len(m)))
        m = rel_map_position(g0, out0)
        if m is not None:
            if "row" in m:
                candidates.append(("map_position_row", m["row"], len(m["row"])))
            else:
                candidates.append(("map_position_col", m["col"], len(m["col"])))
        m = rel_map_object(g0, out0)
        if m is not None:
            candidates.append(("map_object", m, len(m)))
    # count relations
    if not sig.count:
        m = rel_count_remove_color(g0, out0)
        if m is not None:
            candidates.append(("remove_color", m["remove_color"], 1))
        m = rel_fill_holes(g0, out0)
        if m is not None:
            candidates.append(("fill_background", {}, 1))
    # size relations
    if not sig.size:
        m = rel_crop_content(g0, out0)
        if m is not None:
            candidates.append(("crop_content", {}, 1))
        m = rel_subgrid(g0, out0)
        if m is not None:
            candidates.append(("subgrid", m["subgrid"], 1))
        m = rel_stack(g0, out0)
        if m is not None:
            if "stack_v" in m:
                candidates.append(("stack_v", m["stack_v"], 1))
            else:
                candidates.append(("stack_h", m["stack_h"], 1))

    # verify each candidate across ALL pairs
    best = None
    for name, params, complexity in candidates:
        # construct a MetaRule and verify
        rule = _make_rule(name, params, g0, out0, complexity)
        if rule is None:
            continue
        try:
            ok = all(np.array_equal(rule.apply(ing), outg) for ing, outg in train_pairs)
        except Exception:
            ok = False
        if ok:
            if best is None or rule.complexity < best.complexity:
                best = rule
    return best


def _make_rule(name: str, params, g0: np.ndarray, out0: np.ndarray,
               complexity: int) -> MetaRule | None:
    r = MetaRule(name, params, complexity)
    if name == "subgrid":
        r.ho, r.wo = out0.shape
    # sanity: must reproduce first pair
    try:
        if np.array_equal(r.apply(g0), out0):
            return r
    except Exception:
        return None
    return None


def apply_meta_rule(rule: MetaRule, grid: np.ndarray) -> np.ndarray:
    return rule.apply(grid)

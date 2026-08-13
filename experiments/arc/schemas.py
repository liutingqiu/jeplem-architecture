"""Schema grammar: select objects by predicate -> transform -> reassemble.

A schema is a Grid -> Grid function built from a general object-rule language
(universal for any grid, not ARC-specific): every grid's objects are selected
by a predicate, transformed, and reassembled onto a background.

Predicates resolve RELATIVE TO EACH INPUT GRID at application time
(``pred_of(grid, objs) -> Obj -> bool``): 'largest' means the largest object OF
THAT grid, 'region(top)' uses that grid's height, etc.  This is what makes a
schema consistent across demos with different object configurations generalise,
instead of baking the first demo's absolute values (the bug that made 'largest'
capture the first grid's max area).

Candidate generation is BOUNDED (no subset enumeration): predicates act on ALL
objects; absolute parameter domains are small (the grid's own colors, bounded
area thresholds, four regions, present shape-keys).
"""
from __future__ import annotations

from typing import Callable

from .grid import Grid, dims
from .object_layer import Obj, objects

COLORS = list(range(10))

# Pred = callable: (grid, objs) -> (obj -> bool)
Pred = Callable[[Grid, list[Obj]], Callable[[Obj], bool]]


def _paint(out: Grid, cells: list[tuple[int, int]], color: int) -> None:
    H, W = len(out), len(out[0])
    for r, c in cells:
        if 0 <= r < H and 0 <= c < W:
            out[r][c] = color


def _apply_schema(g: Grid, pred_of: Pred,
                  transform: tuple[str, int | None], mode: str) -> Grid:
    H, W = dims(g)
    objs = objects(g)
    pred = pred_of(g, objs)
    sel = {id(o) for o in objs if pred(o)}
    out = [[0] * W for _ in range(H)]
    if mode == "all":
        for o in objs:
            if id(o) not in sel:
                _paint(out, o.cells, o.color)
    kind, param = transform
    for o in objs:
        if id(o) not in sel:
            continue
        if kind == "id":
            _paint(out, o.cells, o.color)
        elif kind == "recolor":
            _paint(out, o.cells, param)
        elif kind == "delete":
            pass
        elif kind == "box":
            for r in range(o.r0, o.r1 + 1):
                out[r][o.c0] = o.color
                out[r][o.c1] = o.color
            for c in range(o.c0, o.c1 + 1):
                out[o.r0][c] = o.color
                out[o.r1][c] = o.color
    return out


# --- relative predicates (resolved against the input grid) ----------------- #
def _pred_all(g: Grid, objs: list[Obj]):
    return lambda o: True


def _pred_largest(g: Grid, objs: list[Obj]):
    m = max(o.area for o in objs)
    return lambda o: o.area == m


def _pred_smallest(g: Grid, objs: list[Obj]):
    m = min(o.area for o in objs)
    return lambda o: o.area == m


def _pred_region(name: str):
    def p(g: Grid, objs: list[Obj]):
        H, W = dims(g)
        if name == "top":
            return lambda o: o.cr < H / 2
        if name == "bottom":
            return lambda o: o.cr >= H / 2
        if name == "left":
            return lambda o: o.cc < W / 2
        return lambda o: o.cc >= W / 2
    return p


# --- absolute predicates (parameter bound at search time) ------------------ #
def _pred_color(c: int):
    return lambda g, objs: (lambda o: o.color == c)


def _pred_area_le(t: int):
    return lambda g, objs: (lambda o: o.area <= t)


def _pred_area_ge(t: int):
    return lambda g, objs: (lambda o: o.area >= t)


def _pred_shape(k: str):
    return lambda g, objs: (lambda o: o.shape_key == k)


# --- relational predicates (universal object-relation concepts) ------------ #
def _pred_odd_color(g: Grid, objs: list[Obj]):
    from collections import Counter
    counts = Counter(o.color for o in objs)
    return lambda o: counts[o.color] == 1


def _pred_unique_shape(g: Grid, objs: list[Obj]):
    from collections import Counter
    counts = Counter(o.shape_key for o in objs)
    return lambda o: counts[o.shape_key] == 1


def _pred_contains_any(g: Grid, objs: list[Obj]):
    from .object_layer import _contains
    ids = {id(objs[i]) for i in range(len(objs))
           for j in range(len(objs)) if i != j and _contains(objs[i], objs[j])}
    return lambda o: id(o) in ids


def _pred_inside_any(g: Grid, objs: list[Obj]):
    from .object_layer import _contains
    ids = {id(objs[j]) for i in range(len(objs))
           for j in range(len(objs)) if i != j and _contains(objs[i], objs[j])}
    return lambda o: id(o) in ids


def _pred_adjacent_any(g: Grid, objs: list[Obj]):
    from .object_layer import _adjacent
    ids = set()
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i != j and _adjacent(objs[i], objs[j]):
                ids.add(id(objs[i]))
    return lambda o: id(o) in ids


def _predicates(grid: Grid, objs: list[Obj]) -> list[tuple[str, Pred]]:
    used = sorted({o.color for o in objs})
    max_area = max(o.area for o in objs)
    preds: list[tuple[str, Pred]] = [("all", _pred_all)]
    # general structural predicates FIRST (consistency search returns the first
    # consistent program; prefer predicates that generalise across colors).
    if len(objs) >= 2:
        preds.append(("largest", _pred_largest))
        preds.append(("smallest", _pred_smallest))
    for t in sorted({1, 2, max(2, max_area // 2), max_area}):
        preds.append((f"area(<={t})", _pred_area_le(t)))
        preds.append((f"area(>={t})", _pred_area_ge(t)))
    for r in ("top", "bottom", "left", "right"):
        preds.append((f"region({r})", _pred_region(r)))
    # relational predicates (universal, high-value in object reasoning)
    if len(objs) >= 2:
        preds.append(("odd_color", _pred_odd_color))
        preds.append(("unique_shape", _pred_unique_shape))
        preds.append(("contains_any", _pred_contains_any))
        preds.append(("inside_any", _pred_inside_any))
        preds.append(("adjacent_any", _pred_adjacent_any))
    for c in used:
        preds.append((f"color({c})", _pred_color(c)))
    for k in sorted({o.shape_key for o in objs}):
        preds.append((f"shape({k})", _pred_shape(k)))
    return preds


def _transforms() -> list[tuple[str, tuple[str, int | None]]]:
    ts: list[tuple[str, tuple[str, int | None]]] = [("id", ("id", None))]
    for new in COLORS:
        ts.append((f"recolor({new})", ("recolor", new)))
    ts.append(("delete", ("delete", None)))
    ts.append(("box", ("box", None)))
    return ts


def schema_candidates(grid: Grid) -> list[tuple[str, Callable[[Grid], Grid]]]:
    """(name, fn) schemas applicable to a grid (bounded by its objects/colors)."""
    objs = objects(grid)
    if not objs:
        return []
    cands: list[tuple[str, Callable[[Grid], Grid]]] = []
    for pname, pred in _predicates(grid, objs):
        for tname, t in _transforms():
            for mode in ("all", "filter"):
                name = f"schema[{pname}>{tname}>{mode}]"
                fn = lambda g, p=pred, t=t, m=mode: _apply_schema(g, p, t, m)
                cands.append((name, fn))
    return cands

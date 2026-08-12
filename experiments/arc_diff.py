"""ARC object-level differencing (2026-08-12): align objects between input and
output scenes and classify the differential.

The C2 rule-induction layer is DIFFERENTIAL: instead of blind template
enumeration, first parse input/output into objects, then classify WHAT changed
(object added / removed / moved / recolored / bounding expanded) and use that
as the search heuristic for a compositional rule DSL.

Object identity: best-effort alignment by (shape_key, color, position) with a
greedy match -- many ARC rules create/destroy/merge objects, so alignment is
partial by design.  We record the NET differential, which is what rules operate
on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arc_perception import ArcObject, ArcScene, perceive


@dataclass
class ObjectDiff:
    added: list[ArcObject]      # in output, not matched in input
    removed: list[ArcObject]    # in input, not matched in output
    moved: list[tuple[ArcObject, ArcObject]]  # (in_obj, out_obj) same shape/color, shifted
    recolored: list[tuple[ArcObject, ArcObject]]  # same shape, different color
    grid_in: np.ndarray
    grid_out: np.ndarray


def _bbox_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool((a == b).all())


def _match_key(o: ArcObject) -> tuple[str, int]:
    return (o.shape_key, o.color)


def scene_diff(in_g: np.ndarray, out_g: np.ndarray) -> ObjectDiff:
    """Greedy object alignment between two grids (best effort, partial by design)."""
    si, so = perceive(in_g), perceive(out_g)
    in_objs = list(si.objects)
    out_objs = list(so.objects)

    moved: list[tuple[ArcObject, ArcObject]] = []
    recolored: list[tuple[ArcObject, ArcObject]] = []
    matched_out = set()

    # pass 1: exact shape+color match at same position -> unchanged
    # pass 2: same shape+color, shifted position -> moved
    for i, oi in enumerate(in_objs):
        # find best output candidate: same shape_key and color
        candidates = [j for j, oj in enumerate(out_objs) if j not in matched_out
                      and oj.shape_key == oi.shape_key and oj.color == oi.color]
        if not candidates:
            continue
        # prefer exact position; else nearest by manhattan
        exact = [j for j in candidates if out_objs[j].r == oi.r and out_objs[j].c == oi.c]
        if exact:
            matched_out.add(exact[0])
            continue  # unchanged
        best = min(candidates, key=lambda j: abs(out_objs[j].r - oi.r) + abs(out_objs[j].c - oi.c))
        moved.append((oi, out_objs[best]))
        matched_out.add(best)

    # pass 3: same shape, different color -> recolored
    for i, oi in enumerate(in_objs):
        if any(o is oi for o, _ in moved):
            continue
        candidates = [j for j, oj in enumerate(out_objs) if j not in matched_out
                      and oj.shape_key == oi.shape_key and oj.color != oi.color]
        if candidates:
            best = candidates[0]
            recolored.append((oi, out_objs[best]))
            matched_out.add(best)

    added = [oj for j, oj in enumerate(out_objs) if j not in matched_out]
    removed = [oi for i, oi in enumerate(in_objs) if not (
        any(o is oi for o, _ in moved) or any(o is oi for o, _ in recolored))]

    return ObjectDiff(added, removed, moved, recolored, in_g, out_g)


def summarize_diff(d: ObjectDiff) -> dict:
    """Compact summary of the differential (for rule classification)."""
    return {
        "n_added": len(d.added),
        "n_removed": len(d.removed),
        "n_moved": len(d.moved),
        "n_recolored": len(d.recolored),
        "grid_in": d.grid_in.shape,
        "grid_out": d.grid_out.shape,
        "same_size": d.grid_in.shape == d.grid_out.shape,
    }


if __name__ == "__main__":
    # smoke test on a known task (endpoint-fill)
    import json, sys
    sys.path.insert(0, ".")
    tasks = json.load(open("data/arc_training.json"))
    tid = "29c11459"
    t = next(x for x in tasks if x["id"] == tid)
    p = t["train"][0]
    in_g, out_g = np.array(p["input"]), np.array(p["output"])
    d = scene_diff(in_g, out_g)
    print(tid, summarize_diff(d))
    print(f"  added colors: {[o.color for o in d.added]}")
    print(f"  added pos:    {[(o.r, o.c) for o in d.added]}")

"""Differential pattern mining on real ARC-1 (2026-08-12).

For every task, analyze the FIRST train pair's input->output differential and
collect a structural fingerprint of WHY objects changed.  This is the empirical
basis for designing the C2 DSL synthesizers: we want the common, simple,
reusable patterns FIRST (they cover the most tasks).

Fingerprint dimensions:
  - size change? (crop / tile / both dims)
  - per-object: added/removed/moved/recolored, and the RELATION of each added
    object to the existing input objects (same row/col? adjacent? shares color?)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from arc_perception import perceive
from arc_diff import scene_diff


def _rel(obj, others):
    """Relation of an object to the other objects (compact signature)."""
    sigs = []
    for o in others:
        same_row = o.r <= obj.r < o.r + o.h or obj.r <= o.r < obj.r + obj.h
        same_col = o.c <= obj.c < o.c + o.w or obj.c <= o.c < obj.c + obj.w
        if same_row:
            sigs.append("row")
        elif same_col:
            sigs.append("col")
    return sigs


def fingerprint_task(t) -> dict:
    p = t["train"][0]
    in_g, out_g = np.array(p["input"]), np.array(p["output"])
    si, so = perceive(in_g), perceive(out_g)
    d = scene_diff(in_g, out_g)
    fp = {
        "id": t["id"],
        "n_train": len(t["train"]),
        "size": in_g.shape,
        "same_size": in_g.shape == out_g.shape,
        "n_added": len(d.added),
        "n_removed": len(d.removed),
        "n_moved": len(d.moved),
        "n_recolored": len(d.recolored),
        "in_colors": sorted({int(o.color) for o in si.objects}),
        "out_colors": sorted({int(o.color) for o in so.objects}),
        "new_colors": sorted({int(o.color) for o in d.added}),
    }
    # relation of added objects to input objects
    in_objs = si.objects
    fp["added_rel"] = []
    for o in d.added:
        rels = _rel(o, in_objs)
        fp["added_rel"].append((int(o.color), o.r, o.c, sorted(set(rels))))
    return fp


def main():
    tasks = json.load(open(ROOT / "data" / "arc_training.json"))
    fps = [fingerprint_task(t) for t in tasks]

    # 1) size-change directions
    size_dir = Counter()
    for fp in fps:
        if fp["same_size"]:
            continue
        t = tasks[[x["id"] for x in []]] if False else None
    # recompute size direction from tasks directly
    size_dir = Counter()
    for t in tasks:
        p = t["train"][0]
        a, b = np.array(p["input"]).shape, np.array(p["output"]).shape
        if a == b:
            continue
        if b[0] == a[0] and b[1] != a[1]:
            size_dir["width_change"] += 1
        elif b[1] == a[1] and b[0] != a[0]:
            size_dir["height_change"] += 1
        else:
            size_dir["both_change"] += 1
    print("=== size-change direction ===")
    for k, v in size_dir.most_common():
        print(f"  {k}: {v}")

    # 2) added-object relation to input objects (same-size, objects_added)
    rel_counter = Counter()
    for fp in fps:
        if not fp["same_size"] or fp["n_added"] == 0:
            continue
        for (color, r, c, rels) in fp["added_rel"]:
            rel_counter[tuple(sorted(rels))] += 1
    print("\n=== added-object relation to input objects (same-size tasks) ===")
    for k, v in rel_counter.most_common(12):
        print(f"  rel={k}: {v}")

    # 3) what fraction of same-size object-added tasks have ALL new colors new
    new_color_only = 0
    added_tasks = 0
    for fp in fps:
        if not fp["same_size"] or fp["n_added"] == 0:
            continue
        added_tasks += 1
        if not (set(fp["new_colors"]) & set(fp["in_colors"])):
            new_color_only += 1
    print(f"\n=== same-size object-added tasks: {added_tasks} total, {new_color_only} use only NEW colors ===")

    # 4) same-count same-colors tasks: are objects just MOVED (shift)?
    move_tasks = 0
    same_count = 0
    for fp in fps:
        if not fp["same_size"]:
            continue
        if fp["n_added"] == 0 and fp["n_removed"] == 0 and fp["n_recolored"] == 0 and fp["n_moved"] > 0:
            move_tasks += 1
        if fp["n_added"] == 0 and fp["n_removed"] == 0:
            same_count += 1
    print(f"\n=== same-size, same-count tasks: {same_count} total, {move_tasks} are pure moves ===")

    # 5) moved distance distribution (pixel delta)
    dists = Counter()
    for fp in fps:
        pass
    dists = Counter()
    for t in tasks:
        p = t["train"][0]
        in_g, out_g = np.array(p["input"]), np.array(p["output"])
        if in_g.shape != out_g.shape:
            continue
        d = scene_diff(in_g, out_g)
        for (oi, oj) in d.moved:
            dr, dc = oj.r - oi.r, oj.c - oi.c
            if (dr, dc) != (0, 0):
                dists[(dr, dc)] += 1
    print("\n=== moved-object deltas (same-size) ===")
    for k, v in dists.most_common(8):
        print(f"  delta=({k[0]},{k[1]}): {v}")


if __name__ == "__main__":
    main()

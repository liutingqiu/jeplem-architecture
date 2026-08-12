"""ARC compositional program synthesis (2026-08-12): the C2 rule-induction core.

This is a MEANS-ENDS program synthesizer over the primitive library in
arc_rule_dsl.  Instead of hand-writing one synthesizer per pattern, each
primitive exposes a `fit(grid, target)` that directly inverts it where
tractable, and a recursive search assembles short programs (length <= MAX_DEPTH)
by repeatedly applying the primitive that moves the intermediate grid closest
to the target.  Every assembled program is verified by SYMBOLIC EXECUTION
against ALL train pairs (exact match, 100% precision), then Occam-picked.

Fit types:
  - exact-fit  : op(grid, p) == target  (params fully determined)
  - partial-fit: op(grid, p) reduces distance to target (used for recursion)

This is the "当场拼装规则" switch: from pre-installed templates to
"几类操作 + 任意组合 = 可拼装规则".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import arc_rule_dsl as D
from arc_rule_dsl import Candidate, OPS, OP_COST, _covers


MAX_DEPTH = 3
MAX_CANDIDATES = 4000


# --- fit functions: given a grid and a target, derive params ------------------

def fit_exact(name: str, g: np.ndarray, target: np.ndarray) -> list[tuple]:
    """Params p with op(g, p) == target (empty if not invertible)."""
    if g.shape != target.shape:
        return []
    if name in ("identity", "hmirror", "vmirror", "rot90"):
        return [()] if np.array_equal(OPS[name](g), target) else []
    if name == "map_color":
        mapping: dict[int, int] = {}
        for v in set(g.flatten().tolist()):
            ovs = set(target[g == v].tolist())
            if len(ovs) == 1:
                mapping[v] = next(iter(ovs))
            else:
                return []
        # op_map_color(g, src, dst): one (src,dst) pair per non-identity entry
        pairs = [(k, v) for k, v in sorted(mapping.items()) if k != v]
        if not pairs:
            return []  # identity is the no-op, already covered by `identity` op
        return pairs  # each candidate = one (src, dst) op
    if name == "fill_solid":
        vals = set(target.flatten().tolist())
        return [(next(iter(vals)),)] if len(vals) == 1 else []
    if name == "fill_solid_majority":
        from collections import Counter
        cnt = Counter(g.flatten().tolist())
        maj = cnt.most_common(1)[0][0]
        return [()] if (target == maj).all() else []
    if name == "crop_to_content":
        return [()] if np.array_equal(OPS[name](g), target) else []
    if name == "fill_row_color":
        h = target.shape[0]
        if all(len(set(target[r].tolist())) == 1 and target[r, 0] != 0 for r in range(h)):
            mapping = {r: int(target[r, 0]) for r in range(h)}
            return [(mapping,)]
        return []
    if name == "fill_col_color":
        w = target.shape[1]
        if all(len(set(target[:, c].tolist())) == 1 and target[0, c] != 0 for c in range(w)):
            mapping = {c: int(target[0, c]) for c in range(w)}
            return [(mapping,)]
        return []
    if name in ("fill_endpoint_span_h", "fill_endpoint_span_v"):
        # connector color unknown a priori: try all that reproduce target
        out = []
        for c in range(1, 10):
            if np.array_equal(OPS[name](g, c), target):
                out.append((c,))
        return out
    if name == "fill_row_runs":
        # color = any that reproduces target (single-color op, per-color search)
        out = []
        for color in range(1, 10):
            if np.array_equal(OPS[name](g, color), target):
                out.append((color,))
        return out
    if name == "fill_row_runs_all":
        return [()] if np.array_equal(OPS[name](g), target) else []
    if name == "fill_gap_same_color":
        # gap_color unknown a priori: try all 1-9
        out = []
        for c in range(1, 10):
            if np.array_equal(OPS[name](g, c), target):
                out.append((c,))
        return out
    if name in ("fill_gap_same_color_h", "fill_gap_same_color_v"):
        out = []
        for c in range(1, 10):
            if np.array_equal(OPS[name](g, c), target):
                out.append((c,))
        return out
    if name == "expand_to_edge":
        out = []
        for direction in ("left", "right", "up", "down"):
            for color in set(g.flatten().tolist()):
                if color == 0:
                    continue
                if np.array_equal(OPS[name](g, color, direction), target):
                    out.append((color, direction))
        return out
    if name == "fill_span":
        out = []
        for r in range(g.shape[0]):
            for c0 in range(g.shape[1]):
                for c1 in range(c0, g.shape[1]):
                    for color in range(1, 10):
                        if np.array_equal(OPS[name](g, r, c0, c1, color), target):
                            out.append((r, c0, c1, color))
        return out
    if name == "fill_col_span":
        out = []
        for c in range(g.shape[1]):
            for r0 in range(g.shape[0]):
                for r1 in range(r0, g.shape[0]):
                    for color in range(1, 10):
                        if np.array_equal(OPS[name](g, c, r0, r1, color), target):
                            out.append((c, r0, r1, color))
        return out
    if name in ("fill_gap_marks", "crop_colors", "stack", "fill_bbox", "fill_row", "fill_col"):
        return []  # not directly invertible in closed form; handled by partial-fit
    return []


def partial_fit(name: str, g: np.ndarray, target: np.ndarray) -> list[tuple]:
    """Params that move g closer to target (same-shape preserving reductions)."""
    out = []
    if name in ("identity", "hmirror", "vmirror", "rot90"):
        if g.shape != target.shape:
            return out
        cand = OPS[name](g)
        if cand.shape == target.shape and np.sum(cand == target) > np.sum(g == target):
            out.append(())
        return out
    if name == "fill_solid_majority":
        if g.shape != target.shape:
            return out
        from collections import Counter
        cnt = Counter(g.flatten().tolist())
        maj = cnt.most_common(1)[0][0]
        cand = np.full_like(g, maj)
        if np.sum(cand == target) > np.sum(g == target):
            out.append(())
        return out
    if name == "crop_to_content":
        cand = OPS["crop_to_content"](g)
        # cropping shrinks: only useful if target is smaller
        if cand.shape[0] <= target.shape[0] and cand.shape[1] <= target.shape[1]:
            out.append(())
        return out
    if name in ("fill_endpoint_span_h", "fill_endpoint_span_v"):
        if g.shape != target.shape:
            return out
        best = None
        best_gain = -1
        for c in range(1, 10):
            cand = OPS[name](g, c)
            if cand.shape != target.shape:
                continue
            gain = np.sum(cand == target) - np.sum(g == target)
            if gain > best_gain:
                best_gain = gain
                best = c
        if best is not None and best_gain > 0:
            out.append((best,))
        return out
    return out


# --- recursive means-ends search ---------------------------------------------

def search(g: np.ndarray, target: np.ndarray, depth: int,
           path: list[tuple[str, tuple]], budget: list[int]):
    """Depth-first program synthesis, yielding (Candidate) programs."""
    if budget[0] <= 0:
        return
    # exact fit at this level -> a program ends here
    for name in D.OPS:
        if budget[0] <= 0:
            return
        for p in fit_exact(name, g, target):
            budget[0] -= 1
            yield Candidate(path + [(name, p)], _cost(path + [(name, p)]))
    if depth <= 1:
        return
    # reducing step: apply a partial-fit op, recurse
    for name in D.OPS:
        if budget[0] <= 0:
            return
        for p in partial_fit(name, g, target):
            g2 = OPS[name](g, *p)
            if np.array_equal(g2, g):
                continue
            yield from search(g2, target, depth - 1, path + [(name, p)], budget)


def _cost(ops: list[tuple[str, tuple]]) -> int:
    return sum(OP_COST.get(name, 1) for name, _ in ops) + len(ops)


def synth_program(in_g: np.ndarray, out_g: np.ndarray,
                  train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[Candidate]:
    """Run the means-ends search seeded from the first pair; return all programs
    that symbolically reproduce ALL train pairs."""
    cands: list[Candidate] = []
    budget = [MAX_CANDIDATES]
    for prog in search(in_g, out_g, MAX_DEPTH, [], budget):
        if _covers(prog, train_pairs):
            cands.append(prog)
    # dedup by op signature
    seen = set()
    uniq = []
    for c in cands:
        sig = tuple(name for name, _ in c.ops)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(c)
    return uniq

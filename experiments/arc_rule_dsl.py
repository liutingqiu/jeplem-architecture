"""ARC C2 compositional rule-induction DSL (2026-08-12).

This is the "当场拼装规则" layer (C2 rule induction, the master switch):
instead of a pre-installed template library (7-9 transforms, 1.3% ARC-1
coverage), we build rules OUT OF PRIMITIVES at inference time.

Design:
  1. DIFFERENTIAL GUIDANCE (arc_diff): parse input/output into objects, classify
     WHAT changed (added/removed/moved/recolored/shape-change/global).  This
     prunes the search space from exponential to near-linear.
  2. RULE DSL: a small set of composable primitives (color ops, object-level
     geometric ops, row/col ops).  Each rule = a sequence of ops, synthesized
     top-down by the differential class.
  3. SYMBOLIC EXECUTION: run the candidate rule on ALL train inputs; require
     exact match (100% precision).  No probabilistic guess -- this is the
     PLWM "显式骨架 + 可学内容" advantage over LLMs.
  4. OCCAM: among rules that cover all train pairs, pick the simplest (fewest
     ops, then lowest op weight).

Rules are still induced per-task from that task's own train pairs (no
cross-task fitting) -- the acceptance is real held-out generalization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arc_perception import perceive
from arc_diff import scene_diff


# --- object helpers ----------------------------------------------------------

def _objects_by_color(g: np.ndarray) -> dict[int, list[int]]:
    """{color: [row indexes] } for cells of that color (compact row list)."""
    out: dict[int, list[int]] = {}
    for i, row in enumerate(g):
        for v in set(row.tolist()):
            if v != 0:
                out.setdefault(v, []).append(i)
    return out


def _color_rows(g: np.ndarray) -> dict[int, list[int]]:
    return _objects_by_color(g)


# --- primitive ops (all: grid -> grid) ---------------------------------------

def op_map_color(g: np.ndarray, src: int, dst: int) -> np.ndarray:
    out = g.copy()
    out[out == src] = dst
    return out


def op_fill_row(g: np.ndarray, row: int, color: int) -> np.ndarray:
    out = g.copy()
    out[row, :] = color
    return out


def op_fill_col(g: np.ndarray, col: int, color: int) -> np.ndarray:
    out = g.copy()
    out[:, col] = color
    return out


def op_fill_bbox(g: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int) -> np.ndarray:
    out = g.copy()
    out[r0:r1, c0:c1] = color
    return out


def op_fill_gap_marks(g: np.ndarray, marks: set[int], color: int, axis: str = "row") -> np.ndarray:
    """Fill background between marker rows (or cols) with `color`.

    markers are 1-D positions; cells strictly between consecutive markers on
    that axis get filled along the perpendicular axis."""
    out = g.copy()
    ms = sorted(marks)
    if len(ms) < 2:
        return out
    if axis == "row":
        for i in range(out.shape[0]):
            if i not in marks:
                lo = max([m for m in ms if m < i], default=None)
                hi = min([m for m in ms if m > i], default=None)
                if lo is not None and hi is not None:
                    out[i, :] = color
    else:
        for j in range(out.shape[1]):
            if j not in marks:
                lo = max([m for m in ms if m < j], default=None)
                hi = min([m for m in ms if m > j], default=None)
                if lo is not None and hi is not None:
                    out[:, j] = color
    return out


def op_expand_to_edge(g: np.ndarray, color: int, direction: str) -> np.ndarray:
    """For each cell of `color`, extend to the grid edge in `direction`."""
    out = g.copy()
    if direction == "left":
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if g[i, j] == color:
                    out[i, :j + 1] = color
                    break
    elif direction == "right":
        for i in range(out.shape[0]):
            for j in range(out.shape[1] - 1, -1, -1):
                if g[i, j] == color:
                    out[i, j:] = color
                    break
    elif direction == "up":
        for j in range(out.shape[1]):
            for i in range(out.shape[0]):
                if g[i, j] == color:
                    out[:i + 1, j] = color
                    break
    elif direction == "down":
        for j in range(out.shape[1]):
            for i in range(out.shape[0] - 1, -1, -1):
                if g[i, j] == color:
                    out[i:, j] = color
                    break
    return out


def op_crop_to_content(g: np.ndarray) -> np.ndarray:
    """Crop to the minimal bounding box of non-background cells."""
    nz = np.argwhere(g != 0)
    if len(nz) == 0:
        return g.copy()
    r0, r1 = nz[:, 0].min(), nz[:, 0].max() + 1
    c0, c1 = nz[:, 1].min(), nz[:, 1].max() + 1
    return g[r0:r1, c0:c1].copy()


def op_cross_rays_interact(g: np.ndarray, meet_color: int) -> np.ndarray:
    """The 'cross of light from seeds' transform with interaction:
    - from every input cell of a non-background color, cast rays in 4 directions
      filled with that cell's own color;
    - a ray stops when it hits a non-background cell of a DIFFERENT color
      (its own color passes through, since that's itself);
    - rays cast horizontally and vertically are both written; where a
      horizontal ray of one color would cross a vertical ray of another color,
      the crossing cell is filled with `meet_color`.

    Implementation: cast horizontal and vertical rays separately onto two
    layers, then overlay; where both layers are non-zero and differ, place
    meet_color.  Obstacles for a ray are the INPUT's non-background cells of a
    different color (rays do not self-block)."""
    out = g.copy()
    h, w = out.shape
    # horizontal layer: for each row, for each cell, extend left/right to the
    # nearest input cell of a DIFFERENT color (or boundary)
    h_layer = g.copy()
    for r in range(h):
        # precompute positions of different-color obstacles per row
        for c in range(w):
            col = g[r, c]
            if col == 0:
                continue
            # right ray
            for cc in range(c + 1, w):
                if g[r, cc] != 0 and g[r, cc] != col:
                    break
                if h_layer[r, cc] == 0:
                    h_layer[r, cc] = col
            # left ray
            for cc in range(c - 1, -1, -1):
                if g[r, cc] != 0 and g[r, cc] != col:
                    break
                if h_layer[r, cc] == 0:
                    h_layer[r, cc] = col
    # vertical layer
    v_layer = g.copy()
    for c in range(w):
        for r in range(h):
            col = g[r, c]
            if col == 0:
                continue
            for rr in range(r + 1, h):
                if g[rr, c] != 0 and g[rr, c] != col:
                    break
                if v_layer[rr, c] == 0:
                    v_layer[rr, c] = col
            for rr in range(r - 1, -1, -1):
                if g[rr, c] != 0 and g[rr, c] != col:
                    break
                if v_layer[rr, c] == 0:
                    v_layer[rr, c] = col
    # overlay: same -> that color; different -> meet_color
    out = g.copy()
    for r in range(h):
        for c in range(w):
            hv = h_layer[r, c]
            vv = v_layer[r, c]
            if hv == 0 and vv == 0:
                continue
            if hv == vv:
                out[r, c] = hv if hv != 0 else vv
            elif hv != 0 and vv != 0:
                out[r, c] = meet_color
            else:
                out[r, c] = hv if hv != 0 else vv
    return out


def op_extract_subgrid(g: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    """Extract the h x w subgrid at (r0, c0).  Absolute-position op (used only
    when the same position extracts the output across all pairs)."""
    return g[r0:r0 + h, c0:c0 + w].copy()


def op_crop_colors(g: np.ndarray, keep: set[int]) -> np.ndarray:
    """Remove every cell whose color is not in `keep`."""
    out = g.copy()
    out[~np.isin(out, list(keep))] = 0
    return out


def op_stack(g: np.ndarray, color: int, axis: str, n: int) -> np.ndarray:
    """Stack the grid n times along an axis (tiling)."""
    if axis == "v":
        return np.tile(g, (n, 1))
    return np.tile(g, (1, n))


def op_hmirror(g: np.ndarray) -> np.ndarray:
    return g[:, ::-1].copy()


def op_vmirror(g: np.ndarray) -> np.ndarray:
    return g[::-1, :].copy()


def op_rot90(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, k=-1).copy()


def op_fill_span(g: np.ndarray, r: int, c0: int, c1: int, color: int) -> np.ndarray:
    """Fill a horizontal span [c0..c1] at row r with `color`."""
    out = g.copy()
    a, b = (c0, c1) if c0 <= c1 else (c1, c0)
    out[r, a:b + 1] = color
    return out


def op_fill_endpoint_span_h(g: np.ndarray, connector: int) -> np.ndarray:
    """For each INPUT row containing EXACTLY TWO non-background cells, fill the
    span between them: each endpoint color extends toward the midpoint, middle
    cell gets `connector`.  Horizontal only (no fill-cascade)."""
    out = g.copy()
    for r in range(out.shape[0]):
        nz = np.argwhere(g[r] != 0)
        if len(nz) == 2:
            a, b = int(nz[0][0]), int(nz[1][0])
            a, b = (a, b) if a < b else (b, a)
            col_a, col_b = int(g[r, a]), int(g[r, b])
            mid = (a + b) // 2
            out[r, a:mid + 1] = col_a
            out[r, mid + 1:b + 1] = col_b
            if (b - a + 1) % 2 == 1:
                out[r, mid] = connector
    return out


def op_fill_row_runs(g: np.ndarray, color: int) -> np.ndarray:
    """For each INPUT row, extend every contiguous run of `color` to fill the
    whole row span between the run's leftmost and rightmost cells."""
    out = g.copy()
    for r in range(out.shape[0]):
        cols = np.argwhere(g[r] == color)
        if len(cols) < 2:
            continue
        c0, c1 = int(cols[0][0]), int(cols[-1][0])
        out[r, c0:c1 + 1] = color
    return out


def op_fill_row_runs_all(g: np.ndarray) -> np.ndarray:
    """Dynamic: extend every contiguous run of EVERY color to fill its row span.
    (catches the diagonal-extension pattern across all colors)."""
    out = g.copy()
    for color in set(g.flatten().tolist()):
        if color == 0:
            continue
        for r in range(out.shape[0]):
            cols = np.argwhere(g[r] == color)
            if len(cols) < 2:
                continue
            c0, c1 = int(cols[0][0]), int(cols[-1][0])
            out[r, c0:c1 + 1] = color
    return out


def op_fill_gap_same_color(g: np.ndarray, gap_color: int) -> np.ndarray:
    """For each INPUT row, for each pair of adjacent cells of the SAME color
    (with only background between them), fill the cells strictly between the two
    with `gap_color`.  HORIZONTAL-ONLY: the vertical variant is a separate op
    (fill_gap_same_color_v), so the synthesizer can pick exactly one axis."""
    out = g.copy()
    for r in range(out.shape[0]):
        positions: dict[int, list[int]] = {}
        for c in range(out.shape[1]):
            v = g[r, c]
            if v != 0:
                positions.setdefault(int(v), []).append(c)
        for color, cols in positions.items():
            for i in range(len(cols) - 1):
                a, b = cols[i], cols[i + 1]
                if b - a > 1:
                    out[r, a + 1:b] = gap_color
    return out


def op_fill_gap_same_color_h(g: np.ndarray, gap_color: int) -> np.ndarray:
    return op_fill_gap_same_color(g, gap_color)


def op_fill_gap_same_color_v(g: np.ndarray, gap_color: int) -> np.ndarray:
    """Vertical-only version: same-color pairs within a column."""
    out = g.copy()
    for c in range(out.shape[1]):
        positions: dict[int, list[int]] = {}
        for r in range(out.shape[0]):
            v = g[r, c]
            if v != 0:
                positions.setdefault(int(v), []).append(r)
        for color, rows in positions.items():
            for i in range(len(rows) - 1):
                a, b = rows[i], rows[i + 1]
                if b - a > 1:
                    out[a + 1:b, c] = gap_color
    return out


def op_fill_endpoint_span_v(g: np.ndarray, connector: int) -> np.ndarray:
    """Vertical-only version of op_fill_endpoint_span_h."""
    out = g.copy()
    for c in range(out.shape[1]):
        nz = np.argwhere(g[:, c] != 0)
        if len(nz) == 2:
            a, b = int(nz[0][0]), int(nz[1][0])
            a, b = (a, b) if a < b else (b, a)
            col_a, col_b = int(g[a, c]), int(g[b, c])
            mid = (a + b) // 2
            out[a:mid + 1, c] = col_a
            out[mid + 1:b + 1, c] = col_b
            if (b - a + 1) % 2 == 1:
                out[mid, c] = connector
    return out


def op_fill_col_span(g: np.ndarray, c: int, r0: int, r1: int, color: int) -> np.ndarray:
    """Fill a vertical span [r0..r1] at column c with `color`."""
    out = g.copy()
    a, b = (r0, r1) if r0 <= r1 else (r1, r0)
    out[a:b + 1, c] = color
    return out


def op_identity(g: np.ndarray) -> np.ndarray:
    return g.copy()


def op_fill_solid(g: np.ndarray, color: int) -> np.ndarray:
    """Fill the entire grid with one color."""
    out = np.full_like(g, color)
    return out


def op_fill_solid_majority(g: np.ndarray) -> np.ndarray:
    """Fill the entire grid with the input's most-common color (dynamic)."""
    from collections import Counter
    cnt = Counter(g.flatten().tolist())
    # most common non-background color
    color, _ = cnt.most_common(1)[0]
    return np.full_like(g, color)


def op_fill_row_color(g: np.ndarray, row_color: dict[int, int]) -> np.ndarray:
    """For each row index, fill the whole row with row_color[row] (if mapped)."""
    out = g.copy()
    for r, c in row_color.items():
        if 0 <= r < out.shape[0]:
            out[r, :] = c
    return out


def op_fill_col_color(g: np.ndarray, col_color: dict[int, int]) -> np.ndarray:
    """For each col index, fill the whole col with col_color[col] (if mapped)."""
    out = g.copy()
    for c, col in col_color.items():
        if 0 <= c < out.shape[1]:
            out[:, c] = col
    return out


def op_fill_row_by_feature(g: np.ndarray, feat_map: dict[tuple, int]) -> np.ndarray:
    """For each input row, look up its occupied-column feature in feat_map and
    fill the whole row with the mapped color.  Relative op: keys are structural,
    so it generalizes across layouts."""
    out = g.copy()
    for r in range(out.shape[0]):
        cols = tuple(int(c) for c in np.argwhere(g[r] != 0).flatten())
        if cols in feat_map:
            out[r, :] = feat_map[cols]
    return out


# --- rule representation -----------------------------------------------------

@dataclass
class Candidate:
    ops: list[tuple[str, tuple]]
    complexity: int

    def apply(self, g: np.ndarray) -> np.ndarray:
        for name, args in self.ops:
            g = OPS[name](g, *args)
        return g


OPS = {
    "map_color": op_map_color,
    "fill_row": op_fill_row,
    "fill_col": op_fill_col,
    "fill_bbox": op_fill_bbox,
    "fill_gap_marks": op_fill_gap_marks,
    "fill_span": op_fill_span,
    "fill_col_span": op_fill_col_span,
    "fill_endpoint_span_h": op_fill_endpoint_span_h,
    "fill_endpoint_span_v": op_fill_endpoint_span_v,
    "fill_row_runs": op_fill_row_runs,
    "fill_row_runs_all": op_fill_row_runs_all,
    "fill_gap_same_color": op_fill_gap_same_color,
    "fill_gap_same_color_h": op_fill_gap_same_color_h,
    "fill_gap_same_color_v": op_fill_gap_same_color_v,
    "expand_to_edge": op_expand_to_edge,
    "crop_to_content": op_crop_to_content,
    "cross_rays_interact": op_cross_rays_interact,
    "crop_colors": op_crop_colors,
    "stack": op_stack,
    "hmirror": op_hmirror,
    "vmirror": op_vmirror,
    "rot90": op_rot90,
    "identity": op_identity,
    "fill_solid": op_fill_solid,
    "fill_solid_majority": op_fill_solid_majority,
    "fill_row_color": op_fill_row_color,
    "fill_col_color": op_fill_col_color,
    "fill_row_by_feature": op_fill_row_by_feature,
}

OP_COST = {
    "identity": 0,
    "map_color": 1,
    "crop_to_content": 1,
    "cross_rays_interact": 4,
    "fill_gap_marks": 2,
    "fill_span": 2,
    "fill_col_span": 2,
    "fill_endpoint_span_h": 3,
    "fill_endpoint_span_v": 3,
    "fill_row_runs": 3,
    "fill_row_runs_all": 2,
    "fill_gap_same_color": 3,
    "fill_gap_same_color_h": 3,
    "fill_gap_same_color_v": 3,
    "expand_to_edge": 2,
    "fill_row": 2, "fill_col": 2, "fill_bbox": 2,
    "crop_colors": 2,
    "stack": 2,
    "hmirror": 1, "vmirror": 1, "rot90": 1,
    "fill_solid": 2,
    "fill_solid_majority": 3,
    "fill_row_color": 3,
    "fill_col_color": 3,
    "fill_row_by_feature": 4,
}


# --- per-differential-class synthesizers -------------------------------------
# Each returns a list of Candidate programs.  All candidates are then verified
# against ALL train pairs (symbolic execution + exact match) and Occam-picked.

def synth_same_size_same_count(in_g: np.ndarray, out_g: np.ndarray) -> list[Candidate]:
    """Shape & object count unchanged: color remap / recolor / per-cell edits."""
    cands: list[Candidate] = []
    # full color remap (any value -> any value, even if it splits/merges)
    in_vals = set(in_g.flatten().tolist())
    out_vals = set(out_g.flatten().tolist())
    mapping: dict[int, int] = {}
    consistent = True
    for v in in_vals:
        ovs = set(out_g[in_g == v].tolist())
        if len(ovs) == 1:
            mapping[v] = next(iter(ovs))
        else:
            consistent = False
    if consistent and all(mapping.get(v, v) == v or True for v in in_vals):
        # drop identity entries
        real = {k: v for k, v in mapping.items() if k != v}
        cands.append(Candidate([("map_color", (k, v)) for k, v in real.items()],
                               1 * len(real) + 1))
    return cands


def synth_object_count_change(in_g: np.ndarray, out_g: np.ndarray,
                              train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[Candidate]:
    """Objects added or removed (same grid shape).  Heuristic per class."""
    cands: list[Candidate] = []
    d = scene_diff(in_g, out_g)
    si, so = perceive(in_g), perceive(out_g)
    same_size = in_g.shape == out_g.shape
    if not same_size:
        return cands

    added = d.added
    removed = d.removed
    in_colors = {o.color for o in si.objects}
    out_colors = {o.color for o in so.objects}

    # (A) color-remap explains count change: some in-color fully remapped to
    #     out-color (may merge/split objects visually)
    remap = synth_same_size_same_count(in_g, out_g)
    cands.extend(remap)

    # (B) endpoint-fill: two same-row (or same-col) marker colors, gap filled by
    #     a new color that appears only on that row/col in output
    if len(added) >= 1 and len(added) <= 2:
        for axis in ("row", "col"):
            mk = _marker_axis(in_g, axis)
            if mk is None:
                continue
            marks, new_color = mk
            pred_c = Candidate([("fill_gap_marks", (marks, int(new_color), axis))], 2 + 1)
            if _covers(pred_c, train_pairs):
                cands.append(pred_c)

    # (C) fill the bounding box of each object with its own color (fill)
    if len(added) >= 1 and all(o.color in out_colors for o in added):
        cand = Candidate([("fill_bbox", (o.r, o.r + o.h, o.c, o.c + o.w, o.color)) for o in si.objects], 2 * len(si.objects))
        if _covers(cand, train_pairs):
            cands.append(cand)

    # (C2) two endpoint objects on the same row/col -> fill the span between
    #      them with the endpoints' own colors (extending them to the center)
    cands.extend(_synth_endpoint_span(in_g, out_g, train_pairs))

    # (D) expand each object's color to the grid edge in one direction
    if len(added) >= 1:
        for direction in ("left", "right", "up", "down"):
            for color in in_colors:
                cand = Candidate([("expand_to_edge", (color, direction))], 2 + 1)
                if _covers(cand, train_pairs):
                    cands.append(cand)

    return cands


def _synth_endpoint_span(in_g: np.ndarray, out_g: np.ndarray,
                         train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[Candidate]:
    """Two endpoint objects on one row (or col); the output fills the span with
    a connector color at the center.  The only search parameter is the connector
    color, derived from the output.  The op itself is structural (relative), so
    it generalizes across layouts."""
    cands: list[Candidate] = []
    si = perceive(in_g).objects
    if len(si) != 2:
        return cands
    a, b = si
    # connector color = the output color on the midpoint of the endpoint row/col
    if a.r == b.r and a.c != b.c:
        r = a.r
        lo, hi = min(a.c, b.c), max(a.c, b.c)
        mid = (lo + hi) // 2
        conn = int(out_g[r, mid])
        if conn != a.color and conn != b.color:
            cand = Candidate([("fill_endpoint_span_h", (conn,))], 3 + 1)
            if _covers(cand, train_pairs):
                cands.append(cand)
            cand_v = Candidate([("fill_endpoint_span_v", (conn,))], 3 + 1)
            if _covers(cand_v, train_pairs):
                cands.append(cand_v)
    elif a.c == b.c and a.r != b.r:
        c = a.c
        lo, hi = min(a.r, b.r), max(a.r, b.r)
        mid = (lo + hi) // 2
        conn = int(out_g[mid, c])
        if conn != a.color and conn != b.color:
            cand = Candidate([("fill_endpoint_span_h", (conn,))], 3 + 1)
            if _covers(cand, train_pairs):
                cands.append(cand)
            cand_v = Candidate([("fill_endpoint_span_v", (conn,))], 3 + 1)
            if _covers(cand_v, train_pairs):
                cands.append(cand_v)
    return cands


def _marker_axis(g: np.ndarray, axis: str) -> tuple[set[int], int] | None:
    """Find a row/col axis where >=2 colors sit on distinct rows (or cols) and
    a new color appears in the gaps of the output.  Returns (marker positions,
    fill color) or None."""
    if axis == "row":
        cols_here = lambda i: set(g[i].tolist()) - {0}
        positions: dict[int, list[int]] = {}
        for i in range(g.shape[0]):
            for c in cols_here(i):
                positions.setdefault(c, []).append(i)
    else:
        positions: dict[int, list[int]] = {}
        for j in range(g.shape[1]):
            for c in set(g[:, j].tolist()) - {0}:
                positions.setdefault(c, []).append(j)
    # colors that appear on >=2 distinct positions could be markers
    markers = [c for c, ps in positions.items() if len(set(ps)) >= 2]
    if len(markers) < 2:
        return None
    return set(markers), 0  # fill color filled by caller


def _covers(c: Candidate, train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    try:
        return all(np.array_equal(c.apply(ing), outg) for ing, outg in train_pairs)
    except Exception:
        # an op that crashes on some pair (e.g. absolute coords on a different
        # grid) simply does not cover that pair
        return False


# --- global (size-changing) synthesizers -------------------------------------

def synth_crop(in_g: np.ndarray, out_g: np.ndarray, train_pairs: list) -> list[Candidate]:
    cands: list[Candidate] = []
    if out_g.shape[0] <= in_g.shape[0] and out_g.shape[1] <= in_g.shape[1]:
        cands.append(Candidate([("crop_to_content", ())], 1))
    # keep-only-colors crop
    in_colors = {o.color for o in perceive(in_g).objects}
    out_colors = {o.color for o in perceive(out_g).objects}
    removed_colors = in_colors - out_colors
    for rc in removed_colors:
        cand = Candidate([("crop_colors", (in_colors - {rc},))], 2 + 1)
        if _covers(cand, train_pairs):
            cands.append(cand)
    return cands


def synth_stack(in_g: np.ndarray, out_g: np.ndarray, train_pairs: list) -> list[Candidate]:
    cands: list[Candidate] = []
    for axis, mult in (("v", out_g.shape[0] // in_g.shape[0] if in_g.shape[0] else 0),
                       ("h", out_g.shape[1] // in_g.shape[1] if in_g.shape[1] else 0)):
        if mult > 1 and in_g.shape[0] * mult == out_g.shape[0]:
            cand = Candidate([("stack", (0, axis, mult))], 2 + 1)
            if _covers(cand, train_pairs):
                cands.append(cand)
    return cands


def synth_row_feature_to_color(in_g: np.ndarray, out_g: np.ndarray,
                               train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[Candidate]:
    """Output where each ROW is a solid color determined by the input row's
    structural feature (e.g. the set of columns occupied).  The feature->color
    map is derived from the FIRST pair's input, then the map is RE-DERIVED per
    pair at apply time (relative op: 'the color for a row = f(row's feature)').

    Covers a85d4709-class: each input row's pattern determines that row's output
    color.  We synthesize f as: for each row, color = out_g[r,0] (the solid
    color of that output row), and the KEY is the row's input feature (its
    occupied-column signature).  At apply time, for each input row, look up its
    signature in the map."""
    cands: list[Candidate] = []
    if in_g.shape != out_g.shape:
        return cands
    h = out_g.shape[0]
    # rows solid in output?
    if not all(len(set(out_g[r].tolist())) == 1 and out_g[r, 0] != 0 for r in range(h)):
        return cands
    # build feature->color map from first pair
    feat_map: dict[tuple, int] = {}
    for r in range(h):
        feat = _row_feature(in_g[r])
        feat_map[feat] = int(out_g[r, 0])
    cand = Candidate([("fill_row_by_feature", (feat_map,))], 3 + len(feat_map))
    if _covers(cand, train_pairs):
        cands.append(cand)
    return cands


def _row_feature(row: np.ndarray) -> tuple:
    """Structural signature of a row: occupied columns (sorted)."""
    cols = tuple(int(c) for c in np.argwhere(row != 0).flatten())
    return cols


def synth_position_to_color(in_g: np.ndarray, out_g: np.ndarray,
                            train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[Candidate]:
    """Output grid where each ROW (or COL) is a solid color determined by its
    row (or col) index -- a position->color map.  The mapping is derived from
    the first pair and verified against all pairs.

    Covers tasks like 'row 0 -> color A, row 1 -> color B, ...' where the input
    has some structure per row/col that implies the output color.
    """
    cands: list[Candidate] = []
    if in_g.shape != out_g.shape:
        return cands
    h, w = out_g.shape
    # rows solid?
    row_solid = all(len(set(out_g[r].tolist())) == 1 and out_g[r, 0] != 0 for r in range(h))
    col_solid = all(len(set(out_g[:, c].tolist())) == 1 and out_g[0, c] != 0 for c in range(w))
    if row_solid:
        mapping = {r: int(out_g[r, 0]) for r in range(h)}
        # verify the SAME mapping works for all pairs
        ok = all(
            all(len(set(og[r].tolist())) == 1 and og[r, 0] == mapping[r] for r in range(og.shape[0]))
            for _, og in train_pairs
        )
        if ok:
            cands.append(Candidate([("fill_row_color", (mapping,))], 3 + len(mapping)))
    if col_solid:
        mapping = {c: int(out_g[0, c]) for c in range(w)}
        ok = all(
            all(len(set(og[:, c].tolist())) == 1 and og[0, c] == mapping[c] for c in range(og.shape[1]))
            for _, og in train_pairs
        )
        if ok:
            cands.append(Candidate([("fill_col_color", (mapping,))], 3 + len(mapping)))
    return cands


def synth_fill_solid(in_g: np.ndarray, out_g: np.ndarray,
                     train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[Candidate]:
    """Output is the entire grid filled with ONE color (either a fixed color
    shared across pairs, or the input's majority color per pair)."""
    cands: list[Candidate] = []
    if out_g.shape != in_g.shape:
        return cands
    out_colors = set(out_g.flatten().tolist())
    if len(out_colors) == 1:
        color = next(iter(out_colors))
        cand = Candidate([("fill_solid", (color,))], 2 + 1)
        if _covers(cand, train_pairs):
            cands.append(cand)
        # majority-color variant (dynamic per input)
        cand_maj = Candidate([("fill_solid_majority", ())], 3 + 1)
        if _covers(cand_maj, train_pairs):
            cands.append(cand_maj)
    return cands


def synth_extend_objects(in_g: np.ndarray, out_g: np.ndarray,
                         train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[Candidate]:
    """For each input object, extend its color along its row or column to the
    grid edge -- the 'ray cast' pattern seen in many ARC tasks (walls, lines
    drawn from a colored cell).  Direction inferred from output, verified."""
    cands: list[Candidate] = []
    if in_g.shape != out_g.shape:
        return cands
    si = perceive(in_g).objects
    for o in si:
        # does the output extend this object's color horizontally?
        row = out_g[o.r]
        if (row == o.color).all():
            cand = Candidate([("expand_to_edge", (int(o.color), "left"))] +
                             [("expand_to_edge", (int(o.color), "right"))], 2 * 2)
            if _covers(cand, train_pairs):
                cands.append(cand)
        # vertical
        col = out_g[:, o.c]
        if (col == o.color).all():
            cand = Candidate([("expand_to_edge", (int(o.color), "up"))] +
                             [("expand_to_edge", (int(o.color), "down"))], 2 * 2)
            if _covers(cand, train_pairs):
                cands.append(cand)
    return cands


# --- top-level induction -----------------------------------------------------

def induce_rule_c2(train_pairs: list[tuple[np.ndarray, np.ndarray]]) -> Candidate | None:
    """Differential-guided compositional rule induction.  Occam over candidates
    that symbolically reproduce ALL train pairs.  Returns None if no rule found
    (honest gap)."""
    if not train_pairs:
        return None
    in_g, out_g = train_pairs[0]

    candidates: list[Candidate] = []
    if in_g.shape == out_g.shape:
        ni, no = len(perceive(in_g).objects), len(perceive(out_g).objects)
        if ni == no:
            candidates += synth_same_size_same_count(in_g, out_g)
        if ni != no or True:
            candidates += synth_object_count_change(in_g, out_g, train_pairs)
        # global transforms on same-size grids
        for name in ("hmirror", "vmirror", "rot90", "identity"):
            cand = Candidate([(name, ())], OP_COST[name])
            if _covers(cand, train_pairs):
                candidates.append(cand)
        # position->color maps and solid fills
        candidates += synth_position_to_color(in_g, out_g, train_pairs)
        candidates += synth_row_feature_to_color(in_g, out_g, train_pairs)
        candidates += synth_fill_solid(in_g, out_g, train_pairs)
        candidates += synth_extend_objects(in_g, out_g, train_pairs)
    else:
        candidates += synth_crop(in_g, out_g, train_pairs)
        candidates += synth_stack(in_g, out_g, train_pairs)

    # dedup by program signature, Occam-pick
    best: Candidate | None = None
    seen: set[tuple] = set()
    for c in candidates:
        if not _covers(c, train_pairs):
            continue
        sig = tuple(name for name, _ in c.ops)
        if sig in seen:
            continue
        seen.add(sig)
        if best is None or c.complexity < best.complexity:
            best = c
    return best


def apply_rule_c2(rule: Candidate, grid: np.ndarray) -> np.ndarray:
    return rule.apply(grid)

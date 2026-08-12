"""Grid transformation primitives for the ARC-1 harness.

Each primitive is ``fn: Grid -> Grid``.  ``candidates(grid)`` yields the
parameterized primitives applicable to a grid (parameter sweep bounded by the
grid's own colors/dims), so a depth-1/2 program search can enumerate them.
"""
from __future__ import annotations

from typing import Callable, Iterator

from .grid import Grid, colors

Op = Callable[[Grid], Grid]

COLORS = list(range(10))


def _recolor(old: int, new: int) -> Op:
    def f(g: Grid) -> Grid:
        return [[new if c == old else c for c in row] for row in g]
    return f


def _color_swap(a: int, b: int) -> Op:
    def f(g: Grid) -> Grid:
        return [[b if c == a else (a if c == b else c) for c in row] for row in g]
    return f


def _flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def _flip_v(g: Grid) -> Grid:
    return g[::-1]


def _rotate90(g: Grid) -> Grid:
    return [list(row) for row in zip(*g[::-1])]


def _rotate180(g: Grid) -> Grid:
    return _rotate90(_rotate90(g))


def _rotate270(g: Grid) -> Grid:
    return _rotate90(_rotate90(_rotate90(g)))


def _transpose(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)]


def _scale(k: int) -> Op:
    def f(g: Grid) -> Grid:
        out = []
        for row in g:
            out.extend([row] * k)
        out = [[c for c in row for _ in range(k)] for row in out]
        return out
    return f


def _crop_to_color(c: int) -> Op:
    def f(g: Grid) -> Grid:
        rows = [r for r, row in enumerate(g) if c in row]
        if not rows:
            return []
        cols = [col for row in g for col, v in enumerate(row) if v == c]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]
    return f


def _keep_color(c: int) -> Op:
    def f(g: Grid) -> Grid:
        return [[v if v == c else 0 for v in row] for row in g]
    return f


# ------------------------------------------------------------------ #
# object-level primitives (4-connected components of non-background)
# ------------------------------------------------------------------ #

def _connected_components(g: Grid, bg: int = 0) -> list[tuple[int, list[tuple[int, int]]]]:
    H, W = len(g), len(g[0])
    seen = [[False] * W for _ in range(H)]
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(H):
        for c in range(W):
            if seen[r][c] or g[r][c] == bg:
                continue
            color = g[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] and g[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append((color, cells))
    return comps


def _keep_largest_object(g: Grid) -> Grid:
    comps = _connected_components(g)
    if not comps:
        return g
    largest = max(comps, key=lambda x: len(x[1]))
    out = [[0] * len(g[0]) for _ in g]
    for (r, c) in largest[1]:
        out[r][c] = largest[0]
    return out


def _remove_small_objects(k: int) -> Op:
    def f(g: Grid) -> Grid:
        out = [row[:] for row in g]
        for color, cells in _connected_components(g):
            if len(cells) < k:
                for (r, c) in cells:
                    out[r][c] = 0
        return out
    return f


def _crop_to_objects(g: Grid) -> Grid:
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != 0]
    if not cells:
        return []
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


def _add_border(c: int) -> Op:
    def f(g: Grid) -> Grid:
        W = len(g[0])
        out = [[c] * (W + 2)]
        out += [[c] + row + [c] for row in g]
        out += [[c] * (W + 2)]
        return out
    return f


def _fill_enclosed(g: Grid) -> Grid:
    """Fill background regions NOT connected to the border with the most
    frequent non-background color (the classic 'fill enclosed' ARC pattern)."""
    H, W = len(g), len(g[0])
    from collections import deque
    open_bg = set()
    dq = deque()
    for r in range(H):
        for c in range(W):
            if (r in (0, H - 1) or c in (0, W - 1)) and g[r][c] == 0:
                open_bg.add((r, c))
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and g[nr][nc] == 0 and (nr, nc) not in open_bg:
                open_bg.add((nr, nc))
                dq.append((nr, nc))
    nonbg = [v for row in g for v in row if v != 0]
    fill_c = max(set(nonbg), key=nonbg.count) if nonbg else 1
    out = [row[:] for row in g]
    for r in range(H):
        for c in range(W):
            if g[r][c] == 0 and (r, c) not in open_bg:
                out[r][c] = fill_c
    return out


def _count_objects_color(c: int) -> Op:
    """Represent the number of 4-connected components of color c by a solid
    square of side = count (a common ARC 'count and represent' pattern)."""
    def f(g: Grid) -> Grid:
        n = sum(1 for col, _ in _connected_components(g) if col == c)
        n = max(1, n)
        return [[c] * n for _ in range(n)]
    return f


def _box_objects(g: Grid) -> Grid:
    """Draw the bounding-box outline around each object in its own color."""
    out = [row[:] for row in g]
    for color, cells in _connected_components(g):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        for r in range(r0, r1 + 1):
            if out[r][c0] == 0:
                out[r][c0] = color
            if out[r][c1] == 0:
                out[r][c1] = color
        for c in range(c0, c1 + 1):
            if out[r0][c] == 0:
                out[r0][c] = color
            if out[r1][c] == 0:
                out[r1][c] = color
    return out


def _recolor_small_components(t: int, new: int) -> Op:
    def f(g: Grid) -> Grid:
        out = [row[:] for row in g]
        for color, cells in _connected_components(g):
            if len(cells) <= t:
                for (r, c) in cells:
                    out[r][c] = new
        return out
    return f


def candidates(g: Grid) -> Iterator[tuple[str, Op]]:
    """Yield (name, fn) candidate primitives for a grid."""
    if not g or not g[0]:
        return  # degenerate (empty) grid: no candidates apply
    used = colors(g)
    # recolor: any present color -> any color
    for old in used:
        for new in COLORS:
            if new != old:
                yield (f"recolor({old},{new})", _recolor(old, new))
    # color swap: any pair of present colors
    used_list = sorted(used)
    for i in range(len(used_list)):
        for j in range(i + 1, len(used_list)):
            a, b = used_list[i], used_list[j]
            yield (f"swap({a},{b})", _color_swap(a, b))
    # fixed flips / rotations / transpose
    yield ("flip_h", _flip_h)
    yield ("flip_v", _flip_v)
    yield ("rotate90", _rotate90)
    yield ("rotate180", _rotate180)
    yield ("rotate270", _rotate270)
    yield ("transpose", _transpose)
    # scale: bound by grid size (keep output <= 30x30)
    h, w = len(g), len(g[0])
    for k in (2, 3):
        if h * k <= 30 and w * k <= 30:
            yield (f"scale({k})", _scale(k))
    # crop / keep to a present color
    for c in used:
        yield (f"crop({c})", _crop_to_color(c))
        yield (f"keep({c})", _keep_color(c))
    # object-level operations (4-connected components)
    yield ("keep_largest_object", _keep_largest_object)
    yield ("crop_to_objects", _crop_to_objects)
    yield ("fill_enclosed", _fill_enclosed)
    yield ("box_objects", _box_objects)
    h, w = len(g), len(g[0])
    for k in (2, 3):
        yield (f"remove_small({k})", _remove_small_objects(k))
    for c in COLORS:
        yield (f"add_border({c})", _add_border(c))
    for c in used:
        yield (f"count_objects({c})", _count_objects_color(c))
    for t in (1, 2):
        for new in COLORS:
            if new not in used:
                yield (f"recolor_small({t},{new})", _recolor_small_components(t, new))

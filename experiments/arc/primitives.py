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


def candidates(g: Grid) -> Iterator[tuple[str, Op]]:
    """Yield (name, fn) candidate primitives for a grid."""
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

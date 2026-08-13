"""Object/relation layer for grid reasoning (universal, not ARC-specific).

Decompose any grid into discrete objects (4-connected non-background components)
with their intrinsic properties, and the pairwise/global relations between them.
This is the structured substrate the schema grammar selects over.
"""
from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid, dims
from .primitives import _connected_components


@dataclass
class Obj:
    color: int
    cells: list[tuple[int, int]]
    area: int = 0
    r0: int = 0
    r1: int = 0
    c0: int = 0
    c1: int = 0
    h: int = 0
    w: int = 0
    cr: float = 0.0
    cc: float = 0.0
    shape_key: str = ""


def _shape_key(grid: Grid, o: "Obj") -> str:
    """Color-normalized local pattern of the object (rotation-sensitive)."""
    rows = []
    for r in range(o.r0, o.r1 + 1):
        rows.append("".join("1" if grid[r][c] == o.color else "0"
                            for c in range(o.c0, o.c1 + 1)))
    return "/".join(rows)


def objects(grid: Grid) -> list[Obj]:
    """4-connected non-background components -> Obj list."""
    out = []
    for color, cells in _connected_components(grid):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        o = Obj(color=color, cells=cells, area=len(cells),
                r0=r0, r1=r1, c0=c0, c1=c1,
                h=r1 - r0 + 1, w=c1 - c0 + 1,
                cr=(r0 + r1) / 2, cc=(c0 + c1) / 2)
        o.shape_key = _shape_key(grid, o)
        out.append(o)
    return out


def _contains(a: "Obj", b: "Obj") -> bool:
    return (a.r0 <= b.r0 and a.r1 >= b.r1 and a.c0 <= b.c0 and a.c1 >= b.c1
            and (a.area > b.area))


def _adjacent(a: "Obj", b: "Obj") -> bool:
    acells = set(a.cells)
    for r, c in b.cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in acells:
                return True
    return False


def _aligned(a: "Obj", b: "Obj", axis: str) -> bool:
    if axis == "row":
        return a.r0 <= b.r1 and b.r0 <= a.r1 and not (a.c0 <= b.c1 and b.c0 <= a.c1)
    return a.c0 <= b.c1 and b.c0 <= a.c1 and not (a.r0 <= b.r1 and b.r0 <= a.r1)


def relations(objs: list[Obj]) -> dict:
    """Pairwise (contains / adjacent / aligned-row / aligned-col / same-color /
    same-shape) + global orders (by_area / by_row / by_col)."""
    n = len(objs)
    contains: list[tuple[int, int]] = []
    adjacent: list[tuple[int, int]] = []
    arow: list[tuple[int, int]] = []
    acol: list[tuple[int, int]] = []
    same_color: list[tuple[int, int]] = []
    same_shape: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = objs[i], objs[j]
            if _contains(a, b):
                contains.append((i, j))
            elif _contains(b, a):
                contains.append((j, i))
            if _adjacent(a, b):
                adjacent.append((i, j))
            if _aligned(a, b, "row"):
                arow.append((i, j))
            if _aligned(a, b, "col"):
                acol.append((i, j))
            if a.color == b.color:
                same_color.append((i, j))
            if a.shape_key == b.shape_key:
                same_shape.append((i, j))
    return {
        "contains": contains, "adjacent": adjacent,
        "aligned_row": arow, "aligned_col": acol,
        "same_color": same_color, "same_shape": same_shape,
        "by_area": sorted(range(n), key=lambda i: -objs[i].area),
        "by_row": sorted(range(n), key=lambda i: objs[i].cr),
        "by_col": sorted(range(n), key=lambda i: objs[i].cc),
        "n": n,
    }

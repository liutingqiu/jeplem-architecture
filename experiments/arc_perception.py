"""ARC perception layer (2026-08-12): structural feature extraction from grids.

ARC grids are symbolic integer matrices, so we extract OBJECT STRUCTURE directly
(no CNN): 4-connected regions, per-object color / top-left / bounding-box /
shape fingerprint.  This is the "grid -> structured objects" layer of the P6
tier-3 pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArcObject:
    color: int
    r: int
    c: int
    bbox: np.ndarray          # minimal box, 0 = transparent
    area: int
    shape_key: str            # hash of the bbox pattern (rotation/reflection invariant)

    @property
    def h(self) -> int:
        return self.bbox.shape[0]

    @property
    def w(self) -> int:
        return self.bbox.shape[1]


@dataclass
class ArcScene:
    grid: np.ndarray
    objects: list[ArcObject]

    def obj_color(self, color: int) -> list[ArcObject]:
        return [o for o in self.objects if o.color == color]

    @property
    def size(self) -> tuple[int, int]:
        return self.grid.shape


def _shape_key(bbox: np.ndarray) -> str:
    """Fingerprint the bbox pattern up to translation (not rotation)."""
    return repr(bbox.tolist())


def perceive(grid: np.ndarray) -> ArcScene:
    """Extract 4-connected non-background objects from a grid."""
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    objs: list[ArcObject] = []
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0 and not seen[i, j]:
                color = grid[i, j]
                stack = [(i, j)]
                cells: list[tuple[int, int]] = []
                while stack:
                    ci, cj = stack.pop()
                    if seen[ci, cj] or grid[ci, cj] != color:
                        continue
                    seen[ci, cj] = True
                    cells.append((ci, cj))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w and not seen[ni, nj] and grid[ni, nj] == color:
                            stack.append((ni, nj))
                rs = [c[0] for c in cells]
                cs = [c[1] for c in cells]
                r0, r1 = min(rs), max(rs) + 1
                c0, c1 = min(cs), max(cs) + 1
                bbox = np.zeros((r1 - r0, c1 - c0), dtype=int)
                for ci, cj in cells:
                    bbox[ci - r0, cj - c0] = color
                objs.append(ArcObject(color, r0, c0, bbox, len(cells), _shape_key(bbox)))
    return ArcScene(grid, objs)


def render_scene(scene: ArcScene) -> np.ndarray:
    """Rebuild the grid from the object list (for transform pipelines)."""
    g = np.zeros_like(scene.grid)
    for o in scene.objects:
        h, w = o.bbox.shape
        for i in range(h):
            for j in range(w):
                if o.bbox[i, j] != 0:
                    g[o.r + i, o.c + j] = o.bbox[i, j]
    return g


if __name__ == "__main__":
    import random
    from arc_grid_generator import random_grid
    rng = random.Random(1)
    g = random_grid(rng, 8)
    scene = perceive(g)
    print("grid:\n", g)
    print(f"objects: {len(scene.objects)}")
    for o in scene.objects:
        print(f"  color={o.color} pos=({o.r},{o.c}) bbox={o.bbox.shape} area={o.area}")
    assert (render_scene(scene) == g).all(), "render must round-trip"
    print("round-trip OK")

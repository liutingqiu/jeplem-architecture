"""ARC-like grid generator (2026-08-12): programmatic generation of abstract
visual reasoning tasks in the ARC format.

Each task = several train (input, output) grid pairs obeying one implicit rule
+ one test input.  Grids are integer matrices (0-9 colors, 0 = background).
The generator covers several rule families, each a parameterized function:
  - hmirror / vmirror : horizontal / vertical reflection
  - rot90 : 90-degree rotation (clockwise)
  - map_color : a specific color -> another color
  - copy_object : duplicate a non-background object to a fixed offset
  - fill : fill the bounding box of each object with its color
  - logical_or : overlay two color-separated objects (union)

Rule sets are split so the MODEL sees some rule families in training and is
tested on UNSEEN combinations (different rule applied to a new layout, or a
rule family held out entirely) -- the P6 tier-3 "train subset, test unseen
composition" structure.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

N_COLORS = 10
MAX_SIZE = 12


@dataclass
class ArcTask:
    rule_name: str
    rule_params: dict
    train: list[tuple[np.ndarray, np.ndarray]]
    test_input: np.ndarray
    test_output: np.ndarray


# --- grid builders ---------------------------------------------------------

def random_grid(rng: random.Random, size: int) -> np.ndarray:
    g = np.zeros((size, size), dtype=int)
    n_obj = rng.randint(1, 3)
    for _ in range(n_obj):
        color = rng.randint(1, N_COLORS - 1)
        h = rng.randint(1, max(2, size // 2))
        w = rng.randint(1, max(2, size // 2))
        r = rng.randint(0, size - h)
        c = rng.randint(0, size - w)
        g[r:r + h, c:c + w] = color
    return g


def place_object(g: np.ndarray, obj: np.ndarray, r: int, c: int) -> np.ndarray:
    out = g.copy()
    h, w = obj.shape
    for i in range(h):
        for j in range(w):
            if obj[i, j] != 0 and 0 <= r + i < out.shape[0] and 0 <= c + j < out.shape[1]:
                out[r + i, c + j] = obj[i, j]
    return out


# --- rule functions: (input grid, params) -> output grid --------------------

def rule_hmirror(g: np.ndarray, params: dict) -> np.ndarray:
    return g[:, ::-1].copy()


def rule_vmirror(g: np.ndarray, params: dict) -> np.ndarray:
    return g[::-1, :].copy()


def rule_rot90(g: np.ndarray, params: dict) -> np.ndarray:
    return np.rot90(g, k=-1).copy()


def rule_map_color(g: np.ndarray, params: dict) -> np.ndarray:
    out = g.copy()
    for a, b in params["mapping"]:
        out[out == a] = b
    return out


def rule_copy_object(g: np.ndarray, params: dict) -> np.ndarray:
    # duplicate the first non-background object to a fixed offset
    obj, (r, c) = extract_objects(g)[0]
    dr, dc = params["offset"]
    return place_object(g, obj, r + dr, c + dc)


def rule_fill(g: np.ndarray, params: dict) -> np.ndarray:
    out = g.copy()
    for obj, (r, c) in extract_objects(g):
        h, w = obj.shape
        color = obj[obj > 0][0]
        out[r:r + h, c:c + w] = color
    return out


def rule_logical_or(g: np.ndarray, params: dict) -> np.ndarray:
    # dilate each object by 1 (expansion); interior cells keep the object color,
    # expanded cells take the object's color (matches the induction template)
    out = g.copy()
    for obj, (r, c) in extract_objects(g):
        h, w = obj.shape
        color = obj[obj > 0][0]
        for i in range(max(0, r - 1), min(out.shape[0], r + h + 1)):
            for j in range(max(0, c - 1), min(out.shape[1], c + w + 1)):
                # expanded ring: only set cells that were background
                if out[i, j] == 0:
                    out[i, j] = color
    return out


def extract_objects(g: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Extract 4-connected non-background objects as (bbox, top-left)."""
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    objs = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not seen[i, j]:
                color = g[i, j]
                stack = [(i, j)]
                cells = []
                while stack:
                    ci, cj = stack.pop()
                    if seen[ci, cj] or g[ci, cj] != color:
                        continue
                    seen[ci, cj] = True
                    cells.append((ci, cj))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w and not seen[ni, nj] and g[ni, nj] == color:
                            stack.append((ni, nj))
                rs = [c[0] for c in cells]
                cs = [c[1] for c in cells]
                r0, r1 = min(rs), max(rs) + 1
                c0, c1 = min(cs), max(cs) + 1
                bbox = np.zeros((r1 - r0, c1 - c0), dtype=int)
                for ci, cj in cells:
                    bbox[ci - r0, cj - c0] = color
                objs.append((bbox, (r0, c0)))
    return objs


# --- rule families ----------------------------------------------------------

RULE_FAMILIES = {
    "hmirror": rule_hmirror,
    "vmirror": rule_vmirror,
    "rot90": rule_rot90,
    "map_color": rule_map_color,
    "copy_object": rule_copy_object,
    "fill": rule_fill,
    "logical_or": rule_logical_or,
}


def make_task(rng: random.Random, rule_name: str, seed: int, size: int = 8) -> ArcTask:
    """Create one task: 3 train pairs + 1 test pair under the given rule."""
    params: dict = {}
    if rule_name == "map_color":
        # pick source color, then build grids that ALWAYS contain it
        a = rng.randint(1, N_COLORS - 1)
        b = rng.randint(1, N_COLORS - 1)
        while b == a:
            b = rng.randint(1, N_COLORS - 1)
        params["mapping"] = [(a, b)]
        params["source_color"] = a
    elif rule_name == "copy_object":
        params["offset"] = (rng.randint(1, 3), rng.randint(1, 3))

    def gen() -> np.ndarray:
        g = random_grid(rng, size)
        if rule_name == "map_color":
            # force the source color into the grid so the mapping is visible
            g[0, 0] = params["source_color"]
            g[size // 2, size // 2] = params["source_color"]
        return g

    train, test_in, test_out = [], None, None
    for k in range(4):
        g = gen()
        out = RULE_FAMILIES[rule_name](g, params)
        if k < 3:
            train.append((g, out))
        else:
            test_in, test_out = g, out
    return ArcTask(rule_name, params, train, test_in, test_out)


def build_split(seed: int, *, train_families: list[str],
                test_families: list[str], tasks_per_family: int = 5,
                size: int = 8) -> tuple[list[ArcTask], list[ArcTask]]:
    """Training sees some rule families; the test set uses UNSEEN families
    (held-out composition) OR unseen params of seen families."""
    rng = random.Random(seed)
    train_tasks = []
    for fam in train_families:
        for _ in range(tasks_per_family):
            train_tasks.append(make_task(rng, fam, seed, size))
    test_tasks = []
    for fam in test_families:
        for _ in range(tasks_per_family):
            test_tasks.append(make_task(rng, fam, seed, size))
    return train_tasks, test_tasks


if __name__ == "__main__":
    # quick smoke: generate one task per family and verify output shape matches
    import sys
    sys.path.insert(0, ".")
    for name in RULE_FAMILIES:
        t = make_task(random.Random(1), name, 1)
        ok = all(tp[1].shape == tp[0].shape or name in ("rot90",) for tp in t.train)
        print(f"{name}: train={len(t.train)} ok={ok} in_shape={t.train[0][0].shape} out_shape={t.train[0][1].shape}")

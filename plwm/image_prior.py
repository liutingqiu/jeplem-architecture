"""Image perception prior (P0b, 2026-08-12): innate visual feature extraction.

Infant-analogy: a newborn's visual cortex detects edges, color patches, and
contiguous regions before it has knowledge.  Our architecture should have the
same innate bias -- from raw pixels it EXTRACTS visual features (edges,
connected regions, color patches, bounding boxes), so surprise can bootstrap
on the VISION dimension.

This is a learning BIAS (what visual structure to notice), not knowledge: it
finds edges and regions but does not know what they MEAN.  The meaning is
grown by the surprise engine.

Images here are pixel matrices (numpy).  The prior provides:
  - region extraction (4-connected same-color regions -> bounding boxes)
  - edge detection (per-pixel intensity change)
  - visual feature dict per frame (n_regions, region sizes, color counts,
    edge density) that the surprise engine can bootstrap on.

The edge/region extraction is deliberately simple and modular -- a real CNN
could replace it later, but the PRIOR (what features matter) stays.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def connected_regions(img: np.ndarray, bg: int = 0) -> list[dict]:
    """4-connected same-value regions.  img: 2D int array.  Returns region
    dicts with color, bbox, area."""
    h, w = img.shape
    seen = np.zeros_like(img, dtype=bool)
    regions = []
    for i in range(h):
        for j in range(w):
            if seen[i, j] or img[i, j] == bg:
                continue
            color = img[i, j]
            stack = [(i, j)]
            cells = []
            while stack:
                ci, cj = stack.pop()
                if seen[ci, cj] or img[ci, cj] != color:
                    continue
                seen[ci, cj] = True
                cells.append((ci, cj))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < h and 0 <= nj < w and not seen[ni, nj] and img[ni, nj] == color:
                        stack.append((ni, nj))
            rs = [c[0] for c in cells]
            cs = [c[1] for c in cells]
            r0, r1 = min(rs), max(rs) + 1
            c0, c1 = min(cs), max(cs) + 1
            regions.append({
                "color": int(color),
                "area": len(cells),
                "r": r0, "c": c0, "h": r1 - r0, "w": c1 - c0,
            })
    return regions


def edge_density(img: np.ndarray) -> float:
    """Fraction of pixels with a neighbor of a different value (edges)."""
    h, w = img.shape
    diff = 0
    total = 0
    for i in range(h):
        for j in range(w):
            v = img[i, j]
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni < h and nj < w:
                    total += 1
                    if img[ni, nj] != v:
                        diff += 1
    return diff / total if total else 0.0


def visual_features(img: np.ndarray) -> dict:
    """Per-frame visual feature dict for the surprise engine."""
    regions = connected_regions(img)
    feats = {
        "n_regions": len(regions),
        "total_area": sum(r["area"] for r in regions),
        "edge_density": round(edge_density(img), 3),
    }
    color_counts = Counter(r["color"] for r in regions)
    for c, n in color_counts.items():
        feats[f"regions_color_{c}"] = n
    # max region size (dominant object)
    feats["max_region_area"] = max((r["area"] for r in regions), default=0)
    return feats


def make_sample_images() -> list[np.ndarray]:
    """Controlled synthetic 'real images': frames with moving object(s)."""
    frames = []
    rng = np.random.RandomState(7)
    for t in range(12):
        img = np.zeros((16, 16), dtype=int)
        # a fixed square object (color 5) at top-left
        img[2:5, 2:5] = 5
        # a moving square (color 7) that shifts each frame
        x = 8 + t
        img[8:11, x:x + 3] = 7
        frames.append(img)
    return frames


if __name__ == "__main__":
    frames = make_sample_images()
    for f in frames:
        print(visual_features(f))
    print("sample regions:", connected_regions(frames[0]))

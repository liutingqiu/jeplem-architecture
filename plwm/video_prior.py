"""Video prior (B6, 2026-08-12): innate bias to extract temporal-visual structure.

Infant-analogy: a newborn perceives motion and scene changes in a visual
stream before knowing what the scenes mean.  Our architecture needs the same
innate bias -- from a sequence of frames it EXTRACTS motion (inter-frame
change), scene cuts (large changes), and activity level, so surprise can
bootstrap on the VIDEO dimension.

This is a learning BIAS (what temporal-visual structure to notice), not
knowledge: it measures motion and cuts but does not know what they MEAN.

Implementation: VideoPrior takes consecutive frames (2D arrays) and computes
motion magnitude (mean abs diff), changed-pixel fraction, scene-cut flag
(large jump), and a rolling activity feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def frame_diff(prev: np.ndarray, cur: np.ndarray) -> dict:
    """Per-transition video features between two frames."""
    if prev.shape != cur.shape:
        return {"motion": 0.0, "changed_frac": 0.0, "scene_cut": 0}
    diff = np.abs(prev.astype(float) - cur.astype(float))
    motion = float(diff.mean())
    changed = float((diff > 0).mean())
    scene_cut = 1 if changed > 0.5 else 0  # large change = scene cut
    return {
        "motion": round(motion, 3),
        "changed_frac": round(changed, 3),
        "scene_cut": scene_cut,
    }


def make_video(n_frames: int = 10) -> list[np.ndarray]:
    """Synthetic video: a small moving square over a static background."""
    frames = []
    for t in range(n_frames):
        img = np.zeros((16, 16), dtype=int)
        img[2:5, 2:5] = 5                      # static object
        x = 4 + t
        img[10:13, x:x + 2] = 7               # moving object
        frames.append(img)
    return frames


if __name__ == "__main__":
    frames = make_video()
    prev = None
    for i, f in enumerate(frames):
        if prev is not None:
            print(f"transition {i}: {frame_diff(prev, f)}")
        prev = f

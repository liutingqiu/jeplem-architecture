"""Spatial-topology prior (P1a, 2026-08-12): innate spatial relations.

The infant-analogy: a newborn has spatial cognition (up/down, near/far,
inside/outside, adjacency) before it has knowledge.  Our architecture should
have the same innate bias -- from a spatial perception (e.g. the Luanti 11x11
vision grid) it EXTRACTS spatial relations (adjacency, distance, direction,
enclosure), so surprise can bootstrap on the SPACE dimension.

This is a learning BIAS (what spatial facts to notice), not knowledge: it
computes distances and directions, but does not know what they MEAN.  The
meaning (e.g. 'key is always on the goal-side of the agent' vs 'decoy is
elsewhere') is grown by the surprise engine.

Implementation: SpatialTopologyPrior computes, per vision frame, the agent's
spatial relations to every other object type (key, decoy, goal): distance,
relative direction, and adjacency flags.  The surprise engine then runs on
these as its fields.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VISION_RADIUS = 5
N = VISION_RADIUS * 2 + 1  # 11


def parse_vision(vision) -> dict:
    """vision: dict with 'cells' (list of 121 tile strings) OR a list.
    Returns {tile_type: set_of_(r,c)} with agent position."""
    if isinstance(vision, dict):
        cells = vision.get("cells", [])
    else:
        cells = vision
    positions = {}
    for i, tile in enumerate(cells):
        r, c = i // N, i % N
        if tile == "air":
            continue
        positions.setdefault(tile, set()).add((r, c))
    return positions


def spatial_features(vision) -> dict:
    """Extract agent-to-other spatial relations as numeric features.

    Returns fields like:
      dist_key (manhattan), dir_key_r/dir_key_c (sign of delta),
      adj_key (bool), same for decoy, goal, wall_min_dist.
    """
    pos = parse_vision(vision)
    agent = pos.get("agent")
    if not agent:
        return {}
    ar, ac = next(iter(agent))
    feats = {}
    for target in ("key", "decoy", "goal", "wall"):
        cells = pos.get(target)
        if not cells:
            continue
        # nearest target cell
        nr, nc = min(cells, key=lambda p: abs(p[0] - ar) + abs(p[1] - ac))
        dist = abs(nr - ar) + abs(nc - ac)
        feats[f"dist_{target}"] = dist
        feats[f"dir_{target}_r"] = 1 if nr > ar else (-1 if nr < ar else 0)
        feats[f"dir_{target}_c"] = 1 if nc > ac else (-1 if nc < ac else 0)
        feats[f"adj_{target}"] = 1 if dist == 1 else 0
    # number of walls (density) as a spatial-context feature
    feats["n_walls"] = len(pos.get("wall", ()))
    feats["n_objects"] = sum(len(v) for k, v in pos.items() if k != "air")
    return feats


def spatial_stream(episodes: list[dict]) -> list[tuple[dict, dict]]:
    """Build a stream of spatial-feature pairs across consecutive visions."""
    pairs = []
    for ep in episodes:
        visions = ep.get("visions") or []
        prev = None
        for v in visions:
            cur = spatial_features(v)
            if not cur:
                continue
            if prev is not None:
                pairs.append((prev, cur))
            prev = cur
    return pairs


if __name__ == "__main__":
    import json
    episodes = [json.loads(l) for l in
                Path(ROOT / "artifacts" / "luanti-composite-rollouts.jsonl").open(encoding="utf-8")]
    stream = spatial_stream(episodes)
    print(f"spatial-feature stream: {len(stream)} transitions")
    if stream:
        print("sample t0:", stream[0][0])

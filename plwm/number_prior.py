"""Number prior (P0a, 2026-08-12): innate quantity bias.

The infant-analogy claim: a newborn counts (has a number sense) before it has
knowledge.  Our architecture should have the same innate bias -- from raw
perception it EXTRACTS quantity features (counts of object types, cell types),
so that surprise can bootstrap on the NUMBER dimension.

This is a learning BIAS (what to notice), not knowledge: it counts, but does
not know what the counts MEAN.  The meaning (e.g. 'wall count is invariant
while agent position changes') is grown by the surprise engine.

Implementation: a NumberPrior that, given a perception (vision cells / object
list), extracts a feature dict of counts.  The surprise engine then runs on
those counts as its fields.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def vision_tile_counts(vision_cells: list) -> dict:
    """Count tile types in a Luanti vision grid."""
    return {f"tile_{k}": int(v) for k, v in Counter(vision_cells).items()}


def object_counts(objects: list) -> dict:
    """Count object properties from a perceived object list (ARC-style).
    objects: list of dicts with 'color'/'type' keys."""
    counts = {}
    colors = Counter(o.get("color") for o in objects)
    for c, n in colors.items():
        counts[f"color_{c}"] = n
    types = Counter(o.get("type") for o in objects)
    for t, n in types.items():
        counts[f"type_{t}"] = n
    counts["n_objects"] = len(objects)
    return counts


class NumberPrior:
    """Extract quantity features from raw perception so surprise can bootstrap
    on the number dimension."""

    def extract(self, raw) -> dict:
        """raw = vision cells (list) OR objects (list of dicts)."""
        if isinstance(raw, list):
            if raw and isinstance(raw[0], str):
                return vision_tile_counts(raw)
            return object_counts(raw)
        if isinstance(raw, dict) and "cells" in raw:
            return vision_tile_counts(raw["cells"])
        return {}


def extract_number_stream(episodes: list[dict]) -> list[tuple[dict, dict]]:
    """From composite episodes (which carry visions), build a stream of
    number-feature pairs (counts at t, counts at t+1) for the surprise engine."""
    pairs = []
    for ep in episodes:
        visions = ep.get("visions") or []
        prev = None
        for v in visions:
            cur = vision_tile_counts(v.get("cells", []))
            if prev is not None:
                pairs.append((prev, cur))
            prev = cur
    return pairs


if __name__ == "__main__":
    import json
    episodes = [json.loads(l) for l in
                Path(ROOT / "artifacts" / "luanti-composite-rollouts.jsonl").open(encoding="utf-8")]
    stream = extract_number_stream(episodes)
    print(f"number-feature stream: {len(stream)} transitions")
    if stream:
        print("sample t0:", stream[0][0])
        print("sample t1:", stream[0][1])

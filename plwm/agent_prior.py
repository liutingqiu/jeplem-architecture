"""Agent prior (P2a, 2026-08-12): innate bias to detect OTHER AGENTS.

Infant-analogy: a newborn attends to agents (faces, moving things) over
inanimate objects -- it can distinguish 'something that moves on its own'
from 'something that is moved'.  Our architecture needs the same innate bias:
from a spatial perception it should notice which objects are AGENTS (self-
propelled, autonomous) vs passive scenery.

The core signal: AUTONOMOUS MOVEMENT.  An object that moves (position changes)
across frames while its identity is preserved is a candidate agent; static
objects are scenery.  This is a learning BIAS (what to attend to), not
knowledge -- it detects agents but does not know what they will do.

Implementation: AgentPrior watches a stream of frames (each a dict of
object positions), and per object tracks whether it moved under its own
accord.  It exposes an 'agent score' per object, which the surprise engine
can bootstrap on.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class AgentPrior:
    """Tracks per-object motion to detect autonomous agents."""

    _moved: dict = field(default_factory=lambda: defaultdict(int))
    _seen: dict = field(default_factory=lambda: defaultdict(int))
    _prev_pos: dict = field(default_factory=dict)

    def observe_frame(self, objects: Mapping[str, tuple]) -> dict:
        """objects: {obj_id: (r, c)}.  Returns {obj_id: agent_score}."""
        scores = {}
        for oid, pos in objects.items():
            self._seen[oid] += 1
            prev = self._prev_pos.get(oid)
            if prev is not None and prev != pos:
                self._moved[oid] += 1
            self._prev_pos[oid] = pos
            moved_frac = self._moved[oid] / self._seen[oid] if self._seen[oid] else 0.0
            scores[oid] = moved_frac
        return scores

    def is_agent(self, oid: str, threshold: float = 0.3) -> bool:
        return self._seen[oid] >= 3 and self._moved[oid] / self._seen[oid] >= threshold


def make_agent_scene(n_frames: int = 10) -> list[dict]:
    """Synthetic scene: object A moves autonomously (agent), B and C static."""
    frames = []
    for t in range(n_frames):
        frames.append({
            "agent_A": (t, 3),       # self-propelled
            "static_B": (5, 5),
            "static_C": (2, 8),
        })
    return frames


if __name__ == "__main__":
    ap = AgentPrior()
    frames = make_agent_scene()
    for f in frames:
        scores = ap.observe_frame(f)
    print("agent scores (moved fraction):")
    for oid in ("agent_A", "static_B", "static_C"):
        print(f"  {oid}: {ap._moved[oid]}/{ap._seen[oid]} -> agent={ap.is_agent(oid)}")

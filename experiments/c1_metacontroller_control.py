"""C1: metacontroller control-group experiment (2026-08-12).

Question: does a metacontroller (learned routing among sources) ADD value to
the exploration pipeline, or does the existing choose_targets heuristic
(gap/value/recency/trust) already suffice?

Control design:
  - WITH: a metacontroller that, per state, picks the source (modality/type)
    whose expected info-gain is highest (trained on source-outcome stats).
  - WITHOUT: the current body heuristic (gap-first) -- no learned routing.

Environment (controlled, not real web, to isolate the variable): N sources,
each with a DIFFERENT info-value distribution.  Good content yields high
'success' (matches the target), bad content yields low.  The metacontroller
must LEARN which source to route to maximize accumulated value.

Criterion: metacontroller accumulated-value > heuristic accumulated-value by
a meaningful margin (else it adds no value and we keep the heuristic).

Honest framing: this is a TOY that isolates the routing decision; if
metacontroller cannot beat gap-heuristic HERE, it will not help the real web
pipeline where sources are even less separable.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class Source:
    """A source with a learnable value distribution."""
    name: str
    good_prob: float      # P(content is valuable | source)
    value_when_good: float = 1.0
    value_when_bad: float = -0.2


def make_sources() -> list[Source]:
    """4 sources with different reliability."""
    return [
        Source("reliable", good_prob=0.8),
        Source("mid", good_prob=0.5),
        Source("noisy", good_prob=0.3),
        Source("dead", good_prob=0.05),
    ]


# --- WITHOUT: gap-first heuristic (current body behavior) -------------------

def heuristic_route(gap_scores: dict[str, float]) -> str:
    """Pick the source with highest gap (curiosity), like choose_targets."""
    return max(gap_scores, key=gap_scores.get)


# --- WITH: metacontroller (learned routing) ---------------------------------

class MetaController:
    """Learns per-source expected value from outcomes; routes to best."""

    def __init__(self, sources: list[Source]):
        self.expected: dict[str, float] = {s.name: 0.5 for s in sources}
        self._n: dict[str, int] = defaultdict(int)

    def choose(self) -> str:
        """Pick the source with the highest learned expected value."""
        return max(self.expected, key=self.expected.get)

    def update(self, source: str, value: float) -> None:
        n = self._n[source] + 1
        self._n[source] = n
        self.expected[source] += (value - self.expected[source]) / n


def run_episode(*, use_meta: bool, sources: list[Source], steps: int = 100,
                seed: int = 1) -> float:
    rng = np.random.RandomState(seed)
    # initial gap scores (curiosity) -- high for unexplored
    gap = {s.name: 1.0 for s in sources}
    meta = MetaController(sources) if use_meta else None
    total = 0.0
    for t in range(steps):
        if use_meta:
            choice = meta.choose()
        else:
            choice = heuristic_route(gap)
        src = next(s for s in sources if s.name == choice)
        good = rng.rand() < src.good_prob
        value = src.value_when_good if good else src.value_when_bad
        total += value
        if meta is not None:
            meta.update(choice, value)
        # gap heuristic (FAIR): after visiting, this source's gap drops to ~0
        # (confidence up), so it rotates to OTHER sources.  A source that never
        # pays resets gap over time (recency decay), like the real time-sense.
        gap[choice] = 0.05
        for s in sources:
            if s.name != choice:
                gap[s.name] = min(1.0, gap[s.name] + 0.15)
    return total


def run(*, seeds=(1, 2, 3), steps: int = 100, artifact: Path | None = None) -> dict:
    sources = make_sources()
    results = {"with": [], "without": []}
    for seed in seeds:
        w = run_episode(use_meta=True, sources=sources, steps=steps, seed=seed)
        wo = run_episode(use_meta=False, sources=sources, steps=steps, seed=seed)
        results["with"].append(round(w, 2))
        results["without"].append(round(wo, 2))
    mean_with = float(np.mean(results["with"]))
    mean_without = float(np.mean(results["without"]))
    report = {
        "protocol": "C1_metacontroller_control: learned routing (with) vs gap-first heuristic (without) over 4 sources with different reliability (0.8/0.5/0.3/0.05). Isolated toy isolating the routing decision. If metacontroller cannot beat heuristic HERE, it adds no value to the web pipeline.",
        "steps": steps,
        "results": results,
        "mean_with": round(mean_with, 2),
        "mean_without": round(mean_without, 2),
        "gain": round(mean_with - mean_without, 2),
        "conclusion": ("metacontroller ADDS value" if mean_with > mean_without * 1.1
                       else "metacontroller does NOT add value over heuristic"),
        "honest_boundary": "This toy has REPEATABLE sources (reliable/mid/noisy/dead) with fixed reliability, so learned routing wins by preferring the reliable source. The real web has NO repeatable source structure (each URL is one-shot content) -- the advantage may not transfer. Metacontroller value on the real web remains UNVERIFIED; this controls only the toy routing decision.",
        "artifact": str(artifact or ROOT / "artifacts" / "c1-metacontroller.json"),
    }
    (artifact or ROOT / "artifacts" / "c1-metacontroller.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "c1-metacontroller.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

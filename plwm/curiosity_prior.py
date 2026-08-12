"""Curiosity anchor (P1b, 2026-08-12): innate priority over surprise.

The infant-analogy: a newborn is attracted to novelty but not equally to all
novelty -- it prefers things that are novel AND learnable (a face it can
eventually parse) over pure noise (static) or the already-known.  Our
architecture needs the same: a curiosity anchor that, over the raw surprise
stream, PRIORITIZES what to explore.

The real-world problem: bare surprise drowns (infinite content, most of it
noise).  A curiosity anchor ranks candidate surprises:

    curiosity(field) = novelty  *  learnability  *  relevance
      novelty    = current surprise rate (how unexpected)
      learnability = can a rule explain it? (a field whose changes are
                     reducible to rules is worth chasing; pure noise is not)
      relevance = how much structure it already explains (momentum)

This is a learning BIAS (where to point attention), not knowledge: it ranks,
but the rules it learns are still grown by surprise.

Implementation: CuriosityPrior wraps the surprise engine and, per field, scores
curiosity; it exposes the ranking so an agent can decide what to observe next
(e.g. which hidden variable deserves a new observation channel).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class CuriosityPrior:
    """Scores per-field curiosity to guide exploration."""

    window: int = 6
    _surprise_rate: dict = field(default_factory=lambda: defaultdict(list))
    _learnable: set = field(default_factory=set)
    _explained_momentum: dict = field(default_factory=lambda: defaultdict(int))

    def observe_field(self, f: str, surprise: bool, explained_by_rule: bool) -> None:
        """Feed per-field surprise + whether a rule already explains it."""
        w = self._surprise_rate[f]
        w.append(surprise)
        if len(w) > self.window:
            w.pop(0)
        if explained_by_rule:
            self._learnable.add(f)
            self._explained_momentum[f] += 1

    def novelty(self, f: str) -> float:
        w = self._surprise_rate[f]
        return sum(w) / len(w) if w else 0.0

    def score(self, f: str) -> float:
        """curiosity = novelty * learnability * relevance.
        Novel but pure noise (not learnable) scores low; novel and reducible to
        rules scores high; already-explained-but-still-changing (momentum)
        stays high until resolved."""
        novelty = self.novelty(f)
        if novelty == 0:
            return 0.0
        learnability = 1.0 if f in self._learnable else 0.5  # unknown lean
        relevance = 1.0 + min(self._explained_momentum[f], 3) / 3.0
        return novelty * learnability * relevance

    def rank(self) -> list[tuple[str, float]]:
        fields = {f for f, w in self._surprise_rate.items() if w}
        return sorted(((f, self.score(f)) for f in fields), key=lambda x: -x[1])

    def top(self, k: int = 3) -> list[str]:
        return [f for f, _ in self.rank()[:k]]

"""Safety prior (P2b, 2026-08-12): innate aversion to danger.

Infant-analogy: a newborn avoids harm (pain, falling) before it has knowledge
of the world.  Our architecture needs the same innate bias -- a safety prior
that flags dangerous signals (strong negative outcomes, terminal events,
destructive change) and biases exploration away from them.

This is a learning BIAS (what to treat as aversive), not knowledge: it flags
'this signal is dangerous' but does not know the world's actual dangers until
the surprise engine learns them.  The prior just gives AVOIDANCE a head start.

Implementation: SafetyPrior watches a stream of (signal) values and flags
dangerous signals (large negative spikes, terminal flags).  It maintains a
per-signal danger score and a total aversion bias that exploration should
respect.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class SafetyPrior:
    """Flags dangerous signals and tracks aversion."""

    danger_threshold: float = -0.5
    terminal_danger: float = 1.0
    _danger_hits: dict = field(default_factory=lambda: defaultdict(int))
    _seen: dict = field(default_factory=lambda: defaultdict(int))
    _worst: dict = field(default_factory=dict)   # most dangerous value seen
    _last_was_terminal: bool = False

    def observe(self, signals: dict) -> dict:
        """signals: {channel: value}.  Returns {channel: danger_score} where
        danger_score = severity of the worst observed value (0..1)."""
        scores = {}
        for ch, val in signals.items():
            self._seen[ch] += 1
            worst = self._worst.get(ch, 0.0)
            if val < worst:
                worst = val
                self._worst[ch] = worst
            if val < self.danger_threshold:
                self._danger_hits[ch] += 1
            # severity: how far below threshold, normalized
            scores[ch] = min(1.0, max(0.0, -worst)) if worst < 0 else 0.0
        return scores

    def aversion(self, channel: str) -> float:
        """Severity-weighted aversion (0 = safe, 1 = catastrophic)."""
        worst = self._worst.get(channel, 0.0)
        return min(1.0, max(0.0, -worst))

    def is_dangerous(self, channel: str, threshold: float = 0.5) -> bool:
        return self._seen[channel] >= 2 and self.aversion(channel) >= threshold


def make_danger_stream(n: int = 10) -> list[dict]:
    """Synthetic: 'reward' mostly positive, occasionally dangerous;
    'touch' always slightly negative (aversive contact)."""
    stream = []
    for t in range(n):
        if t % 4 == 3:
            stream.append({"reward": -2.0, "touch": -0.1, "idle": 0.0})
        else:
            stream.append({"reward": 1.0, "touch": -0.1, "idle": 0.0})
    return stream


if __name__ == "__main__":
    sp = SafetyPrior()
    stream = make_danger_stream()
    for s in stream:
        sp.observe(s)
    print("aversion per channel:")
    for ch in ("reward", "touch", "idle"):
        print(f"  {ch}: {sp.aversion(ch):.2f} dangerous={sp.is_dangerous(ch)}")

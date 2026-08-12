"""World-model drive (I4, 2026-08-12): prediction-error-driven exploration.

The torch world_model (A3) learned a prediction head that drives curiosity
(coverage faster than random).  In the pure-python web pipeline we implement
the equivalent without a neural model: a lightweight WORLD MODEL per modality
that tracks the learned feature distribution, and a PREDICTION-ERROR signal:

  for new content, prediction_error = how far its features are from the
  learned distribution of that modality (a Gaussian-ish / range deviation).

Exploration priority (integrated into choose_targets via a callback):
  - content whose features are HIGHLY predictable (familiar) -> low value
    (already understood)
  - content that is PURE NOISE (far outside any learned structure) -> low
    value (unlearnable, like the surprise engine's hidden vars)
  - content that is MODERATELY novel (near but not in the learned range) ->
    HIGH value (learnable novelty) -- this is the 'curiosity sweet spot' that
    the A3 world model found (curiosity covers faster than random)

This makes exploration active: seek what is learnable-but-unlearned, not
passively wait for surprise.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class ModalityModel:
    """Per-modality learned feature statistics (the world model)."""
    means: dict = field(default_factory=dict)
    stds: dict = field(default_factory=dict)
    n: int = 0

    def update(self, feats: dict) -> None:
        self.n += 1
        for k, v in feats.items():
            if not isinstance(v, (int, float)):
                continue
            if k not in self.means:
                self.means[k] = float(v)
                self.stds[k] = 0.0
            else:
                old = self.means[k]
                self.means[k] = old + (v - old) / self.n
                self.stds[k] = self.stds[k] + (v - old) * (v - self.means[k])

    def prediction_error(self, feats: dict) -> float:
        """Mean RELATIVE deviation of features vs the learned mean.  Uses
        relative change (|v-mean|/|mean|) which is scale-robust -- a 5x word
        count is 'moderate novelty', a 50x is 'extreme'."""
        if self.n == 0 or not feats:
            return 1.0
        errs = []
        for k, v in feats.items():
            if not isinstance(v, (int, float)) or k not in self.means:
                continue
            m = self.means[k]
            if m == 0:
                errs.append(float(v) if v else 0.0)
            else:
                errs.append(abs(v - m) / abs(m))  # relative change
        return float(np.mean(errs)) if errs else 1.0


class WorldModelDrive:
    """Prediction-error-driven exploration."""

    def __init__(self):
        self.models: dict[str, ModalityModel] = defaultdict(ModalityModel)
        self._errors: list[float] = []

    def observe(self, feats: dict, modality: str) -> float:
        """Update the world model; return this content's prediction error."""
        err = self.models[modality].prediction_error(feats)
        self.models[modality].update(feats)
        self._errors.append(err)
        return err

    def curiosity(self, feats: dict, modality: str) -> float:
        """Curiosity score: HIGH for moderately-novel (learnable), LOW for
        familiar (already known) or extreme-noise (unlearnable).

        prediction_error ~ N(0,1) for familiar content (z-scores near 0).
        Curiosity rises with error (novelty) but saturates then falls for
        extreme error (pure noise = unlearnable).  Peak is a soft band."""
        err = self.models[modality].prediction_error(feats)
        # err ~ 0 (familiar) -> low. Rises to a plateau for err in [0.5, 4],
        # then falls for err > 4 (content far outside learned structure).
        if err <= 0.5:
            return float(err * 2.0)  # ramp up from familiar
        if err <= 4.0:
            return 1.0  # learnable novelty band: full curiosity
        return float(max(0.0, 1.0 - (err - 4.0) / 6.0))

    def drive(self, candidate_features: list[tuple[str, dict]]) -> list[float]:
        """Score candidates for exploration priority (used to reorder targets)."""
        return [self.curiosity(f, m) for m, f in candidate_features]


if __name__ == "__main__":
    # demonstrate: familiar content low curiosity, moderate-novel high, noise low
    wm = WorldModelDrive()
    # learn text structure
    for _ in range(20):
        wm.observe({"n_words": 8, "unique_ratio": 0.8, "n_bigrams": 7}, "text")
    familiar = wm.curiosity({"n_words": 8, "unique_ratio": 0.8, "n_bigrams": 7}, "text")
    moderate = wm.curiosity({"n_words": 40, "unique_ratio": 0.3, "n_bigrams": 39}, "text")
    noise = wm.curiosity({"n_words": 500, "unique_ratio": 1.0, "n_bigrams": 499}, "text")
    print(f"familiar curiosity: {familiar:.3f} (should be low)")
    print(f"moderate-novel curiosity: {moderate:.3f} (should be HIGH)")
    print(f"noise curiosity: {noise:.3f} (should be low)")

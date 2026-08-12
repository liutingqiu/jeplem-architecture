"""Self-knowledge layer (I3, 2026-08-12): lightweight capability boundary for
web exploration.

The torch self_model (M3a) learned 'I know my competence boundary' via a
self_head over OOD data.  In the pure-python web pipeline we implement the
same capability-boundary signal WITHOUT a neural head:

    for a new content's features, self_confidence = fraction of features that
    fall in KNOWN structure (invariant fields the engine has seen) vs UNKNOWN
    (fields never seen / that vary unpredictably).

Meaning:
  - high confidence  -> content is structurally familiar (like ID for self_head)
  - low confidence   -> content is novel/unfamiliar (like OOD for self_head)

This lets the resident KNOW 'I don't understand this yet' and (via curiosity)
seek familiar-but-not-yet-mastered content -- the self-model's value in the
free-roaming case.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SelfKnowledge:
    """Capability-boundary signal over the surprise engine's knowledge."""

    def __init__(self, engine):
        self.engine = engine

    def confidence(self, feats: dict, modality: str) -> float:
        """Fraction of this content's features that are KNOWN (invariant).
        1.0 = fully familiar structure, 0.0 = fully novel."""
        if not feats:
            return 0.0
        known = self.engine.knowledge()
        inv = set(known["invariants"])
        ns = f"{modality}::"
        total = 0
        known_count = 0
        for f in feats:
            key = f"{ns}{f}"
            total += 1
            if key in inv:
                known_count += 1
        return known_count / total if total else 0.0

    def novel_fields(self, feats: dict, modality: str) -> list[str]:
        """Fields never seen as invariant (the 'I don't know this yet' list)."""
        known = self.engine.knowledge()
        inv = set(known["invariants"])
        ns = f"{modality}::"
        return [f for f in feats if f"{ns}{f}" not in inv]

    def engine_state(self) -> dict:
        """Expose the capability boundary per modality: how many invariants
        (known) vs hidden (unknown) the engine has per modality."""
        known = self.engine.knowledge()
        from collections import Counter
        inv_by_mod = Counter()
        hid_by_mod = Counter()
        for f in known["invariants"]:
            mod = f.split("::")[0] if "::" in f else "?"
            inv_by_mod[mod] += 1
        for f in known["hidden_vars"]:
            mod = f.split("::")[0] if "::" in f else "?"
            hid_by_mod[mod] += 1
        mods = set(inv_by_mod) | set(hid_by_mod)
        return {
            m: {"known_invariants": inv_by_mod[m], "unknown_hidden": hid_by_mod[m]}
            for m in sorted(mods)
        }


if __name__ == "__main__":
    # demonstrate: familiar text gets high confidence, novel-modality low
    from plwm.modal_surprise_engine import ModalSurpriseEngine
    from plwm.modality_identify import route
    from plwm.perception_hub import PerceptionHub

    hub = PerceptionHub()
    engine = ModalSurpriseEngine(window=4)
    sk = SelfKnowledge(engine)
    prev = None
    for _ in range(4):  # learn text structure
        t = 'python is a readable language created in 1991'
        mod, _ = route(t)
        feats = hub.perceive({'text': t})
        if prev is not None:
            engine.observe(prev, feats, modality='text')
        prev = feats
    # familiar text
    mod, _ = route('python has a standard library')
    familiar = hub.perceive({'text': 'python has a standard library'})
    # image content (never seen -> low confidence)
    import numpy as np
    img_feats = hub.perceive({'image': np.zeros((8, 8), dtype=int)})
    print('familiar text confidence:', round(sk.confidence(familiar, 'text'), 3))
    print('novel image confidence:', round(sk.confidence(img_feats, 'image'), 3))
    print('novel image fields:', sk.novel_fields(img_feats, 'image'))

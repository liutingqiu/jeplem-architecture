"""Web cognitive exploration + narrative monitor (2026-08-12).

The architecture explores the REAL web with its own priors + surprise engine,
and we MONITOR with information (not metrics).  Discipline: we do NOT steer
the exploration -- the body picks targets by its own curiosity; we only watch
and report what it saw and learned.

Layers:
  BodyRunner (sibling worktree body.py)  : sees URLs, senses modalities, keeps
                                           beliefs/emotions/trust
  ModalSurpriseEngine (our plwm)         : the cognitive layer -- from each
                                           sensed content, extract features via
                                           modality routing, feed surprise,
                                           grow invariants/rules/hidden vars

The monitor outputs a NARRATIVE digest each checkpoint:
  - what it saw      : URLs + modalities + a content glimpse
  - what surprised it: which features were unexpected
  - what it learned  : invariants (stable structure), rules (learned deltas),
                       hidden variables (unexplained change) -- in WORDS
  - body state       : belief/trust/emotion drift (bounded)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "zealous-lamport-398898"))

from plwm.modal_surprise_engine import ModalSurpriseEngine
from plwm.modality_identify import route
from plwm.body import BodyRunner, build_default_body
from plwm.web import WebFetchBackend


class WebCognitiveMonitor:
    """The cognitive layer + narrative monitor around a BodyRunner."""

    def __init__(self, runner: BodyRunner, *, window: int = 6):
        self.runner = runner
        self.engine = ModalSurpriseEngine(window=window)
        self.seen: list[dict] = []       # narrative: what it saw
        self.learned: list[dict] = []    # narrative: what it learned
        self._prev_feats: dict | None = None
        self._prev_mod: str | None = None

    def feed_observation(self, url: str, modality: str, payload: dict) -> None:
        """Feed one sensed content into the cognitive layer (route + surprise)."""
        # glimpse for narrative: first ~120 chars of text or content summary
        glimpse = self._glimpse(payload)
        self.seen.append({"url": url, "modality": modality, "glimpse": glimpse})
        # route to the right prior lens
        mod, feats = route(_content_of(payload))
        if feats is None:
            return
        if self._prev_feats is not None and self._prev_mod == mod:
            report = self.engine.observe(self._prev_feats, feats, modality=mod)
            self._check_new_learning(mod, report, url)
        self._prev_feats = feats
        self._prev_mod = mod

    def _check_new_learning(self, mod: str, report: dict, url: str) -> None:
        """Detect when the engine grew structure (invariant/rule/hidden) and
        record it in words."""
        k = self.engine.knowledge()
        # invariants: fields that became invariant recently
        inv_fields = [f for f in k["invariants"] if f.startswith(f"{mod}::")]
        if inv_fields:
            self.learned.append({
                "what": "invariant",
                "where": f"after seeing {url}",
                "detail": f"in {mod} content, these features are STABLE (never change): "
                          + ", ".join(f.split('::')[-1] for f in inv_fields),
            })
        # hidden vars
        hid = [f for f in k["hidden_vars"] if f.startswith(f"{mod}::")]
        if hid:
            self.learned.append({
                "what": "hidden_variable",
                "where": f"after seeing {url}",
                "detail": f"in {mod} content, these features change unpredictably "
                          "(need explanation): "
                          + ", ".join(f.split('::')[-1] for f in hid),
            })

    def _glimpse(self, payload: dict) -> str:
        for key in ("text", "content", "html", "summary", "caption"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v[:120].replace("\n", " ")
        return str(payload)[:120]

    def narrative(self) -> dict:
        """The information-rich report."""
        body = self.runner.body
        k = self.engine.knowledge()
        # beliefs as words
        belief_conf = None
        if hasattr(body.runtime, "belief"):
            try:
                belief_conf = round(float(body.runtime.belief.confidence().item()), 3)
            except Exception:
                belief_conf = None
        mod_seen = {}
        for s in self.seen:
            mod_seen[s["modality"]] = mod_seen.get(s["modality"], 0) + 1
        trust = {}
        try:
            trust = dict(self.runner.trust.snapshot().get("scores", {}))
        except Exception:
            pass
        return {
            "what_it_saw": {
                "n_urls": len(self.seen),
                "by_modality": mod_seen,
                "sample": self.seen[-5:],
            },
            "what_it_learned": self.learned[-10:],
            "cognitive_state": {
                "invariants": [f for f in k["invariants"]],
                "rules": [f for f in k["rules"]],
                "hidden_vars": [f for f in k["hidden_vars"]],
                "belief_confidence": belief_conf,
                "trust": {k2: round(v2, 3) for k2, v2 in list(trust.items())[:5]},
            },
        }


def _content_of(payload: dict):
    """Extract the raw content from a sensed payload for modality routing."""
    for key in ("text", "content", "cells", "image", "audio", "wave", "code"):
        if key in payload:
            return payload[key]
    # nested?
    for v in payload.values():
        if isinstance(v, (str, bytes, list)):
            return v
    return None


if __name__ == "__main__":
    print("Web cognitive monitor module ready.")

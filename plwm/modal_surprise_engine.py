"""Modal SurpriseEngine (task-1, 2026-08-12): surprise stable under modality
switching.

The real-world boundary found in A3: on a full multimodal stream, the surprise
engine learns few invariants because modal switches change the field set (a
field appears/disappears), making identity prediction fail spuriously.

Fix: group learning BY MODALITY.  Each modality maintains its own surprise
state (fields are namespaced `modality::field`), so within a modality the
field set is stable and identity/rule prediction works.  When the modality
switches, we do NOT compare across the old and new field sets -- each modality
learns its own invariants/rules independently.

This mirrors how perception is actually organized: vision has its own
structure, text has its own, and they are learned separately even though the
same engine processes them.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.surprise_engine import SurpriseEngine


class ModalSurpriseEngine(SurpriseEngine):
    """SurpriseEngine with per-modality state (fields namespaced by modality)."""

    def __init__(self, *, window: int = 8, majority: float = 0.6,
                 invariant_threshold: float = 0.99,
                 rule_path: Path | None = None):
        super().__init__(window=window, majority=majority,
                         invariant_threshold=invariant_threshold,
                         rule_path=rule_path)
        self._active_modality: str | None = None

    def observe(self, s0: dict, s1: dict, modality: str | None = None) -> dict:
        """Per-modality observe.  When modality is given, fields are namespaced
        so each modality learns independently; cross-modality field churn is
        ignored."""
        m = modality or self._active_modality or "default"
        self._active_modality = m
        # namespace fields: only feed fields present in BOTH s0 and s1 (stable
        # within a modality), under the modality namespace
        ns_s0 = {}
        ns_s1 = {}
        for f in set(s0) & set(s1):
            ns_s0[f"{m}::{f}"] = s0[f]
            ns_s1[f"{m}::{f}"] = s1[f]
        return super().observe(ns_s0, ns_s1)

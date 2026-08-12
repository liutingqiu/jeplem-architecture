"""SurpriseEngine (2026-08-12): the identity+surprise bootstrap loop, as an
architecture.

This is the seed made concrete.  The engine holds NO preloaded structure -- it
starts with the identity prior (next = prev) and grows structure from surprise:

    identity prior (world stays the same)
      -> predict (next = prev) per field
      -> surprise (prediction vs reality), tracked per field
      -> sustained surprise on a field triggers RULE LEARNING (auto-discover a
         constant-delta rule that eliminates the surprise)
      -> the learned rule is COMPILED into the RuleLedger (System-1 reuse)
      -> later encounters reuse the rule; world changes mark it stale and
         re-induction revises it (P4 semantics via the ledger)

This is the 'tell the model the lowest logic, then give it time' landing: we
do not hand it rule families or conservation detectors; we hand it identity +
surprise, and structure (invariants / rules / hidden variables) grows.

World-agnostic: fields are numeric dict keys; an adapter feeds (s0, s1) pairs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from plwm.rule_ledger import RuleLedger


@dataclass
class SurpriseStats:
    observations: int = 0
    identity_hits: int = 0
    surprise_events: int = 0
    rules_learned: int = 0
    rules_reused: int = 0
    invariants_discovered: int = 0
    hidden_vars: int = 0


class SurpriseEngine:
    """identity + surprise bootstrap loop."""

    def __init__(self, *, window: int = 8, majority: float = 0.6,
                 invariant_threshold: float = 0.99,
                 rule_path: Path | None = None):
        self.ledger = RuleLedger()
        self.stats = SurpriseStats()
        # per-field identity counters
        self._id_correct: dict[str, int] = defaultdict(int)
        self._id_total: dict[str, int] = defaultdict(int)
        # sustained surprise windows per field (for rule-learning trigger)
        self._surprise_window: dict[str, list[bool]] = defaultdict(list)
        # known invariants (identity-consistent fields)
        self._invariants: set[str] = set()
        # hidden variables: fields with irreducible surprise
        self._hidden_vars: set[str] = set()
        self._window = window
        self._majority = majority
        self._invariant_threshold = invariant_threshold
        self.rule_path = rule_path
        self._rule_learners: dict[str, Callable] = {}
        # verification: after learning, track whether the learner eliminates
        # surprise; if not within N observations, reclassify as hidden var.
        self._verify_window: dict[str, int] = defaultdict(int)
        self._verify_hits: dict[str, int] = defaultdict(int)
        self._verify_after = 8

    # --- persistence ---------------------------------------------------------

    def load(self) -> None:
        if self.rule_path and self.rule_path.exists():
            self.ledger.restore(self.rule_path)

    def save(self) -> None:
        if self.rule_path:
            self.ledger.save(self.rule_path)

    # --- the loop ------------------------------------------------------------

    def observe(self, s0: dict, s1: dict) -> dict:
        """One transition.  Per field: predict with identity (or learned rule),
        record surprise, update knowledge.  Returns the surprise report."""
        self.stats.observations += 1
        report = {}
        for f, v1 in s1.items():
            if not isinstance(v1, (int, float)) or isinstance(v1, bool):
                continue
            v0 = s0.get(f)
            # predict: identity (next=prev) or a learned rule
            predicted = self._predict(f, s0, v0)
            surprise = (predicted is None or predicted != v1)
            if surprise:
                self.stats.surprise_events += 1
            else:
                self.stats.identity_hits += 1
            self._id_total[f] += 1
            if not surprise:
                self._id_correct[f] += 1
            report[f] = 1.0 if surprise else 0.0
            # verify a learned rule: does it actually eliminate surprise?
            if f in self._rule_learners:
                self._verify_window[f] += 1
                if not surprise:
                    self._verify_hits[f] += 1
                if self._verify_window[f] >= self._verify_after:
                    acc = self._verify_hits[f] / self._verify_window[f]
                    if acc < 0.5:
                        # the learned rule does NOT explain this field ->
                        # it is a hidden variable; drop the useless learner
                        self._rule_learners.pop(f, None)
                        self._hidden_vars.add(f)
            self._track_surprise(f, surprise, s0, s1)
        return report

    def _predict(self, f: str, s0: dict, v0):
        if f in self._rule_learners:
            try:
                return self._rule_learners[f](s0, v0)
            except Exception:
                return v0
        return v0  # identity: next = prev

    def _track_surprise(self, f: str, surprise: bool, s0: dict, s1: dict) -> None:
        """Window-majority on per-field surprise.  Sustained surprise on a field
        triggers rule learning ONCE; sustained consistency marks it invariant."""
        w = self._surprise_window[f]
        # store the full transition so the learner can find conditional rules
        w.append((surprise, s0, s1))
        if len(w) > self._window:
            w.pop(0)
        if len(w) < self._window:
            return
        frac_surprise = sum(1 for s, _, _ in w if s) / len(w)
        if frac_surprise >= self._majority:
            if f not in self._rule_learners and f not in self._hidden_vars:
                self._learn_field_rule(f, w)
        elif frac_surprise <= 0.05:
            acc = self._id_correct[f] / self._id_total[f] if self._id_total[f] else 0.0
            if acc >= self._invariant_threshold:
                self._invariants.add(f)

    def _learn_field_rule(self, f: str, window: list[tuple]) -> None:
        """Auto-learn a rule for a field with sustained surprise.  Tries, in
        order:
          1. constant delta across the surprising transitions -> simple rule
          2. CONDITIONAL delta: find another field of s0 whose value partitions
             the deltas into constant groups (e.g. x-move depends on
             last_direction).  -> conditional rule
        If neither holds, the field is a HIDDEN variable."""
        transitions = [(s0, s1) for s, s0, s1 in window if s]
        if not transitions:
            self._hidden_vars.add(f)
            return
        deltas = [(s1[f] - s0[f]) for s0, s1 in transitions
                  if isinstance(s0.get(f), (int, float)) and isinstance(s1.get(f), (int, float))]
        if not deltas:
            self._hidden_vars.add(f)
            return
        # 1) constant delta?
        if len(set(round(d, 6) for d in deltas)) == 1:
            self._install_rule(f, deltas[0], cond_field=None, cond_value=None)
            return
        # 2) conditional: partition by another field's value; each partition
        #    must have a constant delta
        cond_fields = sorted({g for s0, _ in transitions
                              for g in s0
                              if isinstance(s0.get(g), (int, float, str))})
        for cf in cond_fields:
            if cf == f:
                continue
            groups = {}
            for s0, s1 in transitions:
                cv = s0.get(cf)
                d = s1[f] - s0[f]
                groups.setdefault(cv, []).append(d)
            if groups and all(len(set(round(g, 6) for g in gs)) == 1 for gs in groups.values()):
                # a conditional rule: delta depends on cf's value
                self._install_rule(f, None, cond_field=cf, cond_value=None, cond_map=groups)
                return
        self._hidden_vars.add(f)

    def _install_rule(self, f: str, delta, cond_field, cond_value, cond_map=None) -> None:
        if cond_map is not None:
            # conditional: delta = cond_map[s0[cond_field]]
            def _learner(s0, v0, _cm=cond_map, _cf=cond_field):
                cv = s0.get(_cf)
                if cv in _cm and isinstance(v0, (int, float)):
                    return v0 + _cm[cv]
                return v0
        elif cond_field is not None:
            def _learner(s0, v0, _d=delta, _cf=cond_field, _cv=cond_value):
                if s0.get(_cf) != _cv:
                    return v0  # condition not met -> identity
                return (v0 + _d) if isinstance(v0, (int, float)) else None
        else:
            def _learner(s0, v0, _d=delta):
                return (v0 + _d) if isinstance(v0, (int, float)) else None
        sig = f"surprise/{f}/delta"
        serialized = {"kind": "surprise_rule", "field": f, "delta": delta}
        self.ledger.compile(sig, serialized, "surprise/rule")
        self._rule_learners[f] = _learner
        self.stats.rules_learned += 1

    def query(self, f: str, s0: dict, v0) -> object:
        """Prediction with the best available model (identity or learned)."""
        return self._predict(f, s0, v0)

    def knowledge(self) -> dict:
        """The structure grown so far: invariants, rules, hidden variables."""
        return {
            "invariants": sorted(self._invariants),
            "rules": sorted(self._rule_learners.keys()),
            "hidden_vars": sorted(self._hidden_vars),
            "ledger": self.ledger.stats(),
        }

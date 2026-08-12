"""MetaEngine (2026-08-12): the experience -> meta-rule runtime loop.

Pulls a stream of (in, out) observation pairs through the pipeline:
    perceive -> conserve (signature) -> rule-ledger lookup (System 1 fast path)
    -> on miss: induce (System 2 slow path) -> verify -> compile -> persist.

This is the architecture's "learning machine": every encounter either reuses a
compiled rule (fast, System 1) or constructs a new one (slow, System 2) which
is then compiled for reuse.  Telemetry measures the compile-once-reuse-many
speed advantage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from plwm.inducer import Inducer
from plwm.rule_ledger import RuleLedger


@dataclass
class EngineStats:
    hits: int = 0                 # System-1 reuse
    inductions: int = 0           # System-2 constructions
    gaps: int = 0                 # honest gaps (nothing induced)
    revises: int = 0              # re-induction after stale
    stale_marks: int = 0          # P4 window-majority marks
    restored: int = 0             # rules loaded from disk


class MetaEngine:
    def __init__(self, adapter, relations: Sequence, *, rule_path: Path | None = None):
        self.adapter = adapter
        self.relations = list(relations)
        self.inducer = Inducer(relations)
        self.ledger = RuleLedger()
        self.stats = EngineStats()
        self.rule_path = rule_path
        self._observed_pairs: list[tuple] = []

    # --- persistence ---------------------------------------------------------

    def load(self) -> None:
        if self.rule_path and self.rule_path.exists():
            self.ledger.restore(self.rule_path)
            self.stats.restored = self.ledger.stats()["n_rules"]

    def save(self) -> None:
        if self.rule_path:
            self.ledger.save(self.rule_path)

    # --- the learning loop ---------------------------------------------------

    def process(self, pairs: Sequence[tuple]) -> None:
        """Consume a stream of (in_raw, out_raw) pairs."""
        for in_raw, out_raw in pairs:
            in_s = self.adapter.perceive(in_raw)
            out_s = self.adapter.perceive(out_raw)
            sig = self.adapter.conserve(in_s, out_s).signature

            def _verify(rule):
                # reuse requires the cached rule to reproduce THIS pair
                pred = _apply_compiled(rule, self.adapter, in_raw)
                return _raw_equal(pred, out_raw, self.adapter)

            cached = self.ledger.lookup(sig, _verify)
            if cached is not None:
                self.stats.hits += 1
                self._observed_pairs.append((in_raw, out_raw))
                continue

            # System-2: slow constructive path
            was_stale = self.ledger.is_stale(sig)
            rule = self.inducer.induce(self.adapter, self._observed_pairs + [(in_raw, out_raw)])
            if rule is None:
                self.stats.gaps += 1
                continue
            if was_stale:
                self.stats.revises += 1
            else:
                self.stats.inductions += 1
            serialized = _serialize_rule(rule)
            self.ledger.compile(sig, serialized, rule.relation_key)
            self._observed_pairs.append((in_raw, out_raw))

    def predict(self, in_raw):
        """Apply the best compiled rule for the current pair, or None."""
        in_s = self.adapter.perceive(in_raw)
        # re-induce if nothing compiled yet (fallback path)
        if not self._observed_pairs:
            return None
        # use the last-seen signature's compiled rule if it fits
        sig = self.adapter.conserve(in_s, self.adapter.perceive(self._observed_pairs[-1][1])).signature
        bucket = self.ledger.current.get(sig)
        if bucket:
            rule = max(bucket, key=lambda r: r.version)
            return _apply_compiled(rule, self.adapter, in_raw)
        return None


def _apply_compiled(compiled, adapter, raw):
    kind = compiled.rule.get("kind")
    if kind == "meta":
        return adapter.apply_meta(compiled.rule["relation"], compiled.rule["params"], raw)
    if kind == "candidate":
        return adapter.apply_ops(compiled.rule["ops"], raw)
    if kind == "synth":
        return adapter.apply_ops(compiled.rule["ops"], raw)
    raise ValueError(f"unknown compiled rule kind: {kind}")


def _serialize_rule(rule) -> dict:
    if hasattr(rule, "relation"):          # MetaRule
        return {"kind": "meta", "relation": rule.relation, "params": _jsonable(rule.params)}
    if hasattr(rule, "ops"):               # Candidate
        return {"kind": "candidate", "ops": _jsonable(rule.ops)}
    return {"kind": "custom", "params": _jsonable(getattr(rule, "params", {}))}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item"):                 # numpy scalar
        return x.item()
    return x


def _raw_equal(pred, out_raw, adapter) -> bool:
    """Compare a predicted raw to the observed raw through the adapter."""
    try:
        ps = adapter.perceive(pred)
        os = adapter.perceive(out_raw)
        from plwm.inducer import _entities_key, _relations_key
        return _entities_key(ps) == _entities_key(os) and _relations_key(ps) == _relations_key(os)
    except Exception:
        # fallback: direct equality for array-like
        try:
            return bool((pred == out_raw).all())
        except Exception:
            return pred == out_raw

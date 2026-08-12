"""MetaEngine (2026-08-12): the experience -> meta-rule runtime loop.

Pulls a stream of (in, out) observation pairs through the pipeline:
    conserve_raw -> rule-ledger lookup (System 1 fast path)
    -> on miss: induce (System 2 slow path) -> verify -> compile -> persist.

This is the architecture's "learning machine": every encounter either reuses a
compiled rule (fast, System 1) or constructs a new one (slow, System 2) which
is then compiled for reuse.  Telemetry measures the compile-once-reuse-many
speed advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

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
    def __init__(self, adapter, relations: Sequence, *, rule_path: Path | None = None,
                 streaming: bool = False):
        self.adapter = adapter
        self.relations = list(relations)
        self.inducer = Inducer(relations)
        self.ledger = RuleLedger()
        self.stats = EngineStats()
        self.rule_path = rule_path
        self._last_compiled: object | None = None
        self._fallback_induce: object | None = None
        self.streaming = streaming
        self._stream_pairs: list[tuple] = []

    def set_fallback(self, fn) -> None:
        """Install a full-pipeline inducer (meta->hand-DSL->synth) as a last
        resort when the relation library cannot construct a rule."""
        self._fallback_induce = fn

    # --- persistence ---------------------------------------------------------

    def load(self) -> None:
        if self.rule_path and self.rule_path.exists():
            self.ledger.restore(self.rule_path)
            self.stats.restored = self.ledger.stats()["n_rules"]

    def save(self) -> None:
        if self.rule_path:
            self.ledger.save(self.rule_path)

    # --- the learning loop ---------------------------------------------------

    def process(self, pairs: Sequence[tuple]) -> object | None:
        """Consume ONE task's stream of (in_raw, out_raw) pairs.

        Each call is an independent induction context (tasks are independent).
        Returns the compiled rule for the task (or None if nothing induced).
        Rules are still stored in the shared ledger keyed by signature, so
        later tasks with the same signature reuse them (compile-once-reuse-many
        ACROSS tasks)."""
        task_pairs = list(pairs)
        rule_out = None
        for in_raw, out_raw in task_pairs:
            if self.streaming:
                self._stream_pairs.append((in_raw, out_raw))
            sig = self.adapter.conserve_raw(in_raw, out_raw).signature

            def _verify(rule):
                pred = _apply_compiled(rule, self.adapter, in_raw)
                return _raw_equal(pred, out_raw)

            cached = self.ledger.lookup(sig, _verify)
            if cached is not None:
                self.stats.hits += 1
                if rule_out is None:
                    rule_out = cached
                continue

            # System-2: slow constructive path.  Non-streaming: induce over the
            # whole task; streaming: induce over pairs seen SO FAR (the rule
            # may be changing over time -- later pairs must not pollute it).
            was_stale = self.ledger.is_stale(sig)
            induce_pairs = self._stream_pairs if self.streaming else task_pairs
            rule = self.inducer.induce(self.adapter, induce_pairs)
            kind = "meta"
            if rule is None and self._fallback_induce is not None:
                fb = self._fallback_induce(task_pairs)
                if fb is not None:
                    rule = fb
                    kind = "custom"
            if rule is None:
                self.stats.gaps += 1
                continue
            if was_stale:
                self.stats.revises += 1
            else:
                self.stats.inductions += 1
            serialized = _serialize_rule(rule)
            if kind == "custom":
                serialized["kind"] = "custom"
            # subgrid rules need the output shape appended so apply can crop
            if serialized.get("relation") == "subgrid":
                out_shape = task_pairs[0][1].shape
                params = serialized.get("params", [])
                if isinstance(params, list) and len(params) == 2:
                    params = list(params) + [out_shape[0], out_shape[1]]
                    serialized["params"] = params
            rel_key = getattr(rule, "relation_key",
                              getattr(rule, "relation", "pipeline"))
            compiled = self.ledger.compile(sig, serialized, rel_key)
            if rule_out is None:
                rule_out = compiled
        self._last_compiled = rule_out
        return rule_out

    def predict(self, in_raw):
        """Apply the rule compiled for THIS task (from process), or None."""
        compiled = getattr(self, "_last_compiled", None)
        if compiled is None:
            return None
        return _apply_compiled(compiled, self.adapter, in_raw)


def _apply_compiled(compiled, adapter, raw):
    kind = compiled.rule.get("kind")
    if kind in ("meta",):
        params = _unjsonable(compiled.rule.get("params", {}))
        relation = compiled.rule.get("relation")
        # logic/compose rules carry an op sequence under params["ops"]
        if relation == "compose" and isinstance(params, dict) and "ops" in params:
            return adapter.apply_ops(params["ops"], raw)
        return adapter.apply_meta(relation, params, raw)
    if kind in ("candidate", "synth", "custom"):
        # custom rules serialize their ops under the same shape as candidate
        ops = _unjsonable(compiled.rule.get("ops", []))
        if ops:
            return adapter.apply_ops(ops, raw)
        params = _unjsonable(compiled.rule.get("params", {}))
        relation = compiled.rule.get("relation")
        if relation:
            if isinstance(params, dict) and "ops" in params:
                return adapter.apply_ops(params["ops"], raw)
            return adapter.apply_meta(relation, params, raw)
    raise ValueError(f"unknown compiled rule kind: {kind}")


def _serialize_rule(rule) -> dict:
    if hasattr(rule, "relation"):          # MetaRule / InducedRule
        params = getattr(rule, "params", {})
        return {"kind": "meta", "relation": rule.relation, "params": _jsonable(params)}
    if hasattr(rule, "ops"):               # Candidate
        return {"kind": "candidate", "ops": _jsonable(rule.ops)}
    return {"kind": "custom", "params": _jsonable(getattr(rule, "params", {}))}


def _jsonable(x):
    """JSON-safe serialization that preserves tuple keys.

    dicts -> [["K", jsonable_key, jsonable_value], ...]; a tuple key is encoded
    as {"__tuple__": [...]}.  The "K" tag lets _unjsonable distinguish an
    int-key entry from a tuple-key entry without ambiguity."""
    if isinstance(x, dict):
        if not x:
            return {"__empty_dict__": True}
        out = []
        for k, v in x.items():
            if isinstance(k, tuple):
                key = {"__tuple__": [_jsonable(e) for e in k]}
            else:
                key = _jsonable(k)
            out.append(["K", key, _jsonable(v)])
        return out
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item"):                 # numpy scalar
        return x.item()
    return x


def _unjsonable(x):
    """Inverse of _jsonable: rebuilds dicts (tuple keys restored)."""
    if isinstance(x, list):
        if x and all(isinstance(e, list) and len(e) == 3 and e[0] == "K" for e in x):
            out = {}
            for tag, k, v in x:
                key = tuple(_unjsonable(e) for e in k["__tuple__"]) \
                    if isinstance(k, dict) and "__tuple__" in k else _unjsonable(k)
                out[key] = _unjsonable(v)
            return out
        return [_unjsonable(v) for v in x]
    if isinstance(x, dict):
        if "__empty_dict__" in x:
            return {}
        return {k: _unjsonable(v) for k, v in x.items()}
    return x


def _raw_equal(pred, out) -> bool:
    try:
        return bool((np.asarray(pred) == np.asarray(out)).all())
    except Exception:
        return pred == out

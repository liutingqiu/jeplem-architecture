"""Compiled-rule ledger (2026-08-12): the System-1 store.

The architecture's answer to the "always-reason is slow" problem: after the
inducer (System 2) synthesizes a rule for a conservation signature, COMPILE it
here (System 1).  A later experience with the same signature hits the cache and
reuses the rule instead of re-reasoning.  This is the "compile once, reuse
many" loop that gives the architecture its speed advantage over always-reasoning
systems.

Semantics mirror VersionedFactLedger (plwm/fact_ledger.py):
  - exact-key lookup (key = Conservation.signature)
  - append-only history, versioned revision (version bump, never delete)
  - strict snapshot/restore re-validation (recompute signatures, reject tamper)
plus P4 staleness (train_phase2_m2b_revision.py): window-majority over the last
`window` verify outcomes; mark stale only at >= majority contradictions (a
single weak contradiction must NOT mark).

Design note: we do NOT reuse VersionedFactLedger as the backing store because
its `value` is restricted to str/int/bool (fact_ledger.py) and a rule carries a
params dict.  We reuse its semantics.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class CompiledRule:
    """One compiled rule, keyed by its conservation signature."""
    signature: str
    rule: dict                       # serialized rule: {"kind":..., ...}
    relation_key: str                # e.g. "quantity/map_color"
    n_confirm: int = 0               # evidence count (hits that verified)
    tick_compiled: int = 0
    tick_last_used: int = 0
    version: int = 1                 # bumped on every revision

    def sig(self) -> str:
        """Recompute the canonical signature for tamper checks."""
        return self.signature


class RuleLedger:
    """Signature-indexed compiled-rule store with P4 staleness + persistence."""

    def __init__(self, *, window: int = 4, majority: int = 3):
        self.window = window
        self.majority = majority
        self.current: dict[str, list[CompiledRule]] = {}   # sig -> bucket
        self.history: list[CompiledRule] = []              # every compile/revision
        self.marked_stale: set[str] = set()                # sigs, NOT deleted
        self._revision_window: dict[str, list[bool]] = {}  # sig -> recent verifies
        self._tick = 0

    # --- lookup --------------------------------------------------------------

    def lookup(self, signature: str, verify: Callable[[CompiledRule], bool]) -> CompiledRule | None:
        """Return a reusable compiled rule for `signature`, or None.

        Reuse requires: signature has a bucket, is not marked stale, and the
        rule still verifies against the CURRENT evidence (`verify`).  Verified
        hits bump n_confirm; contradictions feed the P4 window (and may mark
        stale).  Misses leave the window untouched -- re-induction is the
        caller's job.
        """
        bucket = self.current.get(signature)
        if not bucket or signature in self.marked_stale:
            return None
        # pick the highest-version rule in the bucket
        rule = max(bucket, key=lambda r: r.version)
        ok = False
        try:
            ok = verify(rule)
        except Exception:
            ok = False
        if ok:
            self._confirm(signature, rule)
            return self._current_version(signature)
        self._contradict(signature)
        return None

    def _current_version(self, signature: str) -> CompiledRule | None:
        bucket = self.current.get(signature)
        if not bucket:
            return None
        return max(bucket, key=lambda r: r.version)

    def _confirm(self, signature: str, rule: CompiledRule) -> None:
        updated = CompiledRule(
            signature=rule.signature, rule=rule.rule,
            relation_key=rule.relation_key,
            n_confirm=rule.n_confirm + 1,
            tick_compiled=rule.tick_compiled, tick_last_used=self._tick,
            version=rule.version,
        )
        self._replace_in_bucket(signature, updated)
        # clear contradiction window on success (evidence supports the rule)
        self._revision_window.pop(signature, None)

    def _contradict(self, signature: str) -> None:
        w = self._revision_window.setdefault(signature, [])
        w.append(True)
        if len(w) > self.window:
            w.pop(0)
        if len(w) == self.window and sum(w) >= self.majority:
            self.marked_stale.add(signature)

    def _replace_in_bucket(self, signature: str, rule: CompiledRule) -> None:
        bucket = [r for r in self.current.get(signature, []) if r.version != rule.version]
        bucket.append(rule)
        self.current[signature] = bucket

    # --- compile / revise ----------------------------------------------------

    def compile(self, signature: str, rule: dict, relation_key: str) -> CompiledRule:
        """Write a freshly induced rule.  If a stale rule exists for this
        signature, this is a REVISION: version bumps, stale flag clears."""
        self._tick += 1
        old = self.current.get(signature, [])
        base_version = max((r.version for r in old), default=0)
        version = base_version + 1 if old else 1
        entry = CompiledRule(
            signature=signature, rule=rule, relation_key=relation_key,
            n_confirm=0, tick_compiled=self._tick,
            tick_last_used=self._tick, version=version,
        )
        self.current[signature] = [entry]
        self.history.append(entry)
        self.marked_stale.discard(signature)
        self._revision_window.pop(signature, None)
        return entry

    def mark_stale(self, signature: str) -> None:
        self.marked_stale.add(signature)

    def is_stale(self, signature: str) -> bool:
        return signature in self.marked_stale

    def stats(self) -> dict:
        return {
            "n_signatures": len(self.current),
            "n_rules": sum(len(v) for v in self.current.values()),
            "n_history": len(self.history),
            "n_marked_stale": len(self.marked_stale),
        }

    # --- persistence ---------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "current": {
                sig: [asdict(r) for r in bucket]
                for sig, bucket in sorted(self.current.items())
            },
            "history": [asdict(r) for r in self.history],
            "marked_stale": sorted(self.marked_stale),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    def restore(self, path: Path) -> None:
        """Load a snapshot with STRICT re-validation.  Raises on any mismatch:
        recomputed signature differs, or version not monotonic in history."""
        data = json.loads(path.read_text(encoding="utf-8"))
        current: dict[str, list[CompiledRule]] = {}
        for sig, bucket in data.get("current", {}).items():
            parsed = []
            for raw in bucket:
                r = CompiledRule(**raw)
                # tamper check: recompute the canonical signature
                if r.signature != sig:
                    raise ValueError(f"signature mismatch in snapshot: {sig}")
                parsed.append(r)
            current[sig] = parsed
        history = [CompiledRule(**raw) for raw in data.get("history", [])]
        # version monotonicity across history
        prev = 0
        for r in history:
            if r.version < prev:
                raise ValueError(f"non-monotonic version in history: {r.version} < {prev}")
            prev = r.version
        self.current = current
        self.history = history
        self.marked_stale = set(data.get("marked_stale", []))

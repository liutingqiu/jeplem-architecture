"""Inducer (2026-08-12): conservation-signature-driven relation selection.

Given the adapter's scenes + all observed (in, out) pairs, pick a relation
whose fit derives params that reproduce EVERY pair (symbolic execution), and
Occam-select the simplest.  This is the System-2 path: slow, constructive,
runs only on cache misses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class InducedRule:
    """A candidate rule before it is compiled into the ledger."""
    relation_key: str              # e.g. "quantity/map_color"
    params: object                 # fit-derived params
    complexity: int                # Occam cost
    apply: Callable                # (params, raw_in) -> raw_out


class Inducer:
    def __init__(self, relations: Sequence):
        self.relations = list(relations)

    def induce(self, adapter, pairs: Sequence[tuple]) -> InducedRule | None:
        """Given all observed (in_raw, out_raw) pairs, find the simplest
        relation reproducing ALL of them.  None = honest gap."""
        if not pairs:
            return None
        in0, out0 = pairs[0]
        in_s = adapter.perceive(in0)
        out_s = adapter.perceive(out0)
        sig = adapter.conserve(in_s, out_s)

        best = None
        for rel in self.relations:
            # fire only on the changed dims this relation targets
            if not set(rel.fired_by) & set(sig.changed):
                continue
            try:
                params = rel.fit(in_s, out_s)
            except Exception:
                continue
            if params is None:
                continue
            # verify against ALL pairs
            ok = True
            try:
                for ing, outg in pairs:
                    pred = rel.apply(params, adapter.perceive(ing))
                    if not _scene_equal(pred, adapter.perceive(outg)):
                        ok = False
                        break
            except Exception:
                ok = False
            if not ok:
                continue
            cost = _complexity(rel, params)
            if best is None or cost < best.complexity:
                best = InducedRule(rel.key, params, cost, rel.apply)
        return best


def _scene_equal(a, b) -> bool:
    """Scene equality: same entities (id/type/properties) + relations."""
    return _entities_key(a) == _entities_key(b) and _relations_key(a) == _relations_key(b)


def _entities_key(scene) -> tuple:
    return tuple(
        (e.id, e.type, tuple(sorted(e.properties.items())))
        for e in scene.entities
    )


def _relations_key(scene) -> tuple:
    return tuple(
        (r.rel_type, r.subject, r.object, _hashable(r.value))
        for r in scene.relations
    )


def _hashable(v):
    if isinstance(v, dict):
        return ("dict", tuple(sorted((k, _hashable(x)) for k, x in v.items())))
    if isinstance(v, (list, tuple)):
        return ("list", tuple(_hashable(x) for x in v))
    return v


def _complexity(rel, params) -> int:
    """Occam cost: relation base + param size."""
    base = 1
    try:
        if isinstance(params, dict):
            return base + len(params)
        if isinstance(params, (list, tuple)):
            return base + len(params)
    except Exception:
        pass
    return base + 1

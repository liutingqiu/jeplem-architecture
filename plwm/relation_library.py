"""Six-family relation library (2026-08-12).

A relation is a fit-and-apply pair over EntityScene data:
    fit(in_scene, out_scene) -> params | None   (None = this relation doesn't fire)
    apply(params, in_scene)   -> out_scene      (pure, deterministic)

Invariant: apply(fit(in, out), in) == out for at least one pair; the Inducer
then verifies across ALL observed pairs by symbolic execution (exact match) to
preserve the 100%-precision guarantee.

The six meta-rule families the user endorsed, mapped onto the entity-property-
relation model:
    1. quantity   -- property/count mapping: per-value, per-position, per-feature
    2. causality  -- (cause property of A) -> (effect property of B), invariant
    3. geometry   -- spatial relations: same_row, adjacent, encloses, symmetric
    4. interaction-- binary relation (A,B) -> outcome property, invariant across pairs
    5. social     -- a special interaction over other agents' behavior
    6. logic      -- conditional composition: if property(P, entity) then apply(R)

Family 1 is seeded from the existing ARC relations (arc_meta_rule).  Families
2-6 are registered with stub fit/apply bodies -- the INTERFACE is real, the
implementation is future work.  This module has no dependency on arc_* itself;
the adapter injects the ARC relations via `register`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass(frozen=True)
class RelationFunc:
    family: str
    name: str
    fired_by: tuple[str, ...]       # conservation `changed` keys that fire this
    fit: Callable                   # (in_scene, out_scene) -> params | None
    apply: Callable                 # (params, in_scene) -> out_scene

    @property
    def key(self) -> str:
        return f"{self.family}/{self.name}"


def _stub_fit(in_scene, out_scene):
    raise NotImplementedError("family relation not yet implemented")


def _stub_apply(params, in_scene):
    raise NotImplementedError("family relation not yet implemented")


def relation_catalog() -> list[RelationFunc]:
    """The built-in relation catalog.  Family-1 relations are injected by the
    adapter (they need the arc_meta_rule bodies); families 2-6 are stubs."""
    catalog: list[RelationFunc] = []
    # family 2: causality -- cause_effect: invariant (A-feature -> B-effect) map
    catalog.append(RelationFunc(
        "causality", "cause_effect", ("causes",), _stub_fit, _stub_apply))
    # family 3: geometry -- spatial relation classes
    catalog.append(RelationFunc(
        "geometry", "align", ("positions",), _stub_fit, _stub_apply))
    catalog.append(RelationFunc(
        "geometry", "symmetric", ("positions",), _stub_fit, _stub_apply))
    # family 4: interaction -- binary-relation -> outcome
    catalog.append(RelationFunc(
        "interaction", "interact", ("interaction",), _stub_fit, _stub_apply))
    # family 5: social -- same interface, other-agent outcome
    catalog.append(RelationFunc(
        "social", "social_outcome", ("social",), _stub_fit, _stub_apply))
    # family 6: logic -- conditional composition
    catalog.append(RelationFunc(
        "logic", "compose", ("logical",), _stub_fit, _stub_apply))
    return catalog


def register_family1(catalog: list[RelationFunc],
                     relations: Mapping[str, tuple]) -> list[RelationFunc]:
    """Inject the adapter-provided family-1 (quantity/geometry) relations.

    `relations` maps relation name -> (family, fired_by, fit, apply).  The
    adapter wraps its arc_meta_rule bodies so this module stays arc-free.
    """
    for name, (family, fired_by, fit, apply) in relations.items():
        catalog.append(RelationFunc(family, name, tuple(fired_by), fit, apply))
    return catalog

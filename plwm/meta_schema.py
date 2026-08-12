"""Modality-agnostic meta-rule schema (2026-08-12).

The general "experience -> meta-rule" engine's data model.  This module is the
ONLY one that knows nothing about grids vs voxels vs web: it defines entities,
relations, scenes, and the conservation signature that drives rule induction.

Core idea (the architecture's thesis): the lowest rule in any domain is
"what is conserved, what changed, and how is the change constrained".  A
Conservation is the modality-neutral statement of that: which properties /
relations / counts are invariant under a transition, and which changed.

An adapter (ARC grid, Luanti voxel, web) projects its raw observations onto
this model: perceive(raw) -> EntityScene; conserve(in_scene, out_scene) ->
Conservation.  The engine never touches raw input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MetaEntity:
    """A perceivable object/tile/agent in one scene."""
    id: str                          # stable token within one scene, e.g. "o0"
    type: str                        # modality-chosen: "object"/"tile"/"agent"
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetaRelation:
    """A relation between entities (or a global/scalar statement)."""
    rel_type: str                    # "same_row" | "adjacent" | "causes" | ...
    subject: str                     # entity id, or "self" for scalar/global
    object: str                      # entity id, or "self" for unary/global
    value: Any = True                # e.g. True for adjacency, mapping for causes


@dataclass(frozen=True)
class EntityScene:
    """A perceived state: entities + the relations between them."""
    entities: Sequence[MetaEntity] = ()
    relations: Sequence[MetaRelation] = ()


@dataclass(frozen=True)
class Conservation:
    """The general conservation signature.  `signature` is the rule-cache key.

    - properties: property keys conserved across the transition
      (e.g. "grid_size", "colors", "shapes", "agent_identity")
    - relations:  relation types conserved ("same_row", "adjacent", "causes")
    - counts:     entity-count-by-type conserved ("object-count", "tile-count")
    - changed:    property/relation keys that CHANGED -- this drives which
      relation family fires (family 1 = colors changed, family 2 = causes, ...)
    """
    properties: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    counts: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        """Canonical, order-independent cache key."""
        parts = [
            "P:" + ",".join(sorted(self.properties)),
            "R:" + ",".join(sorted(self.relations)),
            "C:" + ",".join(sorted(self.counts)),
            "X:" + ",".join(sorted(self.changed)),
        ]
        return "|".join(parts)


def to_scene(entities: Sequence[MetaEntity],
             relations: Sequence[MetaRelation] = ()) -> EntityScene:
    return EntityScene(tuple(entities), tuple(relations))

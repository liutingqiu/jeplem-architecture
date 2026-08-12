"""Luanti PerceptionAdapter skeleton (2026-08-12).

Proves the meta-rule engine's PerceptionAdapter protocol is real beyond ARC:
the same engine that learns from ARC grids can attach to the Luanti voxel
world by implementing the same three methods:
    perceive(raw) -> EntityScene
    conserve_raw(in_raw, out_raw) -> Conservation  (signature = cache key)
    familyX_relations() -> relation registry

Luanti raw input = a LuantiState (structured task state) + an 11x11 tile-grid
vision encoding.  This skeleton:
  - perceives the state as an agent entity + vision as tile entities
  - declares the Luanti conservation signature (agent/goal identity, wall
    adjacency invariant under movement)
  - registers family-2 (causality: controls_inverted identity-vs-inversion)
    and family-4 (interaction: key/decoy outcome) relations as STUBS

No live Luanti process is required -- this is the interface, not the rollout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plwm"))

from plwm.meta_schema import Conservation, EntityScene, MetaEntity, MetaRelation

TILES = ("air", "wall", "goal", "key", "decoy", "beacon", "agent")
VISION_RADIUS = 5


class LuantiAdapter:
    """Luanti state+vision -> generic scene projection (skeleton)."""

    # --- perceive ------------------------------------------------------------

    def perceive(self, raw) -> EntityScene:
        """raw = (state_dict, vision_cells) where vision_cells is the 11x11
        tile grid from 'luanti-vision' (list of tile strings)."""
        state, vision = raw
        entities = []
        relations = []

        # agent entity from state
        entities.append(MetaEntity(
            id="agent",
            type="agent",
            properties={
                "x": int(state.get("x", 0)),
                "z": int(state.get("z", 0)),
                "goal_x": int(state.get("goal_x", 0)),
                "goal_z": int(state.get("goal_z", 0)),
                "health": float(state.get("health", 0.0)),
                "completed": bool(state.get("completed", False)),
                "blocked": tuple(sorted(k for k in ("north", "south", "east", "west")
                                       if state.get(f"blocked_{k}"))),
            },
        ))

        # tile entities from vision (exclude the agent tile -- agent is subject)
        n = VISION_RADIUS * 2 + 1
        for i, tile in enumerate(vision):
            r, c = i // n, i % n
            if tile == "agent":
                continue
            if tile == "air":
                continue  # air is background; walls/goals/keys are entities
            entities.append(MetaEntity(
                id=f"tile_{r}_{c}",
                type="tile",
                properties={
                    "tile": tile,
                    "pos": (r - VISION_RADIUS, c - VISION_RADIUS),
                },
            ))
            # adjacency to agent (agent is at vision center 0,0)
            dr, dc = r - VISION_RADIUS, c - VISION_RADIUS
            if abs(dr) <= 1 and abs(dc) <= 1:
                relations.append(MetaRelation("adjacent", "agent", f"tile_{r}_{c}", True))
        return EntityScene(tuple(entities), tuple(relations))

    # --- conserve ------------------------------------------------------------

    def conserve_raw(self, in_raw, out_raw) -> Conservation:
        """Declared Luanti conservation signature: agent identity + goal
        identity + tile counts are conserved under movement; position, blocked
        flags, and collected flags change."""
        in_s = self.perceive(in_raw)
        out_s = self.perceive(out_raw)
        in_agent = next(e for e in in_s.entities if e.type == "agent")
        out_agent = next(e for e in out_s.entities if e.type == "agent")

        properties, counts, changed = [], [], []
        # agent identity conserved (the agent is the same agent)
        properties.append("agent_identity")
        # goal identity conserved (the goal does not move)
        if (in_agent.properties["goal_x"], in_agent.properties["goal_z"]) == \
           (out_agent.properties["goal_x"], out_agent.properties["goal_z"]):
            properties.append("goal_identity")
        else:
            changed.append("goal")
        # wall adjacency conserved (walls don't move)
        in_walls = {e.properties["pos"] for e in in_s.entities if e.properties.get("tile") == "wall"}
        out_walls = {e.properties["pos"] for e in out_s.entities if e.properties.get("tile") == "wall"}
        if in_walls == out_walls:
            properties.append("wall_set")
        else:
            changed.append("walls")
        # position changes
        if (in_agent.properties["x"], in_agent.properties["z"]) != \
           (out_agent.properties["x"], out_agent.properties["z"]):
            changed.append("position")
        # blocked flags change
        if in_agent.properties["blocked"] != out_agent.properties["blocked"]:
            changed.append("blocked")
        # counts
        in_tile_counts = {e.properties["tile"]: 1 for e in in_s.entities if e.type == "tile"}
        out_tile_counts = {e.properties["tile"]: 1 for e in out_s.entities if e.type == "tile"}
        if set(in_tile_counts) == set(out_tile_counts):
            counts.append("tile-count")

        return Conservation(
            properties=tuple(properties),
            relations=tuple(sorted({r.rel_type for r in in_s.relations})),
            counts=tuple(counts),
            changed=tuple(changed),
        )

    # --- apply stubs (family 2 causality, family 4 interaction) --------------

    def apply_meta(self, relation: str, params, raw):
        raise NotImplementedError("Luanti causal/interaction relations not yet implemented")

    def apply_ops(self, ops, raw):
        raise NotImplementedError("Luanti op-sequence rules not yet implemented")

    # --- family relations (stubs: interface is real) -------------------------

    def causality_relations(self) -> dict:
        """Family 2: controls_inverted -- the hidden control-inversion mapping
        (identity vs inverted) is exactly a cause_effect relation: which
        action the agent must take is caused by the (hidden) inversion state.
        Stub body: the interface is real, the implementation is future work."""
        def _fit(in_raw, out_raw):
            raise NotImplementedError("cause_effect fit not yet implemented (Luanti)")

        def _apply(params, in_raw):
            raise NotImplementedError("cause_effect apply not yet implemented (Luanti)")

        return {
            "cause_effect": ("causality", ("blocked", "position"), _fit, _apply),
        }

    def interaction_relations(self) -> dict:
        """Family 4: key/decoy -- whether collecting a colored object is good
        is an interaction relation (A=agent, B=object) -> outcome.  Stub."""
        def _fit(in_raw, out_raw):
            raise NotImplementedError("interact fit not yet implemented (Luanti)")

        def _apply(params, in_raw):
            raise NotImplementedError("interact apply not yet implemented (Luanti)")

        return {
            "interact": ("interaction", ("goal", "position"), _fit, _apply),
        }

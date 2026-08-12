"""ARC PerceptionAdapter (2026-08-12): project ARC grids onto the generic
meta-rule engine.

The engine (plwm/meta_*) never imports arc_*; this adapter is the bridge.  It
implements the PerceptionAdapter protocol over RAW inputs (for ARC, raw = grid):

    perceive(raw) -> EntityScene          (for the inducer's scene-level view)
    conserve_raw(in_raw, out_raw) -> Conservation   (signature = cache key)
    differ(in_raw, out_raw) -> guidance
    apply_meta(relation, params, raw) / apply_ops(ops, raw)  (compiled-rule reuse)
    family1_relations() -> quantity/geometry RelationFuncs over raw grids
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from arc_perception import perceive as arc_perceive
from arc_diff import scene_diff
from plwm.meta_schema import Conservation, EntityScene, MetaEntity, MetaRelation


class ArcAdapter:
    """ARC grid -> generic scene projection."""

    # --- perceive ------------------------------------------------------------

    def perceive(self, grid: np.ndarray) -> EntityScene:
        scene = arc_perceive(np.asarray(grid, dtype=int))
        entities = []
        relations = []
        for i, o in enumerate(scene.objects):
            entities.append(MetaEntity(
                id=f"o{i}",
                type="object",
                properties={
                    "color": int(o.color),
                    "area": int(o.area),
                    "shape_key": o.shape_key,
                    "pos_rc": (int(o.r), int(o.c)),
                    "h": int(o.h),
                    "w": int(o.w),
                },
            ))
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                ar, ac = a.properties["pos_rc"]
                br, bc = b.properties["pos_rc"]
                if ar == br:
                    relations.append(MetaRelation("same_row", a.id, b.id, True))
                if ac == bc:
                    relations.append(MetaRelation("same_col", a.id, b.id, True))
                if abs(ar - br) <= 1 and abs(ac - bc) <= 1:
                    relations.append(MetaRelation("adjacent", a.id, b.id, True))
        return EntityScene(tuple(entities), tuple(relations))

    # --- conserve (raw) ------------------------------------------------------

    def conserve_raw(self, in_g, out_g) -> Conservation:
        """ARC conservation: size/colors/count/shapes, projected onto the
        generic schema.  This is the thin projection of arc_meta_rule.conserve.

        Relations (same_row/adjacent/same_col) are deliberately EXCLUDED from
        the signature: they are scene-structure noise that varies per pair
        within one task (e.g. a pure color-remap task has different relation
        sets across pairs), so including them would break the cache key.  The
        rule is decided by property-level conservation, not relation chance."""
        in_g = np.asarray(in_g, dtype=int)
        out_g = np.asarray(out_g, dtype=int)
        in_s = self.perceive(in_g)
        out_s = self.perceive(out_g)

        size = in_g.shape == out_g.shape
        colors = set(in_g.flatten().tolist()) == set(out_g.flatten().tolist())
        count = len(in_s.entities) == len(out_s.entities)
        in_shapes = sorted(e.properties["shape_key"] for e in in_s.entities)
        out_shapes = sorted(e.properties["shape_key"] for e in out_s.entities)
        shapes = in_shapes == out_shapes

        properties, counts, changed = [], [], []
        if size:
            properties.append("grid_size")
        else:
            changed.append("size")
        if colors:
            properties.append("colors")
        else:
            changed.append("colors")
        if shapes:
            properties.append("shapes")
        else:
            changed.append("shapes")
        if count:
            counts.append("object-count")
        else:
            changed.append("count")

        return Conservation(
            properties=tuple(properties),
            relations=(),              # relations excluded (see docstring)
            counts=tuple(counts),
            changed=tuple(changed),
        )

    # --- differ (guidance) ---------------------------------------------------

    def differ(self, in_g, out_g):
        return scene_diff(np.asarray(in_g, dtype=int), np.asarray(out_g, dtype=int))

    # --- apply dispatch (compiled-rule reuse) --------------------------------

    def apply_meta(self, relation: str, params, raw) -> np.ndarray:
        from arc_meta_rule import MetaRule
        g = np.asarray(raw, dtype=int)
        if relation == "subgrid":
            # params: [r0, c0] or [r0, c0, ho, wo] (shape appended at compile)
            if len(params) >= 4:
                r0, c0, ho, wo = params[0], params[1], params[2], params[3]
                return g[r0:r0 + ho, c0:c0 + wo].copy()
            r0, c0 = params[0], params[1]
            if hasattr(self, "_subgrid_dims"):
                ho, wo = self._subgrid_dims
                return g[r0:r0 + ho, c0:c0 + wo].copy()
            return g
        if relation == "stack":
            # params serialized as {"stack_v": n} or {"stack_h": n}
            if isinstance(params, dict):
                if "stack_v" in params:
                    return np.tile(g, (params["stack_v"], 1))
                if "stack_h" in params:
                    return np.tile(g, (1, params["stack_h"]))
            return g
        return MetaRule(relation, params, 0).apply(g)

    def apply_ops(self, ops, raw) -> np.ndarray:
        from arc_rule_dsl import OPS
        g = np.asarray(raw, dtype=int)
        for name, args in ops:
            g = OPS[name](g, *args)
        return g

    # --- family-1 relations (quantity/geometry) ------------------------------

    def family1_relations(self) -> dict:
        """Wrap arc_meta_rule relations as RelationFunc fit/apply over raw grids."""
        from arc_meta_rule import (
            rel_map_color, rel_map_row_feature, rel_map_position,
            rel_map_object, rel_count_remove_color, rel_fill_holes,
            rel_crop_content, rel_subgrid, rel_stack,
        )

        def _fit(rel_fn):
            return lambda g, og: rel_fn(np.asarray(g, dtype=int), np.asarray(og, dtype=int))

        def _apply_meta(relation):
            def apply(params, g):
                from arc_meta_rule import MetaRule
                return MetaRule(relation, params, 0).apply(np.asarray(g, dtype=int))
            return apply

        def _apply_stack():
            def apply(params, g):
                gg = np.asarray(g, dtype=int)
                if "stack_v" in params:
                    return np.tile(gg, (params["stack_v"], 1))
                if "stack_h" in params:
                    return np.tile(gg, (1, params["stack_h"]))
                return gg
            return apply

        def _apply_subgrid():
            def apply(params, g):
                gg = np.asarray(g, dtype=int)
                r0, c0 = params
                if hasattr(self, "_subgrid_dims"):
                    ho, wo = self._subgrid_dims
                    return gg[r0:r0 + ho, c0:c0 + wo].copy()
                return gg
            return apply

        return {
            "map_color": ("quantity", ("colors",), _fit(rel_map_color), _apply_meta("map_color")),
            "map_object": ("quantity", ("colors",), _fit(rel_map_object), _apply_meta("map_object")),
            "map_row_feature": ("quantity", ("colors",), _fit(rel_map_row_feature), _apply_meta("map_row_feature")),
            "map_position": ("quantity", ("colors",), _fit(rel_map_position), _apply_meta("map_position")),
            "remove_color": ("quantity", ("count",), _fit(rel_count_remove_color), _apply_meta("remove_color")),
            "fill_holes": ("quantity", ("count",), _fit(rel_fill_holes), _apply_meta("fill_background")),
            "crop_content": ("geometry", ("size",), _fit(rel_crop_content), _apply_meta("crop_content")),
            "subgrid": ("geometry", ("size",), _fit(rel_subgrid), _apply_subgrid()),
            "stack": ("geometry", ("size",), _fit(rel_stack), _apply_stack()),
        }

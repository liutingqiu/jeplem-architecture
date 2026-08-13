"""Library growth by compression: recurring 2-step fragments (a grid-op applied
after an object-rule schema, e.g. ``crop_to_objects(schema[largest>recolor(c)>all])``)
are abstracted into NEW atomic primitives with free parameter slots, stored with
provenance in the ledger, and reused.  This is "learn new primitives from data":
the vocabulary grows from experience, enabling deeper compositions within the
search-depth budget.  Re-parameterisation per task uses consistency search (the
G2-verified tool); no dense gradient.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .grid import Grid
from .schemas import (
    _apply_schema, _pred_all, _pred_largest, _pred_smallest,
    _pred_odd_color, _pred_unique_shape, _pred_region,
)
from . import primitives

COLORS = list(range(10))

# relative predicates resolvable by NAME for any grid (no baked parameters)
PRED_BY_NAME = {
    "all": _pred_all,
    "largest": _pred_largest,
    "smallest": _pred_smallest,
    "odd_color": _pred_odd_color,
    "unique_shape": _pred_unique_shape,
    "region(top)": lambda g, o: _pred_region("top")(g, o),
    "region(bottom)": lambda g, o: _pred_region("bottom")(g, o),
    "region(left)": lambda g, o: _pred_region("left")(g, o),
    "region(right)": lambda g, o: _pred_region("right")(g, o),
}

# parameterless grid-ops reusable as the outer step of a compressed fragment
FIXED_OPS = {
    "flip_h": primitives._flip_h,
    "flip_v": primitives._flip_v,
    "rotate90": primitives._rotate90,
    "rotate180": primitives._rotate180,
    "rotate270": primitives._rotate270,
    "transpose": primitives._transpose,
    "keep_largest_object": primitives._keep_largest_object,
    "crop_to_objects": primitives._crop_to_objects,
    "fill_enclosed": primitives._fill_enclosed,
    "box_objects": primitives._box_objects,
}

FRAGMENT_RE = re.compile(
    r"^([a-z0-9_]+)\(schema\[([^>]+)>([a-z]+)(?:\(\d+\))?>([a-z]+)\]\)$")


def fragment_of(name: str) -> tuple | None:
    """crop_to_objects(schema[largest>recolor(5)>all]) ->
    ('crop_to_objects','largest','recolor','all').  None if not a compressible
    2-step fragment over the fixed ops / relative predicates."""
    m = FRAGMENT_RE.match(name)
    if not m:
        return None
    op, pred, tkind, mode = m.group(1), m.group(2), m.group(3), m.group(4)
    if op not in FIXED_OPS or pred not in PRED_BY_NAME:
        return None
    if tkind not in ("recolor", "delete", "box", "id"):
        return None
    return (op, pred, tkind, mode)


@dataclass
class GrownLibrary:
    """Observes fragments from solved tasks, abstracts recurring ones into
    atomic primitives, and yields re-parameterised candidates for search."""

    fragments: dict = field(default_factory=dict)  # fragment -> {'count','tasks'}
    active: set = field(default_factory=set)

    def observe(self, fragment: tuple, task_id: str) -> None:
        e = self.fragments.setdefault(fragment, {"count": 0, "tasks": set()})
        e["count"] += 1
        e["tasks"].add(task_id)

    def learn(self, min_occur: int = 2) -> None:
        self.active = {f for f, e in self.fragments.items() if e["count"] >= min_occur}

    def candidates(self, grid: Grid) -> list[tuple[str, object]]:
        """(name, fn) atomic learned-primitive candidates, parameters swept."""
        out: list[tuple[str, object]] = []
        for op, pred, tkind, mode in sorted(self.active):
            pred_of = PRED_BY_NAME[pred]
            gop = FIXED_OPS[op]
            if tkind == "recolor":
                for c in COLORS:
                    fn = (lambda g, p=pred_of, k=tkind, col=c, o=gop, m=mode:
                          o(_apply_schema(g, p, (k, col), m)))
                    out.append((f"L[{op}({pred}>{tkind}({c})>{mode})]", fn))
            else:
                fn = (lambda g, p=pred_of, k=tkind, o=gop, m=mode:
                      o(_apply_schema(g, p, (k, None), m)))
                out.append((f"L[{op}({pred}>{tkind}>{mode})]", fn))
        return out

    def provenance(self) -> list[dict]:
        return [{"fragment": f, "count": e["count"], "tasks": sorted(e["tasks"])[:5]}
                for f, e in sorted(self.fragments.items())]

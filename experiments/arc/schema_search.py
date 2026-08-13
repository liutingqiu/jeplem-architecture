"""Consistency-based induction over the schema grammar.

Mirrors ``induction.search_program`` (the verified tool: brute-force search +
consistency check reached 100% on expressible families in G2, where the dense
MLP atom mapper failed) but enumerates schema candidates instead of grid
primitives.  Depth extends by composing a schema with a base grid primitive
(grid-op after schema, and schema after grid-op).
"""
from __future__ import annotations

from . import primitives
from .grid import Grid, equal
from .schemas import schema_candidates


def _fits(prog, demos) -> bool:
    for x, y in demos:
        try:
            if not equal(prog(x), y):
                return False
        except Exception:
            return False
    return True


def search_schema(demos, max_steps: int = 2, max_composed: int = 3000,
                  library=None):
    """Return (name, fn) mapping every demo input to its output, else None.
    depth 1 = a single schema (plus any grown-library atomic primitives);
    depth 2 = grid-op(schema(g)) or schema(grid-op(g)).  ``max_composed`` caps
    the composed-candidate sweep."""
    if not demos:
        return None
    x0, y0 = demos[0]
    # depth 1: single schema (+ grown-library primitives as atomic steps).
    # Grown-library primitives are the known-good vocabulary: try them FIRST.
    schemas: list[tuple[str, Grid]] = []
    base_cands = schema_candidates(x0)
    lib_cands = library.candidates(x0) if library is not None else []
    for name, fn in lib_cands + base_cands:
        try:
            if equal(fn(x0), y0) and _fits(fn, demos):
                return (name, fn)
        except Exception:
            continue
        schemas.append((name, fn))
    if max_steps < 2:
        return None
    # non-trivial schemas only for composition (id-transform schemas reduce to
    # plain grid ops, which the base induction already covers)
    nontrivial = [s for s in schemas if not s[0].endswith(">id>all]")
                  and not s[0].endswith(">id>filter]")]
    # depth 2a: grid-op applied AFTER the schema / learned primitive
    n_checked = 0
    for name1, fn1 in nontrivial:
        for name2, fn2 in primitives.candidates(fn1(x0)):
            prog = lambda g, f1=fn1, f2=fn2: f2(f1(g))
            try:
                if _fits(prog, demos):
                    return (f"{name2}({name1})", prog)
            except Exception:
                continue
            n_checked += 1
            if n_checked >= max_composed:
                return None
    # depth 2b: schema applied AFTER a grid-op
    n_checked = 0
    for name2, fn2 in primitives.candidates(x0):
        try:
            g1 = fn2(x0)
        except Exception:
            continue
        for name1, fn1 in schema_candidates(g1):
            if name1.endswith(">id>all]") or name1.endswith(">id>filter]"):
                continue
            prog = lambda g, f1=fn1, f2=fn2: f1(f2(g))
            try:
                if _fits(prog, demos):
                    return (f"{name1}({name2})", prog)
            except Exception:
                continue
            n_checked += 1
            if n_checked >= max_composed:
                return None
    return None


def solve_task(task: dict, max_steps: int = 2, max_composed: int = 3000,
               library=None):
    """task: {"train": [{input, output}], "test": [{input, output}]}
    Returns (program_name, predicted_test_output) or (None, None)."""
    demos = [(g["input"], g["output"]) for g in task["train"]]
    prog = search_schema(demos, max_steps=max_steps, max_composed=max_composed,
                         library=library)
    if prog is None:
        return None, None
    name, fn = prog
    test_in = task["test"][0]["input"]
    return name, fn(test_in)

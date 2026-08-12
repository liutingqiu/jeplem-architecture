"""Brute-force program induction over the primitive library.

``search_program(demos, max_depth)`` finds a (name, fn) whose output equals
every demo's expected output; ``solve_task`` applies it to a held-out test
input.  This is the explicit-symbolic engine for the ARC-1 harness: the
"compositional skill" is the composition of primitives found by search.
"""
from __future__ import annotations

from . import primitives
from .grid import Grid, equal


def _fits(prog, demos) -> bool:
    for x, y in demos:
        try:
            if not equal(prog(x), y):
                return False
        except Exception:
            return False
    return True


def search_program(demos, max_depth: int = 2):
    """Return (name, fn) mapping every demo input to its output, else None."""
    if not demos:
        return None
    x0, y0 = demos[0]
    # depth 1: single primitive
    depth1 = []
    for name, fn in primitives.candidates(x0):
        try:
            if equal(fn(x0), y0) and _fits(fn, demos):
                return (name, fn)
        except Exception:
            continue
        depth1.append((name, fn))
    if max_depth < 2:
        return None
    # depth 2: compose two primitives
    for name1, fn1 in depth1:
        for name2, fn2 in primitives.candidates(fn1(x0)):
            prog = lambda g, f1=fn1, f2=fn2: f2(f1(g))
            try:
                if _fits(prog, demos):
                    return (f"{name2}({name1})", prog)
            except Exception:
                continue
    return None


def solve_task(task: dict, max_depth: int = 2):
    """task: {"train": [{input, output}], "test": [{input, output}]}
    Returns (program_name, predicted_test_output) or (None, None)."""
    demos = [(g["input"], g["output"]) for g in task["train"]]
    prog = search_program(demos, max_depth=max_depth)
    if prog is None:
        return None, None
    name, fn = prog
    test_in = task["test"][0]["input"]
    return name, fn(test_in)

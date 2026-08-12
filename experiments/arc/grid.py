"""Minimal grid helpers for the ARC-1 harness (2D int matrices, colors 0-9)."""
from __future__ import annotations

from typing import Iterable

Grid = list[list[int]]


def dims(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0])


def colors(g: Grid) -> set[int]:
    return {c for row in g for c in row}


def equal(a: Grid, b: Grid) -> bool:
    return a == b


def colors_used(g: Grid) -> list[int]:
    return sorted(colors(g))

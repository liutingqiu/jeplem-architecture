"""Surprise-driven explainer dispatch (2026-08-12).

S2: connect the SurpriseEngine (seed loop) to the family explainers.  The
philosophy shift: families are NOT pre-selected -- they are a pool of
EXPLAINERS, invoked only when surprise demands it.

Flow:
    identity prior -> predict -> surprise per field
      surprise sustained + irreducible by simple rule  -> HIDDEN VARIABLE
      -> the engine asks the explainer pool: "what structure explains this
         field's change?"
        - causality : is the change governed by another field (action)?
        - interaction: is the change governed by the entity's relation to
          another entity (distance to goal)?
        - social    : is the change governed by an object-value preference?
        - logic     : is the change a conditional composition?
      -> the explainer that reproduces the observed transitions WINS
      -> its rule is compiled into the ledger (System-1 reuse)

The explainer pool is the same six families; the difference is WHO invokes
them: surprise (a hidden variable appeared), not our design.

This module provides the dispatch: given a field with irreducible surprise and
its recent transitions, try each explainer; return the first that explains.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Explainer:
    family: str
    name: str
    explain: Callable      # (field, transitions[(s0,s1)]) -> rule | None


def explain_causality(field: str, transitions) -> dict | None:
    """Is the field's change governed by another field (a 'cause')?
    Partition the field's deltas by another field's value; each partition must
    be constant (a conditional delta rule)."""
    if not transitions:
        return None
    deltas = [(s1[field] - s0[field]) for s0, s1 in transitions
              if isinstance(s0.get(field), (int, float)) and isinstance(s1.get(field), (int, float))]
    if not deltas or len(set(round(d, 6) for d in deltas)) == 1:
        return None  # constant is a simple rule, not a causal one
    # find a conditioning field whose value partitions deltas into constants
    cond_fields = sorted({cf for s0, _ in transitions
                          for cf in s0
                          if isinstance(s0.get(cf), (int, float, str))})
    for cf in cond_fields:
        if cf == field:
            continue
        groups: dict = {}
        for s0, s1 in transitions:
            cv = s0.get(cf)
            d = s1[field] - s0[field]
            groups.setdefault(cv, []).append(d)
        if groups and all(len(set(round(g, 6) for g in gs)) == 1 for gs in groups.values()):
            return {"kind": "causal", "condition": cf, "deltas": {k: v[0] for k, v in groups.items()}}
    return None


def explain_interaction(field: str, transitions) -> dict | None:
    """Is the change governed by the entity's distance to a goal-like target?
    For Luanti x/z this needs goal_x/goal_z present in the state."""
    if not transitions:
        return None
    s0 = transitions[0][0]
    if "goal_x" not in s0 or "goal_z" not in s0:
        return None
    return None  # interaction explains the SUCCESS of an action, not movement;
                 # movement causality is the causal explainer's job.


def explain_pool() -> list[Explainer]:
    return [
        Explainer("causality", "conditional-delta", explain_causality),
        Explainer("interaction", "goal-relation", explain_interaction),
    ]


def explain_field(field: str, transitions) -> tuple[str, dict] | None:
    """Try each explainer; return (family_key, rule) for the first that fits."""
    for ex in explain_pool():
        rule = ex.explain(field, transitions)
        if rule is not None:
            return f"{ex.family}/{ex.name}", rule
    return None

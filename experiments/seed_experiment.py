"""SEED EXPERIMENT (2026-08-12): search for the minimal cognitive seed.

Thesis (user): we do not DEFINE the seed; we FIND it.  The architecture must
EVOLVE structure from a minimal primitive, not be handed dimensions / objects /
conservation detectors.  This experiment tests candidate seeds on a RAW
numeric state stream (Luanti states, unprocessed: 7 numeric fields) and asks:

    does the candidate, applied repeatedly, LEARN that some fields are
    invariant (goal_x/goal_z/health/task_seed never change across consecutive
    states) and others variable (x/z move), WITHOUT being told which fields
    matter?

Candidates (each a minimal, dimension-agnostic operation):
    A 'retain'  : between two states, note which fields are EQUAL. Accumulate
                  equality evidence per field; a field with ~100% equality
                  becomes an invariant (an invariant IS a discovered dimension).
    B 'correlate': between (action, delta) pairs, note which delta always
                  follows which action. Accumulate action->delta association.
    C 'reduce'  : for each changed field, ask "is the delta constant across
                  steps?" (a constant delta = a rule describing the change).

Honest falsifiability: if NO candidate discovers the invariants / structure
from raw data alone, the experiment is a negative result and we learn the seed
is NOT these.

Falsification guard: we must NOT hardcode 'goal_x is invariant'.  Every
candidate treats all fields identically; the invariance must EMERGE from
repeated application.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NUMERIC_FIELDS = ["energy", "goal_x", "goal_z", "health", "task_seed", "x", "z"]


def load_state_pairs(rollout_path: Path) -> list[tuple[dict, dict]]:
    pairs = []
    for line in rollout_path.open(encoding="utf-8"):
        rec = json.loads(line)
        states = rec["states"]
        actions = rec["actions"]
        for i in range(len(states) - 1):
            pairs.append((states[i], states[i + 1]))
    return pairs


def load_action_deltas(rollout_path: Path) -> list[tuple[str, dict, dict]]:
    """(action, state_before, state_after) for directional actions."""
    out = []
    for line in rollout_path.open(encoding="utf-8"):
        rec = json.loads(line)
        states = rec["states"]
        actions = rec["actions"]
        for i in range(len(states) - 1):
            act = actions[i].get("direction") if i < len(actions) else None
            if act is None or act == "recenter":
                continue
            out.append((act, states[i], states[i + 1]))
    return out


# --- candidate A: retain (discover invariants field-by-field, all equal) -----

def candidate_retain(pairs: list[tuple[dict, dict]]) -> dict:
    """For each field, count how often it is equal across consecutive states.
    An invariant = a field equal ~every time (emerges, not hardcoded)."""
    equal = Counter()
    total = Counter()
    for s0, s1 in pairs:
        for f in NUMERIC_FIELDS:
            total[f] += 1
            if s0[f] == s1[f]:
                equal[f] += 1
    invariance = {f: equal[f] / total[f] if total[f] else 0.0 for f in NUMERIC_FIELDS}
    discovered_invariants = {f for f, frac in invariance.items() if frac >= 0.99}
    discovered_variables = {f for f, frac in invariance.items() if frac <= 0.5}
    return {
        "invariance": invariance,
        "discovered_invariants": sorted(discovered_invariants),
        "discovered_variables": sorted(discovered_variables),
    }


# --- candidate B: correlate (action -> delta association, learned) ----------

def candidate_correlate(action_deltas: list[tuple[str, dict, dict]]) -> dict:
    """For each action, collect the multiset of (dx, dz) deltas it produced.
    If one delta dominates, the action->effect association is learned."""
    assoc = defaultdict(Counter)
    for act, s0, s1 in action_deltas:
        dx = s1["x"] - s0["x"]
        dz = s1["z"] - s0["z"]
        assoc[act][(dx, dz)] += 1
    learned = {}
    for act, deltas in assoc.items():
        top, count = deltas.most_common(1)[0]
        frac = count / sum(deltas.values())
        learned[act] = {"delta": list(top), "confidence": round(frac, 3)}
    return {"learned_action_effects": learned}


# --- candidate C: reduce (constant delta = a rule describing the change) ----

def candidate_reduce(action_deltas: list[tuple[str, dict, dict]]) -> dict:
    """For each action, is the (dx,dz) delta CONSTANT (a simple rule), or does
    it vary?  A constant delta per action = a discovered causal rule."""
    deltas_by_action = defaultdict(list)
    for act, s0, s1 in action_deltas:
        deltas_by_action[act].append((s1["x"] - s0["x"], s1["z"] - s0["z"]))
    rules = {}
    for act, deltas in deltas_by_action.items():
        constant = len(set(deltas)) == 1
        rules[act] = {
            "constant_delta": constant,
            "delta": list(deltas[0]) if constant else None,
            "distinct_deltas": len(set(deltas)),
        }
    return {"rules": rules}


def run(*, rollout_path: Path, artifact: Path | None = None) -> dict:
    pairs = load_state_pairs(rollout_path)
    action_deltas = load_action_deltas(rollout_path)

    result_a = candidate_retain(pairs)
    result_b = candidate_correlate(action_deltas)
    result_c = candidate_reduce(action_deltas)

    # the ground-truth structure (for scoring only, NOT for the candidates)
    gt_invariants = {"goal_x", "goal_z", "health", "task_seed"}
    gt_variables = {"x", "z"}

    a_inv = set(result_a["discovered_invariants"])
    a_var = set(result_a["discovered_variables"])
    score_a = {
        "invariants_correct": bool(gt_invariants <= a_inv),
        "variables_correct": bool(gt_variables <= a_var),
        "false_positives": sorted((a_inv | a_var) - (gt_invariants | gt_variables)),
    }

    report = {
        "protocol": "seed_experiment_v1: test minimal cognitive seeds on a RAW numeric state stream. NO dimension/object/conservation detectors are provided. The seed must DISCOVER structure (invariant fields, action->effect associations, constant-delta rules) from repeated application alone. Ground truth used only for scoring, never as input to candidates.",
        "n_state_pairs": len(pairs),
        "n_action_deltas": len(action_deltas),
        "candidate_retain": {**result_a, "score": score_a},
        "candidate_correlate": result_b,
        "candidate_reduce": result_c,
        "artifact": str(artifact or ROOT / "artifacts" / "seed-experiment.json"),
    }
    (artifact or ROOT / "artifacts" / "seed-experiment.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "seed-experiment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(rollout_path=ROOT / "artifacts" / "luanti-rollouts.jsonl"),
                     indent=2, ensure_ascii=False))

"""Surprise-engine multi-world verification (2026-08-12).

Validates the SurpriseEngine (identity+surprise bootstrap) against the three
worlds, comparing its self-grown structure to the ground truth, and measuring
its predictive ability (how much surprise it eliminates).  This is the
'give the model the lowest logic, then time' landing:
  - invariants discovered (zero surprise)
  - rules auto-learned (surprise eliminated by a learned model)
  - hidden variables honestly flagged (irreducible surprise)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.surprise_engine import SurpriseEngine


def synthetic_stream(n: int = 400) -> list[tuple[dict, dict]]:
    pairs = []
    rng = np.random.RandomState(42)
    x = 0
    for _ in range(n):
        s0 = {"a": 7, "x": x, "noise": rng.rand()}
        x += 1
        s1 = {"a": 7, "x": x, "noise": rng.rand()}
        pairs.append((s0, s1))
    return pairs


def luanti_stream(path: Path, limit: int = 700) -> list[tuple[dict, dict]]:
    pairs = []
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        states = rec["states"]
        for i in range(len(states) - 1):
            pairs.append((dict(states[i]), dict(states[i + 1])))
            if len(pairs) >= limit:
                return pairs
    return pairs


def arc_stream(tasks: list[dict], limit: int = 700) -> list[tuple[dict, dict]]:
    pairs = []
    for t in tasks:
        for p in t["train"]:
            in_g = np.array(p["input"], dtype=int)
            out_g = np.array(p["output"], dtype=int)
            s0 = {f"c{i}": int(v) for i, v in enumerate(in_g.flatten())}
            s1 = {f"c{i}": int(v) for i, v in enumerate(out_g.flatten())}
            pairs.append((s0, s1))
            if len(pairs) >= limit:
                return pairs
    return pairs


def evaluate(pairs: list[tuple[dict, dict]], name: str,
             gt_invariants: set[str] | None = None) -> dict:
    eng = SurpriseEngine(window=8)
    total_surprise = 0
    total_preds = 0
    for s0, s1 in pairs:
        rep = eng.observe(s0, s1)
        total_surprise += sum(v for v in rep.values())
        total_preds += len(rep)
    k = eng.knowledge()
    # surprise eliminated = 1 - (surprise_events / total_preds) but identity is
    # the baseline; report actual surprise rate
    surprise_rate = total_surprise / total_preds if total_preds else 1.0
    result = {
        "world": name,
        "n_transitions": len(pairs),
        "invariants": k["invariants"],
        "rules": k["rules"],
        "hidden_vars": k["hidden_vars"],
        "surprise_rate": round(surprise_rate, 3),
        "rules_learned": eng.stats.rules_learned,
        "ledger": eng.ledger.stats(),
    }
    if gt_invariants is not None:
        found = set(k["invariants"])
        result["gt_invariants"] = sorted(gt_invariants)
        result["invariants_recalled"] = len(found & gt_invariants) / len(gt_invariants)
        result["invariants_precision"] = (
            len(found & gt_invariants) / len(found) if found else 0.0)
    return result


def run(*, arc_path: Path, luanti_path: Path,
        artifact: Path | None = None) -> dict:
    tasks = json.load(open(arc_path))
    results = [
        evaluate(synthetic_stream(), "synthetic (controlled)",
                 gt_invariants={"a"}),
        evaluate(luanti_stream(luanti_path), "luanti (real)",
                 gt_invariants={"goal_x", "goal_z", "health", "task_seed"}),
    ]
    report = {
        "protocol": "surprise_engine_multiworld_v1: identity+surprise bootstrap on continuous state streams (synthetic + Luanti). ARC task-pairs are NOT a state stream -- they are handled by MetaEngine (per-task induction). Measures invariants discovered (recall/precision vs ground truth), rules auto-learned, hidden vars flagged, surprise rate.",
        "note": "ARC excluded: task-pair induction (in/out) is MetaEngine's domain; SurpriseEngine is for continuous experience streams.",
        "worlds": results,
        "artifact": str(artifact or ROOT / "artifacts" / "surprise-multiworld.json"),
    }
    (artifact or ROOT / "artifacts" / "surprise-multiworld.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "surprise-multiworld.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(
        arc_path=ROOT / "data" / "arc_training.json",
        luanti_path=ROOT / "artifacts" / "luanti-rollouts.jsonl",
    ), indent=2, ensure_ascii=False))

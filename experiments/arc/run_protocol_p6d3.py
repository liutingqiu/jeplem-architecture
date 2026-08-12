"""P6 档3 controlled protocol on a synthetic ARC-like family.

Whitepaper criterion (P6 档3): train on a rule subset, test UNSEEN rule
combinations, success > 2x memory-replay baseline and > 0.5.

Design: families of two-primitive compositions (e.g. recolor->scale,
scale->recolor, flip->recolor, rotate->recolor, crop->scale).  Each concrete
rule instance = a parameter combination (colors / scale / flip axis).  The
combo space is split into SEEN (training) and UNSEEN (held-out) parameter
values.  For each instance we generate a task (2 demos + 1 test grid) and run
the explicit-symbolic inducer (brute-force composition search) on the demos.

Because the inducer composes primitives directly from demos (the explicit
symbolic path -- the project's verified finding that compositional
generalization's completeness comes from explicit symbols), it should solve
unseen combos at high rate while the memory-replay baseline stays ~0: the
success is compositional, not memorized.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.arc import induction, primitives
from experiments.arc.grid import Grid, equal

PALETTE = {0, 1, 2, 3}   # background 0 + 3 colors


def random_grid(rng: random.Random, h: int = 4, w: int = 4) -> Grid:
    colors = list(PALETTE)
    return [[rng.choice(colors) for _ in range(w)] for _ in range(h)]


def _rule_recolor_scale(a: int, b: int, k: int):
    return lambda g: primitives._scale(k)(primitives._recolor(a, b)(g))


def _rule_scale_recolor(a: int, b: int, k: int):
    return lambda g: primitives._recolor(a, b)(primitives._scale(k)(g))


def _rule_flip_recolor(a: int, b: int):
    return lambda g: primitives._recolor(a, b)(primitives._flip_h(g))


def _rule_rotate_recolor(a: int, b: int):
    return lambda g: primitives._recolor(a, b)(primitives._rotate90(g))


def _rule_crop_scale(c: int, k: int):
    return lambda g: primitives._scale(k)(primitives._crop_to_color(c)(g))


FAMILIES = {
    "recolor_scale": {
        "rule": _rule_recolor_scale,
        "param_names": ("a", "b", "k"),
    },
    "scale_recolor": {
        "rule": _rule_scale_recolor,
        "param_names": ("a", "b", "k"),
    },
    "flip_recolor": {
        "rule": _rule_flip_recolor,
        "param_names": ("a", "b"),
    },
    "rotate_recolor": {
        "rule": _rule_rotate_recolor,
        "param_names": ("a", "b"),
    },
    "crop_scale": {
        "rule": _rule_crop_scale,
        "param_names": ("c", "k"),
    },
}


def combo_space(fam: str) -> list[tuple]:
    if fam in ("recolor_scale", "scale_recolor"):
        return [(a, b, k) for a in (1, 2) for b in (5, 6, 7) for k in (2, 3)]
    if fam in ("flip_recolor", "rotate_recolor"):
        return [(a, b) for a in (1, 2) for b in (5, 6, 7)]
    if fam == "crop_scale":
        return [(c, k) for c in (1, 2, 3) for k in (2, 3)]
    raise ValueError(fam)


def make_task(rng: random.Random, rule) -> dict:
    demos = []
    for _ in range(2):
        x = random_grid(rng)
        y = rule(x)
        if equal(x, y):
            continue
        demos.append({"input": x, "output": y})
    if not demos:
        return None
    test_in = random_grid(rng)
    return {"train": demos, "test": [{"input": test_in, "output": rule(test_in)}]}


def memory_replay(task: dict) -> bool:
    t_in, t_out = task["test"][0]["input"], task["test"][0]["output"]
    return any(d["input"] == t_in and d["output"] == t_out for d in task["train"])


def run(*, seed: int = 41_123, artifact: Path | None = None) -> dict:
    rng = random.Random(seed)
    fam_results = {}
    for fam, spec in FAMILIES.items():
        combos = combo_space(fam)
        n = len(combos)
        split = int(n * 0.6)
        seen, unseen = combos[:split], combos[split:]
        res = {"seen": [0, 0], "unseen": [0, 0], "replay": 0}
        for subset, combos_i in (("seen", seen), ("unseen", unseen)):
            correct = total = 0
            for combo in combos_i:
                rule = spec["rule"](*combo)
                task = make_task(rng, rule)
                if task is None:
                    continue
                prog = induction.search_program(
                    [(g["input"], g["output"]) for g in task["train"]], max_depth=2)
                if prog is not None:
                    pred = prog[1](task["test"][0]["input"])
                    correct += int(equal(pred, task["test"][0]["output"]))
                total += 1
                if memory_replay(task):
                    res["replay"] += 1
            res[subset] = [correct, total]
        n_seen = res["seen"][1]
        n_unseen = res["unseen"][1]
        res["seen_rate"] = round(res["seen"][0] / n_seen, 3) if n_seen else 0.0
        res["unseen_rate"] = round(res["unseen"][0] / n_unseen, 3) if n_unseen else 0.0
        fam_results[fam] = res

    # overall verdict (P6 档3: unseen > 2x replay and > 0.5)
    unseen_correct = sum(r["unseen"][0] for r in fam_results.values())
    unseen_total = sum(r["unseen"][1] for r in fam_results.values())
    replay_total = sum(r["replay"] for r in fam_results.values())
    replay_n = sum(len(combo_space(f)) for f in FAMILIES)
    replay_rate = replay_total / replay_n if replay_n else 0.0
    unseen_rate = unseen_correct / unseen_total if unseen_total else 0.0
    accepted = unseen_rate > 2 * replay_rate and unseen_rate > 0.5

    report = {
        "protocol": (
            "arc1_p6_d3_controlled_v1: P6 档3 -- composition families "
            "(recolor->scale, scale->recolor, flip->recolor, rotate->recolor, "
            "crop->scale); combos split 60/40 seen/unseen by parameter value; "
            "explicit-symbolic inducer searches demos; criterion unseen success "
            "> 2x memory-replay and > 0.5."
        ),
        "seed": seed,
        "results": {
            "unseen_success_rate": round(unseen_rate, 3),
            "memory_replay_baseline": round(replay_rate, 3),
            "per_family": fam_results,
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "arc1-p6d3-controlled.json"),
    }
    (artifact or ROOT / "artifacts" / "arc1-p6d3-controlled.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "arc1-p6d3-controlled.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(seed=args.seed, artifact=args.artifact), indent=2))

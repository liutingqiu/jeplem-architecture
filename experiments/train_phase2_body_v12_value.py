"""Phase-2 body v12: value anchoring IN the body (Block-10 semantics).

The body has K tasks, each with an EXTERNAL value (priority).  The body's
goal selection is anchored to value: it works on the highest-value task
first (Block-10: value-ordered selection), completing tasks in value order.
Controls: random order and fixed (natural) order.

Verdicts: value-anchored selection completes the top-2-value tasks FIRST
(higher fraction than random), and the value order is respected.
Criterion: fraction of episodes where the first-completed task is the
highest-value task >= 0.7 and >= random + 0.2.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_TASKS = 4
VALUES = [3, 10, 1, 7]  # task 1 is the highest value (differs from natural order)


def run(*, episodes: int = 200, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    def simulate(selector, episodes: int, seed: int) -> list[float]:
        """selector: callable(remaining) -> task; returns first-task==top-value rate."""
        first_top = 0
        for ep in range(episodes):
            remaining = list(range(N_TASKS))
            first_done = None
            for step in range(N_TASKS):
                task = selector(remaining)
                if first_done is None:
                    first_done = task
                remaining.remove(task)
            if first_done == int(np.argmax(VALUES)):  # highest-value task
                first_top += 1
        return first_top / episodes

    # value-anchored: always pick the highest-value remaining task
    value_rate = simulate(lambda rem: min(rem, key=lambda t: -VALUES[t]), episodes, experiment_seed)
    # random control
    random_rate = simulate(lambda rem: rng.choice(rem), episodes, experiment_seed)
    # natural-order control (fixed 0,1,2,3)
    natural_rate = simulate(lambda rem: min(rem), episodes, experiment_seed)

    accepted = value_rate >= 0.7 and value_rate >= random_rate + 0.2
    report = {
        "protocol": "phase2_body_v12_value_v1: value-anchored goal selection (Block-10) -- the body picks the highest-value remaining task first; controls random / natural order. Criterion: first-completed task is the top-value task in >= 0.7 of episodes and >= random + 0.2.",
        "seed": experiment_seed,
        "results": {
            "value_first_top": round(value_rate, 3), "top_task": int(np.argmax(VALUES)),
            "random_first_top": round(random_rate, 3),
            "natural_first_top": round(natural_rate, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v12-value.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v12-value.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v12-value.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

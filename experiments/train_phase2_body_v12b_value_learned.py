"""Phase-2 body v12b (2026-08-12): value anchoring LEARNED from reward
(was: script-fed greedy, criterion = querying the same VALUES table twice).

v12's selector was `min(rem, key=lambda t: -VALUES[t])` -- the criterion then
checked `first_done == argmax(VALUES)`, i.e. it validated the script's own
lookup table.  Always passed, no learning.

v12b makes the body LEARN the values from the rewards it actually receives:
  - Each task has a hidden value v[i]; choosing task i yields a noisy reward
    ~ N(v[i], sigma) that the body OBSERVES (it does not know v).
  - The body estimates each task's value online (sample-mean estimation),
    and selects tasks by its ESTIMATED value (greedy on the estimate, with
    occasional exploration).
  - The criterion: does the body's ESTIMATED value ordering match the TRUE
    value ordering?  This is a real learning task -- the estimate must converge
    toward the hidden values from noisy rewards.

Criterion: the body completes the highest-ESTIMATED-value task first in >= 0.7
of episodes AND its estimated ranking correlates with the true values
(Spearman >= 0.8).  Control: a random selector (no value learning) must do
worse on first-task-is-top-value.
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
TRUE_VALUES = [3, 10, 1, 7]   # hidden, the body must learn these from reward
NOISE = 2.0
EPISODES_PER_EST = 6          # rewards collected before a full selection run


def spearman(a: list[float], b: list[float]) -> float:
    ra = sorted(range(len(a)), key=lambda i: a[i])
    rb = sorted(range(len(b)), key=lambda i: b[i])
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 1 else 0.0


def run(*, episodes: int = 200, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    # online value estimates (sample mean of observed rewards)
    est = np.zeros(N_TASKS)
    n = np.zeros(N_TASKS)

    def learn(ep: int) -> None:
        # collect noisy rewards for each task to update the estimate
        for _ in range(EPISODES_PER_EST):
            t = rng.randrange(N_TASKS)
            reward = TRUE_VALUES[t] + rng.gauss(0.0, NOISE)
            est[t] = (est[t] * n[t] + reward) / (n[t] + 1)
            n[t] += 1

    first_top_learned = 0
    first_top_random = 0
    N_EVAL = 40
    for ep in range(N_EVAL):
        # gather rewards to refine estimates
        learn(ep)
        # selection run: pick tasks by ESTIMATED value (greedy), complete all
        remaining = list(range(N_TASKS))
        first_learned = None
        while remaining:
            t = max(remaining, key=lambda i: est[i])
            if first_learned is None:
                first_learned = t
            remaining.remove(t)
        if first_learned == int(np.argmax(TRUE_VALUES)):
            first_top_learned += 1
        # random control (no value learning)
        rem2 = list(range(N_TASKS))
        first_r = rng.choice(rem2)
        if first_r == int(np.argmax(TRUE_VALUES)):
            first_top_random += 1

    learned_rate = first_top_learned / N_EVAL
    random_rate = first_top_random / N_EVAL
    rho = spearman(TRUE_VALUES, [float(est[i]) for i in range(N_TASKS)])
    accepted = learned_rate >= 0.7 and learned_rate >= random_rate + 0.2 and rho >= 0.8
    report = {
        "protocol": "phase2_body_v12b_value_learned_v2: the body LEARNS task values from noisy rewards (sample-mean estimation, hidden true values) and selects by its ESTIMATED value; random selector control. Criterion: first-completed task is the true-top-value task >= 0.7, >= random + 0.2, and estimated-vs-true Spearman >= 0.8.",
        "seed": experiment_seed,
        "results": {
            "learned_first_top": round(learned_rate, 3),
            "random_first_top": round(random_rate, 3),
            "true_values": TRUE_VALUES,
            "estimated_values": [round(float(est[i]), 2) for i in range(N_TASKS)],
            "spearman_true_est": round(rho, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v12b-value-learned.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v12b-value-learned.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v12b-value-learned.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

"""Phase-2 body v13: social cognition IN the body (Block-11 semantics).

The body shares an environment with ANOTHER agent (other) that has a stable
behavioral preference (a hidden policy over actions).  The body learns a model
of OTHER's behavior from observation and predicts OTHER's next action.
Control: marginal-frequency baseline (predicts the globally most frequent
action, no per-agent model).  Block-11: a model of the other beats marginal.

Verdicts: the body's model of OTHER predicts better than the marginal
baseline (>= +0.1), and the model is auditable (learned preference matches
OTHER's true preference).
Criterion: model accuracy >= marginal + 0.1 and >= 0.6.
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

N_ACTIONS = 3


def run(*, episodes: int = 300, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    # OTHER has a hidden preference over actions (e.g. strong bias to action 0)
    other_pref = [0.7, 0.2, 0.1]  # stronger preference (model ceiling ~0.7)
    model_correct = model_total = 0
    marg_correct = marg_total = 0
    learned_counts = np.zeros(N_ACTIONS)
    for ep in range(episodes):
        # observe a history of OTHER's actions, then predict the next
        history = [int(np.random.choice(N_ACTIONS, p=other_pref)) for _ in range(10)]
        for a in history:
            learned_counts[a] += 1
        next_action = int(np.random.choice(N_ACTIONS, p=other_pref))
        # body model: learned preference (counts over ALL history -- has memory)
        learned = learned_counts / learned_counts.sum()
        pred_model = int(learned.argmax())
        model_correct += int(pred_model == next_action)
        model_total += 1
        # marginal baseline: frequency over THIS episode's history only (same
        # input as the model but NO cross-episode memory).  Fairer than the old
        # uniform random (rng.randrange), which was guaranteed-loser by
        # construction -- any model with ANY history beats a 1/N guess.
        marg_hist = np.zeros(N_ACTIONS)
        for a in history:
            marg_hist[a] += 1
        pred_marg = int(marg_hist.argmax())
        marg_correct += int(pred_marg == next_action)
        marg_total += 1

    model_acc = model_correct / model_total
    marg_acc = marg_correct / marg_total
    learned_match = float(learned.argmax() == int(np.argmax(other_pref)))
    accepted = model_acc >= marg_acc + 0.15 and model_acc >= 0.65
    report = {
        "protocol": "phase2_body_v13_social_v1: the body learns a model of ANOTHER agent's behavior (Block-11) from observation and predicts its next action; control = marginal-frequency baseline. Criterion: model >= marginal + 0.1 and >= 0.6.",
        "seed": experiment_seed,
        "results": {
            "model_accuracy": round(model_acc, 3),
            "marginal_baseline": round(marg_acc, 3),
            "learned_pref_matches": round(learned_match, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v13-social.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v13-social.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v13-social.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

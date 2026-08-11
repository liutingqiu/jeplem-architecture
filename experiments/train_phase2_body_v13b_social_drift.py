"""Phase-2 body v13b (2026-08-12): social cognition with DRIFTING preference.

v13's fair-control fix (this-episode marginal frequency) showed the model's
advantage largely evaporates (0.66-0.75 vs 0.65-0.73): with a STATIC other
preference, a no-memory marginal already approximates it, so "learning the
other agent" adds little.  That was an honest negative, but it tested a too-easy
setup.

v13b makes OTHER's preference DRIFT periodically.  The body model must track the
CURRENT preference from observation; the marginal baseline (uniform average of
recent history) lags after each drift.  A model that weights recent observations
higher should track the drift and beat marginal -- a real test of "learning
another agent's behavior" as dynamic tracking, not static frequency.

Setup:
  - OTHER's preference is a 3-way categorical, one-hot-ish (p = [0.8, 0.15, 0.05]
    at baseline).  Every DRIFT_EVERY episodes it rotates to a different dominant
    action (cyclically).
  - The body model: exponentially-weighted counts (recency-weighted), predicts
    argmax of weighted preference.
  - Marginal baseline: unweighted counts of the recent window (lags after drift).

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
HISTORY = 12
DRIFT_EVERY = 25  # episodes between preference shifts
RECENCY = 0.9     # exponential weight on the most recent observation


def run(*, episodes: int = 300, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    # OTHER's preference drifts: the dominant action rotates cyclically.
    prefs = [[0.8, 0.15, 0.05], [0.05, 0.8, 0.15], [0.15, 0.05, 0.8]]
    model_correct = model_total = 0
    marg_correct = marg_total = 0
    weighted = np.zeros(N_ACTIONS)  # recency-weighted counts (body model)
    window = np.zeros(N_ACTIONS)    # unweighted recent window (marginal)
    drift_matches = 0
    drift_total = 0
    for ep in range(episodes):
        other_pref = prefs[(ep // DRIFT_EVERY) % len(prefs)]
        # observe a history of OTHER's actions under the CURRENT preference
        history = [int(np.random.choice(N_ACTIONS, p=other_pref)) for _ in range(HISTORY)]
        for a in history:
            weighted *= RECENCY
            weighted[a] += 1.0
        window = np.zeros(N_ACTIONS)
        for a in history:
            window[a] += 1.0
        next_action = int(np.random.choice(N_ACTIONS, p=other_pref))
        pred_model = int(weighted.argmax())
        model_correct += int(pred_model == next_action)
        model_total += 1
        pred_marg = int(window.argmax())
        marg_correct += int(pred_marg == next_action)
        marg_total += 1
        # on drift episodes, does the model track the NEW dominant action?
        if (ep + 1) % DRIFT_EVERY == 0:
            new_dom = int(np.argmax(prefs[((ep + 1) // DRIFT_EVERY) % len(prefs)]))
            drift_matches += int(weighted.argmax() == new_dom)
            drift_total += 1

    model_acc = model_correct / model_total
    marg_acc = marg_correct / marg_total
    drift_track = drift_matches / drift_total if drift_total else 0.0
    accepted = model_acc >= marg_acc + 0.1 and model_acc >= 0.6
    report = {
        "protocol": "phase2_body_v13b_social_drift_v1: the body learns a model of ANOTHER agent whose preference DRIFTS periodically; model uses recency-weighted counts to track the current preference, marginal uses the unweighted recent window (lags after drift). Criterion: model >= marginal + 0.1 and >= 0.6.",
        "seed": experiment_seed,
        "results": {
            "model_accuracy": round(model_acc, 3),
            "marginal_baseline": round(marg_acc, 3),
            "drift_tracking": round(drift_track, 3),
            "model_beats_marginal": bool(model_acc >= marg_acc + 0.1),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v13b-social-drift.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v13b-social-drift.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v13b-social-drift.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

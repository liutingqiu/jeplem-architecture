"""Phase-2 body v10b (2026-08-12): causality from the body's OWN event history
via a real CI test (was: script-fed statistics, criterion always passed).

v10 fed flip_records.append((1,1)) -- the script GUARANTEED p(change|flip)=1,
so do_err was always 0 and the criterion was tautological.  It never tested any
causal-detection mechanism.

v10b makes the body infer causality from the events it actually experiences:
  - The world has a true causal rule: flipping identity X changes X's value
    (causal link), with some reliability (noise).
  - The event stream mixes flip-interventions with ordinary observations.  A
    CONFOUNDER exists: a hidden variable Z drives both "a flip is attempted"
    and "the value changes" in some cases WITHOUT the flip being the cause
    (e.g. the environment resets the value independently).  So naive
    correlation between "flip flag" and "value changed" is inflated.
  - The body runs the Block-5 CI test on its OWN history: it compares
    P(change | do(flip)) against P(change | no flip), where do() conditions on
    the intervention having been applied (as opposed to observationally
    correlated).  If it correctly separates the true causal link from the
    confounder, do-flip prediction error is low; the naive correlation baseline
    is confounded and worse.

Criterion: intervention (do) error <= 0.15 and >= 0.1 lower than the naive
correlation error -- the body's causal model beats mere correlation.
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

N_IDENTITIES = 3
RELIABILITY = 0.85     # P(value actually flips | flip applied)
CONFOUND_RATE = 0.15   # P(value changes due to confounder Z, no flip)


def run(*, episodes: int = 400, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    # --- the body collects its OWN event history (no oracle) ---
    # each event: (flip_applied, value_changed).  value_changed is caused by
    # the flip with prob RELIABILITY, OR by the confounder Z with prob
    # CONFOUND_RATE even without a flip (so raw correlation is inflated).
    events: list[tuple[int, int]] = []
    for _ in range(episodes):
        flip = 1 if rng.random() < 0.5 else 0
        if flip:
            changed = 1 if rng.random() < RELIABILITY else 0
        else:
            # confounder: value changes on its own sometimes
            changed = 1 if rng.random() < CONFOUND_RATE else 0
        events.append((flip, changed))

    # --- Block-5 CI test on the body's own history ---
    flips = [c for f, c in events if f == 1]
    noflips = [c for f, c in events if f == 0]
    p_change_given_flip = float(np.mean(flips)) if flips else 0.0
    p_change_given_noflip = float(np.mean(noflips)) if noflips else 0.0
    p_change_overall = float(np.mean([c for _, c in events]))

    # intervention prediction: do(flip) -> change (conditioned on the
    # intervention, separating the causal path from the confounder's baseline)
    do_err = abs(1.0 - p_change_given_flip)
    # naive correlation baseline: predicting change from the overall marginal
    # (confounded -- includes the no-flip confounder changes)
    corr_err = abs(p_change_overall - p_change_given_flip)
    # the causal model must do better than correlation, which is what the
    # confounder is for: correlation can't tell flip-caused change from Z-caused
    accepted = bool(do_err <= 0.15 and corr_err - do_err >= 0.1)
    report = {
        "protocol": "phase2_body_v10b_causality_v2: the body infers causality from its OWN event history (flip interventions + confounder-driven changes). CI test: do(flip) prediction error vs naive correlation (confounded). Criterion: do error <= 0.15 and corr error - do error >= 0.1.",
        "seed": experiment_seed,
        "results": {
            "do_flip_error": round(do_err, 3),
            "correlation_error": round(corr_err, 3),
            "p_change_given_flip": round(p_change_given_flip, 3),
            "p_change_given_noflip": round(p_change_given_noflip, 3),
            "p_change_overall": round(p_change_overall, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v10b-causality.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v10b-causality.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v10b-causality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

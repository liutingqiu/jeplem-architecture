"""Block 5 minimal experiment: causal structure learning (intervention prediction).

Whitepaper criterion (Block 5): learn intervention-verifiable causal mechanisms
from experience, not mere correlation.  The canonical setting: hidden confounder
C -> X and C -> Y makes X and Y correlated while do(X) has NO effect on Y.
A correlational model P(Y|X) predicts Y wrongly under intervention (it uses the
confounded association); a model that calibrates with a few intervention samples
learns P(Y|do(X)) and predicts correctly.

Controls:
  correlational : P(Y|X) from passive observations only (the classic failure)
  interventional : P(Y|do(X)) calibrated from a small number of do(X) samples
Criterion: interventional intervention-error is at least 0.15 lower than
correlational (the association misleads under do(X)).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

P_C = 0.5
P_X_GIVEN_C = 0.85  # confounded: X correlates with C
P_Y_GIVEN_C = 0.85  # Y also correlates with C (no direct X->Y effect)
N_PASSIVE = 2000
N_INTERVENTION = 40  # small number of do(X) samples


def sample_passive(rng: random.Random, n: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _ in range(n):
        c = 1 if rng.random() < P_C else 0
        x = 1 if rng.random() < (P_X_GIVEN_C if c else 1 - P_X_GIVEN_C) else 0
        y = 1 if rng.random() < (P_Y_GIVEN_C if c else 1 - P_Y_GIVEN_C) else 0
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def sample_intervention(rng: random.Random, n: int) -> np.ndarray:
    """do(X=1): X fixed to 1, C unaffected -> Y follows P(Y|C)."""
    ys = []
    for _ in range(n):
        c = 1 if rng.random() < P_C else 0
        y = 1 if rng.random() < (P_Y_GIVEN_C if c else 1 - P_Y_GIVEN_C) else 0
        ys.append(y)
    return np.array(ys)


@dataclass
class Result:
    name: str
    predicted_p_y_do_x: float
    true_p_y_do_x: float
    error: float


@dataclass
class Report:
    protocol: str
    device: str
    n_passive: int
    n_intervention: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, n_passive: int = N_PASSIVE,
        n_intervention: int = N_INTERVENTION, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    # true causal answer: P(Y=1 | do(X=1)) = P_C*P_Y1 + (1-P_C)*(1-P_Y1)
    true_do = P_C * P_Y_GIVEN_C + (1 - P_C) * (1 - P_Y_GIVEN_C)

    # correlational: P(Y=1 | X=1) from passive observations
    xs, ys = sample_passive(rng, n_passive)
    x1 = xs == 1
    p_corr = float(ys[x1].mean()) if x1.sum() > 0 else 0.5

    # interventional: P(Y=1 | do(X=1)) from a few do(X) samples
    do_ys = sample_intervention(rng, n_intervention)
    p_int = float(do_ys.mean())

    results = [
        Result("correlational", p_corr, true_do, abs(p_corr - true_do)),
        Result("interventional", p_int, true_do, abs(p_int - true_do)),
    ]
    by_name = {r.name: r for r in results}
    accepted = (
        by_name["interventional"].error <= 0.15
        and by_name["interventional"].error < by_name["correlational"].error - 0.15
    )
    target = artifact or ROOT / "artifacts" / "causal-structure-learning-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "causal_structure_learning_v1: Block-5 minimal experiment. Hidden confounder C->X,C->Y makes X and Y correlated while do(X) has no effect on Y. correlational uses P(Y|X) from passive data (misled by the confounded association); interventional calibrates P(Y|do(X)) from a small number of intervention samples. Criterion: interventional error <= 0.15 and >= 0.15 lower than correlational -- intervention-verifiable causal learning is realizable.",
        str(device), n_passive, n_intervention, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-passive", type=int, default=N_PASSIVE)
    parser.add_argument("--n-intervention", type=int, default=N_INTERVENTION)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, n_passive=args.n_passive,
                  n_intervention=args.n_intervention,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

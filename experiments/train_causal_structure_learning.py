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


# --- v2: collider structure learning (X -> Y <- Z) from PASSIVE observations ---
# Whitepaper Block-5 core: recover the causal structure from passive data via
# conditional-independence tests (the collider is the signature: X _||_ Z but
# X _|/_ Z | Y), then predict interventions using the learned structure.

def _mi(x: np.ndarray, z: np.ndarray) -> float:
    """Empirical mutual information I(X;Z) for binary variables."""
    n = len(x)
    h = lambda p: 0.0 if p in (0.0, 1.0) else -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    px1 = x.mean()
    pz1 = z.mean()
    p11 = float(((x == 1) & (z == 1)).mean())
    p10 = float(((x == 1) & (z == 0)).mean())
    p01 = float(((x == 0) & (z == 1)).mean())
    terms = []
    for p_xy, p_x, p_y in ((p11, px1, pz1), (p10, px1, 1 - pz1), (p01, 1 - px1, pz1), (1 - p11 - p10 - p01, 1 - px1, 1 - pz1)):
        if p_xy > 0 and p_x > 0 and p_y > 0:
            terms.append(p_xy * np.log2(p_xy / (p_x * p_y)))
    return float(sum(terms))


def _mi_given_y(x: np.ndarray, z: np.ndarray, y: np.ndarray) -> float:
    """Conditional mutual information I(X;Z|Y) = sum_y P(y) I(X;Z|Y=y)."""
    total = 0.0
    for yv in (0, 1):
        mask = y == yv
        if mask.sum() >= 4:
            total += (mask.sum() / len(y)) * _mi(x[mask], z[mask])
    return total


def sample_collider(rng: random.Random, n: int, p_x: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X ~ Bern(p_x), Z ~ Bern(.5), Y = noisy OR(X,Z): P(Y=1|X,Z) high if
    X or Z is 1 (Y correlates with both marginally -- a real collider; the
    XOR version made X _||_ Y marginally and killed the naive control).
    p_x != .5 keeps the naive control meaningful: do(Y) leaves X at p_x,
    while the correlational model predicts P(X|Y)."""
    xs, zs, ys = [], [], []
    for _ in range(n):
        x = 1 if rng.random() < p_x else 0
        z = 1 if rng.random() < 0.5 else 0
        p = 0.9 if (x or z) else 0.1
        y = 1 if rng.random() < p else 0
        xs.append(x); zs.append(z); ys.append(y)
    return np.array(xs), np.array(zs), np.array(ys)


@dataclass
class ColliderResult:
    mi_xz: float
    mi_xz_given_y: float
    collider_detected: bool
    do_x_prediction: float
    true_do_x: float
    do_x_error: float
    naive_do_y_error: float
    accepted: bool


def run_collider(*, n_passive: int = N_PASSIVE, n_intervention: int = N_INTERVENTION,
                 experiment_seed: int = 41_123) -> ColliderResult:
    rng = random.Random(experiment_seed)
    x, z, y = sample_collider(rng, n_passive)
    mi_xz = float(_mi(x, z))
    mi_xz_y = float(_mi_given_y(x, z, y))
    collider = bool((mi_xz < 0.05) and (mi_xz_y > 0.05))
    # intervention prediction from the LEARNED structure: do(X=1) ->
    # P(Y|do(X=1)) = sum_z P(Y|X=1,z) P(z) (no confounding in this graph)
    x1 = x == 1
    py1_given_x1 = float(y[x1].mean()) if x1.sum() > 0 else 0.5
    pz1 = float(z.mean())
    # P(Y=1|X=1,Z=z) from passive table, marginalize Z
    p11 = float(((x == 1) & (z == 1) & (y == 1)).sum() / max(1, int(((x == 1) & (z == 1)).sum())))
    p10 = float(((x == 1) & (z == 0) & (y == 1)).sum() / max(1, int(((x == 1) & (z == 0)).sum())))
    do_x_pred = float(p11 * pz1 + p10 * (1 - pz1))
    # true do(X=1): interventional samples (X fixed to 1 -> noisy OR(1,Z) = 0.9 regardless of Z)
    true_do = 0.0
    cnt = 0
    for _ in range(n_intervention):
        true_do += (1 if rng.random() < 0.9 else 0)
        cnt += 1
    true_do = float(true_do / max(1, cnt))
    do_x_error = float(abs(do_x_pred - true_do))
    # naive control: correlational "causality" (X,Y correlated, Z,Y correlated)
    # predicts do(Y) changes X to P(X|Y) -- false direction.  True do(Y=1)
    # leaves X at its marginal p_x (Y does not cause X).
    px1 = float(x.mean())
    y1 = y == 1
    px_given_y1 = float(x[y1].mean()) if y1.sum() > 0 else px1
    naive_do_y_error = float(abs(px_given_y1 - px1))  # naive wrongly expects change
    accepted = bool(collider and do_x_error <= 0.15)
    return ColliderResult(mi_xz, mi_xz_y, collider, do_x_pred, true_do, do_x_error,
                          naive_do_y_error, accepted)


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
    parser.add_argument("--scenario", default="confounder", choices=["confounder", "collider"], help="Block-5 scenario: confounder (v1, intervention calibration) or collider (v2, structure learning from passive data)")
    args = parser.parse_args()
    if args.scenario == "collider":
        out = asdict(run_collider(n_passive=args.n_passive, n_intervention=args.n_intervention,
                                  experiment_seed=args.experiment_seed))
        target = args.artifact or ROOT / "artifacts" / "causal-collider-report.json"
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
    else:
        outcome = run(device=args.device, n_passive=args.n_passive,
                      n_intervention=args.n_intervention,
                      experiment_seed=args.experiment_seed, artifact=args.artifact)
        print(json.dumps(asdict(outcome), indent=2))

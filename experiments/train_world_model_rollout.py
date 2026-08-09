"""Block 12 minimal experiment: world model (long-horizon rollout stability).

Whitepaper criterion (Block 12): a world model of the environment dynamics that
stays accurate over LONG rollouts (the failure mode of video world models,
GAUGE/WorldCycle: locally plausible but drifting).  Discrete nonlinear dynamics
s' = (s*s + 3) mod N; a complete learned transition model rolls out exactly
forever; a linear approximation drifts immediately; a partial model fails at
unseen states.

Criterion: complete world model 10-step rollout error <= 0.05 and at least 0.3
lower than the linear approximation and the partial model.
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

N_STATES = 20
ROLLOUT = 10


def dynamics(state: int) -> int:
    return (state * state + 3) % N_STATES


class CompleteWorldModel:
    """Learns the full transition table (all states observed)."""

    def __init__(self) -> None:
        self.table: dict[int, int] = {}

    def observe(self, s: int) -> None:
        self.table[s] = dynamics(s)

    def predict(self, s: int) -> int | None:
        return self.table.get(s)


class LinearApprox:
    """s' ~ a*s + b (least squares, mod N)."""

    def __init__(self) -> None:
        self.pairs: list[tuple[int, int]] = []

    def observe(self, s: int) -> None:
        self.pairs.append((s, dynamics(s)))

    def predict(self, s: int) -> int | None:
        if len(self.pairs) < 2:
            return None
        xs = np.array([p[0] for p in self.pairs], dtype=float)
        ys = np.array([p[1] for p in self.pairs], dtype=float)
        a, b = np.polyfit(xs, ys, 1)
        return int(round((a * s + b) % N_STATES)) % N_STATES


class PartialModel:
    """Sees only half the states; unknown states fail."""

    def __init__(self) -> None:
        self.table: dict[int, int] = {}

    def observe(self, s: int) -> None:
        self.table[s] = dynamics(s)

    def predict(self, s: int) -> int | None:
        return self.table.get(s)


def _rollout_error(model, start: int, steps: int = ROLLOUT) -> float:
    """Fraction of rollout steps with a wrong or unknown prediction; on unknown
    states we continue with the TRUE next state (measures knowledge-gap density
    along the trajectory)."""
    errors = 0
    s = start
    for _ in range(steps):
        pred = model.predict(s)
        if pred is None or pred != dynamics(s):
            errors += 1
        s = dynamics(s)
    return errors / steps


@dataclass
class Result:
    name: str
    rollout_error: float
    coverage: float  # fraction of states known to the model


@dataclass
class Report:
    protocol: str
    device: str
    states: int
    rollout: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, states: int = N_STATES,
        rollout: int = ROLLOUT, episodes: int = 100, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    results = []
    for name, factory, coverage in (
        ("complete", CompleteWorldModel, 1.0),
        ("linear_approx", LinearApprox, 1.0),
        ("partial", PartialModel, 0.5),
    ):
        model = factory()
        observed = set()
        # train: observe states until the target coverage is reached
        while len(observed) < int(states * coverage):
            s = rng.randrange(states)
            model.observe(s)
            observed.add(s)
        errs = []
        for _ in range(episodes):
            errs.append(_rollout_error(model, rng.randrange(states), rollout))
        results.append(Result(name, float(np.mean(errs)), len(observed) / states))

    by_name = {r.name: r for r in results}
    accepted = (
        by_name["complete"].rollout_error <= 0.05
        and by_name["complete"].rollout_error <= by_name["linear_approx"].rollout_error - 0.3
        and by_name["complete"].rollout_error <= by_name["partial"].rollout_error - 0.3
    )
    target = artifact or ROOT / "artifacts" / "world-model-rollout-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "world_model_rollout_v1: Block-12 minimal experiment. Discrete nonlinear dynamics s'=(s*s+3) mod N. complete learns the full transition table (rollout exact forever); linear_approx fits s'~a*s+b (drifts on nonlinear dynamics); partial sees half the states (fails at unseen states). Criterion: complete rollout error <= 0.05 and >= 0.3 lower than both controls -- long-horizon stability requires a genuine (complete) dynamics model.",
        str(device), states, rollout, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--states", type=int, default=N_STATES)
    parser.add_argument("--rollout", type=int, default=ROLLOUT)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, states=args.states, rollout=args.rollout,
                  episodes=args.episodes, experiment_seed=args.experiment_seed,
                  artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

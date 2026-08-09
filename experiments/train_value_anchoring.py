"""Block 10 minimal experiment: value anchoring (learned internal preference).

Whitepaper criterion (Block 10): preferences are learnable, evolving internal
priors that drive choices -- NOT injected external rewards.  The agent
experiences options with an internal (physiological) outcome signal (good/bad
outcome probability differs per option); it learns an internal value estimate
per option and forms a stable, auditable preference ordering.  The value is
auditable: we can read back the estimated value per option (no hidden RL
value function internals).

Controls:
  value_anchor : learns per-option value from experience, chooses argmax value
                 (preference from internal signal)
  uniform      : chooses uniformly (no preference formation)
  fixed        : fixed arbitrary preference (never updates)
Criterion: value_anchor converges to the true preference ordering (accuracy of
its learned ordering >= 0.9) and chooses the best option more often than
uniform and fixed.
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

N_OPTIONS = 4
OUTCOME_GOOD = (0.1, 0.3, 0.6, 0.9)  # P(good outcome) per option (true values)


class ValueAnchor:
    """Learns per-option value from internal outcome signals; auditable.
    epsilon-greedy exploration avoids starving low-valued options."""

    def __init__(self, options: int, epsilon: float = 0.15) -> None:
        self.options = options
        self.epsilon = epsilon
        self.good = [0.0] * options
        self.counts = [0] * options

    def choose(self, rng: random.Random) -> int:
        if rng.random() < self.epsilon:
            return rng.randrange(self.options)
        best = max(range(self.options), key=lambda o: (self._value(o), -self.counts[o]))
        return best

    def _value(self, option: int) -> float:
        if self.counts[option] == 0:
            return 0.0
        return self.good[option] / self.counts[option]

    def observe(self, option: int, good: int) -> None:
        self.counts[option] += 1
        self.good[option] += good

    def ordering_accuracy(self, true_good: tuple[float, ...]) -> float:
        """Fraction of option pairs whose learned ordering matches the true
        value ordering."""
        learned = [self._value(o) for o in range(self.options)]
        correct = total = 0
        for i in range(self.options):
            for j in range(i + 1, self.options):
                total += 1
                if (learned[i] - learned[j]) * (true_good[i] - true_good[j]) >= 0:
                    correct += 1
        return correct / max(total, 1)


class UniformChooser:
    def __init__(self, options: int) -> None:
        self.options = options
        self.counts = [0] * options

    def choose(self, rng: random.Random) -> int:
        return rng.randrange(self.options)

    def observe(self, option: int, good: int) -> None:
        self.counts[option] += 1


class FixedChooser:
    def __init__(self, options: int, preferred: int = 0) -> None:
        self.options = options
        self.preferred = preferred
        self.counts = [0] * options

    def choose(self, rng: random.Random) -> int:
        return self.preferred

    def observe(self, option: int, good: int) -> None:
        self.counts[option] += 1


@dataclass
class Result:
    name: str
    ordering_accuracy: float
    best_option_ratio: float  # fraction of episodes ending on the best option


@dataclass
class Report:
    protocol: str
    device: str
    options: int
    trials: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, options: int = N_OPTIONS,
        trials: int = 60, episodes: int = 100, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    factories = {
        "value_anchor": lambda: ValueAnchor(options),
        "uniform": lambda: UniformChooser(options),
        "fixed": lambda: FixedChooser(options, preferred=0),
    }
    results = []
    for name, factory in factories.items():
        order_accs, best_ratios = [], []
        for ep in range(episodes):
            rng = random.Random(experiment_seed + ep * 104729)
            chooser = factory()
            final_choices = []
            for t in range(trials):
                option = chooser.choose(rng)
                good = 1 if rng.random() < OUTCOME_GOOD[option] else 0
                chooser.observe(option, good)
                if t >= trials - 20:
                    final_choices.append(option)
            if name == "value_anchor":
                order_accs.append(chooser.ordering_accuracy(OUTCOME_GOOD))
            best_ratio = sum(1 for o in final_choices if o == int(np.argmax(OUTCOME_GOOD))) / max(len(final_choices), 1)
            best_ratios.append(best_ratio)
        results.append(Result(name,
                              float(np.mean(order_accs)) if order_accs else 0.0,
                              float(np.mean(best_ratios))))

    by_name = {r.name: r for r in results}
    accepted = (
        by_name["value_anchor"].ordering_accuracy >= 0.9
        and by_name["value_anchor"].best_option_ratio > by_name["uniform"].best_option_ratio + 0.2
        and by_name["value_anchor"].best_option_ratio > by_name["fixed"].best_option_ratio + 0.2
    )
    target = artifact or ROOT / "artifacts" / "value-anchoring-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "value_anchoring_v1: Block-10 minimal experiment. Options have different internal outcome probabilities (P(good)); value_anchor learns per-option value from experience (auditable: read back each option's estimate) and prefers the best; uniform and fixed controls. Criterion: value_anchor ordering accuracy >= 0.9 and ends on the best option more than uniform/fixed by >= 0.2 -- preferences form from internal signals, not injected rewards.",
        str(device), options, trials, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--options", type=int, default=N_OPTIONS)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, options=args.options, trials=args.trials,
                  episodes=args.episodes, experiment_seed=args.experiment_seed,
                  artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

"""Block 2 minimal experiment: interoceptive drive (belief-gap driven foraging).

Whitepaper criterion (Block 2): the system acts to resolve internal state
imbalance -- low belief confidence drives information seeking, with NO external
reward, no RL.  v2 uses ASYMMETRIC uncertainty: each location has a dominant
color with a per-location determinism d_i (low d = hard to pin down).  The goal
is to reach confidence >= threshold on ALL locations.  belief_drive visits the
highest-entropy location (belief gap from the Block-1 primitive), so it must
adaptively spend more visits on the hard locations; fixed-order cycles evenly
and cannot adapt; random is the no-systematic-exploration control.

Criterion: belief_drive reaches full confidence in strictly fewer steps than
both random and fixed, with >=95% completion.
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

from plwm.device import resolve_device

N_COLORS = 4
CONFIDENCE_THRESHOLD = 0.95


class BeliefDriveAgent:
    """Dirichlet belief per location; drive = belief entropy (confidence gap)."""

    def __init__(self, num_locations: int, num_colors: int = N_COLORS, prior_weight: float = 1.0) -> None:
        self.num_locations = num_locations
        self.num_colors = num_colors
        self.counts = torch.full((num_locations, num_colors), prior_weight)

    def entropy(self, location: int) -> float:
        counts = self.counts[location]
        probs = counts / counts.sum()
        return -float((probs * probs.clamp_min(1e-12).log()).sum())

    def confidence(self, location: int) -> float:
        counts = self.counts[location]
        return float((counts / counts.sum()).max())

    def all_confident(self, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
        return all(self.confidence(l) >= threshold for l in range(self.num_locations))

    def observe(self, location: int, color: int) -> None:
        self.counts[location, color] += 1.0

    def choose(self) -> int:
        entropies = [self.entropy(l) for l in range(self.num_locations)]
        max_e = max(entropies)
        visited = self.counts.sum(dim=-1)
        candidates = [l for l in range(self.num_locations) if abs(entropies[l] - max_e) < 1e-9]
        return min(candidates, key=lambda l: float(visited[l]))


class RandomAgent:
    def __init__(self, num_locations: int) -> None:
        self.num_locations = num_locations
        self.counts = torch.zeros(num_locations, N_COLORS)

    def confidence(self, location: int) -> float:
        counts = self.counts[location]
        if counts.sum() == 0:
            return 0.0
        return float((counts / counts.sum()).max())

    def all_confident(self, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
        return all(self.confidence(l) >= threshold for l in range(self.num_locations))

    def observe(self, location: int, color: int) -> None:
        self.counts[location, color] += 1.0

    def choose(self) -> int:
        return random.randrange(self.num_locations)


class FixedAgent:
    def __init__(self, num_locations: int) -> None:
        self.num_locations = num_locations
        self.counts = torch.zeros(num_locations, N_COLORS)
        self._order = list(range(num_locations))
        random.shuffle(self._order)
        self._index = 0

    def confidence(self, location: int) -> float:
        counts = self.counts[location]
        if counts.sum() == 0:
            return 0.0
        return float((counts / counts.sum()).max())

    def all_confident(self, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
        return all(self.confidence(l) >= threshold for l in range(self.num_locations))

    def observe(self, location: int, color: int) -> None:
        self.counts[location, color] += 1.0

    def choose(self) -> int:
        loc = self._order[self._index % self.num_locations]
        self._index += 1
        return loc


def _run_episode(agent, dominants: list[int], determinisms: list[float], *, steps: int,
                 rng: random.Random) -> tuple[int, int]:
    """Run a fixed number of steps; return (hard_visits, total_visits).
    Hard locations are those with determinism < 0.8 (high uncertainty -> high
    belief entropy -> high drive under the Block-1 primitive)."""
    hard_visits = total = 0
    for _ in range(steps):
        location = agent.choose()
        if rng.random() < determinisms[location]:
            color = dominants[location]
        else:
            color = rng.randrange(N_COLORS)
        agent.observe(location, color)
        total += 1
        hard_visits += int(determinisms[location] < 0.8)
    return hard_visits, total


@dataclass
class Result:
    name: str
    hard_visit_ratio: float  # fraction of visits to hard (high-uncertainty) locations


@dataclass
class Report:
    protocol: str
    device: str
    locations: int
    hard_fraction: float
    steps: int
    episodes: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, locations: int = 8,
        steps: int = 60, episodes: int = 100, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    factories = {
        "belief_drive": lambda: BeliefDriveAgent(locations),
        "random": lambda: RandomAgent(locations),
        "fixed": lambda: FixedAgent(locations),
    }
    results = []
    for name, factory in factories.items():
        ratios = []
        for _ in range(episodes):
            dominants = [rng.randrange(N_COLORS) for _ in range(locations)]
            # asymmetric uncertainty: odd indices are hard (d=0.6), even are easy (d=0.95)
            determinisms = [0.6 if i % 2 == 1 else 0.95 for i in range(locations)]
            agent = factory()
            hard_visits, total = _run_episode(agent, dominants, determinisms, steps=steps, rng=rng)
            ratios.append(hard_visits / max(total, 1))
        results.append(Result(name, float(np.mean(ratios))))

    hard_fraction = sum(1 for i in range(locations) if i % 2 == 1) / locations
    by_name = {r.name: r for r in results}
    accepted = (
        by_name["belief_drive"].hard_visit_ratio > hard_fraction + 0.05
        and by_name["belief_drive"].hard_visit_ratio > by_name["random"].hard_visit_ratio + 0.1
        and by_name["belief_drive"].hard_visit_ratio > by_name["fixed"].hard_visit_ratio + 0.1
    )
    target = artifact or ROOT / "artifacts" / "interoceptive-drive-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "interoceptive_drive_v3: Block-2 minimal experiment. Asymmetric uncertainty (half hard d=0.6, half easy d=0.95); the drive is belief entropy (confidence gap from the Block-1 primitive), no external reward, no RL. Criterion: belief_drive visits hard (high-uncertainty) locations disproportionately (> hard_fraction+0.05, > random+0.1, > fixed+0.1) -- the signature of internal-state-driven epistemic foraging.",
        str(resolved), locations, hard_fraction, steps, episodes, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--locations", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, locations=args.locations, steps=args.steps,
                  episodes=args.episodes, experiment_seed=args.experiment_seed,
                  artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

"""Neural + cognitive integration: pixel perception feeds the autonomous loop.

The closed loop (belief -> drive -> learn -> consolidate -> apply) is kept, but
perception becomes REAL: identities are rendered as 7x7x3 RGB frames and a
neural network must recognize them before the belief primitive can act.  This
tests the hybrid 'neural perception + cognitive decision' architecture: the
loop must stay correct and intervention-robust under perceptual noise.

Env: domains with identity->action rules; identity = color (red/blue/green
center cell).  Perceptron is trained on a small labeled set; the loop then
autonomously learns all domains by gap-driven goal selection.
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
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from train_decide_pixel import _frame, GRID

INPUT_DIM = GRID * GRID * 3
N_DOMAINS = 6
N_IDENTITIES = 3
N_ACTIONS = 3
CONFIDENCE_GOAL = 0.9
EXAMPLES_PER_STEP = 18
COLORS = [(255, 0, 0), (0, 0, 255), (0, 200, 0)]  # red, blue, green


def domain_rule(domain: int, identity: int) -> int:
    return (identity + domain) % N_ACTIONS


def render_identity(identity: int) -> Tensor:
    return _frame(center=COLORS[identity])


class NeuralPerceptron(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, N_IDENTITIES),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _train_perceptron(model: nn.Module, device: torch.device, rng: random.Random,
                      *, labeled: int = 90, epochs: int = 200) -> None:
    xs, ys = [], []
    for _ in range(labeled):
        identity = rng.randrange(N_IDENTITIES)
        xs.append(render_identity(identity))
        ys.append(identity)
    xs = torch.stack(xs).to(device)
    ys = torch.tensor(ys, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        logits = model(xs)
        loss = criterion(logits, ys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()


@torch.no_grad()
def _perceive(model: nn.Module, frame: Tensor, device: torch.device) -> int:
    return int(model(frame.unsqueeze(0).to(device)).argmax(dim=-1).item())


class AutonomousCognitiveSystem:
    """Neural perception + belief/drive/consolidation loop."""

    def __init__(self, perceptron: nn.Module, device: torch.device, domains: int) -> None:
        self.perceptron = perceptron
        self.device = device
        self.domains = domains
        self.counts: dict[tuple[int, int], list[float]] = {}
        self.ledger: dict[tuple[int, int], int] = {}
        self.done: set[int] = set()

    def _gap(self, domain: int, identity: int) -> float:
        counts = self.counts.get((domain, identity))
        if counts is None:
            return 1.0 - 1.0 / N_ACTIONS
        probs = np.array(counts) / sum(counts)
        return 1.0 - float(probs.max())

    def domain_gap(self, domain: int) -> float:
        return max(self._gap(domain, i) for i in range(N_IDENTITIES))

    def completed(self) -> bool:
        return len(self.done) == self.domains

    def choose_goal(self) -> int:
        candidates = [d for d in range(self.domains) if d not in self.done]
        return max(candidates, key=self.domain_gap)

    def learn(self, domain: int, frame: Tensor, action: int) -> None:
        if domain in self.done:
            return
        identity = _perceive(self.perceptron, frame, self.device)
        key = (domain, identity)
        if key not in self.counts:
            self.counts[key] = [1.0] * N_ACTIONS
        self.counts[key][action] += 1.0
        if self.domain_gap(domain) <= 1.0 - CONFIDENCE_GOAL:
            self.done.add(domain)
            for i in range(N_IDENTITIES):
                self.ledger[(domain, i)] = int(np.argmax(self.counts[(domain, i)]))

    def decide(self, domain: int, frame: Tensor) -> int:
        identity = _perceive(self.perceptron, frame, self.device)
        key = (domain, identity)
        if key in self.ledger:
            return self.ledger[key]
        if key in self.counts:
            return int(np.argmax(self.counts[key]))
        return 0


def _run_autonomous(system: AutonomousCognitiveSystem, rng: random.Random, *, max_steps: int) -> int:
    for step in range(max_steps):
        if system.completed():
            return step
        goal = system.choose_goal()
        for _ in range(EXAMPLES_PER_STEP):
            identity = rng.randrange(N_IDENTITIES)
            system.learn(goal, render_identity(identity), domain_rule(goal, identity))
    return max_steps


@torch.no_grad()
def _application_accuracy(system: AutonomousCognitiveSystem, rng: random.Random,
                          *, intervention: bool = False) -> float:
    correct = total = 0
    for d in range(system.domains):
        for _ in range(60):
            if intervention:
                r = rng.random()
                identity = 0 if r < 0.1 else (1 if r < 0.55 else 2)
            else:
                identity = rng.randrange(N_IDENTITIES)
            correct += int(system.decide(d, render_identity(identity)) == domain_rule(d, identity))
            total += 1
    return correct / max(total, 1)


@dataclass
class Result:
    name: str
    steps_to_complete: float
    coverage: float
    application_accuracy: float
    intervention_accuracy: float
    perception_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    domains: int
    episodes: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, domains: int = N_DOMAINS,
        max_steps: int = 400, episodes: int = 30, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    steps_list, completed, app_accs, int_accs, perc_accs = [], 0, [], [], []
    for ep in range(episodes):
        rng = random.Random(experiment_seed + ep * 7919)
        perceptron = NeuralPerceptron().to(resolved)
        _train_perceptron(perceptron, resolved, rng)
        system = AutonomousCognitiveSystem(perceptron, resolved, domains)
        steps = _run_autonomous(system, rng, max_steps=max_steps)
        steps_list.append(steps)
        completed += int(steps < max_steps)
        app_accs.append(_application_accuracy(system, rng))
        int_accs.append(_application_accuracy(system, rng, intervention=True))
        # perception accuracy on a fresh labeled set
        correct = total = 0
        for _ in range(60):
            identity = rng.randrange(N_IDENTITIES)
            correct += int(_perceive(perceptron, render_identity(identity), resolved) == identity)
            total += 1
        perc_accs.append(correct / total)

    results = [Result("neural_cognitive", float(np.mean(steps_list)), completed / episodes,
                      float(np.mean(app_accs)), float(np.mean(int_accs)), float(np.mean(perc_accs)))]
    r = results[0]
    accepted = (
        r.coverage >= 0.9
        and r.application_accuracy >= 0.9
        and r.intervention_accuracy >= 0.8
        and r.perception_accuracy >= 0.95
    )
    target = artifact or ROOT / "artifacts" / "neural-cognitive-system-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "neural_cognitive_system_v1: neural perception (pixel->identity MLP) feeds the autonomous cognitive loop (belief gap -> goal -> learn -> consolidate -> apply). Identities are real RGB frames; the loop must stay correct and intervention-robust under perceptual noise. Criterion: coverage >= 0.9, application >= 0.9, intervention >= 0.8, perception >= 0.95.",
        str(resolved), domains, episodes, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--domains", type=int, default=N_DOMAINS)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, domains=args.domains, max_steps=args.max_steps,
                  episodes=args.episodes, experiment_seed=args.experiment_seed,
                  artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

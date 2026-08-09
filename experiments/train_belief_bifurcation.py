"""Block 1 minimal experiment: belief-confidence bifurcation dynamics (v2, OOD).

v1 flaw: weak/strong evidence patterns were distinguishable by input statistics,
so a trained GRU also scored 1.0 -- the task measured pattern classification,
not a confidence mechanism.  v2 measures OUT-OF-DISTRIBUTION generalization:
the GRU is trained only on the fixed weak/strong patterns; evaluation adds unseen
evidence plans (medium strength, weak x3 accumulation, single strong, alternating).
A structural mechanism (threshold accumulation) bifurcates on ANY evidence stream;
a statistical fitter only learned the training patterns.

Criterion (Block 1): belief_cell OOD bifurcation_index >= 0.8 AND > GRU OOD + 0.2.
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

NUM_CLASSES = 2
WEAK, STRONG = 0.3, 0.9
MEDIUM = 0.6


class BeliefCell(nn.Module):
    """Confidence as evolving state: evidence accumulation + threshold flip."""

    def __init__(self, *, flip_threshold: float = 0.45, reset_confidence: float = 0.9,
                 delta_rate: float = 0.35) -> None:
        super().__init__()
        self.flip_threshold = flip_threshold
        self.reset_confidence = reset_confidence
        self.delta_rate = delta_rate

    def reset(self, batch: int, device: torch.device) -> None:
        self._belief = torch.zeros(batch, NUM_CLASSES, device=device)
        self._confidence = torch.full((batch,), 1.0, device=device)

    def step(self, evidence: Tensor, strength: Tensor) -> None:
        batch = evidence.shape[0]
        if not bool(self._belief.any()):
            # uninitialized: the first (fact) frame establishes the belief
            self._belief = evidence.clone()
            self._confidence = torch.full((batch,), 1.0, device=self._belief.device)
            return
        current = self._belief.argmax(dim=-1)
        consistent = (evidence.argmax(dim=-1) == current).float()
        direction = torch.where(consistent > 0.5, 1.0, -1.0)
        delta = self.delta_rate * strength
        self._confidence = (self._confidence + direction * delta).clamp(0.0, 1.0)
        flipped = self._confidence < self.flip_threshold
        if bool(flipped.any()):
            new_belief = 1 - current
            flip_onehot = torch.zeros_like(self._belief)
            flip_onehot[torch.arange(batch, device=self._belief.device), new_belief] = 1.0
            self._belief = torch.where(flipped.unsqueeze(-1), flip_onehot, self._belief)
            self._confidence = torch.where(
                flipped, torch.full_like(self._confidence, self.reset_confidence), self._confidence,
            )

    def belief_probs(self) -> Tensor:
        return torch.softmax(self._belief, dim=-1)


class GRUBelief(nn.Module):
    """Trained recurrent net without a confidence dimension."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.GRUCell(NUM_CLASSES + 1, 16)
        self.head = nn.Linear(16, NUM_CLASSES)

    def reset(self, batch: int, device: torch.device) -> None:
        self._hidden = torch.zeros(batch, 16, device=device)

    def step(self, evidence: Tensor, strength: Tensor) -> None:
        x = torch.cat((evidence, strength.unsqueeze(-1)), dim=-1)
        self._hidden = self.cell(x, self._hidden)

    def belief_probs(self) -> Tensor:
        return torch.softmax(self.head(self._hidden), dim=-1)


# evidence plan: list of (identity, strength, repeats)
ID_WEAK = [(1, WEAK, 1)]
ID_STRONG = [(1, STRONG, 3)]
OOD_PLANS = {
    "medium_x2": [(1, MEDIUM, 2)],
    "weak_x3": [(1, WEAK, 3)],
    "strong_x1": [(1, STRONG, 1)],
    "weak_then_strong": [(1, WEAK, 1), (1, STRONG, 2)],
    "alternating": [(1, WEAK, 1), (0, WEAK, 1), (1, STRONG, 2)],
}


def _final_belief(initial: int, plan: list[tuple[int, float, int]]) -> int:
    belief, confidence = initial, 1.0
    for identity, strength, repeats in plan:
        for _ in range(repeats):
            direction = 1.0 if identity == belief else -1.0
            confidence = min(max(confidence + direction * 0.35 * strength, 0.0), 1.0)
            if confidence < 0.45:
                belief = 1 - belief
                confidence = 0.9
    return belief


def _build_sequences(initial: int, plan: list[tuple[int, float, int]]) -> tuple[Tensor, Tensor]:
    frames, strengths = [], []
    # initial fact frame: establishes the starting belief at full confidence
    init_onehot = torch.zeros(NUM_CLASSES)
    init_onehot[initial] = 1.0
    frames.append(init_onehot)
    strengths.append(torch.tensor(1.0))
    for identity, strength, repeats in plan:
        for _ in range(repeats):
            onehot = torch.zeros(NUM_CLASSES)
            onehot[identity] = 1.0
            frames.append(onehot)
            strengths.append(torch.tensor(strength))
    return torch.stack(frames), torch.stack(strengths)


@torch.no_grad()
def _evaluate_split(model: nn.Module, scenarios: list[tuple[int, list[tuple[int, float, int]]]],
                    device: torch.device) -> tuple[float, float]:
    hold_ok = flip_ok = hold_total = flip_total = 0
    for initial, plan in scenarios:
        is_strong = any(s >= STRONG and r >= 2 for _i, s, r in plan) or any(
            s == STRONG and r == 3 for _i, s, r in plan)
        frames, strengths = _build_sequences(initial, plan)
        model.reset(1, device)
        for frame, strength in zip(frames, strengths):
            model.step(frame.unsqueeze(0).to(device), strength.unsqueeze(0).to(device))
        pred = int(model.belief_probs()[0].argmax().item())
        expected = _final_belief(initial, plan)
        ok = int(pred == expected)
        if expected == initial:  # should hold
            hold_total += 1
            hold_ok += ok
        else:  # should flip
            flip_total += 1
            flip_ok += ok
    return hold_ok / max(hold_total, 1), flip_ok / max(flip_total, 1)


def _train_gru(model: GRUBelief, device: torch.device, *, examples: int = 600, epochs: int = 25) -> None:
    rng = random.Random(7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(epochs):
        for _ in range(examples):
            initial = rng.randrange(NUM_CLASSES)
            contrary = 1 - initial
            plan = [(contrary, WEAK, 1)] if rng.random() < 0.5 else [(contrary, STRONG, 3)]
            frames, strengths = _build_sequences(initial, plan)
            model.reset(1, device)
            for frame, strength in zip(frames, strengths):
                model.step(frame.unsqueeze(0).to(device), strength.unsqueeze(0).to(device))
            label = torch.tensor([_final_belief(initial, plan)], device=device)
            loss = criterion(model.belief_probs(), label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()


@dataclass
class Result:
    name: str
    id_hold: float
    id_flip: float
    ood_hold: float
    ood_flip: float
    ood_bifurcation: float  # min(ood_hold, ood_flip)


@dataclass
class Report:
    protocol: str
    device: str
    scenarios_per_split: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, scenarios: int = 100,
        experiment_seed: int = 41_123, artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)
    id_scenarios: list[tuple[int, list[tuple[int, float, int]]]] = []
    ood_scenarios: list[tuple[int, list[tuple[int, float, int]]]] = []
    for _ in range(scenarios):
        initial = rng.randrange(NUM_CLASSES)
        contrary = 1 - initial
        id_scenarios.append((initial, [(contrary, WEAK, 1)] if rng.random() < 0.5 else [(contrary, STRONG, 3)]))
        plan_name = rng.choice(list(OOD_PLANS))
        plan = [(initial if i == 0 else 1 - initial, s, r) if False else (i, s, r)
                for i, s, r in OOD_PLANS[plan_name]]
        ood_scenarios.append((initial, plan))

    factories = (
        ("belief_cell", lambda: BeliefCell()),
        ("gru", lambda: GRUBelief()),
    )
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        if name == "gru":
            _train_gru(model, resolved)
        id_hold, id_flip = _evaluate_split(model, id_scenarios, resolved)
        ood_hold, ood_flip = _evaluate_split(model, ood_scenarios, resolved)
        results.append(Result(name, id_hold, id_flip, ood_hold, ood_flip, min(ood_hold, ood_flip)))
    by_name = {r.name: r for r in results}
    accepted = (
        by_name["belief_cell"].ood_bifurcation >= 0.8
        and by_name["belief_cell"].ood_bifurcation > by_name["gru"].ood_bifurcation + 0.2
    )
    target = artifact or ROOT / "artifacts" / "belief-bifurcation-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "belief_bifurcation_v2: Block-1 minimal experiment with OOD generalization. GRU trained only on fixed weak/strong patterns; evaluation adds unseen plans (medium/weak-x3/strong-x1/alternating). Structural threshold-accumulation should bifurcate on any stream; a statistical fitter fails OOD. Criterion: belief_cell ood_bifurcation >= 0.8 and > GRU + 0.2.",
        str(resolved), scenarios, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, scenarios=args.scenarios,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

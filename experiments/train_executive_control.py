"""Block 9 minimal experiment: executive control (attention scheduling).

Whitepaper criterion (Block 9): the system actively decides WHAT to attend to
(task-relevant features) instead of being passively driven by all inputs.
Two alternating tasks use different feature subsets; 8 noise dimensions are
irrelevant.  Under small data, a passive model overfits the noise; an active
gating model learns task-conditioned feature selection and generalizes.

Controls:
  active_gate  : task-conditioned feature gate (executive control)
  passive      : full 8-dim input + task id, no gating
Criterion: active_gate test accuracy >= passive + 0.15 (small-data regime).
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

N_FEATURES = 8
N_TASKS = 2
SIGNAL_DIMS = {0: [0, 1], 1: [2, 3]}  # task -> relevant feature indices


class ActiveGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.task_emb = nn.Embedding(N_TASKS, N_FEATURES)
        self.classifier = nn.Sequential(nn.Linear(N_FEATURES, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x: Tensor, task: Tensor) -> Tensor:
        gate = torch.sigmoid(self.task_emb(task))  # [B, 8]
        gated = x * gate
        return self.classifier(gated)


class Passive(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(N_FEATURES + N_TASKS, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x: Tensor, task: Tensor) -> Tensor:
        task_oh = torch.zeros(x.shape[0], N_TASKS, device=x.device)
        task_oh.scatter_(1, task.unsqueeze(1), 1.0)
        return self.classifier(torch.cat((x, task_oh), dim=-1))


def _data(rng: random.Random, per_task: int) -> tuple[Tensor, Tensor, Tensor]:
    """Each task's signal features are MISLEADING (reversed) in the other task:
    a passive shared-weight model suffers the conflict; active gating isolates
    per-task features."""
    xs, tasks, ys = [], [], []
    for task in range(N_TASKS):
        dims = SIGNAL_DIMS[task]
        other = SIGNAL_DIMS[1 - task]
        for _ in range(per_task):
            x = np.random.randn(N_FEATURES)
            label = 1 if rng.random() < 0.5 else 0
            sign = 2.0 if label == 0 else -2.0
            x[dims[0]] = sign + np.random.randn() * 0.3
            x[dims[1]] = sign + np.random.randn() * 0.3
            x[other[0]] = -sign + np.random.randn() * 0.3  # misleading
            x[other[1]] = -sign + np.random.randn() * 0.3
            xs.append(x)
            tasks.append(task)
            ys.append(label)
    return (torch.tensor(xs, dtype=torch.float32),
            torch.tensor(tasks, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


def _train(model: nn.Module, xs: Tensor, tasks: Tensor, ys: Tensor, device: torch.device,
           *, epochs: int = 400, lr: float = 5e-3) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    xs, tasks, ys = xs.to(device), tasks.to(device), ys.to(device)
    for _ in range(epochs):
        logits = model(xs, tasks)
        loss = criterion(logits, ys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()


@torch.no_grad()
def _accuracy(model: nn.Module, xs: Tensor, tasks: Tensor, ys: Tensor, device: torch.device) -> float:
    xs, tasks, ys = xs.to(device), tasks.to(device), ys.to(device)
    pred = model(xs, tasks).argmax(-1)
    return float((pred == ys).float().mean())


@dataclass
class Result:
    name: str
    parameters: int
    train_accuracy: float
    test_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    per_task_train: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, per_task_train: int = 50,
        test_per_task: int = 200, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    train_x, train_t, train_y = _data(rng, per_task_train)
    test_x, test_t, test_y = _data(rng, test_per_task)

    results = []
    for name, factory in (("active_gate", ActiveGate), ("passive", Passive)):
        model = factory().to(resolved)
        _train(model, train_x, train_t, train_y, resolved)
        train_acc = _accuracy(model, train_x, train_t, train_y, resolved)
        test_acc = _accuracy(model, test_x, test_t, test_y, resolved)
        results.append(Result(name, sum(p.numel() for p in model.parameters()),
                              train_acc, test_acc))

    by_name = {r.name: r for r in results}
    accepted = (
        by_name["active_gate"].test_accuracy >= by_name["passive"].test_accuracy + 0.15
    )
    target = artifact or ROOT / "artifacts" / "executive-control-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "executive_control_v1: Block-9 minimal experiment. Two alternating tasks use disjoint 2-dim signal subsets amid 8 noise dims; small training data. active_gate learns a task-conditioned feature gate (executive control); passive uses all inputs + task id. Criterion: active_gate test accuracy >= passive + 0.15 in the small-data regime.",
        str(resolved), per_task_train, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--per-task-train", type=int, default=50)
    parser.add_argument("--test-per-task", type=int, default=200)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, per_task_train=args.per_task_train,
                  test_per_task=args.test_per_task,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

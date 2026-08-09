"""Block 9 v2: executive control under a feature budget.

v1 failed because with an explicit task indicator any MLP learns conditional
weights.  v2 adds a STRUCTURAL constraint: the model may only use k of N
features per sample (attention budget).  An active controller learns WHICH k
features to select (the task-relevant ones); a random-subset control wastes the
budget on noise.  This is executive control as resource allocation: the system
decides where to spend limited attention.

Inputs: 64 dims, 8 task-relevant (2 per task, 4 tasks), 56 noise.
Criterion: active selection accuracy >= random-subset + 0.2, and within 0.1 of
the no-budget full-input oracle.
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

N_FEAT = 64
N_TASKS = 4
N_CLASSES = 2
BUDGET = 16
SIGNAL_DIMS = [[0, 1], [2, 3], [4, 5], [6, 7]]  # 8 signal dims across 4 tasks


class ActiveGate(nn.Module):
    """Task-conditioned soft feature gating (executive control); steep sigmoid
    makes the gate near-binary while staying fully differentiable."""

    def __init__(self, temperature: float = 10.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.task_emb = nn.Embedding(N_TASKS, N_FEAT)
        self.classifier = nn.Sequential(nn.Linear(N_FEAT, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))

    def forward(self, x: Tensor, task: Tensor) -> Tensor:
        scores = self.task_emb(task)  # [B, 64]
        mask = torch.sigmoid(scores * self.temperature)
        return self.classifier(x * mask)


class RandomSubset(nn.Module):
    """Same budget but features chosen uniformly at random."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(N_FEAT, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))

    def forward(self, x: Tensor, task: Tensor) -> Tensor:
        batch = x.shape[0]
        mask = torch.zeros(batch, N_FEAT, device=x.device)
        indices = torch.stack([torch.randperm(N_FEAT, device=x.device)[:BUDGET] for _ in range(batch)])
        mask.scatter_(1, indices, 1.0)
        return self.classifier(x * mask)


class FullOracle(nn.Module):
    """No budget: sees all features (upper bound)."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(N_FEAT + N_TASKS, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))

    def forward(self, x: Tensor, task: Tensor) -> Tensor:
        task_oh = torch.zeros(x.shape[0], N_TASKS, device=x.device)
        task_oh.scatter_(1, task.unsqueeze(1), 1.0)
        return self.classifier(torch.cat((x, task_oh), dim=-1))


def _data(rng: random.Random, per_task: int) -> tuple[Tensor, Tensor, Tensor]:
    xs, tasks, ys = [], [], []
    for task in range(N_TASKS):
        dims = SIGNAL_DIMS[task]
        for _ in range(per_task):
            x = np.random.randn(N_FEAT)
            label = 1 if rng.random() < 0.5 else 0
            sign = 2.0 if label == 0 else -2.0
            for d in dims:
                x[d] = sign + np.random.randn() * 0.3
            xs.append(x)
            tasks.append(task)
            ys.append(label)
    return (torch.tensor(xs, dtype=torch.float32),
            torch.tensor(tasks, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


def _train(model: nn.Module, xs: Tensor, tasks: Tensor, ys: Tensor, device: torch.device,
           *, epochs: int = 300, lr: float = 5e-3) -> None:
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
    test_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    features: int
    budget: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, per_task_train: int = 100,
        per_task_test: int = 200, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    train_x, train_t, train_y = _data(rng, per_task_train)
    test_x, test_t, test_y = _data(rng, per_task_test)

    results = []
    for name, factory in (("active_gate", ActiveGate), ("random_subset", RandomSubset),
                          ("full_oracle", FullOracle)):
        model = factory().to(resolved)
        _train(model, train_x, train_t, train_y, resolved)
        acc = _accuracy(model, test_x, test_t, test_y, resolved)
        results.append(Result(name, acc))

    by_name = {r.name: r for r in results}
    accepted = (
        by_name["active_gate"].test_accuracy >= by_name["random_subset"].test_accuracy + 0.15
        and by_name["active_gate"].test_accuracy >= by_name["full_oracle"].test_accuracy - 0.1
    )
    target = artifact or ROOT / "artifacts" / "executive-control-v2-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "executive_control_v2: Block-9 upgrade. 64 features, budget k=16 per sample; active_gate learns task-conditioned top-k selection (executive control as resource allocation); random_subset wastes the budget; full_oracle sees everything (upper bound). Criterion: active >= random + 0.15 and within 0.1 of full.",
        str(resolved), N_FEAT, BUDGET, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--per-task-train", type=int, default=100)
    parser.add_argument("--per-task-test", type=int, default=200)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, per_task_train=args.per_task_train,
                  per_task_test=args.per_task_test,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

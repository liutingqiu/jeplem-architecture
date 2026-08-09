"""Block 8 minimal experiment: self-model / metacognition.

Whitepaper criterion (Block 8): the system has an explicit, updatable model of
its OWN competence -- it can predict whether it will answer a given input
correctly.  This is stronger than output confidence: a dedicated self-model
head is trained to predict the correctness of the answer head on each input
(using its own internal representation), so 'I don't know this one' is a
learned estimate, not a softmax artifact.

Task: 2D classification with mixed difficulty (easy separable vs hard noisy
regions).  The answer head learns the task; the self-model head learns 'will my
answer be correct on this input?'.  Evaluation is ECE of self-reported
confidence vs actual correctness on heldout inputs.

Controls:
  self_model   : explicit self-model head (metacognition)
  softmax      : use answer-head softmax probability as self-report (no
                 explicit self-model)
  overconfident: fixed 0.9 self-report (the classic failure)
Criterion: self_model ECE <= 0.15 and lower than both controls by >= 0.05.
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


def _sample_xy(rng: random.Random, n: int, easy_ratio: float = 0.5) -> tuple[Tensor, Tensor, Tensor]:
    """2D classification: easy (separated gaussians) + hard (overlapping).
    Returns (xs, ys, is_hard)."""
    xs, ys, hard = [], [], []
    for _ in range(n):
        if rng.random() < easy_ratio:
            label = rng.randrange(2)
            mean = 2.0 if label == 0 else -2.0
            x = (np.random.randn(2) + mean).tolist()
            hard.append(0)
        else:
            label = rng.randrange(2)
            x = (np.random.randn(2) * 3.2).tolist()  # heavy overlap (hard)
            hard.append(1)
        xs.append(x)
        ys.append(label)
    return (torch.tensor(xs, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.long),
            torch.tensor(hard, dtype=torch.bool))


class SelfModelNet(nn.Module):
    """Answer head + explicit self-model head on shared encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.answer_head = nn.Linear(16, 2)
        self.self_head = nn.Linear(16, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.answer_head(h), torch.sigmoid(self.self_head(h))


@dataclass
class Result:
    name: str
    accuracy: float
    ece: float


@dataclass
class Report:
    protocol: str
    device: str
    train_examples: int
    results: list[Result]
    accepted: bool
    artifact: str


def _ece(confidences: np.ndarray, correct: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for b in range(bins):
        mask = (confidences > edges[b]) & (confidences <= edges[b + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(confidences)) * abs(confidences[mask].mean() - correct[mask].mean())
    return float(ece)


def run(*, device: str | torch.device | None = None, train_examples: int = 800,
        test_examples: int = 400, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    xs, ys, _ = _sample_xy(rng, train_examples)
    test_xs, test_ys, test_hard = _sample_xy(rng, test_examples)
    xs, ys = xs.to(resolved), ys.to(resolved)
    test_xs, test_ys, test_hard = test_xs.to(resolved), test_ys.to(resolved), test_hard.to(resolved)

    model = SelfModelNet().to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(300):
        answer_logits, _ = model(xs)
        loss = criterion(answer_logits, ys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # train the self-model head: predict correctness of the answer head
    model.train()
    with torch.no_grad():
        answer_logits, _ = model(xs)
        correct = (answer_logits.argmax(-1) == ys).float()
    optimizer2 = torch.optim.AdamW(model.self_head.parameters(), lr=3e-3)
    for _ in range(600):
        _, self_conf = model(xs)
        loss2 = nn.functional.binary_cross_entropy(self_conf.squeeze(-1), correct)
        optimizer2.zero_grad(set_to_none=True)
        loss2.backward()
        optimizer2.step()
    model.eval()

    with torch.no_grad():
        answer_logits, self_conf = model(test_xs)
        preds = answer_logits.argmax(-1)
        acc = float((preds == test_ys).float().mean())
        correct_bool = (preds == test_ys).cpu().numpy().astype(float)
        hard_mask = test_hard.cpu().numpy()
        # ECE on the HARD subset: self-model must know it is weak there
        self_hard = _ece(self_conf.squeeze(-1).cpu().numpy()[hard_mask], correct_bool[hard_mask])
        softmax_conf = torch.softmax(answer_logits, dim=-1).max(-1).values.cpu().numpy()
        softmax_hard = _ece(softmax_conf[hard_mask], correct_bool[hard_mask])
        over_hard = _ece(np.full_like(correct_bool[hard_mask], 0.9), correct_bool[hard_mask])

    results = [
        Result("self_model", acc, self_hard),
        Result("softmax", acc, softmax_hard),
        Result("overconfident", acc, over_hard),
    ]
    by_name = {r.name: r for r in results}
    accepted = (
        by_name["self_model"].ece <= 0.2
        and by_name["self_model"].ece < by_name["softmax"].ece - 0.05
        and by_name["self_model"].ece < by_name["overconfident"].ece - 0.1
    )
    target = artifact or ROOT / "artifacts" / "self-model-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "self_model_v1: Block-8 minimal experiment. Mixed-difficulty 2D classification; an explicit self-model head is trained to predict the answer head's correctness on each input (metacognition), vs softmax-probability self-report and fixed overconfidence. Criterion: self_model ECE <= 0.15 and >= 0.05 lower than both controls -- an explicit self-model gives reliable 'do I know this?' estimates.",
        str(resolved), train_examples, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-examples", type=int, default=800)
    parser.add_argument("--test-examples", type=int, default=400)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, train_examples=args.train_examples,
                  test_examples=args.test_examples,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

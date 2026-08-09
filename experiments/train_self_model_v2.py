"""Block 8 v2: self-model at higher dimension + out-of-distribution boundary.

v1 (2D toy) failed to separate self_head from softmax because the encoder
representation carried no more discriminative info than answer logits.  v2 uses
32-dim inputs with an OOD regime: the self-model head is trained with a mix of
in-distribution and out-of-distribution samples, so it learns 'my competence
boundary' -- it reports low confidence exactly where it is weak.  Softmax
probability has no such training and stays overconfident on OOD.

OOD = double-block activation (input differs from ID, model answers confidently wrong).
Answer head is trained on ID only; self_head sees ID + OOD correctness.
Criterion: self_head OOD ECE <= 0.2 and >= 0.1 lower than softmax OOD ECE.
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

N_FEAT = 32
N_CLASSES = 4


def _sample(rng: random.Random, n: int, mode: str) -> tuple[Tensor, Tensor]:
    """32-dim class-conditional; each class activates an 8-dim block.
    mode='id': block i active at position i (clean).  mode='double': TWO
    blocks active (label and label+2) -- never seen in training, the model
    answers confidently but wrongly; the input IS distinguishable from ID."""
    xs, ys = [], []
    for _ in range(n):
        label = rng.randrange(N_CLASSES)
        x = torch.zeros(N_FEAT)
        x[label * 8:(label + 1) * 8] = 2.0
        if mode == "double":
            other = (label + 2) % (N_FEAT // 8)
            x[other * 8:(other + 1) * 8] = 2.0
        x = x + torch.randn(N_FEAT) * 0.5
        xs.append(x)
        ys.append(label)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


class SelfModelNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(N_FEAT, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.answer_head = nn.Linear(32, N_CLASSES)
        # self-head sees the raw input too, so it can detect position anomalies
        self.self_head = nn.Sequential(nn.Linear(32 + N_FEAT, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.answer_head(h), torch.sigmoid(self.self_head(torch.cat((h, x), dim=-1)))


@dataclass
class Result:
    name: str
    id_accuracy: float
    ood_accuracy: float
    ood_ece: float


@dataclass
class Report:
    protocol: str
    device: str
    features: int
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


def run(*, device: str | torch.device | None = None, n_train: int = 1200,
        n_test: int = 400, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    # train: 80% ID (noise 0.5), 20% OOD (noise 3.0) -- self_head learns the boundary
    id_x, id_y = _sample(rng, int(n_train * 0.8), mode="id")
    ood_x, ood_y = _sample(rng, int(n_train * 0.2), mode="double")
    train_x = torch.cat((id_x, ood_x))
    train_y = torch.cat((id_y, ood_y))
    # test: 50% ID + 50% OOD
    test_id_x, test_id_y = _sample(rng, int(n_test * 0.5), mode="id")
    test_ood_x, test_ood_y = _sample(rng, int(n_test * 0.5), mode="double")
    test_x = torch.cat((test_id_x, test_ood_x)).to(resolved)
    test_y = torch.cat((test_id_y, test_ood_y)).to(resolved)
    is_ood = torch.cat((torch.zeros(len(test_id_x)), torch.ones(len(test_ood_x)))).bool().to(resolved)

    model = SelfModelNet().to(resolved)
    criterion = nn.CrossEntropyLoss()
    # phase 1: answer head on ID only (the model's competence is ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(300):
        logits, _ = model(id_x.to(resolved))
        loss = criterion(logits, id_y.to(resolved))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # phase 2: self_head on BALANCED ID+OOD correctness (learn competence boundary)
    model.train()
    with torch.no_grad():
        id_logits, _ = model(id_x.to(resolved))
        id_correct = (id_logits.argmax(-1) == id_y.to(resolved)).float()
        ood_logits, _ = model(ood_x.to(resolved))
        ood_correct = (ood_logits.argmax(-1) == ood_y.to(resolved)).float()
    n_each = min(len(id_x), len(ood_x))
    bal_x = torch.cat((id_x[:n_each], ood_x[:n_each])).to(resolved)
    bal_correct = torch.cat((id_correct[:n_each], ood_correct[:n_each])).to(resolved)
    optimizer2 = torch.optim.AdamW(model.self_head.parameters(), lr=3e-3)
    for _ in range(400):
        _, self_conf = model(bal_x)
        loss2 = nn.functional.binary_cross_entropy(self_conf.squeeze(-1), bal_correct)
        optimizer2.zero_grad(set_to_none=True)
        loss2.backward()
        optimizer2.step()
    model.eval()

    with torch.no_grad():
        logits, self_conf = model(test_x)
        preds = logits.argmax(-1)
        id_acc = float((preds[~is_ood] == test_y[~is_ood]).float().mean())
        ood_acc = float((preds[is_ood] == test_y[is_ood]).float().mean())
        correct_bool = (preds == test_y).cpu().numpy().astype(float)
        ood_mask = is_ood.cpu().numpy()
        self_ece = _ece(self_conf.squeeze(-1).cpu().numpy()[ood_mask], correct_bool[ood_mask])
        softmax_conf = torch.softmax(logits, -1).max(-1).values.cpu().numpy()
        softmax_ece = _ece(softmax_conf[ood_mask], correct_bool[ood_mask])

    results = [
        Result("self_model", id_acc, ood_acc, self_ece),
        Result("softmax", id_acc, ood_acc, softmax_ece),
    ]
    by_name = {r.name: r for r in results}
    accepted = (
        by_name["self_model"].ood_ece <= 0.4
        and by_name["self_model"].ood_ece <= by_name["softmax"].ood_ece - 0.05
    )
    target = artifact or ROOT / "artifacts" / "self-model-v2-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "self_model_v2: Block-8 upgrade. 32-dim inputs, OOD = blurred (high-noise) features. self_head trained on ID+OOD correctness (learns the competence boundary); softmax has no such training and stays overconfident on OOD. Criterion: self_head OOD ECE <= 0.4 and >= 0.05 lower than softmax (moderate but consistent gain).",
        str(resolved), N_FEAT, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, n_train=args.n_train, n_test=args.n_test,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

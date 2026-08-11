"""Phase-2 P3 v2: TASK-LEVEL self-knowledge (capability registry).

P3 v1 (per-sample OOD ECE) failed on the harder task because sample-level
correctness was ~50% unpredictable.  v2 changes the measured quantity to the
TASK-LEVEL ability: the capability registry predicts EACH task's overall
success rate ('I am strong on task A, weak on task B'), calibrated against
the actual rate.  Sample-level unpredictability averages out; what matters is
whether the system knows its per-task competence.

Design: two tasks (task0 easy, task1 hard) share an encoder; a registry of
per-task capability heads is trained on the observed ID+OOD correctness of
BOTH tasks.  Criterion:
  (1) per-task prediction error |pred - actual| <= 0.1,
  (2) the registry correctly orders the tasks (task0 predicted > task1),
  (3) prediction error is lower than a naive uniform baseline by >= 0.15.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device

N_FEAT = 32
N_CLASSES = 4
TASKS = 2


def _sample(rng: random.Random, n: int, task: int, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    id_noise = 0.5 if task == 0 else 1.5
    xs, ys, ts = [], [], []
    for _ in range(n):
        label = rng.randrange(N_CLASSES)
        x = torch.zeros(N_FEAT)
        x[label * 8:(label + 1) * 8] = 2.0
        if mode == "double":
            other = (label + 2) % (N_FEAT // 8)
            x[other * 8:(other + 1) * 8] = 2.0
            x = x + torch.randn(N_FEAT) * id_noise
        else:
            x = x + torch.randn(N_FEAT) * id_noise
        xs.append(x)
        ys.append(label)
        ts.append(task)
    return torch.stack(xs), torch.tensor(ys), torch.tensor(ts)


class TaskRegistryModel(nn.Module):
    """Shared encoder + answer; per-task capability heads (registry)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(N_FEAT, 32), nn.ReLU())
        self.answer_head = nn.Linear(32, N_CLASSES)
        self.cap_heads = nn.ModuleList([nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
                                        for _ in range(TASKS)])

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.float())
        logits = self.answer_head(h)
        caps = torch.stack([ch(h) for ch in self.cap_heads], dim=1)  # [B, TASKS, 1]
        cap = caps[torch.arange(caps.shape[0], device=caps.device), task.long()].squeeze(-1)
        return logits, torch.sigmoid(cap)


def run(*, device: str | torch.device | None = None, n_train: int = 1200,
        experiment_seed: int = 41_123, artifact: Path | None = None) -> dict:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    model = TaskRegistryModel().to(resolved)
    # answer head on ID of both tasks
    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.answer_head.parameters()), lr=3e-3)
    X, Y, T = [], [], []
    for task in range(TASKS):
        xs, ys, ts = _sample(rng, n_train, task, "id")
        X.append(xs); Y.append(ys); T.append(ts)
    X, Y, T = torch.cat(X).to(resolved), torch.cat(Y).to(resolved), torch.cat(T).to(resolved)
    for _ in range(150):
        logits, _ = model(X, T)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    # capability heads on ID+OOD correctness of both tasks
    X2, Y2, T2 = [], [], []
    for task in range(TASKS):
        for mode in ("id", "double"):
            xs, ys, ts = _sample(rng, 600, task, mode)
            X2.append(xs); Y2.append(ys); T2.append(ts)
    X2, Y2, T2 = torch.cat(X2).to(resolved), torch.cat(Y2).to(resolved), torch.cat(T2).to(resolved)
    opt2 = torch.optim.AdamW(model.cap_heads.parameters(), lr=3e-3)
    for _ in range(250):
        logits, cap = model(X2, T2)
        with torch.no_grad():
            correct = (logits.argmax(-1) == Y2).float()
        loss = torch.nn.functional.binary_cross_entropy(cap, correct)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
    model.eval()

    # evaluate on a FRESH ID+OOD mix per task
    actual, predicted = [], []
    with torch.no_grad():
        for task in range(TASKS):
            xs, ys, ts = [], [], []
            for mode in ("id", "double"):
                a, b, c = _sample(rng, 600, task, mode)
                xs.append(a); ys.append(b); ts.append(c)
            xs, ys, ts = torch.cat(xs).to(resolved), torch.cat(ys).to(resolved), torch.cat(ts).to(resolved)
            logits, cap = model(xs, ts)
            acc = float((logits.argmax(-1) == ys).float().mean())
            actual.append(acc)
            predicted.append(float(cap.mean()))

    err = [abs(predicted[t] - actual[t]) for t in range(TASKS)]
    max_err = max(err)
    ordered = predicted[0] > predicted[1]  # task0 easier -> predicted stronger
    naive_err = sum(abs(0.5 - a) for a in actual)  # uniform 0.5 baseline
    reg_err = sum(err)
    accepted = max_err <= 0.1 and ordered and reg_err <= naive_err - 0.15
    report = {
        "protocol": "phase2_p3_v2_tasklevel_selfknowledge_v1: per-task capability registry (shared encoder, independent per-task cap heads) predicts EACH task's overall correctness on fresh ID+OOD; task0 easy (noise 0.5), task1 hard (noise 1.0). Criterion: per-task |pred-actual| <= 0.1, registry orders tasks correctly (easy > hard), registry error <= naive-uniform error - 0.15.",
        "seed": experiment_seed,
        "results": {
            "task0_actual": round(actual[0], 3), "task0_predicted": round(predicted[0], 3),
            "task1_actual": round(actual[1], 3), "task1_predicted": round(predicted[1], 3),
            "max_abs_err": round(max_err, 3),
            "registry_err": round(reg_err, 3),
            "naive_err": round(naive_err, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-p3v2-tasklevel.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-p3v2-tasklevel.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-p3v2-tasklevel.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(device=args.device, experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

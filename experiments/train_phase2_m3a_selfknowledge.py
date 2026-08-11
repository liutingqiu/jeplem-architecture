"""Phase-2 M3a / P3 minimal experiment: multi-task self-knowledge.

P3 criterion (JEPLM_PHASE2_WHITEPAPER.md): the system predicts its success
probability per task/regime, calibrated at the BEHAVIOR level, including on
unseen distributions, WITHOUT introspection (external measurable behavior).
Block-8 v2 verified single-task competence boundary; M3a extends to TWO tasks
with DIFFERENT OOD regimes -- the self-model must know 'I am strong on task A,
weak on task B' simultaneously (a capability-model registry).

Design (on Block-8 v2 mechanics): shared encoder + answer head (trained on ID
of BOTH tasks); self_head receives (hidden, input, task-id) and is trained on
ID+OOD correctness of BOTH tasks.  Task A OOD = double-block activation, task
B OOD = high-noise blur (different regimes).  Criterion: for EACH task the
self_head OOD ECE is >= 0.05 lower than the softmax OOD ECE (calibrated
boundary per task, not just overall).
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
TASKS = 2


def _sample(rng: random.Random, n: int, task: int, mode: str) -> tuple[Tensor, Tensor, Tensor]:
    """32-dim class-conditional; each class activates an 8-dim block.
    mode='id': clean block.  OOD task A ('double'): two blocks active (label
    and label+2) -- distinguishable, answered confidently wrong.  OOD task B
    ('blur'): clean block + high noise (x3) -- different failure regime."""
    xs, ys, ts = [], [], []
    id_noise = 0.5 if task == 0 else 1.0  # task 1 harder but learnable
    for _ in range(n):
        label = rng.randrange(N_CLASSES)
        x = torch.zeros(N_FEAT)
        x[label * 8:(label + 1) * 8] = 2.0
        if mode == "double":
            other = (label + 2) % (N_FEAT // 8)
            x[other * 8:(other + 1) * 8] = 2.0
            x = x + torch.randn(N_FEAT) * id_noise
        elif mode == "blur":
            # task B OOD regime: clean block + HIGH noise (x3) -- different
            # failure mode than double-block (was claimed in the docstring but
            # never implemented; task B silently used 'double')
            x = x + torch.randn(N_FEAT) * (id_noise * 3.0)
        else:  # id
            x = x + torch.randn(N_FEAT) * id_noise
        xs.append(x)
        ys.append(label)
        ts.append(task)
    return torch.stack(xs), torch.tensor(ys), torch.tensor(ts)


class MultiTaskModel(nn.Module):
    """Capability-model REGISTRY: one independent self_head PER task (decoupled,
    the whitepaper P3 design), shared encoder + answer head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(N_FEAT, 32), nn.ReLU())
        self.answer_head = nn.Linear(32, N_CLASSES)
        self.self_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(32 + N_FEAT, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in range(TASKS)
        ])

    def forward(self, x: Tensor, task: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x.float())
        logits = self.answer_head(h)
        confs = [torch.sigmoid(sh(torch.cat((h, x.float()), dim=-1))) for sh in self.self_heads]
        confs_st = torch.stack(confs, dim=1)  # [B, TASKS, 1]
        conf = confs_st[torch.arange(confs_st.shape[0], device=confs_st.device), task.long()]  # [B, 1]
        return logits, conf


def _ece(probs: Tensor, correct: Tensor, bins: int = 10) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    confs = probs.max(-1).values if probs.dim() == 2 else probs
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs >= lo) & (confs < hi)
        if bool(m.any()):
            ece += (m.float().sum() / len(confs)) * abs(confs[m].mean() - correct[m].float().mean())
    return float(ece)


def run(*, device: str | torch.device | None = None, n_train: int = 1200,
        experiment_seed: int = 41_123, artifact: Path | None = None) -> dict:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    model = MultiTaskModel().to(resolved)
    # phase 1: answer head on ID of BOTH tasks
    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.answer_head.parameters()), lr=3e-3)
    xs, ys, ts = _sample(rng, n_train, 0, "id")
    xs2, ys2, ts2 = _sample(rng, n_train, 1, "id")
    X, Y, T = torch.cat((xs, xs2)), torch.cat((ys, ys2)), torch.cat((ts, ts2))
    for _ in range(150):
        logits, _ = model(X.to(resolved), T.to(resolved))
        loss = torch.nn.functional.cross_entropy(logits, Y.to(resolved))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    # phase 2: self_head on ID+OOD correctness of BOTH tasks
    idA, yA, tA = _sample(rng, 600, 0, "id")
    oodA, yoA, toA = _sample(rng, 600, 0, "double")
    idB, yB, tB = _sample(rng, 600, 1, "id")
    oodB, yoB, toB = _sample(rng, 600, 1, "blur")
    X2 = torch.cat((idA, oodA, idB, oodB)).to(resolved)
    T2 = torch.cat((tA, toA, tB, toB)).to(resolved)
    Y2 = torch.cat((yA, yoA, yB, yoB)).to(resolved)
    opt2 = torch.optim.AdamW(model.self_heads.parameters(), lr=3e-3)
    for _ in range(250):
        logits, conf = model(X2, T2)
        with torch.no_grad():
            correct = (logits.argmax(-1) == Y2).float()
        loss = torch.nn.functional.binary_cross_entropy(conf.squeeze(-1), correct)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
    model.eval()

    # evaluation per task: OOD ECE of self_head vs softmax
    result = {}
    for task, ood_mode in ((0, "double"), (1, "blur")):
        xs_o, yo, ts_o = _sample(rng, 600, task, ood_mode)
        logits, conf = model(xs_o.to(resolved), ts_o.to(resolved))
        sm_probs = torch.softmax(logits, -1)
        correct = (logits.argmax(-1) == yo.to(resolved)).float()
        result[f"task{task}_self_ood_ece"] = round(_ece(conf.squeeze(-1), correct), 3)
        result[f"task{task}_softmax_ood_ece"] = round(_ece(sm_probs, correct), 3)

    accepted = all(
        result[f"task{t}_softmax_ood_ece"] - result[f"task{t}_self_ood_ece"] >= 0.05
        for t in range(TASKS)
    )
    report = {
        "protocol": "phase2_m3a_p3_multitask_selfknowledge_v1: shared encoder+answer (ID of both tasks); self_head receives (h, x, task-id), trained on ID+OOD correctness of BOTH tasks (A OOD=double-block, B OOD=blur). Criterion: per-task OOD ECE(self_head) >= 0.05 lower than softmax OOD ECE -- calibrated boundary PER task, not just overall.",
        "seed": experiment_seed,
        "results": result,
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-m3a-selfknowledge.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-m3a-selfknowledge.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-m3a-selfknowledge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(device=args.device, experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

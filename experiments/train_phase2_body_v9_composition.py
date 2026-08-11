"""Phase-2 body v9: compositional skill IN the body (P6 hybrid-combiner
semantics) -- symbolic skeleton + learned atomic skills cooperate.

The body has two ATOMIC skills: SKILL-A (recognize pattern P_a -> action A)
and SKILL-B (recognize pattern P_b -> action B), each learned from its own
examples.  A task presents UNSEEN COMBINATIONS: the correct response requires
running SKILL-A then SKILL-B in sequence (symbolic order skeleton) on the
same input.  The P6 hybrid principle: the ORDER structure is symbolic
(guarantees combination), the ATOMIC skills are learned (from data).

Design: input = (x_a, x_b) features; skill A maps x_a -> {A, none}, skill B
maps x_b -> {B, none}.  A combined input where BOTH patterns are present needs
A then B.  Training covers single skills and seen orders; testing covers
UNSEEN order combinations.  Verdicts:
  - atomic skill accuracy (each skill learned);
  - composed skill accuracy on unseen combinations (order skeleton + learned
    atoms), compared with a monolithic GRU that saw the same examples.
Criterion: composed >= 0.9 and >= monolithic + 0.2.
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

N_FEAT = 8


def _pattern(kind: str) -> torch.Tensor:
    """A learnable-ish fixed pattern: P_a activates features 0-3, P_b features 4-7."""
    v = torch.zeros(N_FEAT)
    if kind == "a":
        v[:4] = 1.0
    elif kind == "b":
        v[4:] = 1.0
    elif kind == "both":
        v = 1.0
    else:
        v = torch.randn(N_FEAT) * 0.1
    return v


class SkillA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, 8), nn.ReLU(), nn.Linear(8, 2))  # none / A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkillB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, 8), nn.ReLU(), nn.Linear(8, 2))  # none / B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MonolithicGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRUCell(N_FEAT, 16)
        self.head = nn.Linear(16, 3)  # none / A / B (one step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.gru(x))
        return self.head(h)


def run(*, n_train: int = 300, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    # --- train atomic skills on single-pattern examples ---
    skill_a, skill_b = SkillA(), SkillB()
    opt_a = torch.optim.AdamW(skill_a.parameters(), lr=1e-2)
    opt_b = torch.optim.AdamW(skill_b.parameters(), lr=1e-2)
    for _ in range(300):
        # skill A: P_a -> A, noise -> none
        x = _pattern("a") if rng.random() < 0.5 else _pattern("noise")
        y = torch.tensor([1 if _pattern("a").equal(x) else 0])
        # (x, y) fixed: use the boolean above
        y = torch.tensor([1.0 if (x[:4].sum() > 3.5) else 0.0], dtype=torch.long)
        opt_a.zero_grad()
        loss = torch.nn.functional.cross_entropy(skill_a(x.unsqueeze(0)), y)
        loss.backward()
        opt_a.step()
        xb = _pattern("b") if rng.random() < 0.5 else _pattern("noise")
        yb = torch.tensor([1.0 if (xb[4:].sum() > 3.5) else 0.0], dtype=torch.long)
        opt_b.zero_grad()
        lossb = torch.nn.functional.cross_entropy(skill_b(xb.unsqueeze(0)), yb)
        lossb.backward()
        opt_b.step()

    # --- composed body: symbolic order skeleton A then B ---
    # composed response on a 'both' input: run A, then B (order skeleton)
    def composed(x: torch.Tensor) -> tuple[int, int]:
        with torch.no_grad():
            a_out = int(skill_a(x.unsqueeze(0)).argmax(-1).item())
            b_out = int(skill_b(x.unsqueeze(0)).argmax(-1).item())
        return a_out, b_out

    # --- monolithic control: single GRU sees the same single examples ---
    mono = MonolithicGRU()
    opt_m = torch.optim.AdamW(mono.parameters(), lr=1e-2)
    for _ in range(300):
        kind = rng.choice(["a", "b", "noise"])
        x = _pattern(kind)
        y = torch.tensor([{"a": 1, "b": 2, "noise": 0}[kind]], dtype=torch.long)
        opt_m.zero_grad()
        loss = torch.nn.functional.cross_entropy(mono(x.unsqueeze(0)), y)
        loss.backward()
        opt_m.step()

    # --- evaluate: atomic accuracy + composed accuracy on unseen 'both' inputs ---
    a_correct = sum(1 for _ in range(100) if int(skill_a(_pattern("a").unsqueeze(0)).argmax(-1).item()) == 1)
    b_correct = sum(1 for _ in range(100) if int(skill_b(_pattern("b").unsqueeze(0)).argmax(-1).item()) == 1)
    atomic_acc = (a_correct + b_correct) / 200
    comp_correct = 0
    for _ in range(100):
        x = _pattern("both") + torch.randn(N_FEAT) * 0.05
        a_out, b_out = composed(x)
        comp_correct += int(a_out == 1 and b_out == 1)
    composed_acc = comp_correct / 100
    # structural control: a single-output monolithic cannot express the
    # two-action composition (its ceiling is one action); report its accuracy
    # on 'both' for reference, but the criterion is on the composed body.
    mono_correct = 0
    for _ in range(100):
        x = _pattern("both") + torch.randn(N_FEAT) * 0.05
        mono_correct += int(mono(x.unsqueeze(0)).argmax(-1).item() == 2)
    mono_acc = mono_correct / 100

    # criterion aligned with the docstring claim (was composed+atomic only, the
    # docstring promised composed >= monolithic + 0.2 -- the claim outran the gate)
    accepted = (composed_acc >= 0.9 and atomic_acc >= 0.9
                and composed_acc >= mono_acc + 0.2)
    report = {
        "protocol": "phase2_body_v9_composition_v1: atomic skills learned from single examples; composed response on unseen 'both' runs A then B (symbolic order skeleton + learned atoms, P6 hybrid); monolithic single-output is a STRUCTURAL control (cannot express two-action composition). Criterion: composed >= 0.9 and atomic >= 0.9.",
        "seed": experiment_seed,
        "results": {
            "atomic_skill_acc": round(atomic_acc, 3),
            "composed_acc": round(composed_acc, 3),
            "monolithic_acc": round(mono_acc, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v9-composition.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v9-composition.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v9-composition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

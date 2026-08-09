"""Block 6 minimal experiment: compositional generalization.

Whitepaper criterion (Block 6): compose known concepts/rules into unseen
combinations.  Two attribute axes (color, shape), each with its own rule
(color -> {A,B}, shape -> {C,D}).  Training shows single-attribute objects
only; testing shows two-attribute objects (all 4 unseen color x shape
combinations).  A modular model (independent color/shape heads) should compose;
a monolithic MLP trained on single-attribute inputs sees two-attribute inputs
out of distribution and should fail unless it learned the composition.

Criterion: modular composes to >= 0.9 heldout combination accuracy, beating the
monolithic MLP by >= 0.2.
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

COLORS = 2  # red, blue -> A, B
SHAPES = 2  # circle, square -> C, D
IN_DIM = COLORS + SHAPES
OUT_DIM = COLORS + SHAPES


def color_action(color: int) -> int:
    return color  # red->0, blue->1 (position 0/1 in output)

def shape_action(shape: int) -> int:
    return COLORS + shape  # circle->2, square->3 (position 2/3 in output)


class ModularComposer(nn.Module):
    """Independent color/shape heads; composition is structural."""

    def __init__(self) -> None:
        super().__init__()
        self.color_net = nn.Sequential(nn.Linear(COLORS, 8), nn.ReLU(), nn.Linear(8, COLORS))
        self.shape_net = nn.Sequential(nn.Linear(SHAPES, 8), nn.ReLU(), nn.Linear(8, SHAPES))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.color_net(x[:, :COLORS]), self.shape_net(x[:, COLORS:])


class MonolithicMLP(nn.Module):
    """Shared encoder (mixes both attributes) + two heads: combination inputs
    are out of distribution because the mixed representation is not the simple
    superposition of the single-attribute training representations."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(IN_DIM, 16), nn.ReLU())
        self.color_head = nn.Linear(16, COLORS)
        self.shape_head = nn.Linear(16, SHAPES)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.color_head(h), self.shape_head(h)


def _split_targets(ys: Tensor) -> tuple[Tensor, Tensor]:
    """y: [B, OUT_DIM] one-hot -> (color_target [B,2], shape_target [B,2])."""
    return ys[:, :COLORS], ys[:, COLORS:]


def _single_attribute_examples(rng: random.Random, n: int) -> tuple[Tensor, Tensor]:
    """Train samples: only one attribute active (color XOR shape)."""
    xs, ys = [], []
    for _ in range(n):
        if rng.random() < 0.5:
            color = rng.randrange(COLORS)
            x = torch.zeros(IN_DIM)
            x[color] = 1.0
            y = torch.zeros(OUT_DIM)
            y[color_action(color)] = 1.0
        else:
            shape = rng.randrange(SHAPES)
            x = torch.zeros(IN_DIM)
            x[COLORS + shape] = 1.0
            y = torch.zeros(OUT_DIM)
            y[shape_action(shape)] = 1.0
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def _combination_examples() -> tuple[Tensor, Tensor]:
    """All color x shape combinations (unseen during training)."""
    xs, ys = [], []
    for color in range(COLORS):
        for shape in range(SHAPES):
            x = torch.zeros(IN_DIM)
            x[color] = 1.0
            x[COLORS + shape] = 1.0
            y = torch.zeros(OUT_DIM)
            y[color_action(color)] = 1.0
            y[shape_action(shape)] = 1.0
            xs.append(x)
            ys.append(y)
    return torch.stack(xs), torch.stack(ys)


@dataclass
class Result:
    name: str
    parameters: int
    train_accuracy: float
    combination_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    train_examples: int
    results: list[Result]
    accepted: bool
    artifact: str


def _train(model: nn.Module, xs: Tensor, ys: Tensor, device: torch.device,
           *, epochs: int = 300, lr: float = 1e-2) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    color_t, shape_t = _split_targets(ys)
    has_color = xs[:, :COLORS].sum(dim=-1) > 0.5  # [B]
    has_shape = xs[:, COLORS:].sum(dim=-1) > 0.5
    for _ in range(epochs):
        color_logits, shape_logits = model(xs.to(device))
        loss = torch.tensor(0.0, device=device)
        if bool(has_color.any()):
            loss = loss + criterion(color_logits[has_color], color_t.to(device)[has_color].argmax(dim=-1))
        if bool(has_shape.any()):
            loss = loss + criterion(shape_logits[has_shape], shape_t.to(device)[has_shape].argmax(dim=-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()


@torch.no_grad()
def _accuracy(model: nn.Module, xs: Tensor, ys: Tensor, device: torch.device) -> float:
    color_logits, shape_logits = model(xs.to(device))
    color_t, shape_t = _split_targets(ys)
    color_ok = color_logits.argmax(dim=-1) == color_t.to(device).argmax(dim=-1)
    shape_ok = shape_logits.argmax(dim=-1) == shape_t.to(device).argmax(dim=-1)
    # only score attributes that have label mass (single-attribute samples)
    has_color = color_t.to(device).sum(dim=-1) > 0.5
    has_shape = shape_t.to(device).sum(dim=-1) > 0.5
    per_sample = torch.ones(color_ok.shape[0], dtype=torch.bool, device=device)
    per_sample = per_sample & (~has_color | color_ok)
    per_sample = per_sample & (~has_shape | shape_ok)
    return float(per_sample.float().mean())


def run(*, device: str | torch.device | None = None, train_examples: int = 400,
        experiment_seed: int = 41_123, artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    xs, ys = _single_attribute_examples(rng, train_examples)
    comb_xs, comb_ys = _combination_examples()

    results = []
    for name, factory in (
        ("modular", ModularComposer),
        ("monolithic_mlp", MonolithicMLP),
    ):
        model = factory().to(resolved)
        _train(model, xs, ys, resolved)
        train_acc = _accuracy(model, xs, ys, resolved)
        comb_acc = _accuracy(model, comb_xs, comb_ys, resolved)
        results.append(Result(name, sum(p.numel() for p in model.parameters()),
                              train_acc, comb_acc))

    by_name = {r.name: r for r in results}
    accepted = (
        by_name["modular"].combination_accuracy >= 0.9
        and by_name["modular"].combination_accuracy >= by_name["monolithic_mlp"].combination_accuracy + 0.2
    )
    target = artifact or ROOT / "artifacts" / "compositional-generalization-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "compositional_generalization_v1: Block-6 minimal experiment. Two attribute axes with independent rules; training on single-attribute objects only; testing on all unseen color x shape combinations. modular (independent heads) should compose; monolithic MLP sees combinations out of distribution. Criterion: modular combination accuracy >= 0.9 and >= monolithic + 0.2.",
        str(resolved), train_examples, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-examples", type=int, default=400)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, train_examples=args.train_examples,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))

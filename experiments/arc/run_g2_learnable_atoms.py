"""G2: can the ATOM mapping be LEARNED (vs brute-force searched)?

Phase-3 whitepaper G2 -- the deepest open question: is compositional
generalization a learnable mechanism, or must it be explicit/symbolic?

P6-hybrid semantics: a symbolic skeleton (composition structure) plus a
LEARNED atom mapping (concrete parameters).  Here the skeleton is fixed
(recolor -> scale) and the atom mapper is a network that predicts the concrete
parameters (old, new, k) from a demo pair's grid features.  Trained on some
color combos, tested on UNSEEN combos (colors never seen in training).

Baselines: brute-force search (the current inducer) -- reaches 100% on any
expressible family -- and a color-memorization control (predicts the training
distribution).

Gate (whitepaper): learnable-atom generalization on unseen combos >= search
baseline x 0.9 AND > 0.6 (3 seeds).  Otherwise: "learnable composition"
falsified -- composition must be explicitly searched/symbolic.

The falsification has real value either way: if the atom mapper generalizes to
unseen COLORS, learnable composition is possible and the architecture can learn
to compose from experience; if it fails, composition must be explicit (the
project's long-standing P6 finding).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.arc.grid import colors, dims

N_COLORS = 10
K_CHOICES = [2, 3]
SEEDS = [41_123, 41_124, 41_125]


def demo_features(inp: list[list[int]], out: list[list[int]]) -> np.ndarray:
    """Compact, information-preserving features of one demo pair: color
    histograms of input/output + grid dims.  recolor shifts the histogram
    (old disappears, new appears); scale changes the dims."""
    h_in = np.bincount([c for row in inp for c in row], minlength=N_COLORS)[:N_COLORS].astype(float)
    h_out = np.bincount([c for row in out for c in row], minlength=N_COLORS)[:N_COLORS].astype(float)
    h_in /= h_in.sum() + 1e-9
    h_out /= h_out.sum() + 1e-9
    hi, wi = dims(inp)
    ho, wo = dims(out)
    return np.concatenate([h_in, h_out, [hi, wi, ho, wo]])


def make_demo(rng: random.Random, old: int, new: int, k: int) -> tuple[list[list[int]], list[list[int]]]:
    """Generate one (input, output) demo for recolor(old->new) then scale(k)."""
    H, W = rng.randrange(2, 5), rng.randrange(2, 5)
    palette = [0, old]
    if rng.random() < 0.5:
        palette.append(rng.choice([c for c in range(10) if c not in (0, old)]))
    inp = [[rng.choice(palette) for _ in range(W)] for _ in range(H)]
    rec = [[new if c == old else c for c in row] for row in inp]
    out = [[c for c in row for _ in range(k)] for row in rec for _ in range(k)]
    return inp, out


class AtomMapper(nn.Module):
    """Predicts (old, new, k) from a demo pair's features."""

    def __init__(self, n_feat: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.old_head = nn.Linear(64, N_COLORS)
        self.new_head = nn.Linear(64, N_COLORS)
        self.k_head = nn.Linear(64, len(K_CHOICES))

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.old_head(h), self.new_head(h), self.k_head(h)


def run(*, experiment_seed: int, artifact: Path | None = None) -> dict:
    rng = random.Random(experiment_seed)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)

    # combo space: old in {1,2}, new in {5,6,7}, k in {2,3}
    combos = [(o, n, k) for o in (1, 2) for n in (5, 6, 7) for k in K_CHOICES]
    # train on old=1 (unseen old=2), and hold out one NEW color
    train_combos = [c for c in combos if c[0] == 1]
    seen_new = {5, 6}
    train_combos = [c for c in train_combos if c[1] in seen_new]
    unseen_combos = [c for c in combos if c[0] == 2 or c[1] not in seen_new]
    # also hold out k=3 from training? keep k both seen to isolate the color axis
    train_combos = [c for c in train_combos if c[2] in (2, 3)]

    # build dataset: one demo per (combo, seed) for training; test on unseen combos
    X, Y = [], []
    for _ in range(3):
        for (o, n, k) in train_combos:
            inp, out = make_demo(rng, o, n, k)
            X.append(demo_features(inp, out))
            Y.append((o, n, k))
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y_old = torch.tensor([y[0] for y in Y], dtype=torch.long)
    Y_new = torch.tensor([y[1] for y in Y], dtype=torch.long)
    Y_k = torch.tensor([K_CHOICES.index(y[2]) for y in Y], dtype=torch.long)

    model = AtomMapper()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(800):
        o, n, k = model(X)
        loss = (nn.functional.cross_entropy(o, Y_old) + nn.functional.cross_entropy(n, Y_new)
                + nn.functional.cross_entropy(k, Y_k))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()

    # evaluate on unseen combos: predict (old,new,k), apply, check output
    def predict_params(inp, out) -> tuple[int, int, int]:
        with torch.no_grad():
            fo, fn, fk = model(torch.tensor(demo_features(inp, out), dtype=torch.float32).unsqueeze(0))
        old = int(fo.argmax().item())
        new = int(fn.argmax().item())
        k = K_CHOICES[int(fk.argmax().item())]
        return old, new, k

    def apply_rule(x, old, new, k):
        rec = [[new if c == old else c for c in row] for row in x]
        return [[c for c in row for _ in range(k)] for row in rec for _ in range(k)]

    correct = 0
    param_correct = 0
    n = 0
    for _ in range(5):
        for (o, n2, k) in unseen_combos:
            inp, out = make_demo(rng, o, n2, k)
            po, pn, pk = predict_params(inp, out)
            pred_out = apply_rule(inp, po, pn, pk)
            correct += int(pred_out == out)
            param_correct += int((po, pn, pk) == (o, n2, k))
            n += 1
    learnable_acc = correct / n if n else 0.0
    param_acc = param_correct / n if n else 0.0

    # search baseline: the inducer on the SAME unseen tasks (should be 1.0 for
    # an expressible family) -- it finds the params by brute force
    search_correct = 0
    for _ in range(5):
        for (o, n2, k) in unseen_combos:
            inp, out = make_demo(rng, o, n2, k)
            # search over old/new/k
            found = False
            for po in range(10):
                for pn in range(10):
                    for pk in K_CHOICES:
                        if apply_rule(inp, po, pn, pk) == out:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            search_correct += int(found)
    search_acc = search_correct / n if n else 0.0

    accepted = learnable_acc >= search_acc * 0.9 and learnable_acc > 0.6
    report = {
        "protocol": "phase3_g2_learnable_atoms_v1: AtomMapper network learns recolor->scale parameters (old,new,k) from demo-pair features; trained on old=1/new in {5,6}, tested on UNSEEN old=2 / new=7 combos. Baselines: brute-force search (1.0 on expressible), param-accuracy. Gate: learnable >= search x 0.9 AND > 0.6.",
        "seed": experiment_seed,
        "results": {
            "learnable_output_acc": round(learnable_acc, 3),
            "learnable_param_acc": round(param_acc, 3),
            "search_baseline": round(search_acc, 3),
            "n_test": n,
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "arc-g2-learnable-atoms.json"),
    }
    (artifact or ROOT / "artifacts" / "arc-g2-learnable-atoms.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "arc-g2-learnable-atoms.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

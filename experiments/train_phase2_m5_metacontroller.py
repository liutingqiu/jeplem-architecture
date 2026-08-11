"""Phase-2 M5 (v2): meta-controller needs the PARTS' INTERNAL STATE to route.

The v1 design (route by frame features) was trivial.  This version makes the
routing decision depend on the parts' internal state, so the meta-controller
must 'understand' the parts, not just read shallow frame features.

Task: content c appears TWICE in a stream.  First appearance = a NOISY frame
(the BeliefPart's evidence accumulator is correct -- it accumulates across
repeats).  Second appearance = a STABLE frame (the RecallPart's buffer was
just updated to c, so recall is correct).  The SAME content maps to different
correct parts depending on HISTORY (whether the buffer already holds c).

Meta-controller input variants:
  - WITH state: frame encoding + recall-buffer-state (buffer == c ?) +
    belief-state (evidence count for c) -> can disambiguate the two
    appearances.
  - WITHOUT state (control): frame encoding only -> the two appearances are
    indistinguishable, routing cannot exceed ~0.5.

Criterion: WITH-state routing >= 0.9 and beats WITHOUT-state by >= 0.2 --
routing must consult the parts' internal state (learned on explicit
interfaces, not frozen flowcharts).
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

N_CONTENTS = 10


class RecallPart:
    """Buffer of the last stable content; correct on 'stable' frames."""

    def __init__(self) -> None:
        self.buffer: int | None = None

    def update(self, content: int) -> None:
        self.buffer = content


class BeliefPart:
    """Evidence accumulator per content; correct on 'noisy' frames."""

    def __init__(self) -> None:
        self.evidence: dict[int, int] = {}

    def update(self, content: int) -> None:
        self.evidence[content] = self.evidence.get(content, 0) + 1

    def confident(self, content: int) -> float:
        return min(self.evidence.get(content, 0) / 3.0, 1.0)


class MetaController(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _one_hot(c: int, n: int) -> torch.Tensor:
    v = torch.zeros(n)
    v[c] = 1.0
    return v


def run(*, n_pairs: int = 800, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)
    device = resolve_device(None)

    feats_with, feats_without, labels = [], [], []
    recall, belief = RecallPart(), BeliefPart()
    for _ in range(n_pairs):
        c = rng.randrange(N_CONTENTS)
        # noisy frame first (belief correct), then stable frame (recall correct)
        belief.update(c)
        with_state = torch.cat((_one_hot(c, N_CONTENTS),
                                torch.tensor([1.0 if recall.buffer == c else 0.0]),
                                torch.tensor([belief.confident(c)])))
        without_state = _one_hot(c, N_CONTENTS)
        feats_with.append(with_state)
        feats_without.append(without_state)
        labels.append(1)  # belief part
        # stable frame: recall updates buffer, then recall is correct
        recall.update(c)
        with_state2 = torch.cat((_one_hot(c, N_CONTENTS),
                                 torch.tensor([1.0 if recall.buffer == c else 0.0]),
                                 torch.tensor([belief.confident(c)])))
        feats_with.append(with_state2)
        feats_without.append(_one_hot(c, N_CONTENTS))
        labels.append(0)  # recall part

    y = torch.tensor(labels, dtype=torch.long)
    results = {}
    for tag, feats in (("with_state", feats_with), ("without_state", feats_without)):
        X = torch.stack(feats)
        model = MetaController(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(400):
            logits = model(X.to(device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(X.to(device)).argmax(-1) == y.to(device)).float().mean())
        results[tag] = round(acc, 3)

    accepted = results["with_state"] >= 0.9 and results["with_state"] - results["without_state"] >= 0.2
    report = {
        "protocol": "phase2_m5_metacontroller_v2: routing the same content to different parts depending on the parts' INTERNAL STATE (recall buffer / belief evidence); meta-controller input WITH vs WITHOUT internal state (ablation). Criterion: with-state routing >= 0.9 and beats without-state by >= 0.2.",
        "seed": experiment_seed,
        "results": results,
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-m5-metacontroller.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-m5-metacontroller.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-m5-metacontroller.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))

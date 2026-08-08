"""Check whether sensor-derived Key/Ball labels are separable in one frame."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from experiments.audit_minigrid_delayed_object_identity import _corpus
from plwm.minigrid_features import visible_object_presence


@dataclass
class Report:
    protocol: str
    device: str
    training_examples: int
    updates: int
    heldout_accuracy: float
    accepted_sensor_label_contract: bool
    artifact: str


def run(*, device: str | torch.device | None = None, training_examples: int = 512, updates: int = 200, artifact: Path | None = None) -> Report:
    if training_examples < 2 or updates < 1:
        raise ValueError("training_examples and updates must be positive")
    resolved = resolve_device(device)
    torch.manual_seed(94_337); np.random.seed(94_337); random.seed(94_337)
    train_events, train_labels = _corpus(examples=training_examples, seed_start=7_000_000)
    heldout_events, heldout_labels = _corpus(examples=128, seed_start=8_000_000)
    # This is an auditable perception front end: position-invariant Key/Ball
    # presence derived from the current sensor image, not environment state.
    train_features = visible_object_presence(train_events[:, 0])
    heldout_features = visible_object_presence(heldout_events[:, 0])
    model = nn.Linear(train_features.shape[-1], 2).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(95_337)
    for _update in range(updates):
        indices = torch.randint(training_examples, (32,), generator=generator)
        logits = model(train_features[indices].to(resolved))
        loss = nn.functional.cross_entropy(logits, train_labels[indices].to(resolved))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    with torch.no_grad():
        accuracy = float((model(heldout_features.to(resolved)).argmax(dim=-1).cpu() == heldout_labels).float().mean())
    target = artifact or ROOT / "artifacts" / "minigrid-visible-object-separability-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_visible_object_separability_v2: classify currently visible Key/Ball using a position-invariant presence feature derived only from its current observation image; labels are extracted from that same image with no grid or hidden-state label access.",
        str(resolved), training_examples, updates, accuracy, accuracy >= 0.95, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-examples", type=int, default=512)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, training_examples=args.training_examples, updates=args.updates, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted_sensor_label_contract:
        raise SystemExit("current sensor image did not meet object-label separability gate")

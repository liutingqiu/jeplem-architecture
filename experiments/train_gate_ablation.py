"""Gate-source ablation: external-anchored vs self-referential write gates.

Verification experiment for the design principle distilled from Liu (2026)
"On the Limits of External Control in Self-Referential Computational Systems":
a memory write gate driven by the system's OWN state (self-referential, k -> 1)
should be unstable / fail to selectively store, while a gate anchored to the
current external observation should work.

Three gate sources on the same content-addressable slot architecture:
  input  : gate = sigmoid(linear(encoded_current))   (external-anchored, standard)
  state  : gate = sigmoid(linear(state))             (self-referential: the system
           decides what to write from its own belief, ignoring the current input)
  none   : unconditional FIFO write (no gating at all)

Measured on DecideAfterDelay delay=32 (where gating matters: empty gap frames
must not evict the identity frames).  The self-referential variant is expected to
collapse toward the no-gate baseline or worse.
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
from audit_minigrid_decide_after_delay_scan import Result, _accuracy, _train, make_corpus


class GateAblationEvidence(nn.Module):
    """Evidence-style slots with a swappable write-gate source."""

    def __init__(self, input_dim: int, num_classes: int, *, width: int = 64,
                 slots: int = 8, gate_mode: str = "input") -> None:
        super().__init__()
        if gate_mode not in ("input", "state", "none"):
            raise ValueError("gate_mode must be input/state/none")
        self.width, self.slots, self.gate_mode = width, slots, gate_mode
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.write_gate = nn.Linear(width, 1)
        self.self_gate = nn.Linear(width, 1)
        self.state = nn.GRUCell(width * 2, width)
        self.head = nn.Linear(width * 2, num_classes)

    def forward(self, events: Tensor) -> Tensor:
        batch = events.shape[0]
        state = torch.zeros(batch, self.width, device=events.device)
        keys = torch.zeros(batch, self.slots, self.width, device=events.device)
        values = torch.zeros_like(keys)
        for index in range(events.shape[1]):
            encoded = self.encoder(events[:, index])
            weights = torch.softmax(
                torch.einsum("bd,bmd->bm", self.query(encoded), keys) / self.width**0.5, dim=-1,
            )
            recalled = torch.einsum("bm,bmd->bd", weights, values)
            state = self.state(torch.cat((encoded, recalled), dim=-1), state)
            if self.gate_mode == "input":
                gate = torch.sigmoid(self.write_gate(encoded)).view(batch, 1, 1)
            elif self.gate_mode == "state":
                gate = torch.sigmoid(self.self_gate(state)).view(batch, 1, 1)
            else:  # none: unconditional FIFO
                gate = torch.ones(batch, 1, 1, device=events.device)
            candidate_keys = torch.cat((keys[:, 1:], self.key(encoded).unsqueeze(1)), dim=1)
            candidate_values = torch.cat((values[:, 1:], self.value(encoded).unsqueeze(1)), dim=1)
            keys = keys * (1.0 - gate) + candidate_keys * gate
            values = values * (1.0 - gate) + candidate_values * gate
        final_weights = torch.softmax(
            torch.einsum("bd,bmd->bm", self.query(state), keys) / self.width**0.5, dim=-1,
        )
        recalled = torch.einsum("bm,bmd->bd", final_weights, values)
        return self.head(torch.cat((state, recalled), dim=-1))


@dataclass
class Report:
    protocol: str
    device: str
    delay: int
    gap: int
    identity_count: int
    training_examples: int
    updates: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(
    *, device: str | torch.device | None = None, training_examples: int = 1024,
    updates: int = 400, batch_size: int = 64, delay: int = 32, gap: int = 3,
    identity_count: int = 2, experiment_seed: int = 93_337,
    artifact: Path | None = None,
) -> Report:
    if training_examples < 2 or updates < 1 or batch_size < 1:
        raise ValueError("training_examples, updates, and batch_size must be positive")
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    input_dim = 2 * identity_count
    train_events, train_labels = make_corpus(
        examples=training_examples, seed_start=50_000, delay=delay, gap=gap, identity_count=identity_count,
    )
    heldout_events, heldout_labels = make_corpus(
        examples=256, seed_start=6_000_000, delay=delay, gap=gap, identity_count=identity_count,
    )
    factories = (
        ("gate_input", lambda: GateAblationEvidence(input_dim, identity_count, gate_mode="input")),
        ("gate_state", lambda: GateAblationEvidence(input_dim, identity_count, gate_mode="state")),
        ("gate_none", lambda: GateAblationEvidence(input_dim, identity_count, gate_mode="none")),
    )
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train(model, train_events, train_labels, updates=updates, batch_size=batch_size, device=resolved)
        original = _accuracy(model, heldout_events, heldout_labels, resolved)
        swapped = _accuracy(model, heldout_events, (heldout_labels + 1) % identity_count, resolved)
        results.append(Result(
            name, sum(parameter.numel() for parameter in model.parameters()),
            original, swapped, original - swapped,
        ))
    by_name = {result.name: result for result in results}
    accepted = by_name["gate_input"].heldout_accuracy >= by_name["gate_none"].heldout_accuracy + 0.1
    target = artifact or ROOT / "artifacts" / "gate-ablation-d32-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "gate_ablation_v1: content-addressable slots with swappable write-gate source; input=gate from current observation (external-anchored), state=gate from own recurrent state (self-referential), none=unconditional FIFO; delay=32 where empty gap frames must not evict identity frames; verifies whether a self-referential write gate collapses (Liu 2026 external-control framework) while an externally anchored gate works.",
        str(resolved), delay, gap, identity_count, training_examples, updates,
        results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-examples", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--delay", type=int, default=32)
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument("--identity-count", type=int, default=2)
    parser.add_argument("--experiment-seed", type=int, default=93_337)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, training_examples=args.training_examples,
        updates=args.updates, batch_size=args.batch_size, delay=args.delay,
        gap=args.gap, identity_count=args.identity_count,
        experiment_seed=args.experiment_seed, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))

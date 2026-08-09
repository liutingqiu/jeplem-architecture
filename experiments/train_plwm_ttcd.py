"""PLWM gated memory: surprise-gate vs TTCD-gate at long delay.

Tests whether the TTCD (long-short window prediction difference) write-gate
signal improves PLWM's own gated memory at delay=32, where empty gap frames
must not evict identity frames.  This is the JEPLM v2 write-gate upgrade:
gate_signal = surprise (single-step prediction error) -> TTCD (Princeton
2608.01672), which the standalone gate ablation showed is stabler.
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
from plwm.torch_model import TorchPLWM
from audit_minigrid_decide_after_delay_scan import Result, _accuracy, _train, make_corpus


class PLWMTTCDDecision(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, *, latent: int = 32,
                 memory_slots: int = 8, thought_steps: int = 1,
                 gated_memory: bool = True, ttcd_gate: bool = False) -> None:
        super().__init__()
        self.core = TorchPLWM(
            input_dim=input_dim, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=num_classes,
            gated_memory=gated_memory, ttcd_gate=ttcd_gate,
        )

    def forward(self, events: Tensor) -> Tensor:
        logits, _surprises, _state = self.core.forward(events)
        return logits


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
        ("plwm_ungated", lambda: PLWMTTCDDecision(input_dim, identity_count, gated_memory=False)),
        ("plwm_surprise", lambda: PLWMTTCDDecision(input_dim, identity_count, gated_memory=True, ttcd_gate=False)),
        ("plwm_ttcd", lambda: PLWMTTCDDecision(input_dim, identity_count, gated_memory=True, ttcd_gate=True)),
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
    accepted = by_name["plwm_ttcd"].heldout_accuracy >= by_name["plwm_surprise"].heldout_accuracy
    target = artifact or ROOT / "artifacts" / "plwm-ttcd-d32-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "plwm_ttcd_v1: PLWM gated memory write-gate signal ablation at delay=32; surprise = single-step latent prediction error (standard), ttcd = long-vs-short EMA window prediction difference (Princeton 2608.01672); same corpus, train loop, and swap probe as the scan.",
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

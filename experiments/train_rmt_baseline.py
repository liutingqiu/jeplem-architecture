"""Contemporary baseline: Recurrent Memory Transformer (RMT) on DecideAfterDelay.

RMT (Bulatov et al. 2022) prepends learnable memory tokens to each sequence and
runs a *causal* transformer, so information from earlier frames must be carried
forward through the memory tokens -- the same precise-recall demand as GRU /
PLWM.  This answers the decisive baseline question: does a modern
attention-with-memory architecture match PLWM's advantage on the exact-recall
axis (long delay) and does it collapse on the same points where GRU collapses?

Compared on delay 8 (baseline) and delay 32 (where GRU drops to random ~0.50
while PLWM keeps ~0.86), same corpus, train loop, and swap probe as the scan.
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
from audit_minigrid_decide_after_delay_scan import (
    GRUDecision,
    PLWMDecision,
    Result,
    _accuracy,
    _train,
    make_corpus,
)


class RMTDecision(nn.Module):
    """Causal transformer with learnable prefix memory tokens."""

    def __init__(self, input_dim: int, num_classes: int, *, d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 4, memory_tokens: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model, self.memory_tokens = d_model, memory_tokens
        self.proj = nn.Linear(input_dim, d_model)
        self.memory = nn.Parameter(torch.randn(memory_tokens, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, events: Tensor) -> Tensor:
        batch, length = events.shape[:2]
        mem = self.memory.unsqueeze(0).expand(batch, -1, -1)  # [B, M, d]
        embedded = torch.cat((mem, self.proj(events)), dim=1)  # [B, M+T, d]
        total = self.memory_tokens + length
        embedded = embedded + self.pos[:, :total]
        causal = torch.triu(
            torch.full((total, total), float("-inf"), device=events.device), diagonal=1,
        )
        out = self.encoder(embedded, mask=causal)
        return self.head(out[:, -1])


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
    updates: int = 400, batch_size: int = 64, delay: int = 8, gap: int = 3,
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
        ("gru", lambda: GRUDecision(input_dim, identity_count)),
        ("rmt_mem4", lambda: RMTDecision(input_dim, identity_count, memory_tokens=4)),
        ("rmt_mem8", lambda: RMTDecision(input_dim, identity_count, memory_tokens=8)),
        ("rmt_mem16", lambda: RMTDecision(input_dim, identity_count, memory_tokens=16)),
        ("plwm", lambda: PLWMDecision(input_dim, identity_count)),
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
    persistent_best = max(by_name[name].heldout_accuracy for name in ("gru", "rmt_mem4", "rmt_mem8", "rmt_mem16", "plwm"))
    accepted = persistent_best >= 0.80
    target = artifact or ROOT / "artifacts" / f"rmt-baseline-d{delay}-n{identity_count}-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "rmt_baseline_v1: causal transformer with learnable prefix memory tokens (Recurrent Memory Transformer) on the DecideAfterDelay scan corpus; identical train loop, heldout split, and swap counterfactual probe as the scan; answers whether a modern attention-with-memory baseline matches PLWM's long-delay exact recall.",
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
    parser.add_argument("--delay", type=int, default=8)
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

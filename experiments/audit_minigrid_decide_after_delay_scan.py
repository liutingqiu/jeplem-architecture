"""DecideAfterDelay difficulty scan: delay and identity-count boundaries.

Generalizes the two-identity task.  Each sequence carries two identity frames
(frame 0 and frame ``gap``) and a query frame at ``delay`` whose one-hot channel
selects which identity to output.  Input dimension = 2 * identity_count
(identity one-hot channels + query one-hot channels); output = identity_count
classes.

Scans the delay (8/16/32) and identity-count (2/3/4) axes to locate where the
compressed recurrent hidden state (GRU) starts failing at precise recall while
content-addressable slots (evidence / PLWM) keep succeeding.
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


class GRUDecision(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, width: int = 64) -> None:
        super().__init__()
        self.core = nn.GRU(input_dim, width, batch_first=True)
        self.head = nn.Linear(width, num_classes)

    def forward(self, events: Tensor) -> Tensor:
        outputs, _state = self.core(events)
        return self.head(outputs[:, -1])


class EvidenceDecision(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, *, width: int = 64, slots: int = 8, use_evidence: bool = True) -> None:
        super().__init__()
        self.width, self.slots, self.use_evidence = width, slots, use_evidence
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.write_gate = nn.Linear(width, 1)
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
            recalled = torch.einsum("bm,bmd->bd", weights, values) if self.use_evidence else torch.zeros_like(state)
            state = self.state(torch.cat((encoded, recalled), dim=-1), state)
            if self.use_evidence:
                gate = torch.sigmoid(self.write_gate(encoded)).view(batch, 1, 1)
                candidate_keys = torch.cat((keys[:, 1:], self.key(encoded).unsqueeze(1)), dim=1)
                candidate_values = torch.cat((values[:, 1:], self.value(encoded).unsqueeze(1)), dim=1)
                keys = keys * (1.0 - gate) + candidate_keys * gate
                values = values * (1.0 - gate) + candidate_values * gate
        final_weights = torch.softmax(
            torch.einsum("bd,bmd->bm", self.query(state), keys) / self.width**0.5, dim=-1,
        )
        recalled = torch.einsum("bm,bmd->bd", final_weights, values) if self.use_evidence else torch.zeros_like(state)
        return self.head(torch.cat((state, recalled), dim=-1))


class PLWMDecision(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, *, latent: int = 32, memory_slots: int = 8, thought_steps: int = 1) -> None:
        super().__init__()
        self.core = TorchPLWM(
            input_dim=input_dim, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=num_classes,
        )

    def forward(self, events: Tensor) -> Tensor:
        logits, _surprises, _state = self.core.forward(events)
        return logits


def make_corpus(*, examples: int, seed_start: int, delay: int = 8, gap: int = 3, identity_count: int = 2) -> tuple[Tensor, Tensor]:
    if delay < gap + 1 or gap < 1 or identity_count < 2:
        raise ValueError("delay must exceed gap, gap must be positive, identity_count must be at least two")
    input_dim = 2 * identity_count
    rng = random.Random(91_337 + seed_start)
    sequences, labels = [], []
    for _ in range(examples):
        first = rng.randrange(identity_count)
        second = rng.randrange(identity_count)
        query = rng.randrange(2)  # 0 = query first identity, 1 = query second
        events = torch.zeros(delay + 1, input_dim)
        events[0, first] = 1.0
        events[gap, second] = 1.0
        events[delay, identity_count + query] = 1.0
        label = first if query == 0 else second
        sequences.append(events)
        labels.append(label)
    return torch.stack(sequences), torch.tensor(labels, dtype=torch.long)


@dataclass
class Result:
    name: str
    parameters: int
    heldout_accuracy: float
    swapped_accuracy: float
    memory_evidence: float


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


def _train(model: nn.Module, events: Tensor, labels: Tensor, *, updates: int, batch_size: int, device: torch.device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(92_337)
    model.train()
    for _update in range(updates):
        indices = torch.randint(events.shape[0], (batch_size,), generator=generator)
        logits = model(events[indices].to(device))
        loss = nn.functional.cross_entropy(logits, labels[indices].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def _accuracy(model: nn.Module, events: Tensor, labels: Tensor, device: torch.device) -> float:
    model.eval()
    return float((model(events.to(device)).argmax(dim=-1).cpu() == labels).float().mean())


def run(
    *, device: str | torch.device | None = None, training_examples: int = 1024,
    updates: int = 400, batch_size: int = 64, delay: int = 8, gap: int = 3,
    identity_count: int = 2, experiment_seed: int = 93_337, artifact: Path | None = None,
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
        ("evidence_store", lambda: EvidenceDecision(input_dim, identity_count, use_evidence=True)),
        ("evidence_disabled", lambda: EvidenceDecision(input_dim, identity_count, use_evidence=False)),
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
    persistent_best = max(by_name[name].heldout_accuracy for name in ("gru", "evidence_store", "plwm"))
    accepted = persistent_best >= 0.80
    target = artifact or ROOT / "artifacts" / f"decide-after-delay-scan-d{delay}-n{identity_count}-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_decide_after_delay_scan_v1: two identity frames at positions 0 and gap, one-hot query at position delay selecting which identity to output; input channels = 2*identity_count; counterfactual swap probe flips identity->choice mapping; difficulty axes delay and identity_count; no teacher action, path oracle, seed answer, or reward signal.",
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

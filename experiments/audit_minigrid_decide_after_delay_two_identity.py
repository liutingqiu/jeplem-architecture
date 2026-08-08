"""DecideAfterDelay two-identity: precise identity extraction under query.

Each sequence carries two identity frames (frame 0 and frame 3, each Key or
Ball independently) and a query frame (frame 8) whose one-hot channel selects
which of the two identities the controller must output.  Inputs are four
channels: [key_presence, ball_presence, query_first, query_second].

A controller must retain BOTH identities at their exact positions and route
the query to the requested one.  This is the precise-recall regime where the
ARCHITECTURE.md claim says a single compressed recurrent hidden state blurs
exact recall while content-addressable episodic slots can retrieve the exact
slot.  The swap probe flips the identity->choice mapping at evaluation time.
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


INPUT_DIM = 4  # key, ball, query_first, query_second
SEQUENCE_LENGTH = 9  # identity at 0 and 3, query at 8


class LastFrameDecision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(INPUT_DIM, 2)

    def forward(self, events: Tensor) -> Tensor:
        return self.head(events[:, -1])


class GRUDecision(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.core = nn.GRU(INPUT_DIM, width, batch_first=True)
        self.head = nn.Linear(width, 2)

    def forward(self, events: Tensor) -> Tensor:
        outputs, _state = self.core(events)
        return self.head(outputs[:, -1])


class EvidenceDecision(nn.Module):
    """Content-addressed key/value slots with recurrent state (4-dim input)."""

    def __init__(self, *, width: int = 64, slots: int = 8, use_evidence: bool = True) -> None:
        super().__init__()
        self.width, self.slots, self.use_evidence = width, slots, use_evidence
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, width), nn.Tanh())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.write_gate = nn.Linear(width, 1)
        self.state = nn.GRUCell(width * 2, width)
        self.head = nn.Linear(width * 2, 2)

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
    def __init__(self, *, latent: int = 32, memory_slots: int = 8, thought_steps: int = 1) -> None:
        super().__init__()
        self.core = TorchPLWM(
            input_dim=INPUT_DIM, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=2,
        )

    def forward(self, events: Tensor) -> Tensor:
        logits, _surprises, _state = self.core.forward(events)
        return logits


def _corpus(*, examples: int, seed_start: int) -> tuple[Tensor, Tensor]:
    rng = random.Random(91_337 + seed_start)
    sequences, labels = [], []
    for _ in range(examples):
        first = rng.randrange(2)          # identity of frame 0: 0=key, 1=ball
        second = rng.randrange(2)         # identity of frame 3
        query = rng.randrange(2)          # 0 = query first, 1 = query second
        events = torch.zeros(SEQUENCE_LENGTH, INPUT_DIM)
        events[0, first] = 1.0            # identity frame 0 (key/ball channels 0/1)
        events[3, second] = 1.0           # identity frame 3 (key/ball channels 0/1)
        events[8, 2 + query] = 1.0        # query one-hot (channels 2/3)
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
    updates: int = 400, batch_size: int = 64, experiment_seed: int = 93_337,
    artifact: Path | None = None,
) -> Report:
    if training_examples < 2 or updates < 1 or batch_size < 1:
        raise ValueError("training_examples, updates, and batch_size must be positive")
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    train_events, train_labels = _corpus(examples=training_examples, seed_start=50_000)
    heldout_events, heldout_labels = _corpus(examples=256, seed_start=6_000_000)
    factories = (
        ("no_memory", lambda: LastFrameDecision()),
        ("gru", lambda: GRUDecision()),
        ("evidence_store", lambda: EvidenceDecision(use_evidence=True)),
        ("evidence_disabled", lambda: EvidenceDecision(use_evidence=False)),
        ("plwm", lambda: PLWMDecision()),
    )
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train(model, train_events, train_labels, updates=updates, batch_size=batch_size, device=resolved)
        original = _accuracy(model, heldout_events, heldout_labels, resolved)
        swapped = _accuracy(model, heldout_events, 1 - heldout_labels, resolved)
        results.append(Result(
            name, sum(parameter.numel() for parameter in model.parameters()),
            original, swapped, original - swapped,
        ))
    by_name = {result.name: result for result in results}
    persistent_best = max(by_name[name].heldout_accuracy for name in ("gru", "evidence_store", "plwm"))
    shortcut = max(by_name["no_memory"].heldout_accuracy, by_name["four_event_transformer"].heldout_accuracy) if "four_event_transformer" in by_name else by_name["no_memory"].heldout_accuracy
    accepted = (
        persistent_best >= 0.80
        and persistent_best >= shortcut + 0.20
    )
    target = artifact or ROOT / "artifacts" / "minigrid-decide-after-delay-two-identity-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_decide_after_delay_two_identity_v1: two identity frames (frame 0 and frame 3) plus a one-hot query at frame 8 selecting which identity to output; a controller must retain both identities at exact positions and route the query to the requested one (precise-recall regime); counterfactual swap probe flips the identity->choice mapping; no teacher action, path oracle, seed answer, or reward signal.",
        str(resolved), training_examples, updates, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-examples", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=93_337)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, training_examples=args.training_examples,
        updates=args.updates, batch_size=args.batch_size,
        experiment_seed=args.experiment_seed, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("two-identity decide-after-delay gate not met")

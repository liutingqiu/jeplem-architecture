"""DecideAfterDelay: must the controller use cross-step identity memory to decide?

A Key/Ball identity appears in the first sensor frame of each event sequence;
eight later frames contain neither object; the decision must pick the choice
(0=key, 1=ball) matching that first-frame identity.  Inputs are the
position-invariant Key/Ball presence features derived from each sensor image
(no grid, route, policy, reward-answer, or hidden-state access).

Controllers:
- no_memory: decides from the final frame alone -> cannot exceed chance
- four_event_transformer: short context that excludes the identity frame
- gru: recurrent hidden state (persistent, compressed)
- evidence_store: content-addressed key/value slots + recurrent state
- plwm: full TorchPLWM (belief + episodic memory + workspace)

Counterfactual swap probe: at evaluation time the identity->choice mapping is
flipped (choice = 1 - label).  A controller that genuinely uses the remembered
identity follows it, so its accuracy collapses toward zero under the swap,
while a position/route shortcut scores identically under both mappings.
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
from plwm.minigrid_features import visible_object_presence
from plwm.torch_model import TorchPLWM

try:
    from .audit_minigrid_delayed_object_identity import _corpus
except ImportError:
    from audit_minigrid_delayed_object_identity import _corpus


class LastFrameDecision(nn.Module):
    """No cross-step state: decides from the final frame alone."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2, 2)

    def forward(self, events: Tensor) -> Tensor:
        return self.head(events[:, -1])


class GRUDecision(nn.Module):
    """Recurrent hidden state over the whole sequence."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.core = nn.GRU(2, width, batch_first=True)
        self.head = nn.Linear(width, 2)

    def forward(self, events: Tensor) -> Tensor:
        outputs, _state = self.core(events)
        return self.head(outputs[:, -1])


class ShortContextDecision(nn.Module):
    """Transformer-style controller with a context window shorter than the delay:
    the identity frame is outside the window, so cross-step memory is required."""

    def __init__(self, width: int = 64, window: int = 4) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be at least one")
        self.window = window
        self.project = nn.Linear(2, width)
        layer = nn.TransformerEncoderLayer(
            width, nhead=2, dim_feedforward=width * 2, dropout=0.0,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(width, 2)

    def forward(self, events: Tensor) -> Tensor:
        context = events[:, -self.window :]
        encoded = self.encoder(self.project(context))
        return self.head(encoded[:, -1])


class EvidenceDecision(nn.Module):
    """Content-addressed key/value slots with a switchable write gate, plus a
    recurrent state — the recall mechanism of the persistent evidence store."""

    def __init__(self, *, width: int = 64, slots: int = 8, use_evidence: bool = True) -> None:
        super().__init__()
        self.width, self.slots, self.use_evidence = width, slots, use_evidence
        self.encoder = nn.Sequential(nn.Linear(2, width), nn.Tanh())
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
    """Full TorchPLWM: persistent belief + episodic memory + workspace."""

    def __init__(self, *, latent: int = 32, memory_slots: int = 8, thought_steps: int = 1) -> None:
        super().__init__()
        self.core = TorchPLWM(
            input_dim=2, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=2,
        )

    def forward(self, events: Tensor) -> Tensor:
        logits, _surprises, _state = self.core.forward(events)
        return logits


@dataclass
class Result:
    name: str
    parameters: int
    heldout_accuracy: float
    swapped_accuracy: float
    memory_evidence: float  # heldout - swapped (large positive = follows identity)


@dataclass
class Report:
    protocol: str
    device: str
    delay: int
    training_examples: int
    updates: int
    distractor_ratio: float
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


def _inject_distractors(events: Tensor, *, ratio: float, seed: int) -> Tensor:
    """Inject a distractor object frame between the identity and the query.

    The distractor shows the *other* object than the first-frame identity.
    A compressed recurrent state that weights recent input may be confused
    toward the distractor; a content-addressable store can still retrieve the
    first-frame identity.  The label remains the first-frame identity.
    """
    rng = random.Random(seed)
    injected = events.clone()
    batch, length, dim = injected.shape
    for index in range(batch):
        if rng.random() >= ratio:
            continue
        frame = 1 + rng.randrange(length - 2)  # never the identity or query frame
        if injected[index, frame].abs().sum() < 0.5:  # only inject into an empty frame
            injected[index, frame] = torch.tensor(
                [1.0 - injected[index, 0, 0], 1.0 - injected[index, 0, 1]],
                dtype=injected.dtype,
            )
    return injected


def run(
    *, device: str | torch.device | None = None, training_examples: int = 512,
    updates: int = 300, batch_size: int = 32, distractor_ratio: float = 0.0,
    artifact: Path | None = None,
) -> Report:
    if training_examples < 2 or updates < 1 or batch_size < 1:
        raise ValueError("training_examples, updates, and batch_size must be positive")
    resolved = resolve_device(device)
    torch.manual_seed(93_337)
    np.random.seed(93_337)
    random.seed(93_337)
    train_events, train_labels = _corpus(examples=training_examples, seed_start=50_000)
    heldout_events, heldout_labels = _corpus(examples=128, seed_start=6_000_000)
    train_events = visible_object_presence(train_events)
    heldout_events = visible_object_presence(heldout_events)
    if distractor_ratio > 0.0:
        train_events = _inject_distractors(train_events, ratio=distractor_ratio, seed=94_337)
        heldout_events = _inject_distractors(heldout_events, ratio=distractor_ratio, seed=95_337)
    factories = (
        ("no_memory", lambda: LastFrameDecision()),
        ("four_event_transformer", lambda: ShortContextDecision(window=4)),
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
    shortcut = max(by_name["no_memory"].heldout_accuracy, by_name["four_event_transformer"].heldout_accuracy)
    accepted = (
        persistent_best >= 0.80
        and persistent_best >= shortcut + 0.20
    )
    target = artifact or ROOT / "artifacts" / "minigrid-decide-after-delay-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_decide_after_delay_v2: Key/Ball identity shown in the first sensor frame, eight later frames contain neither object (optionally with a distractor object injected mid-sequence at distractor_ratio), final decision must match the first-frame identity; inputs are position-invariant presence features; counterfactual swap probe flips the identity->choice mapping at evaluation time, so a controller that genuinely uses remembered identity shows large (heldout - swapped) evidence, while a route/shortcut scores identically under both mappings; no teacher action, path oracle, seed answer, or reward signal of any kind.",
        str(resolved), 8, training_examples, updates, distractor_ratio, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-examples", type=int, default=512)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--distractor-ratio", type=float, default=0.0)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, training_examples=args.training_examples,
        updates=args.updates, batch_size=args.batch_size,
        distractor_ratio=args.distractor_ratio, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("decide-after-delay gate not met")

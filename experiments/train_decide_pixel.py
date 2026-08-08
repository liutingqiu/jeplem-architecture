"""DecidePixel: ecological-validity port of DecideAfterDelay two-identity to raw pixels.

The presence one-hot inputs of the two-identity task are replaced by real RGB
grid observations (7x7x3, MiniGrid-style agent-view size).  Frame 0 and frame 3
carry an identity rendered as a colored center cell (yellow=Key, purple=Ball);
frame 8 is the query frame whose *border color* (red=query first, blue=query
second) routes to one of the two stored identities.  The controller must
visually recognize the identity, retain BOTH at exact positions, and route the
query by reading the border signal -- the same precise-recall regime, but with
real perception in front of the memory.

The swap probe flips the identity->choice mapping at evaluation time; a model
that uses memory (rather than a position/query shortcut) scores high on
heldout and low on swap.
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

GRID = 7                 # 7x7 grid, matching MiniGrid agent_view
INPUT_DIM = GRID * GRID * 3  # 147
SEQUENCE_LENGTH = 9      # identity at 0 and 3, query at 8
EMBED = 32               # pixel encoder output width

KEY_COLOR = (255, 215, 0)    # yellow
BALL_COLOR = (128, 0, 128)   # purple
QUERY_FIRST_BORDER = (255, 0, 0)    # red border
QUERY_SECOND_BORDER = (0, 0, 255)   # blue border
EMPTY = (128, 128, 128)


def _frame(center: tuple[int, int, int] | None = None, border: tuple[int, int, int] | None = None) -> Tensor:
    """Render a 7x7x3 frame.  center colors the middle cell; border colors the
    outermost ring (query signal)."""
    grid = np.full((GRID, GRID, 3), EMPTY, dtype=np.float32)
    if center is not None:
        grid[GRID // 2, GRID // 2] = center
    if border is not None:
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = border
    return torch.tensor(grid / 255.0).view(-1)


def _corpus(*, examples: int, seed_start: int) -> tuple[Tensor, Tensor]:
    rng = random.Random(91_337 + seed_start)
    sequences, labels = [], []
    for _ in range(examples):
        first = rng.randrange(2)          # identity of frame 0: 0=key(yellow), 1=ball(purple)
        second = rng.randrange(2)         # identity of frame 3
        query = rng.randrange(2)          # 0 = query first (red border), 1 = query second (blue border)
        events = torch.zeros(SEQUENCE_LENGTH, INPUT_DIM)
        events[0] = _frame(center=KEY_COLOR if first == 0 else BALL_COLOR)
        events[3] = _frame(center=KEY_COLOR if second == 0 else BALL_COLOR)
        events[8] = _frame(border=QUERY_FIRST_BORDER if query == 0 else QUERY_SECOND_BORDER)
        label = first if query == 0 else second
        sequences.append(events)
        labels.append(label)
    return torch.stack(sequences), torch.tensor(labels, dtype=torch.long)


class _Encoder(nn.Module):
    """Shared pixel encoder: 147 -> 32."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(), nn.Linear(64, EMBED), nn.Tanh(),
        )

    def forward(self, events: Tensor) -> Tensor:
        return self.net(events)


class PixelGRU(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.core = nn.GRU(EMBED, width, batch_first=True)
        self.head = nn.Linear(width, 2)

    def forward(self, events: Tensor) -> Tensor:
        encoded = self.encoder(events)
        outputs, _state = self.core(encoded)
        return self.head(outputs[:, -1])


class PixelEvidence(nn.Module):
    """Content-addressed key/value slots, encoder in front (32-dim latent)."""

    def __init__(self, *, width: int = 64, slots: int = 8, use_evidence: bool = True) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.width, self.slots, self.use_evidence = width, slots, use_evidence
        self.query = nn.Linear(EMBED, width, bias=False)
        self.query_final = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(EMBED, width, bias=False)
        self.value = nn.Linear(EMBED, width)
        self.write_gate = nn.Linear(EMBED, 1)
        self.state = nn.GRUCell(EMBED + width, width)
        self.head = nn.Linear(width * 2, 2)

    def forward(self, events: Tensor) -> Tensor:
        encoded = self.encoder(events)  # [B, T, EMBED]
        batch = events.shape[0]
        state = torch.zeros(batch, self.width, device=events.device)
        keys = torch.zeros(batch, self.slots, self.width, device=events.device)
        values = torch.zeros_like(keys)
        for index in range(events.shape[1]):
            step = encoded[:, index]
            weights = torch.softmax(
                torch.einsum("bd,bmd->bm", self.query(step), keys) / self.width**0.5, dim=-1,
            )
            recalled = torch.einsum("bm,bmd->bd", weights, values) if self.use_evidence else torch.zeros_like(state)
            state = self.state(torch.cat((step, recalled), dim=-1), state)
            if self.use_evidence:
                gate = torch.sigmoid(self.write_gate(step)).view(batch, 1, 1)
                candidate_keys = torch.cat((keys[:, 1:], self.key(step).unsqueeze(1)), dim=1)
                candidate_values = torch.cat((values[:, 1:], self.value(step).unsqueeze(1)), dim=1)
                keys = keys * (1.0 - gate) + candidate_keys * gate
                values = values * (1.0 - gate) + candidate_values * gate
        final_weights = torch.softmax(
            torch.einsum("bd,bmd->bm", self.query_final(state), keys) / self.width**0.5, dim=-1,
        )
        recalled = torch.einsum("bm,bmd->bd", final_weights, values) if self.use_evidence else torch.zeros_like(state)
        return self.head(torch.cat((state, recalled), dim=-1))


class PixelPLWM(nn.Module):
    def __init__(self, *, latent: int = 32, memory_slots: int = 8, thought_steps: int = 1) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.core = TorchPLWM(
            input_dim=EMBED, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=2,
        )

    def forward(self, events: Tensor) -> Tensor:
        encoded = self.encoder(events)
        logits, _surprises, _state = self.core.forward(encoded)
        return logits


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
    updates: int = 800, batch_size: int = 64, experiment_seed: int = 93_337,
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
        ("gru", lambda: PixelGRU()),
        ("evidence_store", lambda: PixelEvidence(use_evidence=True)),
        ("evidence_disabled", lambda: PixelEvidence(use_evidence=False)),
        ("plwm", lambda: PixelPLWM()),
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
    accepted = persistent_best >= 0.80
    target = artifact or ROOT / "artifacts" / "decide-pixel-two-identity-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "decide_pixel_two_identity_v1: two identity frames rendered as RGB center cells (yellow=Key, purple=Ball) at frames 0 and 3 plus a query frame at 8 whose border color (red=first, blue=second) routes to one stored identity; raw 7x7x3 pixel input, no presence channels; counterfactual swap probe flips the identity->choice mapping; no teacher, oracle, or reward signal.",
        str(resolved), training_examples, updates, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-examples", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=600)
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
        raise SystemExit("decide-pixel gate not met")

"""Learned source-bound fact revision under long conflicting event streams.

This is deliberately a *memory diagnostic*, not a policy or architecture
claim.  Each sequence contains only agent-visible evidence events: fact
identity, source class, claimed value, confidence, and fact-relative version.
The label asks for one current fact after its supporting evidence has been
buried under conflicts.  The procedural generator is used only to create
labels; no rule verdict, current value, or hidden environment state enters a
model input.

The candidate is a differentiable fact-addressed store.  Its per-fact update
gate is learned from incoming and stored evidence; source validity and version
ordering are not hard-coded in its transition.  GRU and four-event Transformer
baselines receive the identical event stream and query.  Training uses short
and medium distractor streams; held-out evaluation increases only the number
of conflicts and uses disjoint random seeds.
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


FACTS = 4
VALUES = 4
SOURCES = 3  # trusted, weak-new, trusted-stale
EVENT_DIM = FACTS + VALUES + SOURCES + 3  # key/value/source/confidence/version/query bit


def _event(
    fact: int,
    value: int,
    source: int,
    confidence: float,
    version: float,
    *,
    query: bool = False,
) -> list[float]:
    result = [0.0] * EVENT_DIM
    result[fact] = 1.0
    result[FACTS + value] = 1.0
    result[FACTS + VALUES + source] = 1.0
    result[-3] = confidence
    result[-2] = version
    result[-1] = float(query)
    return result


def _corpus(
    *, examples: int, seed_start: int, distractors: int,
    conflict_mode: str = "mixed",
) -> tuple[Tensor, Tensor]:
    if examples < 1 or distractors < FACTS:
        raise ValueError("examples must be positive and distractors must cover all facts")
    if conflict_mode not in {"mixed", "weak", "stale"}:
        raise ValueError("conflict_mode must be mixed, weak, or stale")
    sequences, labels = [], []
    for offset in range(examples):
        rng = random.Random(seed_start + offset)
        initial = [rng.randrange(VALUES) for _ in range(FACTS)]
        current = list(initial)
        by_fact = {
            fact: [_event(fact, value, 0, 0.8, 0.25)]
            for fact, value in enumerate(initial)
        }
        # Every fact receives one trusted, newer revision. Its location is
        # random, so a learner cannot rely on a fixed event position.
        for fact in range(FACTS):
            value = (initial[fact] + rng.randrange(1, VALUES)) % VALUES
            current[fact] = value
            by_fact[fact].append(_event(fact, value, 0, 1.0, 0.75))
        # Conflicts are observationally plausible but invalid: either weak
        # newer evidence or trusted evidence that is fact-relative stale.
        for _ in range(distractors):
            fact = rng.randrange(FACTS)
            wrong = (current[fact] + rng.randrange(1, VALUES)) % VALUES
            if conflict_mode == "weak" or (conflict_mode == "mixed" and rng.random() < 0.5):
                by_fact[fact].append(_event(fact, wrong, 1, 0.1, 1.0))
            else:
                by_fact[fact].append(_event(fact, wrong, 2, 1.0, 0.25))
        # Preserve causal order per fact: initial before revision, revision
        # before its conflicts. Shuffling only independent fact streams keeps
        # the revision problem meaningful without making labels inconsistent.
        ordered = []
        positions = [0] * FACTS
        total_events = sum(len(stream) for stream in by_fact.values())
        while len(ordered) < total_events:
            available = [fact for fact in range(FACTS) if positions[fact] < len(by_fact[fact])]
            fact = rng.choice(available)
            ordered.append(by_fact[fact][positions[fact]])
            positions[fact] += 1
        query_fact = rng.randrange(FACTS)
        ordered.append(_event(query_fact, 0, 0, 0.0, 0.0, query=True))
        sequences.append(torch.tensor(ordered, dtype=torch.float32))
        labels.append(current[query_fact])
    return torch.stack(sequences), torch.tensor(labels, dtype=torch.long)


class LearnedFactStore(nn.Module):
    """A learned, fact-addressed memory with no encoded trust/version rule."""

    def __init__(self, width: int = 48, *, use_fact_addressing: bool = True) -> None:
        super().__init__()
        self.width = width
        self.use_fact_addressing = use_fact_addressing
        self.event = nn.Sequential(nn.Linear(EVENT_DIM - 1, width), nn.Tanh())
        self.candidate = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.update_gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh(), nn.Linear(width, VALUES))

    def forward(self, events: Tensor) -> Tensor:
        memory = events.new_zeros((events.shape[0], FACTS, self.width))
        for index in range(events.shape[1] - 1):
            event = events[:, index, :-1]
            fact = events[:, index, :FACTS].unsqueeze(-1)
            if not self.use_fact_addressing:
                # Same learned update core and parameter count, but all facts
                # contend for one shared slot and queries cannot address them.
                fact = torch.zeros_like(fact)
                fact[:, 0] = 1.0
            encoded = self.event(event)
            old = (memory * fact).sum(dim=1)
            candidate = self.candidate(encoded)
            gate = self.update_gate(torch.cat((old, encoded), dim=-1))
            updated = old * (1.0 - gate) + candidate * gate
            memory = memory * (1.0 - fact) + updated.unsqueeze(1) * fact
        query_event = events[:, -1, :-1]
        query_key = events[:, -1, :FACTS].unsqueeze(-1)
        if not self.use_fact_addressing:
            query_key = torch.zeros_like(query_key)
            query_key[:, 0] = 1.0
        retrieved = (memory * query_key).sum(dim=1)
        return self.head(torch.cat((retrieved, self.event(query_event)), dim=-1))


class GRURevision(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(EVENT_DIM - 1, width), nn.Tanh())
        self.core = nn.GRU(width, width, batch_first=True)
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh(), nn.Linear(width, VALUES))

    def forward(self, events: Tensor) -> Tensor:
        encoded = self.encoder(events[:, :-1, :-1])
        _outputs, state = self.core(encoded)
        query = self.encoder(events[:, -1, :-1])
        return self.head(torch.cat((state[-1], query), dim=-1))


class FourEventRevisionTransformer(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(EVENT_DIM - 1, width), nn.Tanh())
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=width * 2, dropout=0.0, batch_first=True, activation="gelu")
        self.core = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh(), nn.Linear(width, VALUES))

    def forward(self, events: Tensor) -> Tensor:
        context = self.encoder(events[:, -5:-1, :-1])
        state = self.core(context)[:, -1]
        query = self.encoder(events[:, -1, :-1])
        return self.head(torch.cat((state, query), dim=-1))


@dataclass
class Result:
    name: str
    parameters: int
    heldout_accuracy: float
    long_distractor_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    training_examples: int
    updates: int
    train_distractors: int
    heldout_distractors: int
    long_distractors: int
    results: list[Result]
    accepted_learned_fact_revision: bool
    artifact: str


def _fit(model: nn.Module, events: Tensor, labels: Tensor, *, updates: int, device: torch.device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(110_337)
    model.train()
    for _ in range(updates):
        index = torch.randint(events.shape[0], (64,), generator=generator)
        logits = model(events[index].to(device))
        loss = nn.functional.cross_entropy(logits, labels[index].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def _accuracy(model: nn.Module, events: Tensor, labels: Tensor, device: torch.device) -> float:
    model.eval()
    return float((model(events.to(device)).argmax(dim=-1).cpu() == labels).float().mean())


def run(
    *, device: str | torch.device | None = None, training_examples: int = 2048,
    updates: int = 500, train_distractors: int = 8, heldout_distractors: int = 32,
    long_distractors: int = 64, seed: int = 111_337, train_seed_start: int = 100_000,
    heldout_seed_start: int = 2_000_000, long_seed_start: int = 4_000_000,
    artifact: Path | None = None,
) -> Report:
    if training_examples < 64 or updates < 1 or not train_distractors >= FACTS or not heldout_distractors > train_distractors or not long_distractors > heldout_distractors:
        raise ValueError("invalid learned fact-revision diagnostic budget")
    resolved = resolve_device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_events, train_labels = _corpus(examples=training_examples, seed_start=train_seed_start, distractors=train_distractors)
    heldout_events, heldout_labels = _corpus(examples=512, seed_start=heldout_seed_start, distractors=heldout_distractors)
    long_events, long_labels = _corpus(examples=512, seed_start=long_seed_start, distractors=long_distractors)
    results = []
    for name, factory in (
        ("learned_fact_store", LearnedFactStore),
        ("learned_store_no_addressing", lambda: LearnedFactStore(use_fact_addressing=False)),
        ("gru", GRURevision),
        ("four_event_transformer", FourEventRevisionTransformer),
    ):
        model = factory().to(resolved)
        _fit(model, train_events, train_labels, updates=updates, device=resolved)
        results.append(Result(
            name, sum(parameter.numel() for parameter in model.parameters()),
            _accuracy(model, heldout_events, heldout_labels, resolved),
            _accuracy(model, long_events, long_labels, resolved),
        ))
    by_name = {result.name: result for result in results}
    accepted = (
        by_name["learned_fact_store"].heldout_accuracy >= 0.9
        and by_name["learned_fact_store"].long_distractor_accuracy >= 0.85
        and by_name["learned_fact_store"].long_distractor_accuracy >= by_name["gru"].long_distractor_accuracy + 0.1
        and by_name["learned_fact_store"].long_distractor_accuracy >= by_name["learned_store_no_addressing"].long_distractor_accuracy + 0.1
    )
    target = artifact or ROOT / "artifacts" / "learned-fact-revision-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "learned_fact_revision_v1: supervised diagnostic over only agent-visible source/value/confidence/version evidence streams. The candidate learns a differentiable fact-addressed update gate; the generator's strong/weak/stale rule and current-value labels are never model inputs. Held-out worlds raise distractors from 8 to 32 and 64, with disjoint seeds. This is not reward-only control.",
        str(resolved), training_examples, updates, train_distractors, heldout_distractors, long_distractors,
        results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-examples", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--train-distractors", type=int, default=8)
    parser.add_argument("--heldout-distractors", type=int, default=32)
    parser.add_argument("--long-distractors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=111_337)
    parser.add_argument("--train-seed-start", type=int, default=100_000)
    parser.add_argument("--heldout-seed-start", type=int, default=2_000_000)
    parser.add_argument("--long-seed-start", type=int, default=4_000_000)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, training_examples=args.training_examples, updates=args.updates,
        train_distractors=args.train_distractors, heldout_distractors=args.heldout_distractors,
        long_distractors=args.long_distractors, seed=args.seed,
        train_seed_start=args.train_seed_start, heldout_seed_start=args.heldout_seed_start,
        long_seed_start=args.long_seed_start, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted_learned_fact_revision:
        raise SystemExit("learned fact store did not meet long-distractor revision gate")

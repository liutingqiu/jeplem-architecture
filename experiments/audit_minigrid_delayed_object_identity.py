"""Sensor-only delayed Key/Ball identity recall from real MemoryS13 streams."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from torch import Tensor, nn
from minigrid.core.constants import OBJECT_TO_IDX

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from plwm.minigrid_features import visible_object_presence
from experiments.audit_minigrid_delayed_event_recall import EvidenceStoreRecall, FourEventTransformerRecall, GRURecall
from experiments.train_minigrid_memory_actor_critic import INPUT_DIM, encode_observation


ENVIRONMENT = "MiniGrid-MemoryS13-v0"
OBJECT_IDS = {OBJECT_TO_IDX["key"]: 0, OBJECT_TO_IDX["ball"]: 1}


class IdentityHead(nn.Module):
    def __init__(self, encoder: nn.Module, *, feature_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, events: Tensor) -> Tensor:
        return self.head(self.encoder(events))


def _visible_label(observation: dict) -> int | None:
    present = set(observation["image"][..., 0].flatten().tolist())
    labels = [label for object_id, label in OBJECT_IDS.items() if object_id in present]
    return labels[0] if len(labels) == 1 else None


def _corpus(*, examples: int, seed_start: int, sequence_length: int = 9) -> tuple[Tensor, Tensor]:
    rng = random.Random(91_337 + seed_start)
    sequences, labels = [], []
    seed = seed_start
    # Upper bound prevents an accidental infinite loop if an environment change
    # makes the sensor-only sampling predicate impossible.
    while len(sequences) < examples and seed < seed_start + examples * 40:
        environment = gym.make(ENVIRONMENT, max_steps=64)
        try:
            observation, _info = environment.reset(seed=seed)
            stream: list[tuple[Tensor, int | None]] = []
            for _step in range(64):
                label = _visible_label(observation)
                stream.append((encode_observation(observation), label))
                action = rng.randrange(3)
                observation, reward, terminated, truncated, _info = environment.step(action)
                if terminated or truncated:
                    break
            for start in range(len(stream) - sequence_length + 1):
                events = stream[start:start + sequence_length]
                label = events[0][1]
                if label is None or any(event_label is not None for _event, event_label in events[1:]):
                    continue
                sequences.append(torch.stack([event for event, _event_label in events]))
                labels.append(label)
                if len(sequences) == examples:
                    break
        finally:
            environment.close()
        seed += 1
    if len(sequences) < examples:
        raise RuntimeError("insufficient clean sensor-only delayed-object examples")
    return torch.stack(sequences), torch.tensor(labels, dtype=torch.long)


@dataclass
class Result:
    name: str
    parameters: int
    heldout_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    delay: int
    training_examples: int
    updates: int
    results: list[Result]
    accepted_object_identity_recall: bool
    artifact: str


def _train(model: nn.Module, events: Tensor, labels: Tensor, *, updates: int, batch_size: int, device: torch.device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(92_337)
    model.train()
    for _update in range(updates):
        indices = torch.randint(events.shape[0], (batch_size,), generator=generator)
        logits = model(events[indices].to(device))
        loss = nn.functional.cross_entropy(logits, labels[indices].to(device))
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()


@torch.no_grad()
def _accuracy(model: nn.Module, events: Tensor, labels: Tensor, device: torch.device) -> float:
    model.eval()
    return float((model(events.to(device)).argmax(dim=-1).cpu() == labels).float().mean())


def run(
    *, device: str | torch.device | None = None, training_examples: int = 512,
    updates: int = 300, batch_size: int = 32, artifact: Path | None = None,
) -> Report:
    if training_examples < 2 or updates < 1 or batch_size < 1:
        raise ValueError("training_examples, updates, and batch_size must be positive")
    resolved = resolve_device(device)
    torch.manual_seed(93_337); np.random.seed(93_337); random.seed(93_337)
    train_events, train_labels = _corpus(examples=training_examples, seed_start=50_000)
    heldout_events, heldout_labels = _corpus(examples=128, seed_start=6_000_000)
    # The memory module receives only the position-invariant presence feature
    # computed from each current sensor image.  This separates object
    # perception from delayed retention without exposing world internals.
    train_events = visible_object_presence(train_events)
    heldout_events = visible_object_presence(heldout_events)
    try:
        from train_rmt_baseline import RMTDecision
    except ImportError:
        from experiments.train_rmt_baseline import RMTDecision
    factories = (
        ("evidence_store", lambda: IdentityHead(EvidenceStoreRecall(input_dim=2), feature_dim=2)),
        ("evidence_disabled", lambda: IdentityHead(EvidenceStoreRecall(input_dim=2, use_evidence=False), feature_dim=2)),
        ("gru", lambda: IdentityHead(GRURecall(input_dim=2), feature_dim=2)),
        ("four_event_transformer", lambda: IdentityHead(FourEventTransformerRecall(input_dim=2), feature_dim=2)),
        ("rmt_mem4", lambda: RMTDecision(input_dim=2, num_classes=2, memory_tokens=4)),
        ("rmt_mem8", lambda: RMTDecision(input_dim=2, num_classes=2, memory_tokens=8)),
    )
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train(model, train_events, train_labels, updates=updates, batch_size=batch_size, device=resolved)
        results.append(Result(name, sum(parameter.numel() for parameter in model.parameters()), _accuracy(model, heldout_events, heldout_labels, resolved)))
    by_name = {result.name: result for result in results}
    accepted = by_name["evidence_store"].heldout_accuracy >= 0.8 and by_name["evidence_store"].heldout_accuracy >= by_name["evidence_disabled"].heldout_accuracy + 0.1
    target = artifact or ROOT / "artifacts" / "minigrid-delayed-object-identity-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_delayed_object_identity_v2: classify Key versus Ball seen in an initial agent sensor frame after eight later agent-observed frames contain neither object; inputs are position-invariant Key/Ball presence features derived from each sensor image, while labels come exclusively from the initial image with no grid, route, policy, reward-answer, or hidden-state access.",
        str(resolved), 8, training_examples, updates, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-examples", type=int, default=512)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, training_examples=args.training_examples, updates=args.updates, batch_size=args.batch_size, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted_object_identity_recall:
        raise SystemExit("persistent evidence store did not meet delayed-object identity gate")

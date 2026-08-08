"""Self-supervised delayed event recall on real MiniGrid observation streams.

This audit intentionally does not train a policy.  It samples bounded actions
from MemoryS13 and asks each encoder to reconstruct an observed event eight
steps in the past.  The target is a past sensor record, not a grid query,
route, reward answer, or teacher action.  It measures whether a switchable
persistent evidence store can retain and retrieve old observations before it
is reintroduced into reward-only control.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from experiments.train_minigrid_memory_actor_critic import INPUT_DIM, encode_observation


ENVIRONMENT = "MiniGrid-MemoryS13-v0"


class EvidenceStoreRecall(nn.Module):
    """Key/value evidence slots with a switchable learned write gate."""

    def __init__(self, *, input_dim: int = INPUT_DIM, width: int = 64, slots: int = 8, use_evidence: bool = True) -> None:
        super().__init__()
        self.width, self.slots, self.use_evidence = width, slots, use_evidence
        self.input_dim = input_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.write_gate = nn.Linear(width, 1)
        self.state = nn.GRUCell(width * 2, width)
        self.decoder = nn.Linear(width * 2, input_dim)

    def forward(self, events: Tensor) -> Tensor:
        batch = events.shape[0]
        state = torch.zeros(batch, self.width, device=events.device)
        keys = torch.zeros(batch, self.slots, self.width, device=events.device)
        values = torch.zeros_like(keys)
        for index in range(events.shape[1]):
            encoded = self.encoder(events[:, index])
            read_weights = torch.softmax(torch.einsum("bd,bmd->bm", self.query(encoded), keys) / self.width**0.5, dim=-1)
            recalled = torch.einsum("bm,bmd->bd", read_weights, values) if self.use_evidence else torch.zeros_like(state)
            state = self.state(torch.cat((encoded, recalled), dim=-1), state)
            if self.use_evidence:
                gate = torch.sigmoid(self.write_gate(encoded)).view(batch, 1, 1)
                candidate_keys = torch.cat((keys[:, 1:], self.key(encoded).unsqueeze(1)), dim=1)
                candidate_values = torch.cat((values[:, 1:], self.value(encoded).unsqueeze(1)), dim=1)
                keys = keys * (1.0 - gate) + candidate_keys * gate
                values = values * (1.0 - gate) + candidate_values * gate
        final_weights = torch.softmax(torch.einsum("bd,bmd->bm", self.query(state), keys) / self.width**0.5, dim=-1)
        recalled = torch.einsum("bm,bmd->bd", final_weights, values) if self.use_evidence else torch.zeros_like(state)
        return self.decoder(torch.cat((state, recalled), dim=-1))


class GRURecall(nn.Module):
    def __init__(self, *, input_dim: int = INPUT_DIM, width: int = 64) -> None:
        super().__init__()
        self.core = nn.GRU(input_dim, width, batch_first=True)
        self.decoder = nn.Linear(width, input_dim)

    def forward(self, events: Tensor) -> Tensor:
        outputs, _state = self.core(events)
        return self.decoder(outputs[:, -1])


class FourEventTransformerRecall(nn.Module):
    def __init__(self, *, input_dim: int = INPUT_DIM, width: int = 64, window: int = 4) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, width)
        self.position = nn.Parameter(torch.zeros(1, 32, width))
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=width * 2, dropout=0.0, batch_first=True, activation="gelu")
        self.core = nn.TransformerEncoder(layer, num_layers=1)
        self.decoder = nn.Linear(width, input_dim)
        self.window = window

    def forward(self, events: Tensor) -> Tensor:
        size = events.shape[1]
        positions = torch.arange(size, device=events.device)
        mask = positions.unsqueeze(0) > positions.unsqueeze(1)
        mask = mask | ((positions.unsqueeze(1) - positions.unsqueeze(0)) >= self.window)
        encoded = self.core(self.project(events) + self.position[:, :size], mask=mask)
        return self.decoder(encoded[:, -1])


def _corpus(*, episodes: int, sequence_length: int, seed_start: int) -> Tensor:
    rng = random.Random(71_533 + seed_start)
    rows = []
    for index in range(episodes):
        environment = gym.make(ENVIRONMENT, max_steps=64)
        try:
            observation, _info = environment.reset(seed=seed_start + index)
            events = [encode_observation(observation)]
            for _step in range(sequence_length - 1):
                action = rng.randrange(3)
                observation, reward, terminated, truncated, _info = environment.step(action)
                events.append(encode_observation(observation, previous_action=action, previous_reward=float(reward), observation_mode="causal_action_reward"))
                if terminated or truncated:
                    observation, _info = environment.reset(seed=seed_start + index + 10_000_000)
                    events.append(encode_observation(observation))
            rows.append(torch.stack(events[:sequence_length]))
        finally:
            environment.close()
    return torch.stack(rows)


@dataclass
class Result:
    name: str
    parameters: int
    heldout_mse: float


@dataclass
class Report:
    protocol: str
    device: str
    sequence_length: int
    recall_delay: int
    corpus_episodes: int
    updates: int
    results: list[Result]
    accepted_evidence_recall: bool
    artifact: str


def _train(model: nn.Module, corpus: Tensor, *, updates: int, batch_size: int, delay: int, device: torch.device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(81_337)
    model.train()
    for _update in range(updates):
        indices = torch.randint(corpus.shape[0], (batch_size,), generator=generator)
        events = corpus[indices].to(device)
        prediction = model(events)
        target = events[:, -1 - delay]
        loss = (prediction - target).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()


@torch.no_grad()
def _mse(model: nn.Module, corpus: Tensor, *, delay: int, device: torch.device) -> float:
    model.eval()
    events = corpus.to(device)
    return float((model(events) - events[:, -1 - delay]).square().mean().cpu())


def run(
    *, device: str | torch.device | None = None, corpus_episodes: int = 512,
    updates: int = 300, batch_size: int = 32, sequence_length: int = 16,
    recall_delay: int = 8, artifact: Path | None = None,
) -> Report:
    if corpus_episodes < 2 or updates < 1 or batch_size < 1 or not 1 <= recall_delay < sequence_length:
        raise ValueError("invalid delayed-recall audit dimensions")
    resolved = resolve_device(device)
    torch.manual_seed(71_337); np.random.seed(71_337); random.seed(71_337)
    train = _corpus(episodes=corpus_episodes, sequence_length=sequence_length, seed_start=40_000)
    heldout = _corpus(episodes=128, sequence_length=sequence_length, seed_start=5_000_000)
    factories = (("evidence_store", EvidenceStoreRecall), ("evidence_disabled", lambda: EvidenceStoreRecall(use_evidence=False)), ("gru", GRURecall), ("four_event_transformer", FourEventTransformerRecall))
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train(model, train, updates=updates, batch_size=batch_size, delay=recall_delay, device=resolved)
        results.append(Result(name, sum(parameter.numel() for parameter in model.parameters()), _mse(model, heldout, delay=recall_delay, device=resolved)))
    by_name = {result.name: result for result in results}
    accepted = by_name["evidence_store"].heldout_mse < by_name["evidence_disabled"].heldout_mse * 0.8
    target = artifact or ROOT / "artifacts" / "minigrid-delayed-event-recall-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_delayed_event_recall_v1: reconstruct an eight-event-old, agent-observed sensor record from actual MemoryS13 random-action streams; source grid, routes, rewards as answers, and policy labels are excluded.",
        str(resolved), sequence_length, recall_delay, corpus_episodes, updates, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--corpus-episodes", type=int, default=512)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--recall-delay", type=int, default=8)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, corpus_episodes=args.corpus_episodes, updates=args.updates, batch_size=args.batch_size, sequence_length=args.sequence_length, recall_delay=args.recall_delay, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted_evidence_recall:
        raise SystemExit("persistent evidence store did not meet delayed-recall gate")

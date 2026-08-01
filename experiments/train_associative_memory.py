"""Test whether PLWM's episodic memory helps associative recall.

Each episode writes random key/value facts, then queries a key after all values
have disappeared from the observation stream. The no-memory ablation retains
the recurrent belief cell but cannot perform content-addressed reads.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.torch_model import TorchPLWM
from plwm.device import resolve_device


KEYS = 8
INPUT_DIM = 2 + KEYS + 1


def make_batch(batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    values = torch.randint(0, 2, (batch_size, KEYS), device=device, dtype=torch.float32)
    ordering = torch.stack([torch.randperm(KEYS, device=device) for _ in range(batch_size)])
    query_keys = torch.randint(0, KEYS, (batch_size,), device=device)
    observations = torch.zeros(batch_size, KEYS + 1, INPUT_DIM, device=device)
    rows = torch.arange(batch_size, device=device)
    for step in range(KEYS):
        keys = ordering[:, step]
        observations[:, step, 0] = 1.0
        observations[rows, step, 2 + keys] = 1.0
        observations[:, step, -1] = values[rows, keys] * 2.0 - 1.0
    observations[:, -1, 1] = 1.0
    observations[rows, -1, 2 + query_keys] = 1.0
    return observations, values[rows, query_keys].unsqueeze(1)


def train(model: TorchPLWM, *, steps: int, device: torch.device) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    loss_value = 0.0
    for _ in range(steps):
        observations, labels = make_batch(128, device)
        logits, _, _ = model(observations)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return loss_value


@torch.no_grad()
def evaluate(model: TorchPLWM, device: torch.device) -> float:
    model.eval()
    observations, labels = make_batch(4096, device)
    logits, _, _ = model(observations)
    return float(((logits.sigmoid() >= 0.5) == (labels >= 0.5)).float().mean())


@dataclass
class Report:
    memory_accuracy: float
    gated_memory_accuracy: float
    no_memory_accuracy: float
    memory_loss: float
    gated_memory_loss: float
    no_memory_loss: float


def run(steps: int = 550, seed: int = 31, device: str | torch.device | None = None) -> Report:
    torch.manual_seed(seed)
    device = resolve_device(device)
    with_memory = TorchPLWM(
        INPUT_DIM, latent_dim=64, memory_slots=KEYS + 1, thought_steps=3, use_memory=True
    ).to(device)
    without_memory = TorchPLWM(
        INPUT_DIM, latent_dim=64, memory_slots=KEYS + 1, thought_steps=3, use_memory=False
    ).to(device)
    gated_memory = TorchPLWM(
        INPUT_DIM, latent_dim=64, memory_slots=KEYS + 1, thought_steps=3, use_memory=True, gated_memory=True
    ).to(device)
    memory_loss = train(with_memory, steps=steps, device=device)
    gated_memory_loss = train(gated_memory, steps=steps, device=device)
    no_memory_loss = train(without_memory, steps=steps, device=device)
    return Report(
        memory_accuracy=evaluate(with_memory, device),
        gated_memory_accuracy=evaluate(gated_memory, device),
        no_memory_accuracy=evaluate(without_memory, device),
        memory_loss=memory_loss,
        gated_memory_loss=gated_memory_loss,
        no_memory_loss=no_memory_loss,
    )


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if outcome.memory_accuracy < 0.85:
        raise SystemExit("episodic memory did not solve associative recall")
    if outcome.memory_accuracy < outcome.no_memory_accuracy + 0.15:
        raise SystemExit("memory did not improve enough over the recurrent ablation")
    if outcome.gated_memory_accuracy < 0.85:
        raise SystemExit("gated episodic memory did not solve associative recall")
    if outcome.gated_memory_accuracy < outcome.no_memory_accuracy + 0.15:
        raise SystemExit("gated episodic memory did not improve enough over the recurrent ablation")

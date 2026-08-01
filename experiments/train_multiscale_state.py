"""Verify fast adaptation and slow-state resistance to transient contradiction."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.multiscale import MultiScalePLWM


def make_batch(batch: int, steps: int, device: torch.device):
    stable = torch.randint(0, 2, (batch, 1), device=device, dtype=torch.float32)
    volatile = torch.randint(0, 2, (batch, steps), device=device, dtype=torch.float32)
    noise = torch.rand(batch, steps, device=device) < 0.1
    stable_observation = stable.expand(-1, steps).clone()
    stable_observation[noise] = 1.0 - stable_observation[noise]
    observations = torch.stack((stable_observation * 2.0 - 1.0, volatile * 2.0 - 1.0), dim=-1)
    return observations, stable, volatile


def run(steps: int = 700, seed: int = 47):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = MultiScalePLWM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    sequence_length = 12
    for _ in range(steps):
        observations, stable, volatile = make_batch(192, sequence_length, device)
        state = model.initial_state(observations.shape[0], device)
        loss = 0.0
        gates = []
        for index in range(sequence_length):
            state, gate = model.step(observations[:, index], state)
            fast, slow = model.decode(state)
            loss = loss + loss_fn(fast, volatile[:, index : index + 1])
            loss = loss + loss_fn(slow, stable)
            gates.append(gate)
        loss = loss / sequence_length + 0.01 * torch.cat(gates, dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        observations, stable, volatile = make_batch(2048, sequence_length, device)
        state = model.initial_state(observations.shape[0], device)
        for index in range(sequence_length):
            state, _ = model.step(observations[:, index], state)
        fast, slow = model.decode(state)
        fast_accuracy = float(((fast.sigmoid() >= 0.5) == (volatile[:, -1:] >= 0.5)).float().mean())
        slow_accuracy = float(((slow.sigmoid() >= 0.5) == (stable >= 0.5)).float().mean())

        # A three-event contradictory burst should alter volatile tracking but preserve slow belief.
        burst = torch.stack((1.0 - stable.expand(-1, 3), torch.randint(0, 2, (2048, 3), device=device).float()), dim=-1)
        burst = burst * 2.0 - 1.0
        gates = []
        for index in range(3):
            state, gate = model.step(burst[:, index], state)
            gates.append(gate)
        _, slow_after = model.decode(state)
        retained_accuracy = float(((slow_after.sigmoid() >= 0.5) == (stable >= 0.5)).float().mean())
        burst_gate = float(torch.cat(gates, dim=1).mean())
    return Report(fast_accuracy, slow_accuracy, retained_accuracy, burst_gate)


@dataclass
class Report:
    fast_accuracy: float
    slow_accuracy: float
    slow_retention_after_contradiction: float
    mean_slow_gate_during_contradiction: float


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if min(outcome.fast_accuracy, outcome.slow_accuracy, outcome.slow_retention_after_contradiction) < 0.9:
        raise SystemExit("multiscale state separation did not pass")

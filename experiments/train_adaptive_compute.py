"""Validate learned latent-depth allocation under a compute penalty."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.adaptive import AdaptivePonderer


STATES = 8
MAX_STEPS = 4


def make_batch(batch_size: int, device: torch.device):
    start = torch.randint(0, STATES, (batch_size,), device=device)
    difficulty = torch.randint(0, MAX_STEPS + 1, (batch_size,), device=device)
    # The transition to learn is a cyclic successor; depth is required to reach target.
    target = (start + difficulty) % STATES
    return start, difficulty.float() / MAX_STEPS, target, difficulty


@dataclass
class Report:
    accuracy: float
    expected_steps_by_difficulty: list[float]
    fixed_max_compute: float
    learned_mean_compute: float


def run(steps: int = 800, seed: int = 37) -> Report:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = AdaptivePonderer(STATES, latent_dim=32, max_steps=MAX_STEPS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        start, normalized, target, depth_target = make_batch(256, device)
        logits, expected_steps, _, depth_logits = model(start, normalized)
        depth_loss = loss_fn(depth_logits[torch.arange(start.shape[0]), depth_target], target)
        loss = loss_fn(logits, target) + 0.25 * depth_loss + 0.015 * expected_steps.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    expected_by_difficulty = []
    all_predictions = []
    all_targets = []
    all_compute = []
    with torch.no_grad():
        for difficulty in range(MAX_STEPS + 1):
            start = torch.arange(STATES, device=device).repeat(128)
            normalized = torch.full_like(start, difficulty, dtype=torch.float32) / MAX_STEPS
            target = (start + difficulty) % STATES
            logits, expected_steps, _, _ = model(start, normalized)
            expected_by_difficulty.append(float(expected_steps.mean()))
            all_predictions.append(logits.argmax(dim=-1))
            all_targets.append(target)
            all_compute.append(expected_steps)
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    return Report(
        accuracy=float((predictions == targets).float().mean()),
        expected_steps_by_difficulty=expected_by_difficulty,
        fixed_max_compute=float(MAX_STEPS),
        learned_mean_compute=float(torch.cat(all_compute).mean()),
    )


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    monotonic = all(
        right > left for left, right in zip(outcome.expected_steps_by_difficulty, outcome.expected_steps_by_difficulty[1:])
    )
    if outcome.accuracy < 0.98 or not monotonic or outcome.learned_mean_compute >= outcome.fixed_max_compute - 0.5:
        raise SystemExit("adaptive-compute controller did not learn an efficient depth allocation")

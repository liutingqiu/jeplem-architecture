"""Atomic one-frame public-vision side-readout calibration, outside control."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from experiments.train_public_vision_candidate_memory import GRID_SIZE, VISION_CHANNELS, make_batch
from plwm.device import resolve_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEPS = 240
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRAIN_SEED = 13_900_000
DEFAULT_HELDOUT_SEED = 14_000_000
ACCEPTANCE_ACCURACY = 0.90


@dataclass(frozen=True)
class Report:
    protocol: str
    device: str
    train_seed: int
    heldout_seed: int
    train_steps: int
    batch_size: int
    parameters: int
    heldout_accuracy: float
    accepted_fixed_identity_readout: bool


def fixed_goal_side(vision: Tensor) -> Tensor:
    """Read the fixed goal tile's side from public pixels only, outside a model."""
    if vision.ndim != 5 or vision.shape[2:] != (GRID_SIZE, GRID_SIZE, VISION_CHANNELS):
        raise ValueError("expected [batch, time, 11, 11, 7] visual sequences")
    goal = vision[:, 0, :, :, 3]
    if not bool((goal.sum(dim=(1, 2)) == 1).all()):
        raise ValueError("initial public frame must render exactly one goal tile")
    left = goal[:, :, : GRID_SIZE // 2].sum(dim=(1, 2))
    right = goal[:, :, GRID_SIZE // 2 + 1 :].sum(dim=(1, 2))
    if not bool((left + right == 1).all()):
        raise ValueError("goal must be rendered exactly once off centre")
    return right.float()


class FixedIdentitySideMLP(nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(GRID_SIZE * GRID_SIZE * VISION_CHANNELS, width), nn.Tanh(), nn.Linear(width, 1),
        )

    def forward(self, initial_vision: Tensor) -> Tensor:
        if initial_vision.ndim != 4 or initial_vision.shape[1:] != (GRID_SIZE, GRID_SIZE, VISION_CHANNELS):
            raise ValueError("expected [batch, 11, 11, 7] initial public vision")
        return self.network(initial_vision).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train(model: FixedIdentitySideMLP, *, steps: int, batch_size: int, seed: int, device: torch.device) -> None:
    if steps < 1:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(steps):
        batch = make_batch(batch_size, generator=generator, device=device)
        labels = fixed_goal_side(batch.vision)
        loss = nn.functional.binary_cross_entropy_with_logits(model(batch.vision[:, 0]), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def accuracy(model: FixedIdentitySideMLP, *, seed: int, device: torch.device, batch_size: int = 2_048) -> float:
    model.eval()
    batch = make_batch(batch_size, generator=torch.Generator(device=device).manual_seed(seed), device=device)
    return float((model(batch.vision[:, 0]).sigmoid().ge(0.5) == fixed_goal_side(batch.vision).bool()).float().mean())


def run(
    *, steps: int = DEFAULT_STEPS, batch_size: int = DEFAULT_BATCH_SIZE, train_seed: int = DEFAULT_TRAIN_SEED,
    heldout_seed: int = DEFAULT_HELDOUT_SEED, device: str | torch.device | None = None,
) -> Report:
    if train_seed == heldout_seed:
        raise ValueError("train and held-out generator seeds must differ")
    resolved_device = resolve_device(device)
    torch.manual_seed(train_seed)
    model = FixedIdentitySideMLP().to(resolved_device)
    train(model, steps=steps, batch_size=batch_size, seed=train_seed, device=resolved_device)
    heldout_accuracy = accuracy(model, seed=heldout_seed, device=resolved_device)
    return Report(
        "public_vision_fixed_identity_side_calibration_v1: bounded supervised visual readout only; forward receives one initial 11x11x7 public visual frame. Fixed-identity target, coordinates, query, seeds, actions, rewards, teachers, planners, and shaped rewards are excluded from input and never transfer to Luanti control.",
        str(resolved_device), train_seed, heldout_seed, steps, batch_size, parameter_count(model), heldout_accuracy,
        heldout_accuracy >= ACCEPTANCE_ACCURACY,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--heldout-seed", type=int, default=DEFAULT_HELDOUT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "public-vision-fixed-identity-calibration-report.json")
    args = parser.parse_args()
    outcome = run(steps=args.steps, batch_size=args.batch_size, train_seed=args.train_seed, heldout_seed=args.heldout_seed, device=args.device)
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    print(json.dumps(asdict(outcome), indent=2))

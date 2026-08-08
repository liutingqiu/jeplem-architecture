"""Query-conditioned public-vision binding diagnostic, never a control result."""
from __future__ import annotations

import argparse
import copy
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
DEFAULT_TRAIN_SEED = 13_600_000
DEFAULT_HELDOUT_SEED = 13_700_000
ACCEPTANCE_ACCURACY = 0.90
ACCEPTANCE_MARGIN = 0.12


@dataclass(frozen=True)
class ModelResult:
    name: str
    parameters: int
    heldout_accuracy: float


@dataclass(frozen=True)
class Report:
    protocol: str
    device: str
    train_seed: int
    heldout_seed: int
    train_steps: int
    batch_size: int
    parameter_match_error: float
    results: list[ModelResult]
    accepted_query_conditioned_binding: bool


class QueryConditionedVisualBinding(nn.Module):
    """Bind a delayed visual query to a spatial token in the initial frame."""

    def __init__(self, *, query_conditioned: bool, width: int = 16) -> None:
        super().__init__()
        self.query_conditioned = query_conditioned
        self.encoder = nn.Sequential(
            nn.Conv2d(VISION_CHANNELS, width, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv2d(width, width, kernel_size=3, padding=1), nn.Tanh(),
        )
        self.key_projection = nn.Linear(width, width, bias=False)
        self.query_projection = nn.Linear(width, width, bias=False)
        # The unconditioned ablation retains these parameters but scores spatial
        # keys without their relation to the delayed query.
        self.unconditioned_score = nn.Linear(width, 1)
        self.unconditioned_query_bias = nn.Linear(width, 1)
        self.position_readout = nn.Sequential(nn.Linear(GRID_SIZE * GRID_SIZE, width * 4), nn.Tanh(), nn.Linear(width * 4, 1))

    def forward(self, vision: Tensor) -> Tensor:
        if vision.ndim != 5 or vision.shape[2:] != (GRID_SIZE, GRID_SIZE, VISION_CHANNELS):
            raise ValueError("expected [batch, time, 11, 11, 7] public visual sequences")
        initial = self.encoder(vision[:, 0].permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(vision.shape[0], -1, self.key_projection.in_features)
        query_map = self.encoder(vision[:, -1].permute(0, 3, 1, 2))
        query = self.query_projection(query_map.mean(dim=(2, 3)))
        keys = self.key_projection(initial)
        if self.query_conditioned:
            scores = torch.einsum("btd,bd->bt", keys, query) / (keys.shape[-1] ** 0.5)
        else:
            scores = self.unconditioned_score(keys).squeeze(-1) + self.unconditioned_query_bias(query).squeeze(-1).unsqueeze(1)
        return self.position_readout(torch.softmax(scores, dim=1)).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train(model: QueryConditionedVisualBinding, *, steps: int, batch_size: int, seed: int, device: torch.device) -> None:
    if steps < 1:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(steps):
        batch = make_batch(batch_size, generator=generator, device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(model(batch.vision), batch.labels.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def accuracy(model: QueryConditionedVisualBinding, *, seed: int, device: torch.device, batch_size: int = 2_048) -> float:
    model.eval()
    batch = make_batch(batch_size, generator=torch.Generator(device=device).manual_seed(seed), device=device)
    return float((model(batch.vision).sigmoid().ge(0.5) == batch.labels.squeeze(1).bool()).float().mean())


def run(
    *, steps: int = DEFAULT_STEPS, batch_size: int = DEFAULT_BATCH_SIZE, train_seed: int = DEFAULT_TRAIN_SEED,
    heldout_seed: int = DEFAULT_HELDOUT_SEED, device: str | torch.device | None = None,
) -> Report:
    if train_seed == heldout_seed:
        raise ValueError("train and held-out generator seeds must differ")
    resolved_device = resolve_device(device)
    torch.manual_seed(train_seed)
    query_conditioned = QueryConditionedVisualBinding(query_conditioned=True).to(resolved_device)
    late_fusion = copy.deepcopy(query_conditioned)
    late_fusion.query_conditioned = False
    query_parameters, late_parameters = parameter_count(query_conditioned), parameter_count(late_fusion)
    if query_parameters != late_parameters:
        raise RuntimeError("visual-binding ablations must have exactly equal parameter counts")
    train(query_conditioned, steps=steps, batch_size=batch_size, seed=train_seed, device=resolved_device)
    train(late_fusion, steps=steps, batch_size=batch_size, seed=train_seed, device=resolved_device)
    query_accuracy = accuracy(query_conditioned, seed=heldout_seed, device=resolved_device)
    late_accuracy = accuracy(late_fusion, seed=heldout_seed, device=resolved_device)
    results = [ModelResult("query_conditioned", query_parameters, query_accuracy), ModelResult("late_fusion", late_parameters, late_accuracy)]
    return Report(
        "query_conditioned_public_vision_binding_v1: bounded supervised visual readout only; each forward receives initial and delayed-query 11x11x7 public visual frames only. Query/answer labels, identity IDs, coordinates, seeds, actions, rewards, teachers, planners, and shaped rewards are excluded from input and never transfer to Luanti control.",
        str(resolved_device), train_seed, heldout_seed, steps, batch_size, 0.0, results,
        query_accuracy >= ACCEPTANCE_ACCURACY and query_accuracy - late_accuracy >= ACCEPTANCE_MARGIN,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--heldout-seed", type=int, default=DEFAULT_HELDOUT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "query-conditioned-public-vision-binding-report.json")
    args = parser.parse_args()
    report = run(steps=args.steps, batch_size=args.batch_size, train_seed=args.train_seed, heldout_seed=args.heldout_seed, device=args.device)
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()

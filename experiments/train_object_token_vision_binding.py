"""Object-token public-vision binding diagnostic, never a control result."""
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
DEFAULT_TRAIN_SEED = 14_100_000
DEFAULT_HELDOUT_SEED = 14_200_000
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
    accepted_object_token_binding: bool


class ObjectTokenVisualBinding(nn.Module):
    """Match the one delayed public object against four initial object tokens.

    ``object_mask`` is computed from the raw identity-tile part of the public
    raster.  It only distinguishes an object tile from air/agent; it neither
    reads nor exposes a selected identity, coordinates, answer, or label.
    """

    def __init__(self, *, query_conditioned: bool, width: int = 16) -> None:
        super().__init__()
        self.query_conditioned = query_conditioned
        self.encoder = nn.Sequential(
            nn.Conv2d(VISION_CHANNELS, width, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv2d(width, width, kernel_size=3, padding=1), nn.Tanh(),
        )
        self.key_projection = nn.Linear(width, width, bias=False)
        self.query_projection = nn.Linear(width, width, bias=False)
        # This branch remains present in both models for an exact parameter
        # match.  The ablation cannot use the delayed query to rank objects.
        self.unconditioned_score = nn.Linear(width, 1)
        self.unconditioned_query_bias = nn.Linear(width, 1)
        self.position_readout = nn.Sequential(
            nn.Linear(GRID_SIZE * GRID_SIZE, width * 4), nn.Tanh(), nn.Linear(width * 4, 1),
        )

    @staticmethod
    def _object_mask(frame: Tensor) -> Tensor:
        if frame.ndim != 4 or frame.shape[1:] != (GRID_SIZE, GRID_SIZE, VISION_CHANNELS):
            raise ValueError("expected [batch, 11, 11, 7] public visual frame")
        return frame[..., 3:].sum(dim=-1).gt(0)

    @staticmethod
    def _masked_pool(tokens: Tensor, mask: Tensor, *, expected_count: int) -> Tensor:
        if mask.shape != tokens.shape[:2] or not bool((mask.sum(dim=1) == expected_count).all()):
            raise ValueError("public visual object-token contract is violated")
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1) / float(expected_count)
        return (tokens * weights).sum(dim=1)

    def forward(self, vision: Tensor) -> Tensor:
        if vision.ndim != 5 or vision.shape[2:] != (GRID_SIZE, GRID_SIZE, VISION_CHANNELS):
            raise ValueError("expected [batch, time, 11, 11, 7] public visual sequences")
        initial_frame, query_frame = vision[:, 0], vision[:, -1]
        initial_mask, query_mask = self._object_mask(initial_frame), self._object_mask(query_frame)
        initial_tokens = self.encoder(initial_frame.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(vision.shape[0], -1, self.key_projection.in_features)
        query_tokens = self.encoder(query_frame.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(vision.shape[0], -1, self.query_projection.in_features)
        initial_mask = initial_mask.reshape(vision.shape[0], -1)
        query_mask = query_mask.reshape(vision.shape[0], -1)
        if not bool((initial_mask.sum(dim=1) == 4).all()):
            raise ValueError("initial public frame must expose exactly four object tokens")
        query = self.query_projection(self._masked_pool(query_tokens, query_mask, expected_count=1))
        keys = self.key_projection(initial_tokens)
        if self.query_conditioned:
            scores = torch.einsum("btd,bd->bt", keys, query) / (keys.shape[-1] ** 0.5)
        else:
            scores = self.unconditioned_score(keys).squeeze(-1) + self.unconditioned_query_bias(query).squeeze(-1).unsqueeze(1)
        # The public object mask prevents background locations from competing
        # with the four visible facts.  It contains no object identity value.
        scores = scores.masked_fill(~initial_mask, torch.finfo(scores.dtype).min)
        return self.position_readout(torch.softmax(scores, dim=1)).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train(model: ObjectTokenVisualBinding, *, steps: int, batch_size: int, seed: int, device: torch.device) -> None:
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
def accuracy(model: ObjectTokenVisualBinding, *, seed: int, device: torch.device, batch_size: int = 2_048) -> float:
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
    object_token = ObjectTokenVisualBinding(query_conditioned=True).to(resolved_device)
    unconditioned = copy.deepcopy(object_token)
    unconditioned.query_conditioned = False
    object_parameters, ablation_parameters = parameter_count(object_token), parameter_count(unconditioned)
    if object_parameters != ablation_parameters:
        raise RuntimeError("object-token visual-binding ablations must have exactly equal parameter counts")
    train(object_token, steps=steps, batch_size=batch_size, seed=train_seed, device=resolved_device)
    train(unconditioned, steps=steps, batch_size=batch_size, seed=train_seed, device=resolved_device)
    object_accuracy = accuracy(object_token, seed=heldout_seed, device=resolved_device)
    ablation_accuracy = accuracy(unconditioned, seed=heldout_seed, device=resolved_device)
    results = [
        ModelResult("object_token_query_matching", object_parameters, object_accuracy),
        ModelResult("query_independent_object_tokens", ablation_parameters, ablation_accuracy),
    ]
    return Report(
        "object_token_public_vision_binding_v1: bounded supervised visual readout only; each forward receives initial and delayed-query 11x11x7 public visual frames. Object masks are derived only from raw public tile pixels and only distinguish object from air/agent. Labels, identity IDs, coordinates, seeds, actions, rewards, teachers, planners, and shaped rewards are excluded from input and never transfer to Luanti control.",
        str(resolved_device), train_seed, heldout_seed, steps, batch_size, 0.0, results,
        object_accuracy >= ACCEPTANCE_ACCURACY and object_accuracy - ablation_accuracy >= ACCEPTANCE_MARGIN,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--heldout-seed", type=int, default=DEFAULT_HELDOUT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "object-token-public-vision-binding-report.json")
    args = parser.parse_args()
    outcome = run(steps=args.steps, batch_size=args.batch_size, train_seed=args.train_seed, heldout_seed=args.heldout_seed, device=args.device)
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    print(json.dumps(asdict(outcome), indent=2))

"""Bounded public-vision sparse-candidate storage/readout diagnostic.

This is deliberately not a Luanti control result.  It tests whether several
latent candidates derived from one public 11x11x7 visual event are more useful
than an exactly parameter-matched one-candidate aggregation.  Query identity
and answer side are generated labels only: neither is provided to a model
forward pass.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from plwm.device import resolve_device
from plwm.torch_model import TorchPLWM


ROOT = Path(__file__).resolve().parents[1]
GRID_SIZE = 11
VISION_CHANNELS = 7
IDENTITIES = 4
ACTION_COUNT = 4
START_ACTION = ACTION_COUNT
DELAY_EVENTS = 4
EVENTS = DELAY_EVENTS + 2
DEFAULT_LATENT_DIM = 32
DEFAULT_MEMORY_SLOTS = 8
DEFAULT_STEPS = 1_200
DEFAULT_BATCH_SIZE = 96
DEFAULT_TRAIN_SEED = 13_200_000
DEFAULT_HELDOUT_SEED = 13_300_000
ACCEPTANCE_ACCURACY = 0.90
ACCEPTANCE_MARGIN = 0.12


@dataclass(frozen=True)
class DiagnosticBatch:
    vision: Tensor
    previous_actions: Tensor
    previous_rewards: Tensor
    labels: Tensor


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
    accepted_mechanism_diagnostic: bool


def make_batch(batch_size: int, *, generator: torch.Generator, device: torch.device) -> DiagnosticBatch:
    """Create delayed queries from raw tile grids without adding model inputs.

    The generator's side/query arrays are used only to render raw visual tiles
    and form the terminal classification label.  The returned model inputs do
    not include coordinates, identity IDs, seeds, answers, or teacher actions.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    vision = torch.zeros(batch_size, EVENTS, GRID_SIZE, GRID_SIZE, VISION_CHANNELS, device=device)
    vision[..., 0] = 1.0  # Public air tile.
    vision[:, :, GRID_SIZE // 2, GRID_SIZE // 2, 0] = 0.0
    vision[:, :, GRID_SIZE // 2, GRID_SIZE // 2, 2] = 1.0  # Public agent tile.
    sides = torch.randint(0, 2, (batch_size, IDENTITIES), generator=generator, device=device)
    queries = torch.randint(0, IDENTITIES, (batch_size,), generator=generator, device=device)
    rows = torch.stack([torch.randperm(IDENTITIES, generator=generator, device=device) * 2 + 1 for _ in range(batch_size)])
    for batch_index in range(batch_size):
        for identity in range(IDENTITIES):
            x = 2 if int(sides[batch_index, identity]) == 0 else GRID_SIZE - 3
            y = int(rows[batch_index, identity])
            vision[batch_index, 0, y, x] = 0.0
            vision[batch_index, 0, y, x, 3 + identity] = 1.0
        query = int(queries[batch_index])
        vision[batch_index, -1, 1, GRID_SIZE // 2] = 0.0
        vision[batch_index, -1, 1, GRID_SIZE // 2, 3 + query] = 1.0
    previous_actions = torch.full((batch_size, EVENTS), START_ACTION, dtype=torch.long, device=device)
    previous_actions[:, 1:] = 1  # The same public forced-south action event as the delayed course.
    previous_rewards = torch.zeros(batch_size, EVENTS, device=device)
    return DiagnosticBatch(vision, previous_actions, previous_rewards, sides.gather(1, queries.unsqueeze(1)).float())


class PublicVisionCandidateMemory(nn.Module):
    """Shared visual writer, with either separate or aggregated candidate writes."""

    def __init__(
        self,
        *,
        multi_candidate: bool,
        latent_dim: int = DEFAULT_LATENT_DIM,
        memory_slots: int = DEFAULT_MEMORY_SLOTS,
        candidate_count: int = IDENTITIES,
    ) -> None:
        super().__init__()
        if not 1 <= candidate_count <= memory_slots:
            raise ValueError("candidate_count must fit the episodic memory")
        self.multi_candidate = multi_candidate
        self.candidate_count = candidate_count
        self.spatial = nn.Sequential(
            nn.Conv2d(VISION_CHANNELS, latent_dim, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1), nn.Tanh(),
        )
        self.event_context = nn.Sequential(nn.Linear(ACTION_COUNT + 2, latent_dim), nn.Tanh())
        self.event_projection = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Tanh())
        self.candidate_projection = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh())
        self.candidate_score = nn.Linear(latent_dim, 1)
        self.core = TorchPLWM(
            latent_dim, latent_dim=latent_dim, memory_slots=memory_slots, thought_steps=0,
            output_dim=1, use_memory=True, gated_memory=True,
        )

    def _public_event(self, vision: Tensor, action: Tensor, reward: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if vision.shape[1:] != (GRID_SIZE, GRID_SIZE, VISION_CHANNELS):
            raise ValueError("expected one 11x11x7 public vision event")
        if action.shape != (vision.shape[0],) or reward.shape != (vision.shape[0],):
            raise ValueError("previous action and reward must align with the visual batch")
        if bool((action < 0).any()) or bool((action > START_ACTION).any()):
            raise ValueError("previous action is outside the public action vocabulary")
        action_features = torch.nn.functional.one_hot(action, num_classes=ACTION_COUNT + 1).to(dtype=vision.dtype)
        context = self.event_context(torch.cat((action_features, reward.unsqueeze(1)), dim=1))
        spatial = self.spatial(vision.permute(0, 3, 1, 2)) + context.unsqueeze(-1).unsqueeze(-1)
        tokens = self.candidate_projection(spatial.permute(0, 2, 3, 1).reshape(vision.shape[0], -1, context.shape[1]))
        scores = self.candidate_score(tokens).squeeze(-1)
        selected_scores, selected_indices = scores.topk(self.candidate_count, dim=1)
        candidates = tokens.gather(1, selected_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        event = self.event_projection(torch.cat((spatial.mean(dim=(2, 3)), context), dim=1))
        # This per-candidate allocation mask is a predicate over raw public
        # pixels only. It says that a selected spatial candidate overlaps a
        # non-static tile, never which identity or side that tile represents.
        visible_object = vision[..., 3:].sum(dim=-1).reshape(vision.shape[0], -1).gt(0)
        write_mask = visible_object.gather(1, selected_indices).to(dtype=vision.dtype)
        return event, candidates, write_mask

    def forward(self, vision: Tensor, previous_actions: Tensor, previous_rewards: Tensor) -> Tensor:
        if vision.ndim != 5 or previous_actions.shape != vision.shape[:2] or previous_rewards.shape != vision.shape[:2]:
            raise ValueError("expected aligned [batch, time] public event sequences")
        state = self.core.initial_state(vision.shape[0], vision.device)
        for index in range(vision.shape[1]):
            event, candidates, write_mask = self._public_event(
                vision[:, index], previous_actions[:, index], previous_rewards[:, index]
            )
            if self.multi_candidate:
                write_candidates = candidates
            else:
                weights = torch.softmax(self.candidate_score(candidates).squeeze(-1), dim=1)
                write_candidates = torch.einsum("bk,bkd->bd", weights, candidates).unsqueeze(1)
                write_mask = write_mask[:, :1]
            state, _ = self.core.step(event, state, write_candidates=write_candidates, write_mask=write_mask)
        return self.core.classify(state).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train(model: PublicVisionCandidateMemory, *, steps: int, batch_size: int, seed: int, device: torch.device) -> None:
    if steps < 1:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(steps):
        batch = make_batch(batch_size, generator=generator, device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(model(batch.vision, batch.previous_actions, batch.previous_rewards), batch.labels.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if hasattr(model, 'core') and hasattr(model.core, 'enforce_key_orthogonality'):
            model.core.enforce_key_orthogonality()


@torch.no_grad()
def accuracy(model: PublicVisionCandidateMemory, *, seed: int, device: torch.device, batch_size: int = 2_048) -> float:
    model.eval()
    batch = make_batch(batch_size, generator=torch.Generator(device=device).manual_seed(seed), device=device)
    predictions = model(batch.vision, batch.previous_actions, batch.previous_rewards).sigmoid().ge(0.5)
    return float((predictions == batch.labels.squeeze(1).bool()).float().mean())


def run(
    *, steps: int = DEFAULT_STEPS, batch_size: int = DEFAULT_BATCH_SIZE, train_seed: int = DEFAULT_TRAIN_SEED,
    heldout_seed: int = DEFAULT_HELDOUT_SEED, device: str | torch.device | None = None,
) -> Report:
    if train_seed == heldout_seed:
        raise ValueError("train and held-out generator seeds must differ")
    resolved_device = resolve_device(device)
    torch.manual_seed(train_seed)
    multi = PublicVisionCandidateMemory(multi_candidate=True).to(resolved_device)
    single = copy.deepcopy(multi)
    single.multi_candidate = False
    multi_parameters, single_parameters = parameter_count(multi), parameter_count(single)
    if multi_parameters != single_parameters:
        raise RuntimeError("candidate ablations must have exactly equal parameter counts")
    train(multi, steps=steps, batch_size=batch_size, seed=train_seed, device=resolved_device)
    train(single, steps=steps, batch_size=batch_size, seed=train_seed, device=resolved_device)
    multi_accuracy = accuracy(multi, seed=heldout_seed, device=resolved_device)
    single_accuracy = accuracy(single, seed=heldout_seed, device=resolved_device)
    results = [ModelResult("multi_candidate", multi_parameters, multi_accuracy), ModelResult("single_candidate", single_parameters, single_accuracy)]
    return Report(
        "public_vision_sparse_candidate_memory_v2: bounded supervised storage/readout diagnostic only; each forward receives only current 11x11x7 public vision, previous chosen action, and previous scalar reward. Sparse per-candidate allocation is derived from public pixels only. Query/answer labels, coordinates, seeds, teachers, planners, and shaped rewards are excluded from model input and never transfer to Luanti control.",
        str(resolved_device), train_seed, heldout_seed, steps, batch_size, 0.0,
        results, multi_accuracy >= ACCEPTANCE_ACCURACY and multi_accuracy - single_accuracy >= ACCEPTANCE_MARGIN,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--heldout-seed", type=int, default=DEFAULT_HELDOUT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "public-vision-sparse-candidate-memory-report.json")
    args = parser.parse_args()
    report = run(steps=args.steps, batch_size=args.batch_size, train_seed=args.train_seed, heldout_seed=args.heldout_seed, device=args.device)
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()

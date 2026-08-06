"""Object-token episodic-memory diagnostic, explicitly outside control."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from experiments.train_public_vision_candidate_memory import (
    ACTION_COUNT, GRID_SIZE, START_ACTION, VISION_CHANNELS, make_batch,
)
from plwm.device import resolve_device
from plwm.torch_model import TorchPLWM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEPS = 240
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRAIN_SEED = 14_300_000
DEFAULT_HELDOUT_SEED = 14_400_000
ACCEPTANCE_ACCURACY = 0.90
ACCEPTANCE_MARGIN = 0.12
LATENT_DIM = 32
MEMORY_SLOTS = 8


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
    accepted_object_token_memory: bool


class ObjectTokenEpisodicMemory(nn.Module):
    """Write four public object tokens, then retrieve one from a delayed query."""

    def __init__(self, *, multi_writer: bool, use_memory: bool) -> None:
        super().__init__()
        self.multi_writer = multi_writer
        self.spatial = nn.Sequential(
            nn.Conv2d(VISION_CHANNELS, LATENT_DIM, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv2d(LATENT_DIM, LATENT_DIM, kernel_size=3, padding=1), nn.Tanh(),
        )
        # This is an internal learned grid-index parameter, not an input
        # coordinate field.  It permits a written visual fact to retain side.
        self.initial_position = nn.Parameter(torch.empty(GRID_SIZE * GRID_SIZE, LATENT_DIM))
        nn.init.normal_(self.initial_position, std=0.02)
        self.event_context = nn.Sequential(nn.Linear(ACTION_COUNT + 2, LATENT_DIM), nn.Tanh())
        self.candidate_score = nn.Linear(LATENT_DIM, 1)
        self.core = TorchPLWM(
            LATENT_DIM, latent_dim=LATENT_DIM, memory_slots=MEMORY_SLOTS, thought_steps=0,
            output_dim=1, use_memory=use_memory, gated_memory=True,
        )

    @staticmethod
    def _object_mask(frame: Tensor) -> Tensor:
        if frame.ndim != 4 or frame.shape[1:] != (GRID_SIZE, GRID_SIZE, VISION_CHANNELS):
            raise ValueError("expected [batch, 11, 11, 7] public visual frame")
        return frame[..., 3:].sum(dim=-1).gt(0).reshape(frame.shape[0], -1)

    def _visual_tokens(self, frame: Tensor) -> Tensor:
        return self.spatial(frame.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(frame.shape[0], -1, LATENT_DIM)

    @staticmethod
    def _pool(tokens: Tensor, mask: Tensor, *, expected_count: int) -> Tensor:
        if mask.shape != tokens.shape[:2] or not bool((mask.sum(dim=1) == expected_count).all()):
            raise ValueError("public object-token contract is violated")
        return (tokens * mask.to(dtype=tokens.dtype).unsqueeze(-1)).sum(dim=1) / float(expected_count)

    def _context(self, action: Tensor, reward: Tensor, *, dtype: torch.dtype) -> Tensor:
        if action.ndim != 1 or reward.shape != action.shape or bool((action < 0).any()) or bool((action > START_ACTION).any()):
            raise ValueError("previous public action/reward event is invalid")
        features = torch.nn.functional.one_hot(action, num_classes=ACTION_COUNT + 1).to(dtype=dtype)
        return self.event_context(torch.cat((features, reward.unsqueeze(1).to(dtype=dtype)), dim=1))

    def forward(self, vision: Tensor, previous_actions: Tensor, previous_rewards: Tensor) -> Tensor:
        if vision.ndim != 5 or vision.shape[2:] != (GRID_SIZE, GRID_SIZE, VISION_CHANNELS):
            raise ValueError("expected [batch, time, 11, 11, 7] public visual sequences")
        if previous_actions.shape != vision.shape[:2] or previous_rewards.shape != vision.shape[:2]:
            raise ValueError("public action/reward sequences must align with vision")
        state = self.core.initial_state(vision.shape[0], vision.device)
        for index in range(vision.shape[1]):
            frame = vision[:, index]
            tokens = self._visual_tokens(frame)
            context = self._context(previous_actions[:, index], previous_rewards[:, index], dtype=vision.dtype)
            object_mask = self._object_mask(frame)
            if index == 0:
                positioned_tokens = tokens + self.initial_position.unsqueeze(0)
                event = self._pool(positioned_tokens, object_mask, expected_count=4) + context
                # ``topk`` over a boolean public mask extracts exactly the four
                # visible object locations, never a semantic identity index.
                _, object_indices = object_mask.to(dtype=vision.dtype).topk(4, dim=1)
                write_candidates = positioned_tokens.gather(
                    1, object_indices.unsqueeze(-1).expand(-1, -1, LATENT_DIM)
                )
                write_mask = torch.ones(vision.shape[0], 4, dtype=vision.dtype, device=vision.device)
            elif index == vision.shape[1] - 1:
                event = self._pool(tokens, object_mask, expected_count=1) + context
                write_candidates = torch.zeros(vision.shape[0], 1, LATENT_DIM, dtype=vision.dtype, device=vision.device)
                write_mask = torch.zeros(vision.shape[0], 1, dtype=vision.dtype, device=vision.device)
            else:
                if bool(object_mask.any()):
                    raise ValueError("delay events must not expose an object token")
                event = tokens.mean(dim=1) + context
                write_candidates = torch.zeros(vision.shape[0], 1, LATENT_DIM, dtype=vision.dtype, device=vision.device)
                write_mask = torch.zeros(vision.shape[0], 1, dtype=vision.dtype, device=vision.device)
            if not self.multi_writer:
                weights = torch.softmax(self.candidate_score(write_candidates).squeeze(-1), dim=1)
                write_candidates = torch.einsum("bk,bkd->bd", weights, write_candidates).unsqueeze(1)
                write_mask = write_mask[:, :1]
            state, _ = self.core.step(event, state, write_candidates=write_candidates, write_mask=write_mask)
        return self.core.classify(state).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train(model: ObjectTokenEpisodicMemory, *, steps: int, batch_size: int, seed: int, device: torch.device) -> None:
    if steps < 1:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(steps):
        batch = make_batch(batch_size, generator=generator, device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(batch.vision, batch.previous_actions, batch.previous_rewards), batch.labels.squeeze(1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if hasattr(model, 'core') and hasattr(model.core, 'enforce_key_orthogonality'):
            model.core.enforce_key_orthogonality()


@torch.no_grad()
def accuracy(model: ObjectTokenEpisodicMemory, *, seed: int, device: torch.device, batch_size: int = 2_048) -> float:
    model.eval()
    batch = make_batch(batch_size, generator=torch.Generator(device=device).manual_seed(seed), device=device)
    prediction = model(batch.vision, batch.previous_actions, batch.previous_rewards).sigmoid().ge(0.5)
    return float((prediction == batch.labels.squeeze(1).bool()).float().mean())


def run(
    *, steps: int = DEFAULT_STEPS, batch_size: int = DEFAULT_BATCH_SIZE, train_seed: int = DEFAULT_TRAIN_SEED,
    heldout_seed: int = DEFAULT_HELDOUT_SEED, device: str | torch.device | None = None,
) -> Report:
    if train_seed == heldout_seed:
        raise ValueError("train and held-out generator seeds must differ")
    resolved_device = resolve_device(device)
    torch.manual_seed(train_seed)
    multi = ObjectTokenEpisodicMemory(multi_writer=True, use_memory=True).to(resolved_device)
    single = copy.deepcopy(multi)
    single.multi_writer = False
    memory_disabled = copy.deepcopy(multi)
    memory_disabled.core.use_memory = False
    models = [
        ("object_token_multi_writer", multi),
        ("object_token_single_writer", single),
        ("object_token_memory_disabled", memory_disabled),
    ]
    parameters = [parameter_count(model) for _, model in models]
    if len(set(parameters)) != 1:
        raise RuntimeError("object-token memory ablations must have exactly equal parameter counts")
    results = []
    for name, model in models:
        train(model, steps=steps, batch_size=batch_size, seed=train_seed, device=resolved_device)
        results.append(ModelResult(name, parameter_count(model), accuracy(model, seed=heldout_seed, device=resolved_device)))
    by_name = {result.name: result for result in results}
    multi_accuracy = by_name["object_token_multi_writer"].heldout_accuracy
    accepted = (
        multi_accuracy >= ACCEPTANCE_ACCURACY
        and multi_accuracy - by_name["object_token_single_writer"].heldout_accuracy >= ACCEPTANCE_MARGIN
        and multi_accuracy - by_name["object_token_memory_disabled"].heldout_accuracy >= ACCEPTANCE_MARGIN
    )
    return Report(
        "object_token_episodic_memory_v1: bounded supervised storage/readout diagnostic only; each forward receives only current 11x11x7 public vision, previous chosen action, and previous scalar reward. Object masks derive solely from raw public tile pixels and only distinguish object from air/agent; learned grid-index parameters are internal model weights, not coordinate inputs. Labels, identity IDs, coordinates, seeds, teachers, planners, and shaped rewards are excluded from input and never transfer to Luanti control.",
        str(resolved_device), train_seed, heldout_seed, steps, batch_size, 0.0, results, accepted,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--heldout-seed", type=int, default=DEFAULT_HELDOUT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "object-token-episodic-memory-report.json")
    args = parser.parse_args()
    outcome = run(steps=args.steps, batch_size=args.batch_size, train_seed=args.train_seed, heldout_seed=args.heldout_seed, device=args.device)
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    print(json.dumps(asdict(outcome), indent=2))

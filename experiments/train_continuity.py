"""Train a PLWM on a partially observable cue-recall task.

The task gives one early binary cue, distractor events, and a final query that
contains no cue. A stateless model cannot beat chance; a persistent model must
retain and recover the cue after the interruption point.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.torch_model import StatelessBaseline, TorchPLWM
from plwm.device import resolve_device


INPUT_DIM = 4


def make_batch(batch_size: int, distractors: int, device: torch.device) -> tuple[Tensor, Tensor]:
    labels = torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32)
    steps = distractors + 2
    observations = torch.zeros(batch_size, steps, INPUT_DIM, device=device)
    observations[:, 0, 0] = 1.0  # cue phase
    observations[:, 0, 3] = labels[:, 0] * 2.0 - 1.0
    observations[:, 1:-1, 1] = 1.0  # distractor phase
    observations[:, 1:-1, 3] = torch.randn(batch_size, distractors, device=device)
    observations[:, -1, 2] = 1.0  # query phase, deliberately cue-free
    return observations, labels


@dataclass
class Report:
    plwm_accuracy: float
    baseline_accuracy: float
    checkpoint_exact: bool
    plwm_loss: float
    baseline_loss: float


def accuracy(logits: Tensor, labels: Tensor) -> float:
    return float(((logits.sigmoid() >= 0.5) == (labels >= 0.5)).float().mean())


def train_model(model: nn.Module, *, steps: int, distractors: int, device: torch.device) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    last_loss = 0.0
    for _ in range(steps):
        observations, labels = make_batch(128, distractors, device)
        logits = model(observations)[0] if isinstance(model, TorchPLWM) else model(observations)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return last_loss


@torch.no_grad()
def evaluate(model: nn.Module, *, distractors: int, device: torch.device) -> float:
    model.eval()
    observations, labels = make_batch(4096, distractors, device)
    logits = model(observations)[0] if isinstance(model, TorchPLWM) else model(observations)
    return accuracy(logits, labels)


@torch.no_grad()
def checkpoint_equivalence(model: TorchPLWM, device: torch.device, distractors: int) -> bool:
    observations, _ = make_batch(64, distractors, device)
    state = model.initial_state(observations.shape[0], device)
    split = observations.shape[1] - 1
    for index in range(split):
        state, _ = model.step(observations[:, index], state)
    with TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "runtime-state.pt"
        state.checkpoint(checkpoint_path)
        checkpoint = type(state).restore(checkpoint_path, device)
    uninterrupted, _ = model.step(observations[:, -1], state, think=True)
    resumed, _ = model.step(observations[:, -1], checkpoint, think=True)
    return bool(torch.allclose(model.classify(uninterrupted), model.classify(resumed), atol=1e-7))


def run(steps: int = 350, distractors: int = 5, seed: int = 23, device: str | torch.device | None = None) -> Report:
    torch.manual_seed(seed)
    device = resolve_device(device)
    plwm = TorchPLWM(INPUT_DIM, latent_dim=48, memory_slots=8, thought_steps=3).to(device)
    baseline = StatelessBaseline(INPUT_DIM, hidden_dim=48).to(device)
    plwm_loss = train_model(plwm, steps=steps, distractors=distractors, device=device)
    baseline_loss = train_model(baseline, steps=steps, distractors=distractors, device=device)
    report = Report(
        plwm_accuracy=evaluate(plwm, distractors=distractors, device=device),
        baseline_accuracy=evaluate(baseline, distractors=distractors, device=device),
        checkpoint_exact=checkpoint_equivalence(plwm, device, distractors),
        plwm_loss=plwm_loss,
        baseline_loss=baseline_loss,
    )
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if outcome.plwm_accuracy < 0.9:
        raise SystemExit("PLWM did not reach the feasibility threshold")
    if outcome.baseline_accuracy > 0.65:
        raise SystemExit("stateless baseline unexpectedly exceeded chance")
    if not outcome.checkpoint_exact:
        raise SystemExit("checkpoint recovery changed the prediction")

"""Train and test multi-step imagined rollouts in a hidden-wind environment."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.world_model import RolloutWorldModel


STATES = 5
HORIZON = 3


def transition(state: torch.Tensor, wind: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    direction = action * 2 - 1
    return (state + wind * direction) % STATES


def make_batch(batch_size: int, device: torch.device):
    wind = torch.randint(0, 2, (batch_size,), device=device) * 2 - 1
    start = torch.randint(0, STATES, (batch_size,), device=device)
    actions = torch.randint(0, 2, (batch_size, HORIZON), device=device)
    states = []
    current = start
    for index in range(HORIZON):
        current = transition(current, wind, actions[:, index])
        states.append(current)
    return wind.unsqueeze(1).float(), start, actions, torch.stack(states, dim=1)


def train(model: RolloutWorldModel, *, steps: int, device: torch.device) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    last_loss = 0.0
    model.train()
    for _ in range(steps):
        context, start, actions, targets = make_batch(256, device)
        belief = model.encode_context(context)
        state = torch.nn.functional.one_hot(start, STATES).float()
        loss = 0.0
        for index in range(HORIZON):
            logits = model.predict(belief, state, actions[:, index])
            loss = loss + loss_fn(logits, targets[:, index])
            state = torch.nn.functional.one_hot(targets[:, index], STATES).float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return last_loss


@torch.no_grad()
def evaluate(model: RolloutWorldModel, device: torch.device) -> tuple[float, float]:
    context, start, actions, targets = make_batch(4096, device)
    belief = model.encode_context(context)
    rollout = model.rollout(belief, start, actions)
    transition_accuracy = float((rollout.argmax(dim=-1) == targets).float().mean())
    goal = targets[:, -1]
    plan, _ = model.plan(belief, start, goal, HORIZON)
    actual = start
    wind = context[:, 0].long()
    for index in range(HORIZON):
        actual = transition(actual, wind, plan[:, index])
    return transition_accuracy, float((actual == goal).float().mean())


@dataclass
class Report:
    persistent_transition_accuracy: float
    persistent_rollout_success: float
    stateless_transition_accuracy: float
    stateless_rollout_success: float
    persistent_loss: float
    stateless_loss: float


def run(steps: int = 400, seed: int = 29) -> Report:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    persistent = RolloutWorldModel(use_context=True).to(device)
    stateless = RolloutWorldModel(use_context=False).to(device)
    persistent_loss = train(persistent, steps=steps, device=device)
    stateless_loss = train(stateless, steps=steps, device=device)
    persistent_transition, persistent_success = evaluate(persistent, device)
    stateless_transition, stateless_success = evaluate(stateless, device)
    return Report(
        persistent_transition,
        persistent_success,
        stateless_transition,
        stateless_success,
        persistent_loss,
        stateless_loss,
    )


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if outcome.persistent_transition_accuracy < 0.98 or outcome.persistent_rollout_success < 0.98:
        raise SystemExit("persistent multi-step rollout failed")
    if outcome.stateless_rollout_success > 0.65:
        raise SystemExit("stateless model unexpectedly planned hidden-wind rollouts")

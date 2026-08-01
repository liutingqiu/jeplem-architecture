"""Learn action outcomes, then plan from a persistent latent context.

The first event reveals a hidden wind direction. The later decision event only
contains a goal. Actions have opposite effects under opposite winds. The model
gets supervision for transition prediction, never for the planning action.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.world_model import ActionConditionedPLWM, StatelessWorldModel
from plwm.device import resolve_device


def state_index(signed_state: Tensor) -> Tensor:
    # -1 -> 0, +1 -> 2. State 1 is intentionally unused but keeps a 3-state head.
    return torch.where(signed_state == -1, torch.zeros_like(signed_state), torch.full_like(signed_state, 2))


def make_batch_fixed(batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    wind = torch.randint(0, 2, (batch_size,), device=device) * 2 - 1
    goal = torch.randint(0, 2, (batch_size,), device=device) * 2 - 1
    action = torch.randint(0, 2, (batch_size,), device=device)
    next_state = wind * (action * 2 - 1)
    optimal_action = (goal == wind).long()
    context = torch.stack((wind.float(), torch.zeros_like(wind).float(), torch.zeros_like(wind).float()), dim=-1)
    decision = torch.stack((torch.zeros_like(goal).float(), torch.zeros_like(goal).float(), goal.float()), dim=-1)
    return context, decision, action, state_index(next_state), optimal_action


def train_persistent(model: ActionConditionedPLWM, *, steps: int, device: torch.device) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    last_loss = 0.0
    model.train()
    for _ in range(steps):
        context, decision, action, next_state, _ = make_batch_fixed(256, device)
        state = model.advance(context, model.initial_state(context.shape[0], device))
        state = model.advance(decision, state)
        loss = loss_fn(model.imagine(state, action), next_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return last_loss


def train_stateless(model: StatelessWorldModel, *, steps: int, device: torch.device) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    last_loss = 0.0
    model.train()
    for _ in range(steps):
        _, decision, action, next_state, _ = make_batch_fixed(256, device)
        loss = loss_fn(model(decision, action), next_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return last_loss


@torch.no_grad()
def evaluate_persistent(model: ActionConditionedPLWM, device: torch.device) -> tuple[float, float]:
    context, decision, action, next_state, optimal_action = make_batch_fixed(4096, device)
    state = model.advance(context, model.initial_state(context.shape[0], device))
    state = model.advance(decision, state)
    predicted = model.imagine(state, action).argmax(dim=-1)
    goal_state = state_index(torch.where(decision[:, 2] > 0, torch.ones_like(action), -torch.ones_like(action)))
    planned, _ = model.plan(state, goal_state)
    return float((predicted == next_state).float().mean()), float((planned == optimal_action).float().mean())


@torch.no_grad()
def evaluate_stateless(model: StatelessWorldModel, device: torch.device) -> tuple[float, float]:
    _, decision, action, next_state, optimal_action = make_batch_fixed(4096, device)
    predicted = model(decision, action).argmax(dim=-1)
    goal_state = state_index(torch.where(decision[:, 2] > 0, torch.ones_like(action), -torch.ones_like(action)))
    planned = model.plan(decision, goal_state)
    return float((predicted == next_state).float().mean()), float((planned == optimal_action).float().mean())


@dataclass
class Report:
    persistent_transition_accuracy: float
    persistent_planning_accuracy: float
    stateless_transition_accuracy: float
    stateless_planning_accuracy: float
    persistent_loss: float
    stateless_loss: float


def run(steps: int = 300, seed: int = 13, device: str | torch.device | None = None) -> Report:
    torch.manual_seed(seed)
    device = resolve_device(device)
    persistent = ActionConditionedPLWM(3).to(device)
    stateless = StatelessWorldModel(3).to(device)
    persistent_loss = train_persistent(persistent, steps=steps, device=device)
    stateless_loss = train_stateless(stateless, steps=steps, device=device)
    persistent_transition, persistent_plan = evaluate_persistent(persistent, device)
    stateless_transition, stateless_plan = evaluate_stateless(stateless, device)
    return Report(
        persistent_transition,
        persistent_plan,
        stateless_transition,
        stateless_plan,
        persistent_loss,
        stateless_loss,
    )


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if outcome.persistent_transition_accuracy < 0.95 or outcome.persistent_planning_accuracy < 0.95:
        raise SystemExit("persistent world model did not pass the causal planning gate")
    if outcome.stateless_planning_accuracy > 0.65:
        raise SystemExit("stateless ablation unexpectedly solved hidden-context planning")

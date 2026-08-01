"""Train an action-conditioned model, then exercise it through persistent runtime."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_causal_world import make_batch_fixed, state_index, train_persistent
from plwm.commitments import Commitment
from plwm.device import resolve_device
from plwm.world_model import ActionConditionedPLWM
from plwm.world_runtime import WorldModelPersistentRuntime


@dataclass
class WorldRuntimeResult:
    runtime_planning_accuracy: float
    checkpoint_action_exact: bool
    completion_requires_evidence: bool
    commitment_switches_after_evidence: bool
    mean_selected_utility: float


def run(*, steps: int = 300, seed: int = 47, device: str | torch.device | None = None) -> WorldRuntimeResult:
    torch.manual_seed(seed)
    device = resolve_device(device)
    model = ActionConditionedPLWM(3).to(device)
    train_persistent(model, steps=steps, device=device)
    context, decision, _action, _next_state, optimal = make_batch_fixed(128, device)
    goals = state_index(torch.where(decision[:, 2] > 0, torch.ones_like(optimal), -torch.ones_like(optimal)))
    correct = 0
    utility = []
    checkpoint_exact = True
    evidence_only = True
    with torch.no_grad():
        for index in range(context.shape[0]):
            goal = int(goals[index].item())
            runtime = WorldModelPersistentRuntime(model, belief_states=3)
            runtime.add_commitment(Commitment("goal", target_state=goal, priority=1.0))
            runtime.ingest(context[index].tolist())
            runtime.ingest(decision[index].tolist())
            before_completion = runtime.ledger.commitments[0].status.value
            action, scores = runtime.propose(goal, [0.02, 0.02])
            correct += action == int(optimal[index].item())
            utility.append(scores[action])
            if index == 0:
                restored = WorldModelPersistentRuntime.restore(runtime.snapshot(), model, belief_states=3)
                resumed_action, resumed_scores = restored.propose(goal, [0.02, 0.02])
                checkpoint_exact = action == resumed_action and scores == resumed_scores
            likelihood = [-8.0, -8.0, -8.0]
            likelihood[goal] = 0.0
            runtime.ingest([0.0, 0.0, 0.0], likelihood)
            evidence_only = evidence_only and before_completion == "active" and runtime.ledger.commitments[0].status.value == "completed"
    priority_runtime = WorldModelPersistentRuntime(model, belief_states=3)
    priority_runtime.add_commitment(Commitment("urgent", target_state=0, priority=2.0))
    priority_runtime.add_commitment(Commitment("later", target_state=2, priority=1.0))
    priority_runtime.ingest(context[0].tolist())
    before_id, _before_action, _before_scores = priority_runtime.propose_selected([0.02, 0.02])
    priority_runtime.ingest([0.0, 0.0, 0.0], [0.0, -8.0, -8.0])
    after_id, _after_action, _after_scores = priority_runtime.propose_selected([0.02, 0.02])
    return WorldRuntimeResult(
        correct / len(optimal),
        checkpoint_exact,
        evidence_only,
        before_id == "urgent" and after_id == "later",
        sum(utility) / len(utility),
    )


if __name__ == "__main__":
    print(asdict(run()))

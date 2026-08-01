"""Persistent runtime that uses a trainable action-conditioned world model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import torch

from .belief import CategoricalBelief
from .commitments import Commitment, CommitmentLedger, CommitmentStatus
from .runtime import PersistentRuntime
from .world_model import ActionConditionedPLWM, WorldState


@dataclass
class WorldRuntimeTick:
    selected_commitment: str | None
    belief_confidence: float
    completed_commitments: int


class WorldModelPersistentRuntime:
    """Joins latent dynamics, normalized belief, and auditable commitments."""

    SNAPSHOT_VERSION = 1

    def __init__(self, model: ActionConditionedPLWM, belief_states: int):
        self.model = model
        self.world_state = model.initial_state(1, "cpu")
        self.belief = CategoricalBelief.uniform(1, belief_states)
        self.ledger = CommitmentLedger()

    def add_commitment(self, commitment: Commitment) -> None:
        self.ledger.add(commitment)

    def ingest(self, observation: Sequence[float], log_likelihood: Sequence[float] | None = None) -> WorldRuntimeTick:
        if log_likelihood is not None:
            self.belief = self.belief.observe(torch.tensor([list(log_likelihood)], dtype=torch.float32))
        self.world_state = self.model.advance(torch.tensor([list(observation)], dtype=torch.float32), self.world_state)
        probs = self.belief.probs.squeeze(0).tolist()
        self.ledger.tick()
        self.ledger.reconcile(probs)
        selected = self.ledger.select(probs)
        completed = sum(item.status == CommitmentStatus.COMPLETED for item in self.ledger.commitments)
        return WorldRuntimeTick(selected.id if selected else None, float(self.belief.confidence().item()), completed)

    @torch.no_grad()
    def propose(self, goal_state: int, action_costs: Sequence[float]) -> tuple[int, list[float]]:
        action, utility = self.model.plan_costed(
            self.world_state,
            torch.tensor([goal_state]),
            torch.tensor(list(action_costs), dtype=torch.float32),
        )
        return int(action.item()), utility.squeeze(0).tolist()

    @torch.no_grad()
    def propose_selected(self, action_costs: Sequence[float]) -> tuple[str | None, int | None, list[float] | None]:
        """Plan for the ledger-selected active commitment, if any."""
        selected = self.ledger.select(self.belief.probs.squeeze(0).tolist())
        if selected is None:
            return None, None, None
        action, utility = self.propose(selected.target_state, action_costs)
        return selected.id, action, utility

    def snapshot(self) -> dict:
        snapshot = {
            "version": self.SNAPSHOT_VERSION,
            "world_belief": self.world_state.belief.detach().cpu().tolist(),
            "belief_log_probs": self.belief.log_probs.tolist(),
            "ledger": self.ledger.to_dict(),
        }
        snapshot["integrity"] = PersistentRuntime._integrity(snapshot)
        return snapshot

    @classmethod
    def restore(cls, snapshot: dict, model: ActionConditionedPLWM, belief_states: int) -> "WorldModelPersistentRuntime":
        if snapshot.get("version") != cls.SNAPSHOT_VERSION:
            raise ValueError("unsupported world runtime snapshot version")
        if snapshot.get("integrity") != PersistentRuntime._integrity(snapshot):
            raise ValueError("world runtime snapshot integrity check failed")
        runtime = cls(model, belief_states)
        runtime.world_state = WorldState(torch.tensor(snapshot["world_belief"], dtype=torch.float32))
        runtime.belief = CategoricalBelief(torch.tensor(snapshot["belief_log_probs"], dtype=torch.float32))
        ledger_data = snapshot["ledger"]
        runtime.ledger.time = int(ledger_data["time"])
        runtime.ledger.commitments = [
            Commitment(
                id=item["id"],
                target_state=int(item["target_state"]),
                priority=float(item["priority"]),
                deadline=item.get("deadline"),
                status=CommitmentStatus(item["status"]),
                evidence=list(item.get("evidence", [])),
            )
            for item in ledger_data["commitments"]
        ]
        return runtime

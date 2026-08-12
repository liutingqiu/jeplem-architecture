"""Unified persistent runtime joining latent state, belief, and commitments."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import torch

from .belief import CategoricalBelief
from .commitments import Commitment, CommitmentLedger, CommitmentStatus
from .core import CognitiveAction, PLWM, PLWMConfig, PersistentState, StepResult
from .continual import EndToEndPersistentPLWM


@dataclass
class RuntimeTick:
    action: CognitiveAction
    belief_entropy: float
    belief_confidence: float
    selected_commitment: str | None
    completed_commitments: int
    latent: StepResult


class PersistentRuntime:
    """A small executable contract for PLWM's state layers.

    It intentionally keeps latent state, probabilistic facts, and commitments
    separate. This avoids converting a vector similarity into an unsupported
    claim that an external goal has completed.
    """

    def __init__(self, input_dim: int, belief_states: int, *, latent_dim: int = 16):
        self.core = PLWM(PLWMConfig(input_dim=input_dim, latent_dim=latent_dim))
        self.belief = CategoricalBelief.uniform(1, belief_states)
        self.ledger = CommitmentLedger()

    SNAPSHOT_VERSION = 1

    def add_commitment(self, commitment: Commitment) -> None:
        self.ledger.add(commitment)

    def ingest(self, observation: Sequence[float] | None, log_likelihood: Sequence[float] | None = None) -> RuntimeTick:
        if log_likelihood is not None:
            likelihood = torch.tensor([list(log_likelihood)], dtype=torch.float32)
            self.belief = self.belief.observe(likelihood)
        probs = self.belief.probs.squeeze(0).tolist()
        self.ledger.tick()
        self.ledger.reconcile(probs)
        selected = self.ledger.select(probs)
        entropy = float(self.belief.entropy().item())
        confidence = float(self.belief.confidence().item())
        if observation is None and selected is not None and entropy > 0.2:
            action = CognitiveAction.OBSERVE
        elif selected is not None and entropy > 0.2:
            action = CognitiveAction.THINK
        else:
            action = CognitiveAction.SLEEP
        latent = self.core.step(observation, action=action)
        completed = sum(item.status.value == "completed" for item in self.ledger.commitments)
        return RuntimeTick(action, entropy, confidence, selected.id if selected else None, completed, latent)

    def snapshot(self) -> dict:
        snapshot = {
            "version": self.SNAPSHOT_VERSION,
            "latent_state": self.core.state.to_dict(),
            "belief_log_probs": self.belief.log_probs.tolist(),
            "ledger": self.ledger.to_dict(),
        }
        snapshot["integrity"] = self._integrity(snapshot)
        return snapshot

    @staticmethod
    def _integrity(snapshot: dict) -> str:
        payload = {key: value for key, value in snapshot.items() if key != "integrity"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def restore(cls, snapshot: dict, input_dim: int, belief_states: int, *, latent_dim: int = 16) -> "PersistentRuntime":
        if snapshot.get("version") != cls.SNAPSHOT_VERSION:
            raise ValueError("unsupported persistent runtime snapshot version")
        if snapshot.get("integrity") != cls._integrity(snapshot):
            raise ValueError("persistent runtime snapshot integrity check failed")
        runtime = cls(input_dim, belief_states, latent_dim=latent_dim)
        runtime.core.state = PersistentState.from_dict(snapshot["latent_state"])
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


@dataclass
class TrainableRuntimeTick:
    """Observable result from a trainable runtime tick."""

    action: CognitiveAction
    belief_entropy: float
    belief_confidence: float
    selected_commitment: str | None
    completed_commitments: int
    prediction_error: float


class TrainablePersistentRuntime:
    """Persistent runtime whose latent transition is backed by a trainable PLWM.

    The trainable model proposes latent transitions; normalized belief evidence
    and the commitment ledger remain authoritative for uncertainty and goal
    completion.  The checkpoint stores all runtime layers but deliberately does
    not serialize model weights, which are deployment artifacts rather than
    mutable episode state.
    """

    SNAPSHOT_VERSION = 1

    def __init__(self, model: EndToEndPersistentPLWM, belief_states: int):
        self.model = model
        self.latent = model.core.initial_state(1, "cpu")
        self.belief = CategoricalBelief.uniform(1, belief_states)
        self.ledger = CommitmentLedger()

    def add_commitment(self, commitment: Commitment) -> None:
        self.ledger.add(commitment)

    def ingest(self, observation: Sequence[float], log_likelihood: Sequence[float] | None = None) -> TrainableRuntimeTick:
        if log_likelihood is not None:
            likelihood = torch.tensor([list(log_likelihood)], dtype=torch.float32)
            self.belief = self.belief.observe(likelihood)
        probs = self.belief.probs.squeeze(0).tolist()
        self.ledger.tick()
        self.ledger.reconcile(probs)
        selected = self.ledger.select(probs)
        entropy = float(self.belief.entropy().item())
        confidence = float(self.belief.confidence().item())
        action = CognitiveAction.THINK if selected is not None and entropy > 0.2 else CognitiveAction.SLEEP
        event = torch.tensor([list(observation)], dtype=torch.float32)
        self.latent, surprise = self.model.core.step(event, self.latent, think=action == CognitiveAction.THINK)
        completed = sum(item.status == CommitmentStatus.COMPLETED for item in self.ledger.commitments)
        return TrainableRuntimeTick(action, entropy, confidence, selected.id if selected else None, completed, float(surprise.item()))

    def think_only(self, *, steps: int | None = None) -> TrainableRuntimeTick:
        """Allocate latent compute while preserving external evidence and goals."""
        self.ledger.tick()
        self.latent = self.model.core.think_only(self.latent, steps=steps)
        probs = self.belief.probs.squeeze(0).tolist()
        selected = self.ledger.select(probs)
        completed = sum(item.status == CommitmentStatus.COMPLETED for item in self.ledger.commitments)
        return TrainableRuntimeTick(
            CognitiveAction.THINK,
            float(self.belief.entropy().item()),
            float(self.belief.confidence().item()),
            selected.id if selected else None,
            completed,
            0.0,
        )

    def snapshot(self) -> dict:
        snapshot = {
            "version": self.SNAPSHOT_VERSION,
            "latent": {
                "belief": self.latent.belief.detach().cpu().tolist(),
                "workspace": self.latent.workspace.detach().cpu().tolist(),
                "keys": self.latent.keys.detach().cpu().tolist(),
                "values": self.latent.values.detach().cpu().tolist(),
                "confidences": self.latent.confidences.detach().cpu().tolist(),
            },
            "belief_log_probs": self.belief.log_probs.tolist(),
            "ledger": self.ledger.to_dict(),
        }
        snapshot["integrity"] = PersistentRuntime._integrity(snapshot)
        return snapshot

    @classmethod
    def restore(cls, snapshot: dict, model: EndToEndPersistentPLWM, belief_states: int) -> "TrainablePersistentRuntime":
        if snapshot.get("version") != cls.SNAPSHOT_VERSION:
            raise ValueError("unsupported trainable runtime snapshot version")
        if snapshot.get("integrity") != PersistentRuntime._integrity(snapshot):
            raise ValueError("trainable runtime snapshot integrity check failed")
        runtime = cls(model, belief_states)
        latent = snapshot["latent"]
        runtime.latent = type(runtime.latent)(
            belief=torch.tensor(latent["belief"], dtype=torch.float32),
            workspace=torch.tensor(latent["workspace"], dtype=torch.float32),
            keys=torch.tensor(latent["keys"], dtype=torch.float32),
            values=torch.tensor(latent["values"], dtype=torch.float32),
            confidences=torch.tensor(latent["confidences"], dtype=torch.float32),
        )
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

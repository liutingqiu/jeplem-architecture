"""Persistent commitments: priority, evidence, completion, and cancellation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class CommitmentStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Commitment:
    id: str
    target_state: int
    priority: float
    deadline: int | None = None
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    evidence: list[str] = field(default_factory=list)

    def urgency(self, now: int) -> float:
        if self.deadline is None:
            return 0.0
        return max(0.0, 1.0 - max(0, self.deadline - now) / max(1, self.deadline))


@dataclass
class CommitmentLedger:
    commitments: list[Commitment] = field(default_factory=list)
    time: int = 0

    def add(self, commitment: Commitment) -> None:
        if any(item.id == commitment.id for item in self.commitments):
            raise ValueError(f"duplicate commitment id: {commitment.id}")
        self.commitments.append(commitment)

    def tick(self) -> None:
        self.time += 1

    def reconcile(self, belief_probs: list[float], *, completion_threshold: float = 0.9) -> None:
        for commitment in self.commitments:
            if commitment.status == CommitmentStatus.ACTIVE and belief_probs[commitment.target_state] >= completion_threshold:
                commitment.status = CommitmentStatus.COMPLETED
                commitment.evidence.append(f"belief={belief_probs[commitment.target_state]:.3f} at t={self.time}")

    def cancel(self, commitment_id: str, reason: str) -> None:
        for commitment in self.commitments:
            if commitment.id == commitment_id:
                if commitment.status != CommitmentStatus.ACTIVE:
                    raise ValueError(f"cannot cancel {commitment.status} commitment")
                commitment.status = CommitmentStatus.CANCELLED
                commitment.evidence.append(f"cancelled at t={self.time}: {reason}")
                return
        raise KeyError(commitment_id)

    def select(self, belief_probs: list[float]) -> Commitment | None:
        active = [item for item in self.commitments if item.status == CommitmentStatus.ACTIVE]
        if not active:
            return None
        # Progress is discounted, but not treated as completion without reconcile.
        return max(
            active,
            key=lambda item: item.priority * (1.0 + item.urgency(self.time)) * (1.0 - belief_probs[item.target_state]),
        )

    def to_dict(self) -> dict:
        return {"time": self.time, "commitments": [{**asdict(item), "status": item.status.value} for item in self.commitments]}

"""Freshness and provenance guards for cross-modal observations.

The guard is deliberately small and modality-agnostic.  A visual frame and a
language goal can be represented by the same envelope, but neither is allowed
to cross a world or mechanism revision boundary without being re-observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class VersionedEvidence:
    """One causally scoped item delivered by a sensor or language adapter."""

    world_id: str
    mechanism_version: int
    tick: int
    modality: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.world_id or not isinstance(self.world_id, str):
            raise ValueError("evidence requires a non-empty world_id")
        if isinstance(self.mechanism_version, bool) or self.mechanism_version < 0:
            raise ValueError("mechanism_version must be a non-negative integer")
        if isinstance(self.tick, bool) or self.tick < 0:
            raise ValueError("evidence tick must be a non-negative integer")
        if not self.modality or not isinstance(self.modality, str):
            raise ValueError("evidence requires a modality")
        if not isinstance(self.payload, Mapping):
            raise ValueError("evidence payload must be a mapping")


@dataclass(frozen=True)
class EvidenceDecision:
    accepted: bool
    reason: str


class EvidenceFreshnessGate:
    """Reject stale or cross-world evidence before it reaches a learner.

    ``advance_mechanism`` is called only after a causally verified rule change.
    Timestamps are tracked independently per modality so a visual frame and a
    language constraint from the same tick may both be consumed once.
    """

    def __init__(self, world_id: str, *, mechanism_version: int = 0) -> None:
        if not world_id:
            raise ValueError("world_id is required")
        if mechanism_version < 0:
            raise ValueError("mechanism_version must be non-negative")
        self.world_id = world_id
        self.mechanism_version = mechanism_version
        self._last_tick: dict[str, int] = {}
        self.rejected_stale = 0
        self.accepted_count = 0

    def advance_mechanism(self) -> int:
        self.mechanism_version += 1
        return self.mechanism_version

    def accept(self, evidence: VersionedEvidence) -> EvidenceDecision:
        if evidence.world_id != self.world_id:
            self.rejected_stale += 1
            return EvidenceDecision(False, "world-mismatch")
        if evidence.mechanism_version != self.mechanism_version:
            self.rejected_stale += 1
            return EvidenceDecision(False, "mechanism-version-stale")
        last_tick = self._last_tick.get(evidence.modality, -1)
        if evidence.tick <= last_tick:
            self.rejected_stale += 1
            return EvidenceDecision(False, "tick-stale")
        self._last_tick[evidence.modality] = evidence.tick
        self.accepted_count += 1
        return EvidenceDecision(True, "accepted")


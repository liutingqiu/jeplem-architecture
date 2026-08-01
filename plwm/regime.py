"""Auditable belief over changing environment dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegimeEvidence:
    world_id: str
    regime: int
    confidence: float
    verified: bool
    source: str
    time: int


@dataclass
class RegimeMemory:
    """Keeps a separately versioned causal rule for every world instance."""

    current: dict[str, int] = field(default_factory=dict)
    history: dict[str, list[RegimeEvidence]] = field(default_factory=dict)
    time: int = 0

    def ingest(self, world_id: str, regime: int, *, confidence: float, verified: bool, source: str) -> None:
        if regime not in (-1, 1):
            raise ValueError("regime must be -1 or 1")
        self.time += 1
        evidence = RegimeEvidence(world_id, regime, max(0.0, min(1.0, confidence)), verified, source, self.time)
        self.history.setdefault(world_id, []).append(evidence)
        if verified and evidence.confidence >= 0.85:
            self.current[world_id] = regime

    def regime(self, world_id: str) -> int | None:
        return self.current.get(world_id)

    def plan_action(self, world_id: str, goal: int) -> int | None:
        """Select action 0/1 for a signed goal under the verified local rule."""
        regime = self.regime(world_id)
        if regime is None or goal not in (-1, 1):
            return None
        return int(goal == regime)


@dataclass
class LastObservationRegime:
    """Weak ablation that overwrites causal state with every observation."""

    current: dict[str, int] = field(default_factory=dict)

    def ingest(self, world_id: str, regime: int, **_: object) -> None:
        self.current[world_id] = regime

    def regime(self, world_id: str) -> int | None:
        return self.current.get(world_id)

    def plan_action(self, world_id: str, goal: int) -> int | None:
        regime = self.regime(world_id)
        return None if regime is None else int(goal == regime)

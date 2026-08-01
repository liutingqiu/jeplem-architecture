"""Versioned episodic memory for lifelong, non-stationary fact learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class FactEvidence:
    key: int
    value: int
    confidence: float
    source: str
    timestamp: int
    verified: bool = False


@dataclass
class VersionedFactMemory:
    """Memory that separates current belief, provenance, and untrusted candidates."""

    current: dict[int, int] = field(default_factory=dict)
    history: dict[int, list[FactEvidence]] = field(default_factory=dict)
    pending: list[FactEvidence] = field(default_factory=list)
    time: int = 0
    consolidation_count: int = 0

    def ingest(self, key: int, value: int, *, confidence: float, source: str, verified: bool = False) -> None:
        self.time += 1
        evidence = FactEvidence(key, value, min(1.0, max(0.0, confidence)), source, self.time, verified)
        self.history.setdefault(key, []).append(evidence)
        if verified and confidence >= 0.85:
            self.current[key] = value
        else:
            self.pending.append(evidence)

    def sleep(self, *, replay_limit: int = 32) -> None:
        """Consolidate only repeated, mutually consistent pending evidence."""
        grouped: dict[tuple[int, int], list[FactEvidence]] = {}
        for evidence in self.pending[-replay_limit:]:
            grouped.setdefault((evidence.key, evidence.value), []).append(evidence)
        for (key, value), records in grouped.items():
            if len(records) >= 3 and sum(item.confidence for item in records) / len(records) >= 0.8:
                self.current[key] = value
                self.consolidation_count += 1
        self.pending.clear()

    def retrieve(self, key: int) -> int | None:
        return self.current.get(key)

    def provenance(self, key: int) -> list[FactEvidence]:
        return list(self.history.get(key, []))

    def snapshot(self) -> dict:
        return {
            "current": {str(key): value for key, value in self.current.items()},
            "history": {str(key): [asdict(item) for item in records] for key, records in self.history.items()},
            "pending": [asdict(item) for item in self.pending],
            "time": self.time,
            "consolidation_count": self.consolidation_count,
        }


@dataclass
class LastWriteMemory:
    """A deliberately weak ablation: every observation overwrites the fact."""

    current: dict[int, int] = field(default_factory=dict)

    def ingest(self, key: int, value: int, **_: object) -> None:
        self.current[key] = value

    def sleep(self, **_: object) -> None:
        return None

    def retrieve(self, key: int) -> int | None:
        return self.current.get(key)

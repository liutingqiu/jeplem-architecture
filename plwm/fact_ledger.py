"""Versioned, source-bound fact storage with weak-evidence isolation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FactRecord:
    key: str
    value: str | int | bool
    source: str
    tick: int
    confidence: float
    version: int
    request_id: str | None = None


class VersionedFactLedger:
    """Accept strong newer evidence; isolate weak or stale conflicts.

    This is an auditable memory interface, not a learned controller.  It makes
    factual ownership and revision explicit so downstream policies cannot
    silently treat a weak observation as a durable truth.
    """

    def __init__(self, *, trusted_sources: set[str], revision_confidence: float = 0.9) -> None:
        if not trusted_sources or not all(isinstance(item, str) and item for item in trusted_sources):
            raise ValueError("trusted_sources must be non-empty strings")
        if not 0.0 < revision_confidence <= 1.0:
            raise ValueError("revision_confidence must be in (0, 1]")
        self.trusted_sources = frozenset(trusted_sources)
        self.revision_confidence = revision_confidence
        self.current: dict[str, FactRecord] = {}
        self.history: list[FactRecord] = []
        self.isolated: list[FactRecord] = []

    def observe(self, record: FactRecord) -> str:
        if not record.key or not record.source or record.tick < 0 or not 0.0 <= record.confidence <= 1.0 or record.version < 1:
            raise ValueError("invalid fact record")
        existing = self.current.get(record.key)
        strong = record.source in self.trusted_sources and record.confidence >= self.revision_confidence
        if existing is None:
            if strong:
                self.current[record.key] = record
                self.history.append(record)
                return "accepted"
            self.isolated.append(record)
            return "isolated-weak"
        if record.tick <= existing.tick:
            self.isolated.append(record)
            return "isolated-stale"
        if record.value == existing.value:
            if strong:
                self.current[record.key] = record
                self.history.append(record)
                return "refreshed"
            self.isolated.append(record)
            return "isolated-weak"
        if not strong:
            self.isolated.append(record)
            return "isolated-conflict"
        revised = FactRecord(record.key, record.value, record.source, record.tick, record.confidence, existing.version + 1, record.request_id)
        self.current[record.key] = revised
        self.history.append(revised)
        return "revised"

    def snapshot(self) -> dict[str, Any]:
        return {
            "trusted_sources": sorted(self.trusted_sources),
            "revision_confidence": self.revision_confidence,
            "current": {key: asdict(value) for key, value in self.current.items()},
            "history": [asdict(value) for value in self.history],
            "isolated": [asdict(value) for value in self.isolated],
        }

    @staticmethod
    def _record(payload: Mapping[str, Any]) -> FactRecord:
        key, value, source = payload.get("key"), payload.get("value"), payload.get("source")
        tick, confidence, version, request_id = payload.get("tick"), payload.get("confidence"), payload.get("version"), payload.get("request_id")
        if (not isinstance(key, str) or not key or not isinstance(value, (str, int, bool)) or isinstance(value, float)
                or not isinstance(source, str) or not source or isinstance(tick, bool) or not isinstance(tick, int)
                or tick < 0 or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0 or isinstance(version, bool) or not isinstance(version, int)
                or version < 1 or request_id is not None and not isinstance(request_id, str)):
            raise ValueError("invalid fact ledger snapshot record")
        return FactRecord(key, value, source, tick, float(confidence), version, request_id)

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any]) -> "VersionedFactLedger":
        sources, confidence = snapshot.get("trusted_sources"), snapshot.get("revision_confidence")
        if not isinstance(sources, list) or isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("invalid fact ledger snapshot")
        ledger = cls(trusted_sources=set(sources), revision_confidence=float(confidence))
        current, history, isolated = snapshot.get("current"), snapshot.get("history"), snapshot.get("isolated")
        if not isinstance(current, Mapping) or not isinstance(history, list) or not isinstance(isolated, list):
            raise ValueError("invalid fact ledger snapshot")
        ledger.current = {str(key): cls._record(value) for key, value in current.items() if isinstance(value, Mapping)}
        if len(ledger.current) != len(current):
            raise ValueError("invalid current fact ledger records")
        ledger.history = [cls._record(value) for value in history if isinstance(value, Mapping)]
        ledger.isolated = [cls._record(value) for value in isolated if isinstance(value, Mapping)]
        if len(ledger.history) != len(history) or len(ledger.isolated) != len(isolated):
            raise ValueError("invalid fact ledger history")
        # A restored snapshot must obey the same ownership guarantees as an
        # online ledger. Otherwise a tampered ``current`` map could smuggle a
        # fact under a different key or bypass the trusted-evidence history.
        if any(key != record.key for key, record in ledger.current.items()):
            raise ValueError("current fact key does not match its record")
        previous: dict[str, FactRecord] = {}
        for record in ledger.history:
            if record.source not in ledger.trusted_sources or record.confidence < ledger.revision_confidence:
                raise ValueError("untrusted record in fact ledger history")
            earlier = previous.get(record.key)
            if earlier is not None:
                if record.tick <= earlier.tick or record.version < earlier.version:
                    raise ValueError("non-monotonic fact ledger history")
                if record.value != earlier.value and record.version != earlier.version + 1:
                    raise ValueError("invalid fact ledger revision version")
            previous[record.key] = record
        if set(ledger.current) != set(previous):
            raise ValueError("current fact ledger records do not match history")
        if any(ledger.current[key] != record for key, record in previous.items()):
            raise ValueError("current fact ledger records are not latest history")
        return ledger


class LuantiRewardFactLedger:
    """Translate verified Luanti bridge reward envelopes into ledger facts.

    The adapter deliberately recognizes only the existing bounded prerequisite
    reward semantics.  It never inspects the Luanti task seed, collection
    fields, goal state, or planner output.
    """

    def __init__(self) -> None:
        self.ledger = VersionedFactLedger(trusted_sources={"luanti-reward"})

    def observe_reward(self, reward: Mapping[str, Any] | None) -> str | None:
        if reward is None:
            return None
        value = reward.get("value")
        envelope = reward.get("_plwm_provenance")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        if not isinstance(envelope, Mapping):
            raise ValueError("verified Luanti fact reward requires provenance")
        source, tick, request_id = envelope.get("source"), envelope.get("tick"), envelope.get("request_id")
        if (source != "luanti-reward" or isinstance(tick, bool) or not isinstance(tick, int) or tick < 0
                or not isinstance(request_id, str) or not request_id):
            raise ValueError("invalid Luanti reward fact provenance")
        if value >= 0.19:
            return self.ledger.observe(FactRecord("key_verified", True, source, tick, 1.0, 1, request_id))
        if 0.145 <= value < 0.185:
            return self.ledger.observe(FactRecord("charge_verified", True, source, tick, 1.0, 1, request_id))
        return None

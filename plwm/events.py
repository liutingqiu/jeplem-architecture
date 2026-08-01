"""Modality-neutral event protocol for persistent environment interaction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping


EventKind = Literal["observation", "action", "outcome", "reward", "commitment", "checkpoint"]


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class EnvironmentEvent:
    """One causally ordered environment/runtime interaction event.

    ``modality`` describes the source representation, while payload stays
    structured. The cognitive core receives encoded tensors later; raw content
    remains in the trace for replay, audit, and future multimodal encoders.
    """

    episode_id: str
    index: int
    kind: EventKind
    modality: str
    payload: Mapping[str, Any]
    previous_hash: str
    event_hash: str

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        index: int,
        kind: EventKind,
        modality: str,
        payload: Mapping[str, Any],
        previous_hash: str,
    ) -> "EnvironmentEvent":
        unsigned = {
            "episode_id": episode_id,
            "index": index,
            "kind": kind,
            "modality": modality,
            "payload": dict(payload),
            "previous_hash": previous_hash,
        }
        return cls(**unsigned, event_hash=hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest())

    def verify(self) -> bool:
        unsigned = {
            "episode_id": self.episode_id,
            "index": self.index,
            "kind": self.kind,
            "modality": self.modality,
            "payload": dict(self.payload),
            "previous_hash": self.previous_hash,
        }
        return self.event_hash == hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()


@dataclass
class EpisodeTrace:
    """Append-only hash-chained trace shared by simulators and game bridges."""

    episode_id: str
    events: list[EnvironmentEvent] = field(default_factory=list)

    GENESIS_HASH = "0" * 64

    def append(self, kind: EventKind, modality: str, payload: Mapping[str, Any]) -> EnvironmentEvent:
        event = EnvironmentEvent.create(
            episode_id=self.episode_id,
            index=len(self.events),
            kind=kind,
            modality=modality,
            payload=payload,
            previous_hash=self.events[-1].event_hash if self.events else self.GENESIS_HASH,
        )
        self.events.append(event)
        return event

    def verify(self) -> None:
        previous = self.GENESIS_HASH
        for index, event in enumerate(self.events):
            if event.episode_id != self.episode_id or event.index != index:
                raise ValueError("episode trace ordering mismatch")
            if event.previous_hash != previous or not event.verify():
                raise ValueError("episode trace integrity check failed")
            previous = event.event_hash

    def write_jsonl(self, path: str | Path) -> None:
        self.verify()
        with Path(path).open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(_canonical(asdict(event)) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "EpisodeTrace":
        events: list[EnvironmentEvent] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(EnvironmentEvent(**json.loads(line)))
        if not events:
            raise ValueError("episode trace cannot be empty")
        trace = cls(events[0].episode_id, events)
        trace.verify()
        return trace


def observation_payload(*, vector: Iterable[float] | None = None, text: str | None = None, image_ref: str | None = None, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create an observation payload without conflating raw modalities."""
    payload: dict[str, Any] = {}
    if vector is not None:
        payload["vector"] = [float(value) for value in vector]
    if text is not None:
        payload["text"] = text
    if image_ref is not None:
        payload["image_ref"] = image_ref
    if state is not None:
        payload["state"] = dict(state)
    if not payload:
        raise ValueError("observation requires at least one modality payload")
    return payload

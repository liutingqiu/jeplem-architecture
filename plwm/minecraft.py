"""Auditable local file bridge between the Fabric mod and PLWM runtime."""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .events import EpisodeTrace, observation_payload
from .runtime import TrainablePersistentRuntime, TrainableRuntimeTick


@dataclass
class MinecraftFileBridge:
    """Reads Fabric bridge records and writes only allowlisted local actions."""

    root: Path

    ACTIONS = frozenset({"observe", "set_episode", "reset_episode"})
    EVENT_KINDS = frozenset({"observation", "outcome", "reward", "commitment", "checkpoint"})
    REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def observations_path(self) -> Path:
        return self.root / "observations.jsonl"

    @property
    def actions_path(self) -> Path:
        return self.root / "actions.jsonl"

    def write_action(
        self,
        name: str,
        *,
        episode_id: str | None = None,
        request_id: str | None = None,
    ) -> str:
        """Append one capability-limited action and return its stable request ID."""
        if name not in self.ACTIONS:
            raise ValueError(f"action is not allowed by the Minecraft bridge: {name}")
        request_id = request_id or uuid.uuid4().hex
        if not self.REQUEST_ID_PATTERN.fullmatch(request_id):
            raise ValueError("request_id must be 1-128 ASCII identifier characters")
        payload: dict[str, Any] = {"name": name}
        payload["request_id"] = request_id
        if name == "set_episode":
            if not episode_id:
                raise ValueError("set_episode requires a nonempty episode_id")
            payload["episode_id"] = episode_id
        self.root.mkdir(parents=True, exist_ok=True)
        with self.actions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        return request_id

    def read_episode(self, episode_id: str) -> EpisodeTrace:
        if not self.observations_path.exists():
            raise FileNotFoundError(self.observations_path)
        trace = EpisodeTrace(episode_id)
        with self.observations_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("episode_id") != episode_id:
                    continue
                kind = record.get("kind")
                modality = record.get("modality")
                payload = record.get("payload")
                if kind not in self.EVENT_KINDS or not isinstance(modality, str) or not isinstance(payload, Mapping):
                    raise ValueError("invalid Fabric bridge record")
                trace.append(kind, modality, payload)
        if not trace.events:
            raise ValueError(f"no events found for episode {episode_id}")
        trace.verify()
        return trace


class MinecraftStateEncoder:
    """Stable five-value starter encoding for trainable runtime experiments."""

    DIMENSION = 5

    @classmethod
    def encode(cls, payload: Mapping[str, Any]) -> list[float]:
        health = float(payload.get("health", 0.0)) / 20.0
        food = float(payload.get("food", 0.0)) / 20.0
        x = float(payload.get("x", 0.0)) / 256.0
        z = float(payload.get("z", 0.0)) / 256.0
        time = float(payload.get("game_time", 0.0)) * (2.0 * math.pi / 24000.0)
        return [health, food, x + math.sin(time), z + math.cos(time), 1.0]


@dataclass
class MinecraftRuntimeSession:
    """Connects verified game observations to one persistent PLWM episode."""

    bridge: MinecraftFileBridge
    runtime: TrainablePersistentRuntime
    episode_id: str
    trace: EpisodeTrace = field(init=False)
    pending_action_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.trace = EpisodeTrace(self.episode_id)

    def begin(self) -> str:
        request_id = self.bridge.write_action("set_episode", episode_id=self.episode_id)
        self._record_action(request_id, "set_episode", episode_id=self.episode_id)
        return request_id

    def request_observation(self) -> str:
        request_id = self.bridge.write_action("observe")
        self._record_action(request_id, "observe")
        return request_id

    def reset_episode(self) -> str:
        request_id = self.bridge.write_action("reset_episode")
        self._record_action(request_id, "reset_episode")
        return request_id

    def ingest_outcome(self, payload: Mapping[str, Any]) -> None:
        """Accept only a terminal Fabric outcome linked to this live session."""
        request_id = payload.get("request_id")
        status = payload.get("status")
        if not isinstance(request_id, str) or request_id not in self.pending_action_ids:
            raise ValueError("outcome does not match a pending Minecraft action")
        if status not in {"accepted", "rejected"}:
            raise ValueError("Minecraft action outcome must be accepted or rejected")
        self.pending_action_ids.remove(request_id)
        self.trace.append("outcome", "minecraft-bridge", dict(payload))

    def _record_action(self, request_id: str, name: str, *, episode_id: str | None = None) -> None:
        payload: dict[str, Any] = {"name": name, "request_id": request_id}
        if episode_id is not None:
            payload["episode_id"] = episode_id
        self.pending_action_ids.add(request_id)
        self.trace.append("action", "minecraft-action", payload)

    def ingest_observation(self, payload: Mapping[str, Any]) -> TrainableRuntimeTick:
        self.trace.append("observation", "minecraft-state", observation_payload(state=payload))
        return self.runtime.ingest(MinecraftStateEncoder.encode(payload))

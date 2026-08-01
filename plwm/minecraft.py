"""Auditable local file bridge between the Fabric mod and PLWM runtime."""

from __future__ import annotations

import json
import math
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

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def observations_path(self) -> Path:
        return self.root / "observations.jsonl"

    @property
    def actions_path(self) -> Path:
        return self.root / "actions.jsonl"

    def write_action(self, name: str, *, episode_id: str | None = None) -> None:
        if name not in self.ACTIONS:
            raise ValueError(f"action is not allowed by the Minecraft bridge: {name}")
        payload: dict[str, Any] = {"name": name}
        if name == "set_episode":
            if not episode_id:
                raise ValueError("set_episode requires a nonempty episode_id")
            payload["episode_id"] = episode_id
        self.root.mkdir(parents=True, exist_ok=True)
        with self.actions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")

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

    def __post_init__(self) -> None:
        self.trace = EpisodeTrace(self.episode_id)

    def begin(self) -> None:
        self.bridge.write_action("set_episode", episode_id=self.episode_id)
        self.trace.append("action", "minecraft-action", {"name": "set_episode", "episode_id": self.episode_id})

    def request_observation(self) -> None:
        self.bridge.write_action("observe")
        self.trace.append("action", "minecraft-action", {"name": "observe"})

    def ingest_observation(self, payload: Mapping[str, Any]) -> TrainableRuntimeTick:
        self.trace.append("observation", "minecraft-state", observation_payload(state=payload))
        return self.runtime.ingest(MinecraftStateEncoder.encode(payload))

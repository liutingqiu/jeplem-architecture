"""Auditable local file bridge between a Luanti training world and PLWM."""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .events import EpisodeTrace, observation_payload
from .runtime import TrainablePersistentRuntime, TrainableRuntimeTick


@dataclass
class LuantiFileBridge:
    """Reads Luanti records and writes only capability-limited local actions."""

    root: Path

    ACTIONS = frozenset({"observe", "set_episode", "reset_episode", "recenter", "step"})
    EVENT_KINDS = frozenset({"observation", "outcome", "reward", "commitment", "checkpoint"})
    DIRECTIONS = frozenset({"north", "south", "east", "west"})
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
        task_seed: int | None = None,
        direction: str | None = None,
        request_id: str | None = None,
    ) -> str:
        """Append an allowlisted action and return its stable request ID."""
        if name not in self.ACTIONS:
            raise ValueError(f"action is not allowed by the Luanti bridge: {name}")
        if name == "set_episode" and not episode_id:
            raise ValueError("set_episode requires a nonempty episode_id")
        if task_seed is not None and (
            name != "set_episode"
            or isinstance(task_seed, bool)
            or not isinstance(task_seed, int)
            or not 0 <= task_seed < 2_147_483_647
        ):
            raise ValueError("task_seed is a 31-bit nonnegative integer for set_episode only")
        if name == "step" and direction not in self.DIRECTIONS:
            raise ValueError("step requires one of north, south, east, or west")
        if name != "step" and direction is not None:
            raise ValueError("direction is only valid for step")

        request_id = request_id or uuid.uuid4().hex
        if not self.REQUEST_ID_PATTERN.fullmatch(request_id):
            raise ValueError("request_id must be 1-128 ASCII identifier characters")

        payload: dict[str, Any] = {"name": name, "request_id": request_id}
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if task_seed is not None:
            payload["task_seed"] = task_seed
        if direction is not None:
            payload["direction"] = direction
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
                    raise ValueError("invalid Luanti bridge record")
                trace.append(kind, modality, payload)
        if not trace.events:
            raise ValueError(f"no events found for episode {episode_id}")
        trace.verify()
        return trace


class LuantiStateEncoder:
    """Stable starter encoding for the virtual Luanti training body."""

    DIMENSION = 5

    @classmethod
    def encode(cls, payload: Mapping[str, Any]) -> list[float]:
        health = float(payload.get("health", 0.0)) / 20.0
        energy = float(payload.get("energy", 0.0)) / 20.0
        x = float(payload.get("x", 0.0)) / 32.0
        z = float(payload.get("z", 0.0)) / 32.0
        time = float(payload.get("game_time", 0.0)) * (2.0 * math.pi / 24000.0)
        return [health, energy, x + math.sin(time), z + math.cos(time), 1.0]


@dataclass
class LuantiRuntimeSession:
    """Connects verified Luanti observations to one persistent PLWM episode."""

    bridge: LuantiFileBridge
    runtime: TrainablePersistentRuntime
    episode_id: str
    task_seed: int = 0
    trace: EpisodeTrace = field(init=False)
    pending_action_ids: set[str] = field(default_factory=set, init=False)
    acknowledged_action_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.trace = EpisodeTrace(self.episode_id)

    def begin(self) -> str:
        request_id = self.bridge.write_action("set_episode", episode_id=self.episode_id, task_seed=self.task_seed)
        self._record_action(request_id, "set_episode", episode_id=self.episode_id, task_seed=self.task_seed)
        return request_id

    def request_observation(self) -> str:
        request_id = self.bridge.write_action("observe")
        self._record_action(request_id, "observe")
        return request_id

    def step(self, direction: str) -> str:
        request_id = self.bridge.write_action("step", direction=direction)
        self._record_action(request_id, "step", direction=direction)
        return request_id

    def reset_episode(self) -> str:
        request_id = self.bridge.write_action("reset_episode")
        self._record_action(request_id, "reset_episode")
        return request_id

    def recenter(self) -> str:
        request_id = self.bridge.write_action("recenter")
        self._record_action(request_id, "recenter")
        return request_id

    def ingest_outcome(self, payload: Mapping[str, Any]) -> None:
        """Accept only a terminal outcome linked to this live session."""
        request_id = payload.get("request_id")
        status = payload.get("status")
        if not isinstance(request_id, str) or request_id not in self.pending_action_ids:
            raise ValueError("outcome does not match a pending Luanti action")
        if status not in {"accepted", "rejected"}:
            raise ValueError("Luanti action outcome must be accepted or rejected")
        self.pending_action_ids.remove(request_id)
        self.acknowledged_action_ids.add(request_id)
        self.trace.append("outcome", "luanti-bridge", dict(payload))

    def _record_action(
        self,
        request_id: str,
        name: str,
        *,
        episode_id: str | None = None,
        task_seed: int | None = None,
        direction: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"name": name, "request_id": request_id}
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if task_seed is not None:
            payload["task_seed"] = task_seed
        if direction is not None:
            payload["direction"] = direction
        self.pending_action_ids.add(request_id)
        self.trace.append("action", "luanti-action", payload)

    def ingest_observation(self, payload: Mapping[str, Any]) -> TrainableRuntimeTick:
        caused_by = payload.get("caused_by")
        if caused_by is not None and (
            not isinstance(caused_by, str) or caused_by not in self.acknowledged_action_ids
        ):
            raise ValueError("observation claims an unknown Luanti action cause")
        self.trace.append("observation", "luanti-state", observation_payload(state=payload))
        return self.runtime.ingest(LuantiStateEncoder.encode(payload))

    def ingest_reward(self, payload: Mapping[str, Any]) -> None:
        """Record only a numeric reward causally linked to an acknowledged action."""
        caused_by = payload.get("caused_by")
        value = payload.get("value")
        if (
            not isinstance(caused_by, str)
            or caused_by not in self.acknowledged_action_ids
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError("reward must be numeric and linked to an acknowledged Luanti action")
        self.trace.append("reward", "luanti-reward", dict(payload))

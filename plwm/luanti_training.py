"""Real-world rollout primitives for the capability-limited Luanti bridge."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .luanti import LuantiFileBridge
from .torch_model import TorchCognitiveState, TorchPLWM


DIRECTIONS = ("north", "south", "east", "west")
DIRECTION_TO_INDEX = {direction: index for index, direction in enumerate(DIRECTIONS)}
VISION_TILES = ("air", "wall", "agent", "goal", "key", "decoy", "beacon")
VISION_WIDTH = 11
VISION_FEATURE_DIM = VISION_WIDTH * VISION_WIDTH * len(VISION_TILES)
DELTAS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


def _next_seed(value: int) -> int:
    return (value * 48271) % 2_147_483_647


def gate_x(task_seed: int) -> int:
    """Reproduce the Lua generator for the teacher and held-out evaluator only."""
    return -3 if _next_seed(task_seed + 13) % 2 == 0 else 3


def controls_inverted(task_seed: int) -> bool:
    """The hidden control rule is never placed in the model observation."""
    return (_next_seed(task_seed + 29) // 65_536) % 2 == 1


def is_blocked(task_seed: int, x: int, z: int) -> bool:
    return abs(x) >= 8 or abs(z) >= 8 or (z == 0 and x != gate_x(task_seed))


def encode_vision(payload: Mapping[str, Any]) -> tuple[float, ...]:
    """Validate and one-hot encode a local Luanti visual observation."""
    if payload.get("encoding") != "tile-grid-v1":
        raise ValueError("unsupported Luanti vision encoding")
    width, height, cells = payload.get("width"), payload.get("height"), payload.get("cells")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width < 1
        or height < 1
        or not isinstance(cells, list)
        or len(cells) != width * height
    ):
        raise ValueError("invalid Luanti vision grid dimensions")
    if any(cell not in VISION_TILES for cell in cells):
        raise ValueError("Luanti vision grid contains an unknown tile")
    if cells.count("agent") != 1:
        raise ValueError("Luanti vision grid must contain exactly one agent tile")
    return tuple(float(cell == tile) for cell in cells for tile in VISION_TILES)


def visual_frame_features(state: LuantiState, vision: Mapping[str, Any]) -> tuple[float, ...]:
    """Combine non-local symbolic goal context with visual local geometry."""
    symbolic = list(state.features())
    symbolic[7:11] = [0.0, 0.0, 0.0, 0.0]
    return tuple(symbolic) + encode_vision(vision)


def object_visual_frame_features(state: ObjectTaskState, vision: Mapping[str, Any]) -> tuple[float, ...]:
    """Object-task symbolic context plus local visual geometry."""
    symbolic = list(state.features())
    symbolic[7:11] = [0.0, 0.0, 0.0, 0.0]
    return tuple(symbolic) + encode_vision(vision)


def object_reward_visual_frame_features(state: ObjectTaskState, vision: Mapping[str, Any]) -> tuple[float, ...]:
    symbolic = list(state.memory_features())
    symbolic[7:11] = [0.0, 0.0, 0.0, 0.0]
    return tuple(symbolic) + encode_vision(vision)


def object_fact_memory_visual_frame_features(
    state: ObjectTaskState,
    vision: Mapping[str, Any],
    *,
    verified_key: bool,
) -> tuple[float, ...]:
    return object_reward_visual_frame_features(state, vision)[: ObjectGoalDirectedPLWM.input_dim + 1] + (float(verified_key),) + encode_vision(vision)


def object_two_fact_memory_visual_frame_features(
    state: ObjectTaskState,
    vision: Mapping[str, Any],
    *,
    verified_key: bool,
    verified_charge: bool,
) -> tuple[float, ...]:
    """Expose only externally verified charge/key facts to a learned head."""
    return (
        object_reward_visual_frame_features(state, vision)[: ObjectGoalDirectedPLWM.input_dim + 1]
        + (float(verified_key), float(verified_charge))
        + encode_vision(vision)
    )


def object_polarity_fact_memory_visual_frame_features(
    state: ObjectTaskState,
    vision: Mapping[str, Any],
    *,
    verified_key: bool,
    charge_mode: str | None,
) -> tuple[float, ...]:
    """Encode a reward-verified polarity without exposing a world-side answer."""
    if charge_mode not in {None, "key", "decoy"}:
        raise ValueError("unsupported verified charge mode")
    return (
        object_reward_visual_frame_features(state, vision)[: ObjectGoalDirectedPLWM.input_dim + 1]
        + (float(verified_key), float(charge_mode == "key"), float(charge_mode == "decoy"))
        + encode_vision(vision)
    )


def _multi_object_coordinates(state: ObjectTaskState) -> tuple[float, ...]:
    if len(state.candidate_positions) != 4:
        raise ValueError("multi-object curriculum requires exactly four public candidates")
    return tuple(coordinate / 7.0 for candidate in state.candidate_positions for coordinate in candidate)


def object_multi_choice_public_visual_frame_features(
    state: ObjectTaskState, vision: Mapping[str, Any]
) -> tuple[float, ...]:
    """Public event evidence for every multi-object baseline."""
    return object_reward_visual_frame_features(state, vision)[: ObjectGoalDirectedPLWM.input_dim + 1] + _multi_object_coordinates(state) + encode_vision(vision)


def object_multi_choice_fact_memory_visual_frame_features(
    state: ObjectTaskState,
    vision: Mapping[str, Any],
    *,
    verified_key: bool,
    charge_choice: int | None,
) -> tuple[float, ...]:
    """Add only the reward-verified four-way choice to public evidence."""
    if charge_choice not in {None, 1, 2, 3, 4}:
        raise ValueError("unsupported verified multi-object choice")
    return (
        object_reward_visual_frame_features(state, vision)[: ObjectGoalDirectedPLWM.input_dim + 1]
        + _multi_object_coordinates(state)
        + (float(verified_key), *[float(charge_choice == choice) for choice in range(1, 5)])
        + encode_vision(vision)
    )


def reconcile_visual_sensors(state: LuantiState, vision: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """Repair only the four local visual cells covered by fresh sensors.

    This is a cross-modal consistency operation, not a map oracle: it may read
    the public, immediate blocked sensors but has no access to the task seed,
    gate location, future state, or control mapping.  All other visual cells
    remain untouched, so the visual task still requires seeing the distant gap.
    """
    encode_vision(vision)
    payload = dict(vision)
    cells = list(payload["cells"])
    center = (int(payload["width"]) // 2, int(payload["height"]) // 2)
    offsets = {
        "north": (0, -1),
        "south": (0, 1),
        "east": (1, 0),
        "west": (-1, 0),
    }
    corrections = 0
    for direction, (delta_x, delta_z) in offsets.items():
        x, z = center[0] + delta_x, center[1] + delta_z
        index = z * int(payload["width"]) + x
        expected = "wall" if _sensor_is_blocked(state, direction) else "air"
        if cells[index] != expected:
            cells[index] = expected
            corrections += 1
    payload["cells"] = cells
    return payload, corrections


def temporal_consensus_vision(
    previous_state: LuantiState,
    previous_vision: Mapping[str, Any],
    current_state: LuantiState,
    current_vision: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """Fuse a fresh frame with the action-result-aligned preceding frame.

    A cell from the preceding view is projected into the current egocentric
    grid using the *observed* body displacement.  Where a non-agent cell
    conflicts, the predicted cell is provisionally retained and the number of
    such low-confidence conflicts is returned.  No task generator property or
    action request is consulted; the physical outcome is the sole alignment
    signal.  The caller must still enforce evidence freshness before invoking
    this function.
    """
    encode_vision(previous_vision)
    encode_vision(current_vision)
    if previous_vision["width"] != current_vision["width"] or previous_vision["height"] != current_vision["height"]:
        raise ValueError("temporal vision frames must share dimensions")
    width, height = int(current_vision["width"]), int(current_vision["height"])
    shift_x = current_state.x - previous_state.x
    shift_z = current_state.z - previous_state.z
    previous_cells = list(previous_vision["cells"])
    fused = list(current_vision["cells"])
    conflicts = 0
    for z in range(height):
        for x in range(width):
            previous_x, previous_z = x + shift_x, z + shift_z
            if not (0 <= previous_x < width and 0 <= previous_z < height):
                continue
            current_index = z * width + x
            previous_index = previous_z * width + previous_x
            current_cell, predicted_cell = fused[current_index], previous_cells[previous_index]
            if current_cell == "agent" or predicted_cell == "agent" or current_cell == predicted_cell:
                continue
            fused[current_index] = predicted_cell
            conflicts += 1
    payload = dict(current_vision)
    payload["cells"] = fused
    return payload, conflicts


def augment_visual_observations(
    observations: Tensor,
    *,
    flip_rate: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply sensor-only tile corruption while preserving the agent tile.

    The function accepts CPU training windows and never changes symbolic values
    or action labels. A selected air/wall cell is swapped; other non-agent
    semantic tiles are occluded to air, matching the controlled noise used by
    the live degradation audit.
    """
    if not 0.0 <= flip_rate <= 1.0:
        raise ValueError("visual flip_rate must be between 0 and 1")
    if observations.ndim != 3 or observations.shape[-1] != VisionGoalDirectedPLWM.input_dim:
        raise ValueError("visual observations must have [batch, time, vision-input] shape")
    if flip_rate == 0.0:
        return observations.clone()
    augmented = observations.clone()
    visual = augmented[:, :, GoalDirectedPLWM.input_dim :].reshape(
        augmented.shape[0], augmented.shape[1], VISION_WIDTH * VISION_WIDTH, len(VISION_TILES)
    )
    tile_indices = visual.argmax(dim=-1)
    selected = torch.rand(tile_indices.shape, generator=generator, device=observations.device) < flip_rate
    selected &= tile_indices != VISION_TILES.index("agent")
    rows = selected.reshape(-1).nonzero(as_tuple=False).flatten()
    flat = visual.reshape(-1, len(VISION_TILES))
    original = tile_indices.reshape(-1)[rows]
    replacement = torch.where(
        original == VISION_TILES.index("air"),
        torch.full_like(original, VISION_TILES.index("wall")),
        torch.full_like(original, VISION_TILES.index("air")),
    )
    flat[rows] = 0.0
    flat[rows, replacement] = 1.0
    return augmented


@dataclass(frozen=True)
class LuantiState:
    """Validated task state emitted by the Luanti training game."""

    health: float
    energy: float
    x: int
    z: int
    goal_x: int
    goal_z: int
    task_seed: int
    completed: bool
    blocked_north: bool
    blocked_south: bool
    blocked_east: bool
    blocked_west: bool
    last_direction: str | None
    last_blocked: bool
    failed: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LuantiState":
        required_integers = ("x", "z", "goal_x", "goal_z", "task_seed")
        if any(isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int) for key in required_integers):
            raise ValueError("Luanti observation has invalid integer task fields")
        boolean_fields = ("completed", "blocked_north", "blocked_south", "blocked_east", "blocked_west", "last_blocked")
        if any(not isinstance(payload.get(key), bool) for key in boolean_fields):
            raise ValueError("Luanti observation has invalid completion field")
        if "failed" in payload and not isinstance(payload["failed"], bool):
            raise ValueError("Luanti observation has invalid failure field")
        last_direction = payload.get("last_direction")
        if last_direction is not None and last_direction not in DIRECTIONS:
            raise ValueError("Luanti observation has invalid last direction")
        return cls(
            health=float(payload.get("health", 0.0)),
            energy=float(payload.get("energy", 0.0)),
            x=int(payload["x"]),
            z=int(payload["z"]),
            goal_x=int(payload["goal_x"]),
            goal_z=int(payload["goal_z"]),
            task_seed=int(payload["task_seed"]),
            completed=bool(payload["completed"]),
            blocked_north=bool(payload["blocked_north"]),
            blocked_south=bool(payload["blocked_south"]),
            blocked_east=bool(payload["blocked_east"]),
            blocked_west=bool(payload["blocked_west"]),
            last_direction=last_direction,
            last_blocked=bool(payload["last_blocked"]),
            failed=bool(payload.get("failed", False)),
        )

    @property
    def distance(self) -> int:
        return abs(self.goal_x - self.x) + abs(self.goal_z - self.z)

    def features(self) -> tuple[float, ...]:
        """Goal-conditioned, bounded observations for the trainable core."""
        return (
            self.health / 20.0,
            self.energy / 20.0,
            self.x / 7.0,
            self.z / 7.0,
            self.goal_x / 7.0,
            self.goal_z / 7.0,
            float(self.completed),
            float(self.blocked_north),
            float(self.blocked_south),
            float(self.blocked_east),
            float(self.blocked_west),
            *[float(self.last_direction == direction) for direction in DIRECTIONS],
            float(self.last_blocked),
        )

    def oracle_actual_direction(self) -> str | None:
        """Shortest traversable direction before applying hidden controls."""
        if self.completed:
            return None
        # A fixed safe first probe lets a recurrent policy infer the hidden
        # control inversion from its following position change.
        if self.last_direction is None:
            return "east"
        frontier = [(self.x, self.z)]
        parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {(self.x, self.z): None}
        goal = (self.goal_x, self.goal_z)
        while frontier:
            x, z = frontier.pop(0)
            if (x, z) == goal:
                break
            for actual_direction, (delta_x, delta_z) in DELTAS.items():
                candidate = (x + delta_x, z + delta_z)
                if candidate not in parents and not is_blocked(self.task_seed, *candidate):
                    parents[candidate] = ((x, z), actual_direction)
                    frontier.append(candidate)
        if goal not in parents:
            raise RuntimeError("generated Luanti task has no path to its goal")
        cursor = goal
        while parents[cursor] is not None and parents[cursor][0] != (self.x, self.z):
            cursor = parents[cursor][0]
        _parent, actual_direction = parents[cursor]  # type: ignore[misc]
        return actual_direction

    def oracle_direction(self) -> str | None:
        """Teacher request after applying the hidden control mechanism."""
        if self.completed:
            return None
        if self.last_direction is None:
            return "east"
        actual_direction = self.oracle_actual_direction()
        if actual_direction is None:
            return None
        if controls_inverted(self.task_seed):
            return {"north": "south", "south": "north", "east": "west", "west": "east"}[actual_direction]
        return actual_direction


@dataclass(frozen=True)
class ObjectTaskState:
    """Extended state for the long-horizon key/decoy curriculum."""

    base: LuantiState
    key_x: int
    key_z: int
    decoy_x: int
    decoy_z: int
    key_collected: bool
    decoy_collected: bool
    last_reward: float = 0.0
    candidate_positions: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ObjectTaskState":
        if payload.get("object_mode") is not True:
            raise ValueError("Luanti observation is not an object-task state")
        integer_fields = ("key_x", "key_z", "decoy_x", "decoy_z")
        if any(isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int) for key in integer_fields):
            raise ValueError("Luanti object observation has invalid object coordinates")
        if not isinstance(payload.get("key_collected"), bool) or not isinstance(payload.get("decoy_collected"), bool):
            raise ValueError("Luanti object observation has invalid collection flags")
        raw_candidates = payload.get("candidate_positions", [])
        # Lua's JSON encoder represents an empty table as null. Candidate
        # coordinates are optional outside the four-choice curriculum.
        if raw_candidates is None:
            raw_candidates = []
        if not isinstance(raw_candidates, list):
            raise ValueError("Luanti object observation has invalid candidate positions")
        candidates: list[tuple[int, int]] = []
        for index, candidate in enumerate(raw_candidates, start=1):
            if not isinstance(candidate, Mapping) or candidate.get("index") != index:
                raise ValueError("Luanti object observation has invalid candidate identity")
            x, z = candidate.get("x"), candidate.get("z")
            if isinstance(x, bool) or not isinstance(x, int) or isinstance(z, bool) or not isinstance(z, int):
                raise ValueError("Luanti object observation has invalid candidate coordinate")
            candidates.append((x, z))
        return cls(
            base=LuantiState.from_payload(payload),
            key_x=int(payload["key_x"]),
            key_z=int(payload["key_z"]),
            decoy_x=int(payload["decoy_x"]),
            decoy_z=int(payload["decoy_z"]),
            key_collected=bool(payload["key_collected"]),
            decoy_collected=bool(payload["decoy_collected"]),
            last_reward=float(payload.get("last_reward", 0.0)),
            candidate_positions=tuple(candidates),
        )

    @property
    def completed(self) -> bool:
        return self.base.completed

    def features(self) -> tuple[float, ...]:
        return self.base.features() + (
            self.key_x / 7.0,
            self.key_z / 7.0,
            self.decoy_x / 7.0,
            self.decoy_z / 7.0,
            float(self.key_collected),
            float(self.decoy_collected),
        )

    def memory_features(self) -> tuple[float, ...]:
        """Add the immediate causal reward while keeping legacy features stable."""
        return self.features() + (self.last_reward,)

    def oracle_direction(self) -> str | None:
        # The first requested east probe is intentionally invariant so its
        # observed displacement identifies the hidden mapping.
        if self.base.last_direction is None:
            return "east"
        actual_direction = self.oracle_actual_direction()
        if actual_direction is None:
            return None
        if controls_inverted(self.base.task_seed):
            return CausalControlWorldModel._inverted_direction[actual_direction]
        return actual_direction

    def oracle_actual_direction(self) -> str | None:
        """Shortest physical direction before any hidden request mapping."""
        if self.completed:
            return None
        if self.base.last_direction is None:
            return "east"
        target = (self.base.goal_x, self.base.goal_z) if self.key_collected else (self.key_x, self.key_z)
        frontier = [(self.base.x, self.base.z)]
        parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {(self.base.x, self.base.z): None}
        while frontier:
            x, z = frontier.pop(0)
            if (x, z) == target:
                break
            for actual_direction, (delta_x, delta_z) in DELTAS.items():
                candidate = (x + delta_x, z + delta_z)
                if candidate not in parents and not is_blocked(self.base.task_seed, *candidate):
                    parents[candidate] = ((x, z), actual_direction)
                    frontier.append(candidate)
        if target not in parents:
            raise RuntimeError("generated Luanti object task has no path to its target")
        cursor = target
        while parents[cursor] is not None and parents[cursor][0] != (self.base.x, self.base.z):
            cursor = parents[cursor][0]
        _parent, actual_direction = parents[cursor]  # type: ignore[misc]
        return actual_direction


@dataclass(frozen=True)
class LuantiInteraction:
    request_id: str
    outcome: Mapping[str, Any]
    observation: Mapping[str, Any] | None
    vision: Mapping[str, Any] | None
    reward: Mapping[str, Any] | None


class LuantiLiveClient:
    """Serializes local bridge actions and waits for their causally linked records."""

    def __init__(self, bridge: LuantiFileBridge, *, timeout_seconds: float = 5.0, poll_seconds: float = 0.01) -> None:
        self.bridge = bridge
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._offset = bridge.observations_path.stat().st_size if bridge.observations_path.exists() else 0

    def _drain_records(self) -> list[dict[str, Any]]:
        path = self.bridge.observations_path
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("rb") as handle:
            handle.seek(self._offset)
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.seek(position)
                    break
                self._offset = handle.tell()
                if line.strip():
                    records.append(json.loads(line))
        return records

    def submit_batch(
        self,
        actions: Iterable[Mapping[str, Any]],
    ) -> tuple[dict[str, LuantiInteraction], list[dict[str, Any]]]:
        """Submit an ordered action batch and return only matched terminal records."""
        requested: list[str] = []
        required_observations: set[str] = set()
        for action in actions:
            request_id = self.bridge.write_action(
                str(action["name"]),
                episode_id=action.get("episode_id"),
                task_seed=action.get("task_seed"),
                direction=action.get("direction"),
            )
            requested.append(request_id)
            if bool(action.get("expect_observation")):
                required_observations.add(request_id)
        outcomes: dict[str, Mapping[str, Any]] = {}
        observations: dict[str, Mapping[str, Any]] = {}
        visions: dict[str, Mapping[str, Any]] = {}
        rewards: dict[str, Mapping[str, Any]] = {}
        raw_records: list[dict[str, Any]] = []
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            records = self._drain_records()
            raw_records.extend(records)
            for record in records:
                payload = record.get("payload")
                if not isinstance(payload, Mapping):
                    continue
                if record.get("kind") == "outcome" and payload.get("request_id") in requested:
                    outcomes[str(payload["request_id"])] = payload
                elif record.get("kind") == "observation" and payload.get("caused_by") in requested:
                    if record.get("modality") == "luanti-state":
                        observations[str(payload["caused_by"])] = payload
                    elif record.get("modality") == "luanti-vision":
                        visions[str(payload["caused_by"])] = payload
                elif record.get("kind") == "reward" and payload.get("caused_by") in requested:
                    # Preserve the bridge envelope with the reward so durable
                    # facts can retain their causal source instead of only a
                    # scalar value.
                    enriched_reward = dict(payload)
                    enriched_reward["_plwm_provenance"] = {
                        "episode_id": record.get("episode_id"),
                        "tick": record.get("tick"),
                        "source": record.get("modality"),
                        "request_id": payload.get("caused_by"),
                        "fact_schema_version": 1,
                    }
                    rewards[str(payload["caused_by"])] = enriched_reward
            if (
                set(requested).issubset(outcomes)
                and required_observations.issubset(observations)
                and required_observations.issubset(visions)
            ):
                return (
                    {
                        request_id: LuantiInteraction(
                            request_id=request_id,
                            outcome=outcomes[request_id],
                            observation=observations.get(request_id),
                            vision=visions.get(request_id),
                            reward=rewards.get(request_id),
                        )
                        for request_id in requested
                    },
                    raw_records,
                )
            time.sleep(self.poll_seconds)
        missing_outcomes = sorted(set(requested) - set(outcomes))
        missing_observations = sorted(required_observations - set(observations))
        missing_visions = sorted(required_observations - set(visions))
        raise TimeoutError(
            f"Luanti bridge timeout: outcomes={missing_outcomes}, observations={missing_observations}, visions={missing_visions}"
        )

    def begin_episode(self, episode_id: str, task_seed: int) -> LuantiState:
        actions = (
            {"name": "set_episode", "episode_id": episode_id, "task_seed": task_seed},
            {"name": "reset_episode", "expect_observation": True},
        )
        interactions, _records = self.submit_batch(actions)
        reset = list(interactions.values())[-1]
        if reset.outcome.get("status") != "accepted" or reset.observation is None:
            raise RuntimeError(f"Luanti reset was not accepted: {reset.outcome}")
        return LuantiState.from_payload(reset.observation)

    def begin_episode_with_vision(self, episode_id: str, task_seed: int) -> tuple[LuantiState, Mapping[str, Any]]:
        actions = (
            {"name": "set_episode", "episode_id": episode_id, "task_seed": task_seed},
            {"name": "reset_episode", "expect_observation": True},
        )
        interactions, _records = self.submit_batch(actions)
        reset = list(interactions.values())[-1]
        if reset.outcome.get("status") != "accepted" or reset.observation is None or reset.vision is None:
            raise RuntimeError(f"Luanti reset vision was not accepted: {reset.outcome}")
        return LuantiState.from_payload(reset.observation), reset.vision

    def begin_object_episode(self, episode_id: str, task_seed: int) -> ObjectTaskState:
        actions = (
            {"name": "set_episode", "episode_id": episode_id, "task_seed": task_seed},
            {"name": "reset_episode", "expect_observation": True},
        )
        interactions, _records = self.submit_batch(actions)
        reset = list(interactions.values())[-1]
        if reset.outcome.get("status") != "accepted" or reset.observation is None:
            raise RuntimeError(f"Luanti object reset was not accepted: {reset.outcome}")
        return ObjectTaskState.from_payload(reset.observation)

    def begin_object_episode_with_vision(self, episode_id: str, task_seed: int) -> tuple[ObjectTaskState, Mapping[str, Any]]:
        actions = (
            {"name": "set_episode", "episode_id": episode_id, "task_seed": task_seed},
            {"name": "reset_episode", "expect_observation": True},
        )
        interactions, _records = self.submit_batch(actions)
        reset = list(interactions.values())[-1]
        if reset.outcome.get("status") != "accepted" or reset.observation is None or reset.vision is None:
            raise RuntimeError(f"Luanti object reset vision was not accepted: {reset.outcome}")
        return ObjectTaskState.from_payload(reset.observation), reset.vision

    def step(self, direction: str) -> tuple[LuantiInteraction, LuantiState | None]:
        actions = ({"name": "step", "direction": direction, "expect_observation": True},)
        interactions, _records = self.submit_batch(actions)
        interaction = next(iter(interactions.values()))
        state = LuantiState.from_payload(interaction.observation) if interaction.observation is not None else None
        return interaction, state

    def step_with_vision(self, direction: str) -> tuple[LuantiInteraction, LuantiState | None, Mapping[str, Any] | None]:
        actions = ({"name": "step", "direction": direction, "expect_observation": True},)
        interactions, _records = self.submit_batch(actions)
        interaction = next(iter(interactions.values()))
        state = LuantiState.from_payload(interaction.observation) if interaction.observation is not None else None
        return interaction, state, interaction.vision

    def step_object(self, direction: str) -> tuple[LuantiInteraction, ObjectTaskState | None]:
        actions = ({"name": "step", "direction": direction, "expect_observation": True},)
        interactions, _records = self.submit_batch(actions)
        interaction = next(iter(interactions.values()))
        state = ObjectTaskState.from_payload(interaction.observation) if interaction.observation is not None else None
        return interaction, state

    def step_object_with_vision(self, direction: str) -> tuple[LuantiInteraction, ObjectTaskState | None, Mapping[str, Any] | None]:
        actions = ({"name": "step", "direction": direction, "expect_observation": True},)
        interactions, _records = self.submit_batch(actions)
        interaction = next(iter(interactions.values()))
        state = ObjectTaskState.from_payload(interaction.observation) if interaction.observation is not None else None
        return interaction, state, interaction.vision

    def observe_object_with_vision(self) -> tuple[LuantiInteraction, ObjectTaskState | None, Mapping[str, Any] | None]:
        """Request a fresh object-task observation without taking an action.

        Delayed-memory evaluations use this to insert real environment time and
        evidence records after a fact has been hidden, rather than duplicating
        an old Python-side state.
        """
        actions = ({"name": "observe", "expect_observation": True},)
        interactions, _records = self.submit_batch(actions)
        interaction = next(iter(interactions.values()))
        state = ObjectTaskState.from_payload(interaction.observation) if interaction.observation is not None else None
        return interaction, state, interaction.vision

    def recenter(self) -> tuple[LuantiInteraction, LuantiState | None]:
        actions = ({"name": "recenter", "expect_observation": True},)
        interactions, _records = self.submit_batch(actions)
        interaction = next(iter(interactions.values()))
        state = LuantiState.from_payload(interaction.observation) if interaction.observation is not None else None
        return interaction, state

    def recenter_with_vision(self) -> tuple[LuantiInteraction, LuantiState | None, Mapping[str, Any] | None]:
        actions = ({"name": "recenter", "expect_observation": True},)
        interactions, _records = self.submit_batch(actions)
        interaction = next(iter(interactions.values()))
        state = LuantiState.from_payload(interaction.observation) if interaction.observation is not None else None
        return interaction, state, interaction.vision

    def recenter_object(self) -> tuple[LuantiInteraction, ObjectTaskState | None]:
        actions = ({"name": "recenter", "expect_observation": True},)
        interactions, _records = self.submit_batch(actions)
        interaction = next(iter(interactions.values()))
        state = ObjectTaskState.from_payload(interaction.observation) if interaction.observation is not None else None
        return interaction, state

    def recenter_object_with_vision(self) -> tuple[LuantiInteraction, ObjectTaskState | None, Mapping[str, Any] | None]:
        actions = ({"name": "recenter", "expect_observation": True},)
        interactions, _records = self.submit_batch(actions)
        interaction = next(iter(interactions.values()))
        state = ObjectTaskState.from_payload(interaction.observation) if interaction.observation is not None else None
        return interaction, state, interaction.vision


class GoalDirectedPLWM(nn.Module):
    """A goal-conditioned action head over PLWM's persistent cognitive state."""

    input_dim = 16

    def __init__(self, latent_dim: int = 32, *, history_length: int = 4) -> None:
        super().__init__()
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = history_length
        self.core = TorchPLWM(
            self.input_dim,
            latent_dim=latent_dim,
            memory_slots=8,
            thought_steps=2,
            output_dim=len(DIRECTIONS),
            use_memory=True,
            gated_memory=True,
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        logits, surprises, _state = self.core(observations, think_on_last=True)
        return logits, surprises

    @torch.no_grad()
    def choose(self, history: Sequence[LuantiState]) -> str:
        if not history:
            raise ValueError("policy requires at least one observation")
        device = next(self.parameters()).device
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        sequence = torch.tensor([state.features() for state in context], dtype=torch.float32, device=device).unsqueeze(0)
        logits, _surprise = self(sequence)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]

    @torch.no_grad()
    def begin_episode(self, initial_observation: LuantiState) -> "GoalDirectedPolicySession":
        """Create a live policy session with state retained across game actions."""
        return GoalDirectedPolicySession(self, initial_observation)


class GoalDirectedPolicySession:
    """One episode of policy inference over PLWM's continuous cognitive state."""

    def __init__(self, model: GoalDirectedPLWM, initial_observation: LuantiState) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.cognitive_state = model.core.initial_state(1, self.device)
        # Match the left-padded context used for supervised training before
        # processing subsequent observations only once.
        encoded = torch.tensor(initial_observation.features(), dtype=torch.float32, device=self.device).unsqueeze(0)
        for index in range(model.history_length):
            self.cognitive_state, _surprise = model.core.step(
                encoded,
                self.cognitive_state,
                think=index == model.history_length - 1,
            )

    @torch.no_grad()
    def observe(self, observation: LuantiState) -> None:
        encoded = torch.tensor(observation.features(), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.cognitive_state, _surprise = self.model.core.step(encoded, self.cognitive_state, think=True)

    @torch.no_grad()
    def choose(self) -> str:
        logits = self.model.core.classify(self.cognitive_state)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class NoMemoryGoalPolicy(nn.Module):
    """Fair policy baseline restricted to the current observation only."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(GoalDirectedPLWM.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(DIRECTIONS)),
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.network(observations[:, -1, :])
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[LuantiState]) -> str:
        if not history:
            raise ValueError("policy requires at least one observation")
        device = next(self.parameters()).device
        observation = torch.tensor(history[-1].features(), dtype=torch.float32, device=device).view(1, 1, -1)
        logits, _surprise = self(observation)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class MaskedSensorGoalPolicy(GoalDirectedPLWM):
    """PLWM state baseline with local symbolic obstacle sensors removed."""

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        masked = observations.clone()
        masked[:, :, 7:11] = 0.0
        return super().forward(masked)


class GRUContextGoalPolicy(nn.Module):
    """Standard finite-context recurrent policy baseline with the same inputs."""

    def __init__(self, hidden_dim: int = 32, *, history_length: int = 4) -> None:
        super().__init__()
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = history_length
        self.encoder = nn.GRU(GoalDirectedPLWM.input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, len(DIRECTIONS))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        encoded, _state = self.encoder(observations)
        logits = self.head(encoded[:, -1, :])
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[LuantiState]) -> str:
        if not history:
            raise ValueError("policy requires at least one observation")
        device = next(self.parameters()).device
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        observations = torch.tensor([state.features() for state in context], dtype=torch.float32, device=device).unsqueeze(0)
        logits, _surprise = self(observations)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class ObjectGoalDirectedPLWM(nn.Module):
    """PLWM policy for long-horizon key/decoy consequences."""

    input_dim = 22

    def __init__(self, latent_dim: int = 40, *, history_length: int = 4) -> None:
        super().__init__()
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = history_length
        self.core = TorchPLWM(
            self.input_dim,
            latent_dim=latent_dim,
            memory_slots=8,
            thought_steps=2,
            output_dim=len(DIRECTIONS),
            use_memory=True,
            gated_memory=True,
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        logits, surprises, _state = self.core(observations, think_on_last=True)
        return logits, surprises

    @torch.no_grad()
    def choose(self, history: Sequence[ObjectTaskState]) -> str:
        if not history:
            raise ValueError("object policy requires at least one observation")
        device = next(self.parameters()).device
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        observations = torch.tensor([state.features() for state in context], dtype=torch.float32, device=device).unsqueeze(0)
        logits, _surprise = self(observations)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


def padded_object_histories(
    states: Sequence[ObjectTaskState], *, history: int, skip_label_indices: frozenset[int] = frozenset()
) -> tuple[Tensor, Tensor]:
    if history < 1:
        raise ValueError("history must be positive")
    windows: list[list[tuple[float, ...]]] = []
    labels: list[int] = []
    for index, state in enumerate(states):
        if index in skip_label_indices:
            continue
        direction = state.oracle_direction()
        if direction is None:
            continue
        prefix = list(states[max(0, index - history + 1) : index + 1])
        prefix = [prefix[0]] * (history - len(prefix)) + prefix
        windows.append([item.features() for item in prefix])
        labels.append(DIRECTION_TO_INDEX[direction])
    if not windows:
        return torch.empty((0, history, ObjectGoalDirectedPLWM.input_dim)), torch.empty((0,), dtype=torch.long)
    return torch.tensor(windows, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class NoMemoryObjectPolicy(nn.Module):
    """Current-object-state-only baseline for the long-horizon task."""

    def __init__(self, hidden_dim: int = 40) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(ObjectGoalDirectedPLWM.input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, len(DIRECTIONS)))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.network(observations[:, -1, :])
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[ObjectTaskState]) -> str:
        device = next(self.parameters()).device
        observation = torch.tensor(history[-1].features(), dtype=torch.float32, device=device).view(1, 1, -1)
        logits, _surprise = self(observation)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class GRUObjectPolicy(nn.Module):
    """Finite-context GRU baseline for object consequences."""

    def __init__(self, hidden_dim: int = 40, *, history_length: int = 4) -> None:
        super().__init__()
        self.history_length = history_length
        self.encoder = nn.GRU(ObjectGoalDirectedPLWM.input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, len(DIRECTIONS))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        encoded, _state = self.encoder(observations)
        logits = self.head(encoded[:, -1, :])
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[ObjectTaskState]) -> str:
        device = next(self.parameters()).device
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        observations = torch.tensor([state.features() for state in context], dtype=torch.float32, device=device).unsqueeze(0)
        logits, _surprise = self(observations)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class TransformerObjectPolicy(nn.Module):
    """Finite-context Transformer baseline over the public object history."""

    input_dim = ObjectGoalDirectedPLWM.input_dim

    def __init__(self, *, history_length: int = 16, model_dim: int = 48) -> None:
        super().__init__()
        self.history_length = history_length
        self.input_projection = nn.Linear(self.input_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, history_length, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(model_dim, len(DIRECTIONS))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        if observations.shape[1] > self.history_length:
            raise ValueError("object sequence exceeds Transformer context budget")
        encoded = self.encoder(self.input_projection(observations) + self.position[:, : observations.shape[1]])
        logits = self.head(encoded[:, -1])
        surprise = torch.zeros(
            (observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device
        )
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[ObjectTaskState]) -> str:
        if not history:
            raise ValueError("Transformer object policy requires at least one observation")
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        device = next(self.parameters()).device
        observations = torch.tensor([item.features() for item in context], dtype=torch.float32, device=device).unsqueeze(0)
        logits, _surprise = self(observations)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class VisionGoalDirectedPLWM(nn.Module):
    """PLWM policy whose local obstacle evidence comes from a tile grid."""

    input_dim = GoalDirectedPLWM.input_dim + VISION_FEATURE_DIM

    def __init__(self, latent_dim: int = 48, *, history_length: int = 4) -> None:
        super().__init__()
        self.history_length = history_length
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32),
            nn.Tanh(),
        )
        self.core = TorchPLWM(
            GoalDirectedPLWM.input_dim + 32,
            latent_dim=latent_dim,
            memory_slots=8,
            thought_steps=2,
            output_dim=len(DIRECTIONS),
            use_memory=True,
            gated_memory=True,
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, time_steps = observations.shape[:2]
        symbolic = observations[:, :, : GoalDirectedPLWM.input_dim]
        vision = observations[:, :, GoalDirectedPLWM.input_dim :].reshape(
            batch_size * time_steps, len(VISION_TILES), VISION_WIDTH, VISION_WIDTH
        )
        visual_embedding = self.vision_encoder(vision).reshape(batch_size, time_steps, -1)
        logits, surprises, _state = self.core(torch.cat((symbolic, visual_embedding), dim=-1), think_on_last=True)
        return logits, surprises

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[LuantiState, Mapping[str, Any]]]) -> str:
        if not history:
            raise ValueError("visual policy requires at least one observation")
        device = next(self.parameters()).device
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        observations = torch.tensor(
            [visual_frame_features(state, vision) for state, vision in context],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        logits, _surprise = self(observations)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class VisionObjectGoalDirectedPLWM(nn.Module):
    """Object-goal policy whose obstacle evidence remains visual and local."""

    input_dim = ObjectGoalDirectedPLWM.input_dim + VISION_FEATURE_DIM

    def __init__(self, latent_dim: int = 48, *, history_length: int = 4) -> None:
        super().__init__()
        self.history_length = history_length
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32),
            nn.Tanh(),
        )
        self.core = TorchPLWM(
            ObjectGoalDirectedPLWM.input_dim + 32,
            latent_dim=latent_dim,
            memory_slots=8,
            thought_steps=2,
            output_dim=len(DIRECTIONS),
            use_memory=True,
            gated_memory=True,
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, time_steps = observations.shape[:2]
        symbolic = observations[:, :, : ObjectGoalDirectedPLWM.input_dim]
        vision = observations[:, :, ObjectGoalDirectedPLWM.input_dim :].reshape(
            batch_size * time_steps, len(VISION_TILES), VISION_WIDTH, VISION_WIDTH
        )
        visual_embedding = self.vision_encoder(vision).reshape(batch_size, time_steps, -1)
        logits, surprises, _state = self.core(torch.cat((symbolic, visual_embedding), dim=-1), think_on_last=True)
        return logits, surprises

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]]) -> str:
        if not history:
            raise ValueError("visual object policy requires at least one observation")
        device = next(self.parameters()).device
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        observations = torch.tensor(
            [object_visual_frame_features(state, vision) for state, vision in context],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        logits, _surprise = self(observations)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class NoMemoryVisionObjectPolicy(nn.Module):
    """Current-frame visual object baseline with no persistent state."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(ObjectGoalDirectedPLWM.input_dim + 32, 48),
            nn.Tanh(),
            nn.Linear(48, len(DIRECTIONS)),
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        symbolic = observations[:, -1, : ObjectGoalDirectedPLWM.input_dim]
        vision = observations[:, -1, ObjectGoalDirectedPLWM.input_dim :].reshape(
            observations.shape[0], len(VISION_TILES), VISION_WIDTH, VISION_WIDTH
        )
        logits = self.head(torch.cat((symbolic, self.vision_encoder(vision)), dim=-1))
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]]) -> str:
        if not history:
            raise ValueError("no-memory visual object policy requires one observation")
        state, vision = history[-1]
        device = next(self.parameters()).device
        observation = torch.tensor(object_visual_frame_features(state, vision), dtype=torch.float32, device=device).view(1, 1, -1)
        logits, _surprise = self(observation)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class RewardMemoryVisionObjectPLWM(nn.Module):
    """Persistent object policy that receives causal reward but no lasting key flag."""

    input_dim = ObjectGoalDirectedPLWM.input_dim + 1 + VISION_FEATURE_DIM

    def __init__(self, latent_dim: int = 48, *, history_length: int = 16) -> None:
        super().__init__()
        self.history_length = history_length
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32),
            nn.Tanh(),
        )
        self.core = TorchPLWM(
            ObjectGoalDirectedPLWM.input_dim + 1 + 32,
            latent_dim=latent_dim,
            memory_slots=16,
            thought_steps=2,
            output_dim=len(DIRECTIONS),
            use_memory=True,
            gated_memory=True,
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        batch, steps = observations.shape[:2]
        symbolic = observations[:, :, : ObjectGoalDirectedPLWM.input_dim + 1]
        vision = observations[:, :, ObjectGoalDirectedPLWM.input_dim + 1 :].reshape(
            batch * steps, len(VISION_TILES), VISION_WIDTH, VISION_WIDTH
        )
        embedding = self.vision_encoder(vision).reshape(batch, steps, -1)
        logits, surprises, _state = self.core(torch.cat((symbolic, embedding), dim=-1), think_on_last=True)
        return logits, surprises

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]]) -> str:
        if not history:
            raise ValueError("reward-memory visual policy requires an observation")
        device = next(self.parameters()).device
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        observations = torch.tensor(
            [object_reward_visual_frame_features(state, vision) for state, vision in context],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        logits, _surprise = self(observations)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class NoMemoryRewardVisionObjectPolicy(nn.Module):
    """Current-frame baseline for the hidden-collection-fact curriculum."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(ObjectGoalDirectedPLWM.input_dim + 1 + 32, 48),
            nn.Tanh(),
            nn.Linear(48, len(DIRECTIONS)),
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        symbolic = observations[:, -1, : ObjectGoalDirectedPLWM.input_dim + 1]
        # A fact-memory policy may append verified facts before the vision
        # block. This baseline deliberately ignores them and always reads the
        # fixed-size visual block from the end of the observation.
        vision = observations[:, -1, -VISION_FEATURE_DIM:].reshape(
            observations.shape[0], len(VISION_TILES), VISION_WIDTH, VISION_WIDTH
        )
        logits = self.head(torch.cat((symbolic, self.vision_encoder(vision)), dim=-1))
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]]) -> str:
        if not history:
            raise ValueError("no-memory reward visual policy requires an observation")
        state, vision = history[-1]
        device = next(self.parameters()).device
        observation = torch.tensor(object_reward_visual_frame_features(state, vision), dtype=torch.float32, device=device).view(1, 1, -1)
        logits, _surprise = self(observation)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


@dataclass(frozen=True)
class VerifiedFactProvenance:
    """Causal envelope retained with a verified delayed fact."""

    fact: str
    value: str | int | bool
    source: str
    episode_id: str | None
    tick: int
    request_id: str | None
    fact_schema_version: int
    confidence: float = 1.0


class VerifiedObjectFactMemory:
    """Retain only causally confirmed delayed-object facts."""

    def __init__(self) -> None:
        self.key_verified = False
        self.charge_verified = False
        self.charge_mode: str | None = None
        self.charge_choice: int | None = None
        self.provenance: list[VerifiedFactProvenance] = []
        self._fallback_tick = 0

    def _record(self, fact: str, value: str | int | bool, reward: Mapping[str, Any]) -> None:
        envelope = reward.get("_plwm_provenance")
        if not isinstance(envelope, Mapping):
            envelope = {}
        self._fallback_tick += 1
        source = envelope.get("source", "luanti-reward")
        episode_id = envelope.get("episode_id")
        tick = envelope.get("tick", self._fallback_tick)
        request_id = envelope.get("request_id")
        version = envelope.get("fact_schema_version", 1)
        if (
            not isinstance(source, str) or not source
            or episode_id is not None and not isinstance(episode_id, str)
            or isinstance(tick, bool) or not isinstance(tick, int) or tick < 0
            or request_id is not None and not isinstance(request_id, str)
            or isinstance(version, bool) or not isinstance(version, int) or version < 1
        ):
            raise ValueError("invalid verified fact provenance")
        self.provenance.append(VerifiedFactProvenance(
            fact, value, source, episode_id, tick, request_id, version,
        ))

    def observe_reward(self, reward: Mapping[str, Any] | None) -> None:
        if reward is None:
            return
        value = reward.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value >= 0.19:
                self.key_verified = True
                self._record("key_verified", True, reward)
            elif 0.145 <= value < 0.185:
                self.charge_verified = True
                self.charge_choice = int(round((value - 0.15) / 0.01)) + 1
                self.charge_mode = {1: "key", 2: "decoy"}.get(self.charge_choice)
                self._record("charge_choice", self.charge_choice, reward)

    def snapshot(self) -> dict[str, Any]:
        return {
            "key_verified": self.key_verified,
            "charge_verified": self.charge_verified,
            "charge_mode": self.charge_mode,
            "charge_choice": self.charge_choice,
            "provenance": [
                {
                    "fact": item.fact,
                    "value": item.value,
                    "source": item.source,
                    "episode_id": item.episode_id,
                    "tick": item.tick,
                    "request_id": item.request_id,
                    "fact_schema_version": item.fact_schema_version,
                    "confidence": item.confidence,
                }
                for item in self.provenance
            ],
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any]) -> "VerifiedObjectFactMemory":
        key = snapshot.get("key_verified")
        # Snapshots written before the richer course have no charge field.
        charge = snapshot.get("charge_verified", False)
        charge_mode = snapshot.get("charge_mode")
        charge_choice = snapshot.get("charge_choice")
        records = snapshot.get("provenance", [])
        if (
            not isinstance(key, bool)
            or not isinstance(charge, bool)
            or charge_mode not in {None, "key", "decoy"}
            or charge_choice not in {None, 1, 2, 3, 4}
            or not isinstance(records, list)
        ):
            raise ValueError("invalid object fact memory snapshot")
        if charge_choice is None and charge != (charge_mode is not None):
            raise ValueError("object fact memory charge mode is inconsistent")
        if charge_choice is not None and not charge:
            raise ValueError("object fact memory choice requires verified charge")
        memory = cls()
        memory.key_verified = key
        memory.charge_verified = charge
        memory.charge_mode = charge_mode
        memory.charge_choice = charge_choice
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("invalid object fact provenance record")
            fact = record.get("fact")
            value = record.get("value")
            source = record.get("source")
            episode_id = record.get("episode_id")
            tick = record.get("tick")
            request_id = record.get("request_id")
            version = record.get("fact_schema_version")
            confidence = record.get("confidence")
            if (
                fact not in {"key_verified", "charge_choice"}
                or not isinstance(value, (str, int, bool))
                or not isinstance(source, str) or not source
                or episode_id is not None and not isinstance(episode_id, str)
                or isinstance(tick, bool) or not isinstance(tick, int) or tick < 0
                or request_id is not None and not isinstance(request_id, str)
                or isinstance(version, bool) or not isinstance(version, int) or version < 1
                or isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError("invalid object fact provenance record")
            memory.provenance.append(VerifiedFactProvenance(
                fact, value, source, episode_id, tick, request_id, version, float(confidence),
            ))
            memory._fallback_tick = max(memory._fallback_tick, tick)
        return memory


class VerifiedFactObjectPlanner:
    """Inspectable symbolic planner over public object coordinates.

    It chooses a physical direction only. The causal control model remains
    responsible for translating it into a request after observing mappings.
    The planner receives no seed, gate, hidden collection flag, or visual map.
    """

    def choose_actual(
        self,
        state: ObjectTaskState,
        *,
        verified_key: bool,
        verified_charge: bool | None = None,
        charge_mode: str | None = None,
        charge_choice: int | None = None,
    ) -> str | None:
        if state.completed:
            return None
        if verified_charge is None:
            target_x, target_z = (
                (state.base.goal_x, state.base.goal_z) if verified_key else (state.key_x, state.key_z)
            )
        elif not verified_charge:
            target_x, target_z = state.base.goal_x, state.base.goal_z
        elif not verified_key:
            if state.candidate_positions:
                if charge_choice not in {1, 2, 3, 4}:
                    raise ValueError("multi-object planner requires a verified candidate choice")
                target_x, target_z = state.candidate_positions[charge_choice - 1]
            else:
                target_x, target_z = (
                    (state.decoy_x, state.decoy_z) if charge_mode == "decoy" else (state.key_x, state.key_z)
                )
        else:
            target_x, target_z = state.base.goal_x, state.base.goal_z
        if state.base.x < target_x:
            return "east"
        if state.base.x > target_x:
            return "west"
        if state.base.z < target_z:
            return "south"
        if state.base.z > target_z:
            return "north"
        # A current position at an unreported key is possible only before the
        # causally linked reward arrives; probing leaves the action boundary
        # unchanged and lets the fact memory update from that reward.
        return "east"


class FactMemoryVisionObjectPLWM(nn.Module):
    """Visual object policy augmented only with causally verified fact memory."""

    input_dim = ObjectGoalDirectedPLWM.input_dim + 2 + VISION_FEATURE_DIM

    def __init__(self, latent_dim: int = 48, *, history_length: int = 16) -> None:
        super().__init__()
        self.history_length = history_length
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32), nn.Tanh(),
        )
        self.core = TorchPLWM(
            ObjectGoalDirectedPLWM.input_dim + 2 + 32,
            latent_dim=latent_dim,
            memory_slots=16,
            thought_steps=2,
            output_dim=len(DIRECTIONS),
            use_memory=True,
            gated_memory=True,
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        batch, steps = observations.shape[:2]
        symbolic = observations[:, :, : ObjectGoalDirectedPLWM.input_dim + 2]
        vision = observations[:, :, ObjectGoalDirectedPLWM.input_dim + 2 :].reshape(batch * steps, len(VISION_TILES), VISION_WIDTH, VISION_WIDTH)
        embedding = self.vision_encoder(vision).reshape(batch, steps, -1)
        logits, surprises, _state = self.core(torch.cat((symbolic, embedding), dim=-1), think_on_last=True)
        return logits, surprises

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]], *, verified_key: bool) -> str:
        if not history:
            raise ValueError("fact-memory visual policy requires an observation")
        device = next(self.parameters()).device
        context = list(history[-self.history_length :])
        context = [context[0]] * (self.history_length - len(context)) + context
        observations = torch.tensor(
            [object_fact_memory_visual_frame_features(state, vision, verified_key=verified_key) for state, vision in context],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        logits, _surprise = self(observations)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class FactMemoryCurrentVisionObjectPolicy(nn.Module):
    """Current-frame controller whose sole cross-step input is verified fact memory."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32), nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(ObjectGoalDirectedPLWM.input_dim + 2 + 32, 48), nn.Tanh(), nn.Linear(48, len(DIRECTIONS))
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        symbolic = observations[:, -1, : ObjectGoalDirectedPLWM.input_dim + 2]
        vision = observations[:, -1, -VISION_FEATURE_DIM:].reshape(observations.shape[0], len(VISION_TILES), VISION_WIDTH, VISION_WIDTH)
        logits = self.head(torch.cat((symbolic, self.vision_encoder(vision)), dim=-1))
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]], *, verified_key: bool) -> str:
        if not history:
            raise ValueError("fact-memory current policy requires an observation")
        state, vision = history[-1]
        device = next(self.parameters()).device
        observation = torch.tensor(
            object_fact_memory_visual_frame_features(state, vision, verified_key=verified_key),
            dtype=torch.float32,
            device=device,
        ).view(1, 1, -1)
        logits, _surprise = self(observation)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class TwoFactMemoryCurrentVisionObjectPolicy(nn.Module):
    """Current-frame learned controller with two causal fact inputs."""

    input_dim = ObjectGoalDirectedPLWM.input_dim + 3 + VISION_FEATURE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(ObjectGoalDirectedPLWM.input_dim + 3 + 32, 48),
            nn.Tanh(),
            nn.Linear(48, len(DIRECTIONS)),
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        symbolic = observations[:, -1, : ObjectGoalDirectedPLWM.input_dim + 3]
        vision = observations[:, -1, -VISION_FEATURE_DIM:].reshape(
            observations.shape[0], len(VISION_TILES), VISION_WIDTH, VISION_WIDTH
        )
        logits = self.head(torch.cat((symbolic, self.vision_encoder(vision)), dim=-1))
        surprise = torch.zeros(
            (observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device
        )
        return logits, surprise

    @torch.no_grad()
    def choose(
        self,
        history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]],
        *,
        verified_key: bool,
        verified_charge: bool,
    ) -> str:
        if not history:
            raise ValueError("two-fact policy requires an observation")
        device = next(self.parameters()).device
        state, vision = history[-1]
        observation = torch.tensor(
            object_two_fact_memory_visual_frame_features(
                state, vision, verified_key=verified_key, verified_charge=verified_charge
            ),
            dtype=torch.float32,
            device=device,
        ).view(1, 1, -1)
        logits, _surprise = self(observation)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class PolarityFactMemoryCurrentVisionObjectPolicy(nn.Module):
    """Current-frame learned controller with a causally verified polarity."""

    input_dim = ObjectGoalDirectedPLWM.input_dim + 4 + VISION_FEATURE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(ObjectGoalDirectedPLWM.input_dim + 4 + 32, 48),
            nn.Tanh(),
            nn.Linear(48, len(DIRECTIONS)),
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        symbolic = observations[:, -1, : ObjectGoalDirectedPLWM.input_dim + 4]
        vision = observations[:, -1, -VISION_FEATURE_DIM:].reshape(
            observations.shape[0], len(VISION_TILES), VISION_WIDTH, VISION_WIDTH
        )
        logits = self.head(torch.cat((symbolic, self.vision_encoder(vision)), dim=-1))
        surprise = torch.zeros(
            (observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device
        )
        return logits, surprise

    @torch.no_grad()
    def choose(
        self,
        history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]],
        *,
        verified_key: bool,
        charge_mode: str | None,
    ) -> str:
        if not history:
            raise ValueError("polarity policy requires an observation")
        device = next(self.parameters()).device
        state, vision = history[-1]
        observation = torch.tensor(
            object_polarity_fact_memory_visual_frame_features(
                state, vision, verified_key=verified_key, charge_mode=charge_mode
            ),
            dtype=torch.float32,
            device=device,
        ).view(1, 1, -1)
        logits, _surprise = self(observation)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


class MultiChoiceFactMemoryCurrentVisionObjectPolicy(nn.Module):
    """Current-frame controller with a reward-verified four-way choice fact."""

    symbolic_dim = ObjectGoalDirectedPLWM.input_dim + 1 + 8 + 5
    input_dim = symbolic_dim + VISION_FEATURE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(len(VISION_TILES), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * VISION_WIDTH * VISION_WIDTH, 32),
            nn.Tanh(),
        )
        self.head = nn.Sequential(nn.Linear(self.symbolic_dim + 32, 56), nn.Tanh(), nn.Linear(56, len(DIRECTIONS)))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        symbolic = observations[:, -1, : self.symbolic_dim]
        vision = observations[:, -1, -VISION_FEATURE_DIM:].reshape(
            observations.shape[0], len(VISION_TILES), VISION_WIDTH, VISION_WIDTH
        )
        logits = self.head(torch.cat((symbolic, self.vision_encoder(vision)), dim=-1))
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(
        self,
        history: Sequence[tuple[ObjectTaskState, Mapping[str, Any]]],
        *,
        verified_key: bool,
        charge_choice: int | None,
    ) -> str:
        if not history:
            raise ValueError("multi-choice policy requires an observation")
        state, vision = history[-1]
        device = next(self.parameters()).device
        observation = torch.tensor(
            object_multi_choice_fact_memory_visual_frame_features(
                state, vision, verified_key=verified_key, charge_choice=charge_choice
            ),
            dtype=torch.float32,
            device=device,
        ).view(1, 1, -1)
        logits, _surprise = self(observation)
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


def padded_visual_histories(
    states: Sequence[LuantiState],
    visions: Sequence[Mapping[str, Any]],
    *,
    history: int,
    skip_label_indices: frozenset[int] = frozenset(),
    actual_direction_labels: bool = False,
) -> tuple[Tensor, Tensor]:
    if len(states) != len(visions):
        raise ValueError("visual and symbolic trajectories must have equal lengths")
    windows: list[list[tuple[float, ...]]] = []
    labels: list[int] = []
    for index, state in enumerate(states):
        if index in skip_label_indices:
            continue
        direction = state.oracle_actual_direction() if actual_direction_labels else state.oracle_direction()
        if direction is None:
            continue
        prefix_start = max(0, index - history + 1)
        state_prefix = list(states[prefix_start : index + 1])
        vision_prefix = list(visions[prefix_start : index + 1])
        while len(state_prefix) < history:
            state_prefix.insert(0, state_prefix[0])
            vision_prefix.insert(0, vision_prefix[0])
        windows.append([visual_frame_features(item, visual) for item, visual in zip(state_prefix, vision_prefix)])
        labels.append(DIRECTION_TO_INDEX[direction])
    if not windows:
        return torch.empty((0, history, VisionGoalDirectedPLWM.input_dim)), torch.empty((0,), dtype=torch.long)
    return torch.tensor(windows, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def padded_object_visual_histories(
    states: Sequence[ObjectTaskState],
    visions: Sequence[Mapping[str, Any]],
    *,
    history: int,
    skip_label_indices: frozenset[int] = frozenset(),
    actual_direction_labels: bool = False,
) -> tuple[Tensor, Tensor]:
    if len(states) != len(visions):
        raise ValueError("object visual and symbolic trajectories must have equal lengths")
    windows: list[list[tuple[float, ...]]] = []
    labels: list[int] = []
    for index, state in enumerate(states):
        if index in skip_label_indices:
            continue
        direction = state.oracle_actual_direction() if actual_direction_labels else state.oracle_direction()
        if direction is None:
            continue
        start = max(0, index - history + 1)
        state_prefix = list(states[start : index + 1])
        vision_prefix = list(visions[start : index + 1])
        while len(state_prefix) < history:
            state_prefix.insert(0, state_prefix[0])
            vision_prefix.insert(0, vision_prefix[0])
        windows.append([object_visual_frame_features(item, vision) for item, vision in zip(state_prefix, vision_prefix)])
        labels.append(DIRECTION_TO_INDEX[direction])
    if not windows:
        return torch.empty((0, history, VisionObjectGoalDirectedPLWM.input_dim)), torch.empty((0,), dtype=torch.long)
    return torch.tensor(windows, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class ActionConditionedWorldModel(nn.Module):
    """Predict the next spatial outcome of a requested Luanti action.

    This component is intentionally separate from the goal policy: it can be
    wrong while the teacher-driven policy is right, and vice versa.  Its input
    includes only the observation history plus the proposed action, never the
    hidden seed rule.
    """

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(len(DIRECTIONS), 8)
        self.outcome = nn.Sequential(
            # This is deliberately a short-horizon causal model.  The most
            # recent observed displacement is evidence about a hidden control
            # mapping, computed from observations rather than supplied as it.
            nn.Linear(8 + GoalDirectedPLWM.input_dim * 2, latent_dim),
            nn.Tanh(),
            # x displacement, z displacement, and blocked are all discrete
            # outcomes in this world.  Classifying them prevents uncertain
            # regression values from becoming impossible fractional moves.
            nn.Linear(latent_dim, 7),
        )

    def forward(self, histories: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        current = histories[:, -1, :]
        previous = histories[:, -2, :]
        recent_change = current - previous
        prediction = self.outcome(torch.cat((self.action_embedding(actions), current, recent_change), dim=-1))
        surprises = torch.zeros((histories.shape[0], histories.shape[1]), dtype=histories.dtype, device=histories.device)
        return prediction[:, :3], prediction[:, 3:6], prediction[:, 6], surprises


@dataclass(frozen=True)
class PredictedTransition:
    """A discrete, inspectable next-state prediction for one requested action."""

    delta_x: int
    delta_z: int
    blocked: bool


def _sensor_is_blocked(state: LuantiState, direction: str) -> bool:
    return {
        "north": state.blocked_north,
        "south": state.blocked_south,
        "east": state.blocked_east,
        "west": state.blocked_west,
    }[direction]


class CausalControlWorldModel:
    """Infer a per-episode control mechanism from causally linked outcomes.

    The candidate mechanisms are identity and full directional inversion, both
    derived from the task's observable action interface. No task seed, gate
    location, or hidden-rule value enters this model. A single unblocked
    action can eliminate one candidate; blocked actions are evaluated using
    the local obstacle sensors before contributing evidence.
    """

    _inverted_direction = {"north": "south", "south": "north", "east": "west", "west": "east"}

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._consistent = {False: True, True: True}
        self.revision_count = 0

    def snapshot(self) -> dict[str, Any]:
        """Serialize only the evidence-supported control hypothesis state."""
        return {
            "consistent_identity": self._consistent[False],
            "consistent_inversion": self._consistent[True],
            "revision_count": self.revision_count,
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any]) -> "CausalControlWorldModel":
        identity = snapshot.get("consistent_identity")
        inversion = snapshot.get("consistent_inversion")
        revisions = snapshot.get("revision_count")
        if not isinstance(identity, bool) or not isinstance(inversion, bool) or not (identity or inversion):
            raise ValueError("invalid causal control snapshot hypotheses")
        if isinstance(revisions, bool) or not isinstance(revisions, int) or revisions < 0:
            raise ValueError("invalid causal control snapshot revision count")
        model = cls()
        model._consistent = {False: identity, True: inversion}
        model.revision_count = revisions
        return model

    @property
    def inferred_inversion(self) -> bool | None:
        if self._consistent[False] and not self._consistent[True]:
            return False
        if self._consistent[True] and not self._consistent[False]:
            return True
        return None

    def seed_mapping(self, inverted: bool) -> None:
        self._consistent = {False: not inverted, True: inverted}

    @classmethod
    def _actual_direction(cls, requested: str, inverted: bool) -> str:
        if requested not in DIRECTIONS:
            raise ValueError(f"invalid requested direction: {requested}")
        return cls._inverted_direction[requested] if inverted else requested

    def _predict_for_mapping(self, state: LuantiState, requested: str, inverted: bool) -> PredictedTransition:
        actual = self._actual_direction(requested, inverted)
        if _sensor_is_blocked(state, actual):
            return PredictedTransition(0, 0, True)
        delta_x, delta_z = DELTAS[actual]
        return PredictedTransition(delta_x, delta_z, False)

    def predict(self, state: LuantiState, requested: str) -> PredictedTransition:
        # A tie resolves to the no-inversion prior. The first probe may be
        # wrong, but its observed result resolves the tie for this episode.
        inverted = self._consistent[True] and not self._consistent[False]
        return self._predict_for_mapping(state, requested, inverted)

    def request_for_actual(self, actual: str) -> str:
        if actual not in DIRECTIONS:
            raise ValueError(f"invalid actual direction: {actual}")
        return self._inverted_direction[actual] if self.inferred_inversion is True else actual

    def observe(self, before: LuantiState, requested: str, after: LuantiState) -> None:
        observed = PredictedTransition(after.x - before.x, after.z - before.z, after.last_blocked)
        for inverted in tuple(self._consistent):
            if self._consistent[inverted] and self._predict_for_mapping(before, requested, inverted) != observed:
                self._consistent[inverted] = False
        if not any(self._consistent.values()):
            # A formerly supported mechanism was invalidated. Re-evaluate the
            # just-observed transition under a fresh hypothesis set rather
            # than carrying an impossible belief into later actions.
            self.revision_count += 1
            self._consistent = {False: True, True: True}
            for inverted in tuple(self._consistent):
                if self._predict_for_mapping(before, requested, inverted) != observed:
                    self._consistent[inverted] = False
            if not any(self._consistent.values()):
                raise RuntimeError("observed Luanti transition matches no supported control mechanism")


class StatelessControlBaseline:
    """Same local sensors but no retained action-result evidence."""

    def reset(self) -> None:
        return None

    def predict(self, state: LuantiState, requested: str) -> PredictedTransition:
        return CausalControlWorldModel()._predict_for_mapping(state, requested, False)

    def observe(self, before: LuantiState, requested: str, after: LuantiState) -> None:
        return None


class ControlMechanismMemory:
    """Stores only verified control mappings, scoped to a stable world ID."""

    def __init__(self) -> None:
        self._mappings: dict[str, bool] = {}

    def begin(self, world_id: str) -> CausalControlWorldModel:
        model = CausalControlWorldModel()
        if world_id in self._mappings:
            model.seed_mapping(self._mappings[world_id])
        return model

    def commit(self, world_id: str, model: CausalControlWorldModel) -> None:
        mapping = model.inferred_inversion
        if mapping is not None:
            self._mappings[world_id] = mapping

    def mapping_for(self, world_id: str) -> bool | None:
        return self._mappings.get(world_id)


def padded_histories(
    states: Sequence[LuantiState], *, history: int, skip_label_indices: frozenset[int] = frozenset()
) -> tuple[Tensor, Tensor]:
    """Return fixed-size history windows and oracle action labels for a trajectory."""
    if history < 1:
        raise ValueError("history must be positive")
    windows: list[list[tuple[float, ...]]] = []
    labels: list[int] = []
    for index, state in enumerate(states):
        if index in skip_label_indices:
            continue
        direction = state.oracle_direction()
        if direction is None:
            continue
        prefix = list(states[max(0, index - history + 1) : index + 1])
        prefix = [prefix[0]] * (history - len(prefix)) + prefix
        windows.append([item.features() for item in prefix])
        labels.append(DIRECTION_TO_INDEX[direction])
    if not windows:
        return torch.empty((0, history, GoalDirectedPLWM.input_dim)), torch.empty((0,), dtype=torch.long)
    return torch.tensor(windows, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

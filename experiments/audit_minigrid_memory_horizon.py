"""Measure the causal memory horizon of MiniGrid Memory environments.

This is an environment-structure audit, not a learner or a training source.
It uses grid access only to construct a shortest safe path that first reaches
a state where the initial-room object is visible, then reaches the known
success position.  The recorded delay is from the final source observation to
the action that commits to the upper or lower terminal branch.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments


ROOT = Path(__file__).resolve().parents[1]
LEFT, RIGHT, FORWARD = 0, 1, 2
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass(frozen=True)
class State:
    x: int
    y: int
    direction: int


@dataclass
class EpisodeHorizon:
    seed: int
    source_seen: bool
    source_to_branch_actions: int | None
    route_actions: int


@dataclass
class EnvironmentHorizon:
    environment: str
    episodes: list[EpisodeHorizon]
    minimum_source_to_branch_actions: int | None
    median_source_to_branch_actions: float | None


@dataclass
class Report:
    protocol: str
    seed_start: int
    environments: list[EnvironmentHorizon]
    artifact: str


def _transition(environment: gym.Env, state: State, action: int) -> State:
    if action == LEFT:
        return State(state.x, state.y, (state.direction - 1) % 4)
    if action == RIGHT:
        return State(state.x, state.y, (state.direction + 1) % 4)
    if action != FORWARD:
        raise ValueError("only movement actions are valid in the horizon audit")
    dx, dy = DIRS[state.direction]
    nx, ny = state.x + dx, state.y + dy
    cell = environment.unwrapped.grid.get(nx, ny)
    if cell is None or cell.can_overlap():
        return State(nx, ny, state.direction)
    return state


def _path(
    environment: gym.Env,
    start: State,
    goal: Callable[[State], bool],
) -> list[int]:
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        state, actions = queue.popleft()
        if goal(state):
            return actions
        for action in (LEFT, RIGHT, FORWARD):
            candidate = _transition(environment, state, action)
            if candidate not in visited:
                visited.add(candidate)
                queue.append((candidate, [*actions, action]))
    raise RuntimeError("no path to the required memory-audit state")


def _source_visible(environment: gym.Env, state: State, source: tuple[int, int]) -> bool:
    unwrapped = environment.unwrapped
    original_pos, original_dir = unwrapped.agent_pos.copy(), unwrapped.agent_dir
    try:
        unwrapped.agent_pos[:] = (state.x, state.y)
        unwrapped.agent_dir = state.direction
        return bool(unwrapped.agent_sees(*source))
    finally:
        unwrapped.agent_pos[:] = original_pos
        unwrapped.agent_dir = original_dir


def audit_episode(environment_id: str, seed: int) -> EpisodeHorizon:
    environment = gym.make(environment_id)
    try:
        environment.reset(seed=seed)
        unwrapped = environment.unwrapped
        source = (1, unwrapped.height // 2 - 1)
        start = State(int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1]), int(unwrapped.agent_dir))
        observe_path = _path(environment, start, lambda state: _source_visible(environment, state, source))
        state = start
        for action in observe_path:
            state = _transition(environment, state, action)
        target = tuple(int(value) for value in unwrapped.success_pos)
        finish_path = _path(environment, state, lambda candidate: (candidate.x, candidate.y) == target)
        actions = [*observe_path, *finish_path]
        states = [start]
        for action in actions:
            states.append(_transition(environment, states[-1], action))
        last_seen = max(
            index for index, state in enumerate(states[:-1])
            if _source_visible(environment, state, source)
        )
        center = unwrapped.height // 2
        branch = next(
            index for index, action in enumerate(actions)
            if index >= last_seen
            and states[index].y == center
            and states[index + 1].y != center
            and action == FORWARD
        )
        return EpisodeHorizon(seed, True, branch - last_seen, len(actions))
    finally:
        environment.close()


def run(
    *,
    environments: tuple[str, ...] = ("MiniGrid-MemoryS7-v0", "MiniGrid-MemoryS13-v0", "MiniGrid-MemoryS17Random-v0"),
    episodes: int = 64,
    seed_start: int = 3_000_000,
    artifact: Path | None = None,
) -> Report:
    if episodes < 1 or seed_start < 0:
        raise ValueError("episodes must be positive and seed_start must be non-negative")
    results: list[EnvironmentHorizon] = []
    for environment in environments:
        rows = [audit_episode(environment, seed_start + index) for index in range(episodes)]
        delays = sorted(row.source_to_branch_actions for row in rows if row.source_to_branch_actions is not None)
        results.append(EnvironmentHorizon(
            environment, rows,
            delays[0] if delays else None,
            (delays[(len(delays) - 1) // 2] + delays[len(delays) // 2]) / 2 if delays else None,
        ))
    target = artifact or ROOT / "artifacts" / "minigrid-memory-horizon-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "minigrid_memory_horizon_v1: oracle route is used solely to inspect environment observability; no route, action, grid state, or success label may train a controller",
        seed_start, results, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=3_000_000)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(asdict(run(
        episodes=args.episodes, seed_start=args.seed_start, artifact=args.artifact,
    )), indent=2))

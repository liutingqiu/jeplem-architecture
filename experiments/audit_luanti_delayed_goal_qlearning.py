"""Screen the delayed-goal Luanti course with reactive and retained-cue Q tables.

This is a task-diagnostic baseline, not a PLWM experiment.  Both policies use
only public local vision-derived features and reward.  The retained-cue table
is an explicit standard-memory upper bound: it stores the last public side
seen while the goal tile was visible.  It establishes whether the course truly
requires a persistent state before neural controller comparisons begin.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audit_luanti_micro_navigation_qlearning import ACTION_COUNT, GOAL_NOT_VISIBLE, State, _greedy_action, public_goal_delta
from plwm.luanti_reward_only_env import DEFAULT_BRIDGE_ROOT, GymVisionOnlyLuanti, VISION_SHAPE


MemoryState = tuple[State, int]


@dataclass(frozen=True)
class Evaluation:
    episodes: int
    successes: int
    mean_return: float
    mean_steps: float
    action_counts: list[int]

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes


@dataclass(frozen=True)
class Report:
    protocol: str
    executor: str
    training_episodes: int
    max_steps: int
    training_seed_range: list[int]
    heldout_seed_range: list[int]
    action_seed_base: int
    random_evaluation: Evaluation
    reactive_q_evaluation: Evaluation
    retained_cue_q_evaluation: Evaluation
    reactive_states: int
    retained_cue_states: int
    accepted_memory_diagnostic: bool
    artifact: str


def _visible_side(state: State) -> int | None:
    if state == GOAL_NOT_VISIBLE:
        return None
    if state[1] == 0:
        raise ValueError("delayed-goal course cannot place the goal on the agent column")
    return 1 if state[1] > 0 else -1


def public_valid_actions(observation: np.ndarray) -> tuple[int, ...]:
    """Read only adjacent public wall tiles as an action availability mask."""
    if not isinstance(observation, np.ndarray) or observation.shape != VISION_SHAPE or observation.dtype != np.uint8:
        raise ValueError("expected a public Luanti vision tensor")
    agents = np.argwhere(observation[:, :, 2] == 1)
    if agents.shape != (1, 2):
        raise ValueError("public vision must contain exactly one agent")
    row, column = agents[0]
    offsets = ((-1, 0), (1, 0), (0, 1), (0, -1))
    available = tuple(index for index, (delta_row, delta_column) in enumerate(offsets) if observation[row + delta_row, column + delta_column, 1] == 0)
    if not available:
        raise ValueError("public vision has no valid cardinal action")
    return available


def _select(values: np.ndarray, available: tuple[int, ...]) -> int:
    best = max(float(values[action]) for action in available)
    return min(action for action in available if float(values[action]) == best)


def _step_values(
    table: dict[object, np.ndarray], key: object, *, action: int, reward: float,
    next_key: object | None, terminal: bool,
) -> None:
    values = table.setdefault(key, np.zeros(ACTION_COUNT, dtype=np.float32))
    target = float(reward) if terminal else float(reward) + 0.95 * float(table.setdefault(next_key, np.zeros(ACTION_COUNT, dtype=np.float32)).max())
    values[action] += 0.35 * (target - float(values[action]))


def _train_reactive(*, episodes: int, max_steps: int, seed_start: int, action_seed: int, bridge_root: Path, masked: bool) -> dict[State, np.ndarray]:
    table: dict[State, np.ndarray] = {}
    generator = random.Random(action_seed)
    environment = GymVisionOnlyLuanti(seed=seed_start, max_steps=max_steps, bridge_root=bridge_root)
    try:
        for episode in range(episodes):
            observation, info = environment.reset(seed=seed_start + episode)
            if info != {}:
                raise RuntimeError("reactive Q received non-public info")
            epsilon = 0.35 - 0.30 * episode / max(1, episodes - 1)
            for _step in range(max_steps):
                state = public_goal_delta(observation)
                values = table.setdefault(state, np.zeros(ACTION_COUNT, dtype=np.float32))
                available = public_valid_actions(observation) if masked else tuple(range(ACTION_COUNT))
                action = generator.choice(available) if generator.random() < epsilon else _select(values, available)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                if info != {}:
                    raise RuntimeError("reactive Q received non-public info")
                _step_values(table, state, action=action, reward=reward, next_key=None if terminated or truncated else public_goal_delta(next_observation), terminal=terminated or truncated)
                observation = next_observation
                if terminated or truncated:
                    break
    finally:
        environment.close()
    return table


def _train_retained_cue(*, episodes: int, max_steps: int, seed_start: int, action_seed: int, bridge_root: Path, masked: bool) -> dict[MemoryState, np.ndarray]:
    table: dict[MemoryState, np.ndarray] = {}
    generator = random.Random(action_seed)
    environment = GymVisionOnlyLuanti(seed=seed_start, max_steps=max_steps, bridge_root=bridge_root)
    try:
        for episode in range(episodes):
            observation, info = environment.reset(seed=seed_start + episode)
            if info != {}:
                raise RuntimeError("retained-cue Q received non-public info")
            remembered_side = 0
            epsilon = 0.35 - 0.30 * episode / max(1, episodes - 1)
            for _step in range(max_steps):
                state = public_goal_delta(observation)
                remembered_side = _visible_side(state) or remembered_side
                key = (state, remembered_side)
                values = table.setdefault(key, np.zeros(ACTION_COUNT, dtype=np.float32))
                available = public_valid_actions(observation) if masked else tuple(range(ACTION_COUNT))
                action = generator.choice(available) if generator.random() < epsilon else _select(values, available)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                if info != {}:
                    raise RuntimeError("retained-cue Q received non-public info")
                if terminated or truncated:
                    next_key = None
                else:
                    next_state = public_goal_delta(next_observation)
                    next_side = _visible_side(next_state) or remembered_side
                    next_key = (next_state, next_side)
                _step_values(table, key, action=action, reward=reward, next_key=next_key, terminal=terminated or truncated)
                observation = next_observation
                if terminated or truncated:
                    break
    finally:
        environment.close()
    return table


def _evaluate(
    select_action, *, episodes: int, seed_start: int, max_steps: int, bridge_root: Path,
) -> Evaluation:
    environment = GymVisionOnlyLuanti(seed=seed_start, max_steps=max_steps, bridge_root=bridge_root)
    returns, steps, successes, action_counts = [], [], 0, [0] * ACTION_COUNT
    try:
        for episode in range(episodes):
            observation, info = environment.reset(seed=seed_start + episode)
            if info != {}:
                raise RuntimeError("delayed-goal evaluation received non-public info")
            controller_state = None
            total = 0.0
            for step in range(max_steps):
                action, controller_state = select_action(observation, controller_state)
                observation, reward, terminated, truncated, info = environment.step(action)
                if info != {}:
                    raise RuntimeError("delayed-goal evaluation received non-public info")
                action_counts[action] += 1
                total += reward
                if terminated or truncated:
                    successes += int(terminated and reward > 0.0)
                    steps.append(step + 1)
                    break
            else:
                steps.append(max_steps)
            returns.append(total)
    finally:
        environment.close()
    return Evaluation(episodes, successes, float(np.mean(returns)), float(np.mean(steps)), action_counts)


def run(
    *, training_episodes: int = 128, max_steps: int = 10, evaluation_episodes: int = 16,
    training_seed_start: int = 7_100_000, heldout_seed_start: int = 7_200_000,
    course: str = "long_delay", action_seed_base: int = 942_337,
    bridge_root: Path = DEFAULT_BRIDGE_ROOT, artifact: Path | None = None,
) -> Report:
    course_ranges = {
        "long_delay": (7_000_000, 8_000_000),
        "compact_binary": (8_000_000, 9_000_000),
        "gated_compact_binary": (9_000_000, 10_000_000),
        "gated_compact_binary_replication": (9_000_000, 10_000_000),
    }
    if course not in course_ranges:
        raise ValueError("unsupported delayed-goal course")
    minimum_seed, maximum_seed = course_ranges[course]
    masked = course.startswith("gated_compact_binary")
    if (
        training_episodes < 32 or max_steps < 4 or evaluation_episodes < 8
        or not isinstance(action_seed_base, int)
        or not minimum_seed <= training_seed_start < maximum_seed
        or not minimum_seed <= heldout_seed_start < maximum_seed
        or set(range(training_seed_start, training_seed_start + training_episodes)).intersection(range(heldout_seed_start, heldout_seed_start + evaluation_episodes))
    ):
        raise ValueError("invalid delayed-goal Q-learning contract")
    reactive = _train_reactive(
        episodes=training_episodes, max_steps=max_steps, seed_start=training_seed_start,
        action_seed=action_seed_base, bridge_root=bridge_root, masked=masked,
    )
    retained = _train_retained_cue(
        episodes=training_episodes, max_steps=max_steps, seed_start=training_seed_start,
        action_seed=action_seed_base + 10_000, bridge_root=bridge_root, masked=masked,
    )
    random_generator = random.Random(action_seed_base + 20_000)
    random_evaluation = _evaluate(
        lambda observation, state: (random_generator.choice(public_valid_actions(observation) if masked else tuple(range(ACTION_COUNT))), state),
        episodes=evaluation_episodes, seed_start=heldout_seed_start,
        max_steps=max_steps, bridge_root=bridge_root,
    )
    reactive_evaluation = _evaluate(
        lambda observation, _state: (_select(
            reactive.get(public_goal_delta(observation), np.zeros(ACTION_COUNT, dtype=np.float32)),
            public_valid_actions(observation) if masked else tuple(range(ACTION_COUNT)),
        ), None),
        episodes=evaluation_episodes, seed_start=heldout_seed_start,
        max_steps=max_steps, bridge_root=bridge_root,
    )
    def retained_action(observation, remembered_side):
        state = public_goal_delta(observation)
        side = _visible_side(state) or (remembered_side or 0)
        return _select(
            retained.get((state, side), np.zeros(ACTION_COUNT, dtype=np.float32)),
            public_valid_actions(observation) if masked else tuple(range(ACTION_COUNT)),
        ), side
    retained_evaluation = _evaluate(
        retained_action, episodes=evaluation_episodes, seed_start=heldout_seed_start,
        max_steps=max_steps, bridge_root=bridge_root,
    )
    target = artifact or ROOT / "artifacts" / f"luanti-delayed-goal-{course}-qlearning-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        f"luanti_delayed_goal_qlearning_{course}_v1: real Luanti delayed-goal micro-navigation. The initial 11x11 public vision reveals goal side, corridor walls force one forward action, then the goal is absent from later vision. Reactive and retained-cue tabular Q policies receive only public vision-derived state, their own action, scalar environment reward, done, and empty Gym info. The retained-cue baseline stores only the last public goal side; no bridge state, seed, teacher trajectory, external planner, shaped reward, or PLWM is used. Task diagnostic only, not an architecture result.",
        "local-cpu", training_episodes, max_steps,
        list(range(training_seed_start, training_seed_start + training_episodes)),
        list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)),
        action_seed_base,
        random_evaluation, reactive_evaluation, retained_evaluation,
        len(reactive), len(retained),
        retained_evaluation.success_rate >= 0.75 and retained_evaluation.success_rate >= reactive_evaluation.success_rate + 0.25,
        str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-episodes", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=16)
    parser.add_argument("--training-seed-start", type=int, default=7_100_000)
    parser.add_argument("--heldout-seed-start", type=int, default=7_200_000)
    parser.add_argument("--course", choices=("long_delay", "compact_binary", "gated_compact_binary", "gated_compact_binary_replication"), default="long_delay")
    parser.add_argument("--action-seed-base", type=int, default=942_337)
    parser.add_argument("--bridge-root", type=Path, default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument("--artifact", type=Path, default=None)
    arguments = parser.parse_args()
    print(json.dumps(asdict(run(
        training_episodes=arguments.training_episodes, max_steps=arguments.max_steps,
        evaluation_episodes=arguments.evaluation_episodes,
        training_seed_start=arguments.training_seed_start,
        heldout_seed_start=arguments.heldout_seed_start,
        course=arguments.course, action_seed_base=arguments.action_seed_base,
        bridge_root=arguments.bridge_root, artifact=arguments.artifact,
    )), indent=2))

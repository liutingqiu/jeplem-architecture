"""Leakage-restricted real-Luanti environment for reward-only control tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np

from .luanti import LuantiFileBridge
from .luanti_training import DIRECTIONS, LuantiLiveClient, encode_vision


DEFAULT_BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")
VISION_SHAPE = (11, 11, 7)


def vision_only_observation(vision: Mapping[str, Any]) -> np.ndarray:
    """Convert the public local visual record to a detached uint8 tensor.

    ``encode_vision`` validates the complete bridge payload, while this adapter
    retains none of the accompanying symbolic state record.  Copying prevents
    callers from retaining a mutable bridge-owned structure.
    """
    encoded = np.asarray(encode_vision(vision), dtype=np.uint8)
    if encoded.size != int(np.prod(VISION_SHAPE)):
        raise ValueError("Luanti reward-only adapter received an unexpected vision shape")
    return encoded.reshape(VISION_SHAPE).copy()


class VisionOnlyLuanti:
    """One serial Luanti episode with no symbolic observation side channel."""

    action_count = len(DIRECTIONS)

    def __init__(self, client: LuantiLiveClient, *, task_seed: int, episode_id: str, max_steps: int) -> None:
        if (
            isinstance(task_seed, bool) or not isinstance(task_seed, int) or task_seed < 0
            or not isinstance(episode_id, str) or not episode_id
            or isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
        ):
            raise ValueError("invalid reward-only Luanti episode parameters")
        self._client = client
        self._task_seed = task_seed
        self._episode_id = episode_id
        self._max_steps = max_steps
        self._started = False
        self._finished = False
        self._steps = 0

    def reset(self) -> np.ndarray:
        # The visual boundary does not depend on a course exposing object
        # state.  Use the generic bridge route and discard its state record so
        # visible-goal curricula cannot accidentally acquire a new side input.
        _unsafe_state, vision = self._client.begin_episode_with_vision(self._episode_id, self._task_seed)
        self._started = True
        self._finished = False
        self._steps = 0
        return vision_only_observation(vision)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        if not self._started:
            raise RuntimeError("reset must be called before stepping a Luanti episode")
        if self._finished:
            raise RuntimeError("cannot step a completed Luanti episode")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < self.action_count:
            raise ValueError("invalid reward-only Luanti action")
        interaction, _unsafe_state, vision = self._client.step_with_vision(DIRECTIONS[action])
        if interaction.outcome.get("status") != "accepted" or vision is None:
            raise RuntimeError(f"Luanti reward-only action was not accepted: {interaction.outcome}")
        reward_payload = interaction.reward or {}
        reward = reward_payload.get("value")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise RuntimeError("Luanti reward-only action omitted a numeric reward")
        self._steps += 1
        terminated = bool(interaction.outcome.get("terminal", False))
        truncated = self._steps >= self._max_steps and not terminated
        self._finished = terminated or truncated
        return vision_only_observation(vision), float(reward), terminated, truncated


class GymVisionOnlyLuanti(gym.Env[np.ndarray, int]):
    """Gymnasium wrapper exposing only vision/action/reward/terminal fields.

    It intentionally discards every symbolic Luanti observation, including
    agent/goal/object coordinates, task seed, hidden collection state, and
    control diagnostics.  The empty info dictionary is part of the contract.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int,
        max_steps: int = 40,
        bridge_root: Path = DEFAULT_BRIDGE_ROOT,
        client_factory: Callable[[Path], LuantiLiveClient] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._seed = seed
        self._max_steps = max_steps
        self._bridge_root = Path(bridge_root)
        factory = client_factory or (lambda root: LuantiLiveClient(LuantiFileBridge(root), timeout_seconds=8.0))
        self._client = factory(self._bridge_root)
        self._episode_index = 0
        self._environment: VisionOnlyLuanti | None = None
        self.observation_space = gym.spaces.Box(0, 1, shape=VISION_SHAPE, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(len(DIRECTIONS))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer")
            self._seed = seed
            self._episode_index = 0
        episode_index = self._episode_index
        self._episode_index += 1
        self._environment = VisionOnlyLuanti(
            self._client,
            task_seed=self._seed + episode_index,
            episode_id=f"reward-only-{self._seed}-{episode_index}",
            max_steps=self._max_steps,
        )
        return self._environment.reset(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._environment is None:
            raise RuntimeError("reset must be called before stepping GymVisionOnlyLuanti")
        observation, reward, terminated, truncated = self._environment.step(int(action))
        return observation, reward, terminated, truncated, {}

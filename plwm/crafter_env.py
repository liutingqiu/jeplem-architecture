"""Leakage-restricted Crafter adapter for public interaction experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class CrafterTransition:
    observation: np.ndarray
    reward: float
    done: bool


class PixelsOnlyCrafter:
    """Expose only RGB observations, chosen actions, reward, and done state.

    Crafter's info dict contains semantic maps, inventory, and achievement
    state.  They are intentionally discarded at the adapter boundary so no
    policy, pretrainer, or evaluator can receive them accidentally.
    """

    def __init__(self, *, seed: int, length: int = 256) -> None:
        if seed < 0 or length < 1:
            raise ValueError("seed must be non-negative and length positive")
        import crafter

        self._environment = crafter.Env(length=length, seed=seed)
        self.action_count = int(self._environment.action_space.n)

    @staticmethod
    def _validate(observation: Any) -> np.ndarray:
        if not isinstance(observation, np.ndarray) or observation.shape != (64, 64, 3) or observation.dtype != np.uint8:
            raise ValueError("Crafter adapter expected a 64x64 uint8 RGB observation")
        return observation.copy()

    def reset(self) -> np.ndarray:
        return self._validate(self._environment.reset())

    def step(self, action: int) -> CrafterTransition:
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < self.action_count:
            raise ValueError("invalid Crafter action")
        observation, reward, done, _unsafe_info = self._environment.step(action)
        return CrafterTransition(self._validate(observation), float(reward), bool(done))


class GymPixelsOnlyCrafter(gym.Env[np.ndarray, int]):
    """Gymnasium boundary that deliberately exposes no Crafter info fields."""

    metadata = {"render_modes": []}

    def __init__(self, *, seed: int, length: int = 256) -> None:
        super().__init__()
        self._seed = seed
        self._length = length
        self._episode = 0
        self._environment: PixelsOnlyCrafter | None = None
        self.observation_space = gym.spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(17)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self._episode = 0
        self._environment = PixelsOnlyCrafter(seed=self._seed + self._episode, length=self._length)
        self._episode += 1
        return self._environment.reset(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._environment is None:
            raise RuntimeError("reset must be called before step")
        transition = self._environment.step(int(action))
        # no ``info`` value is forwarded from the underlying environment.
        return transition.observation, transition.reward, transition.done, False, {}

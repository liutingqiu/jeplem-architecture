import unittest

import numpy as np

from plwm.luanti_reward_only_env import GymVisionOnlyLuanti, VisionOnlyLuanti, VISION_SHAPE, vision_only_observation
from plwm.luanti_training import LuantiInteraction


def _vision():
    cells = ["air"] * 121
    cells[60] = "agent"
    return {"encoding": "tile-grid-v1", "width": 11, "height": 11, "cells": cells}


class _Client:
    def __init__(self) -> None:
        self.starts = []
        self.directions = []

    def begin_episode_with_vision(self, episode_id, task_seed):
        self.starts.append((episode_id, task_seed))
        # State intentionally contains tempting symbolic data; the adapter must
        # return only the paired vision record.
        return _ForbiddenState(), _vision()

    def step_with_vision(self, direction):
        self.directions.append(direction)
        return (
            LuantiInteraction("request", {"status": "accepted", "terminal": False}, {"goal_x": 7}, _vision(), {"value": -0.01}),
            _ForbiddenState(),
            _vision(),
        )


class _ForbiddenState:
    """Fails the test if the reward-only adapter touches symbolic state."""

    def __getattribute__(self, _name):
        raise AssertionError("reward-only adapter accessed symbolic Luanti state")


class LuantiRewardOnlyEnvironmentTests(unittest.TestCase):
    def test_observation_contains_only_validated_vision_tensor(self):
        observation = vision_only_observation(_vision())
        self.assertEqual(observation.shape, VISION_SHAPE)
        self.assertEqual(observation.dtype, np.uint8)
        self.assertEqual(int(observation.sum()), 121)

    def test_serial_adapter_discards_symbolic_state_and_enforces_horizon(self):
        client = _Client()
        environment = VisionOnlyLuanti(client, task_seed=4_000_000, episode_id="test", max_steps=1)
        self.assertEqual(environment.reset().shape, VISION_SHAPE)
        observation, reward, terminated, truncated = environment.step(2)
        self.assertEqual(observation.shape, VISION_SHAPE)
        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(client.starts, [("test", 4_000_000)])
        self.assertEqual(client.directions, ["east"])
        with self.assertRaises(RuntimeError):
            environment.step(0)

    def test_gym_boundary_hides_info_and_advances_world_seed(self):
        client = _Client()
        environment = GymVisionOnlyLuanti(seed=4_000_000, max_steps=2, client_factory=lambda _root: client)
        observation, info = environment.reset()
        self.assertEqual(observation.shape, VISION_SHAPE)
        self.assertEqual(info, {})
        _observation, _reward, _terminated, _truncated, info = environment.step(0)
        self.assertEqual(info, {})
        environment.reset()
        self.assertEqual([seed for _episode, seed in client.starts], [4_000_000, 4_000_001])

    def test_invalid_action_is_rejected(self):
        environment = VisionOnlyLuanti(_Client(), task_seed=4_000_000, episode_id="test", max_steps=2)
        environment.reset()
        with self.assertRaises(ValueError):
            environment.step(4)


if __name__ == "__main__":
    unittest.main()

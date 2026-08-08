import unittest

import numpy as np

from experiments.collect_luanti_public_dynamics import ACTION_COUNT, collect
from plwm.luanti_reward_only_env import VISION_SHAPE


class _PublicOnlyEnvironment:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.actions: list[int] = []
        self._step = 0

    def reset(self, *, seed: int):
        self.seeds.append(seed)
        self._step = 0
        return np.zeros(VISION_SHAPE, dtype=np.uint8), {}

    def step(self, action: int):
        self.actions.append(action)
        self._step += 1
        observation = np.full(VISION_SHAPE, self._step % 2, dtype=np.uint8)
        return observation, -0.01, self._step == 2, False, {}


class LuantiPublicDynamicsCollectionTests(unittest.TestCase):
    def test_collects_only_public_causal_transition_arrays(self):
        environment = _PublicOnlyEnvironment()
        corpus = collect(environment, episodes=3, max_steps=4, seed_start=4_700_000, action_seed=7)
        self.assertEqual(corpus["observations"].shape, (6, *VISION_SHAPE))
        self.assertEqual(corpus["next_observations"].shape, (6, *VISION_SHAPE))
        self.assertEqual(corpus["actions"].shape, (6,))
        self.assertTrue(np.all((0 <= corpus["actions"]) & (corpus["actions"] < ACTION_COUNT)))
        self.assertEqual(environment.seeds, [4_700_000, 4_700_001, 4_700_002])
        self.assertEqual(corpus["episode_indices"].tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(corpus["step_indices"].tolist(), [0, 1, 0, 1, 0, 1])

    def test_invalid_collection_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            collect(_PublicOnlyEnvironment(), episodes=0, max_steps=4, seed_start=4_700_000, action_seed=1)


if __name__ == "__main__":
    unittest.main()

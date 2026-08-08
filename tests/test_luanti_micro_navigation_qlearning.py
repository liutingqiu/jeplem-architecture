import unittest

import numpy as np

from experiments.audit_luanti_micro_navigation_qlearning import GOAL_NOT_VISIBLE, _greedy_action, public_goal_delta, run
from plwm.luanti_reward_only_env import VISION_SHAPE


class LuantiMicroNavigationQLearningTests(unittest.TestCase):
    def test_public_goal_delta_reads_only_rendered_tiles(self):
        observation = np.zeros(VISION_SHAPE, dtype=np.uint8)
        observation[5, 5, 2] = 1
        observation[3, 7, 3] = 1
        self.assertEqual(public_goal_delta(observation), (-2, 2))
        observation[3, 7, 3] = 0
        self.assertEqual(public_goal_delta(observation), GOAL_NOT_VISIBLE)
        observation[5, 5, 2] = 0
        with self.assertRaises(ValueError):
            public_goal_delta(observation)

    def test_greedy_action_is_stably_deterministic(self):
        self.assertEqual(_greedy_action(np.array([0.0, 1.0, 1.0, 0.0])), 1)

    def test_invalid_contract_is_rejected_before_connecting_to_luanti(self):
        with self.assertRaises(ValueError):
            run(training_episodes=8)
        with self.assertRaises(ValueError):
            run(training_seed_start=6_300_000, heldout_seed_start=6_300_000)


if __name__ == "__main__":
    unittest.main()

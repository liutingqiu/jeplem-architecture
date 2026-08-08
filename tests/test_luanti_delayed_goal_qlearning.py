import unittest

import numpy as np

from experiments.audit_luanti_delayed_goal_qlearning import _visible_side, public_valid_actions, run
from experiments.audit_luanti_micro_navigation_qlearning import GOAL_NOT_VISIBLE
from plwm.luanti_reward_only_env import VISION_SHAPE


class LuantiDelayedGoalQLearningTests(unittest.TestCase):
    def test_visible_side_preserves_the_public_not_visible_state(self):
        self.assertEqual(_visible_side((2, -2)), -1)
        self.assertEqual(_visible_side((2, 2)), 1)
        self.assertIsNone(_visible_side(GOAL_NOT_VISIBLE))
        with self.assertRaises(ValueError):
            _visible_side((2, 0))

    def test_invalid_contract_is_rejected_before_connecting(self):
        with self.assertRaises(ValueError):
            run(training_episodes=16)
        with self.assertRaises(ValueError):
            run(training_seed_start=7_100_000, heldout_seed_start=7_100_000)
        with self.assertRaises(ValueError):
            run(course="unknown")

    def test_public_action_mask_uses_only_adjacent_wall_tiles(self):
        observation = np.zeros(VISION_SHAPE, dtype=np.uint8)
        observation[5, 5, 2] = 1
        observation[4, 5, 1] = 1
        observation[5, 6, 1] = 1
        observation[5, 4, 1] = 1
        self.assertEqual(public_valid_actions(observation), (1,))


if __name__ == "__main__":
    unittest.main()

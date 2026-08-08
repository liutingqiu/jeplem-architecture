import unittest

import torch

from plwm.minigrid_features import BALL_CODE, KEY_CODE, visible_object_presence


class MiniGridFeaturesTests(unittest.TestCase):
    def test_detects_object_identity_without_position_assumption(self):
        events = torch.zeros((2, 159))
        events[0, 3 * 3] = KEY_CODE
        events[1, 21 * 3] = BALL_CODE
        self.assertTrue(torch.equal(visible_object_presence(events), torch.tensor([[1.0, 0.0], [0.0, 1.0]])))

    def test_rejects_non_minigrid_prefix(self):
        with self.assertRaises(ValueError):
            visible_object_presence(torch.zeros((2, 10)))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from experiments.audit_luanti_addressed_visual_contract import IDENTITIES, identity_counts
from plwm.luanti_reward_only_env import VISION_SHAPE
from plwm.luanti_training import VISION_TILES


class LuantiAddressedVisualContractTests(unittest.TestCase):
    def test_identity_counts_reads_only_public_tile_channels(self):
        observation = np.zeros(VISION_SHAPE, dtype=np.uint8)
        for index, identity in enumerate(IDENTITIES):
            observation[index, index, VISION_TILES.index(identity)] = 1
        self.assertEqual(identity_counts(observation), {identity: 1 for identity in IDENTITIES})

    def test_identity_counts_rejects_a_non_public_tensor_shape(self):
        with self.assertRaises(ValueError):
            identity_counts(np.zeros((11, 11, 6), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()

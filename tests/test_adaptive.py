import unittest

import torch

from plwm.adaptive import AdaptivePonderer


class AdaptivePondererTests(unittest.TestCase):
    def test_halt_distribution_and_depth_outputs_have_expected_shapes(self):
        model = AdaptivePonderer(state_count=8, latent_dim=12, max_steps=4)
        logits, expected, weights, depth_logits = model(
            torch.tensor([0, 1, 2]), torch.tensor([0.0, 0.5, 1.0])
        )
        self.assertEqual(tuple(logits.shape), (3, 8))
        self.assertEqual(tuple(expected.shape), (3,))
        self.assertEqual(tuple(weights.shape), (3, 5))
        self.assertEqual(tuple(depth_logits.shape), (3, 5, 8))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3)))


if __name__ == "__main__":
    unittest.main()

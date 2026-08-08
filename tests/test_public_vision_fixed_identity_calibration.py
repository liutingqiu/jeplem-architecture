import unittest

import torch

from experiments.train_public_vision_candidate_memory import EVENTS, GRID_SIZE, VISION_CHANNELS, make_batch
from experiments.train_public_vision_fixed_identity_calibration import FixedIdentitySideMLP, fixed_goal_side, run


class PublicVisionFixedIdentityCalibrationTests(unittest.TestCase):
    def test_public_pixel_target_reads_the_generated_goal_side(self):
        batch = make_batch(16, generator=torch.Generator().manual_seed(5), device=torch.device("cpu"))
        labels = fixed_goal_side(batch.vision)
        self.assertEqual(tuple(labels.shape), (16,))
        self.assertTrue(((labels == 0.0) | (labels == 1.0)).all())

    def test_model_accepts_one_initial_public_frame(self):
        model = FixedIdentitySideMLP()
        self.assertEqual(tuple(model(torch.zeros(2, GRID_SIZE, GRID_SIZE, VISION_CHANNELS)).shape), (2,))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, EVENTS, GRID_SIZE, GRID_SIZE, VISION_CHANNELS))

    def test_run_rejects_identical_train_and_heldout_seeds(self):
        with self.assertRaises(ValueError):
            run(steps=1, batch_size=2, train_seed=1, heldout_seed=1, device="cpu")


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from experiments.train_public_vision_binding_calibration import TwoFramePublicVisionBindingCNN, run
from experiments.train_public_vision_candidate_memory import EVENTS, GRID_SIZE, VISION_CHANNELS


class PublicVisionBindingCalibrationTests(unittest.TestCase):
    def test_model_reads_only_aligned_public_visual_sequences(self):
        model = TwoFramePublicVisionBindingCNN()
        logits = model(torch.zeros(3, EVENTS, GRID_SIZE, GRID_SIZE, VISION_CHANNELS))
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaises(ValueError):
            model(torch.zeros(3, EVENTS, GRID_SIZE, GRID_SIZE, VISION_CHANNELS - 1))

    def test_run_rejects_identical_train_and_heldout_seeds(self):
        with self.assertRaises(ValueError):
            run(steps=1, batch_size=2, train_seed=7, heldout_seed=7, device="cpu")


if __name__ == "__main__":
    unittest.main()

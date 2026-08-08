import unittest

import torch

from experiments.audit_public_vision_binding_contract import public_pixel_labels, run
from experiments.train_public_vision_candidate_memory import GRID_SIZE, VISION_CHANNELS, make_batch


class PublicVisionBindingContractTests(unittest.TestCase):
    def test_public_pixel_oracle_reconstructs_generated_labels(self):
        batch = make_batch(32, generator=torch.Generator().manual_seed(8), device=torch.device("cpu"))
        self.assertTrue(torch.equal(public_pixel_labels(batch.vision), batch.labels))

    def test_public_pixel_oracle_rejects_missing_query(self):
        vision = torch.zeros(1, 2, GRID_SIZE, GRID_SIZE, VISION_CHANNELS)
        with self.assertRaises(ValueError):
            public_pixel_labels(vision)

    def test_contract_audit_accepts_public_label_reconstruction(self):
        self.assertTrue(run(episodes=32, seed=9).accepted)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from experiments.train_public_vision_candidate_memory import EVENTS, GRID_SIZE, VISION_CHANNELS
from experiments.train_query_conditioned_vision_binding import QueryConditionedVisualBinding, parameter_count, run


class QueryConditionedVisionBindingTests(unittest.TestCase):
    def test_both_ablation_modes_read_only_public_visual_sequences(self):
        vision = torch.zeros(2, EVENTS, GRID_SIZE, GRID_SIZE, VISION_CHANNELS)
        for conditioned in (False, True):
            logits = QueryConditionedVisualBinding(query_conditioned=conditioned)(vision)
            self.assertEqual(tuple(logits.shape), (2,))
            self.assertTrue(torch.isfinite(logits).all())

    def test_ablation_parameter_counts_match_exactly(self):
        self.assertEqual(
            parameter_count(QueryConditionedVisualBinding(query_conditioned=True)),
            parameter_count(QueryConditionedVisualBinding(query_conditioned=False)),
        )

    def test_run_rejects_identical_train_and_heldout_seeds(self):
        with self.assertRaises(ValueError):
            run(steps=1, batch_size=2, train_seed=3, heldout_seed=3, device="cpu")


if __name__ == "__main__":
    unittest.main()

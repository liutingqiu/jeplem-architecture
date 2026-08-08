import unittest

import torch

from experiments.train_public_vision_candidate_memory import (
    EVENTS,
    GRID_SIZE,
    VISION_CHANNELS,
    PublicVisionCandidateMemory,
    make_batch,
    parameter_count,
    run,
)


class PublicVisionCandidateMemoryTests(unittest.TestCase):
    def test_batch_exposes_only_visual_events_and_prior_action_reward(self):
        batch = make_batch(3, generator=torch.Generator().manual_seed(1), device=torch.device("cpu"))
        self.assertEqual(tuple(batch.vision.shape), (3, EVENTS, GRID_SIZE, GRID_SIZE, VISION_CHANNELS))
        self.assertEqual(tuple(batch.previous_actions.shape), (3, EVENTS))
        self.assertEqual(tuple(batch.previous_rewards.shape), (3, EVENTS))
        self.assertEqual(tuple(batch.labels.shape), (3, 1))

    def test_candidate_and_single_writer_have_exactly_matched_parameters(self):
        multi = PublicVisionCandidateMemory(multi_candidate=True)
        single = PublicVisionCandidateMemory(multi_candidate=False)
        self.assertEqual(parameter_count(multi), parameter_count(single))

    def test_both_writer_modes_read_aligned_public_sequences(self):
        batch = make_batch(2, generator=torch.Generator().manual_seed(2), device=torch.device("cpu"))
        for multi_candidate in (False, True):
            model = PublicVisionCandidateMemory(multi_candidate=multi_candidate)
            logits = model(batch.vision, batch.previous_actions, batch.previous_rewards)
            self.assertEqual(tuple(logits.shape), (2,))
            self.assertTrue(torch.isfinite(logits).all())

    def test_run_rejects_identical_train_and_heldout_seeds(self):
        with self.assertRaises(ValueError):
            run(steps=1, batch_size=2, train_seed=7, heldout_seed=7, device="cpu")


if __name__ == "__main__":
    unittest.main()

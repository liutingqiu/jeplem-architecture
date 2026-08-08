import unittest

import torch

from experiments.train_object_token_memory import ObjectTokenEpisodicMemory, parameter_count, run
from experiments.train_public_vision_candidate_memory import make_batch


class ObjectTokenMemoryTests(unittest.TestCase):
    def test_forward_uses_only_public_event_sequences(self):
        batch = make_batch(2, generator=torch.Generator().manual_seed(12), device=torch.device("cpu"))
        output = ObjectTokenEpisodicMemory(multi_writer=True, use_memory=True)(
            batch.vision, batch.previous_actions, batch.previous_rewards
        )
        self.assertEqual(tuple(output.shape), (2,))

    def test_all_comparison_modes_are_parameter_matched(self):
        multi = ObjectTokenEpisodicMemory(multi_writer=True, use_memory=True)
        single = ObjectTokenEpisodicMemory(multi_writer=False, use_memory=True)
        disabled = ObjectTokenEpisodicMemory(multi_writer=True, use_memory=False)
        self.assertEqual(parameter_count(multi), parameter_count(single))
        self.assertEqual(parameter_count(multi), parameter_count(disabled))

    def test_rejects_missing_initial_object_tokens(self):
        batch = make_batch(1, generator=torch.Generator().manual_seed(13), device=torch.device("cpu"))
        batch.vision[:, 0, :, :, 3:] = 0.0
        with self.assertRaises(ValueError):
            ObjectTokenEpisodicMemory(multi_writer=True, use_memory=True)(
                batch.vision, batch.previous_actions, batch.previous_rewards
            )

    def test_run_rejects_shared_seed(self):
        with self.assertRaises(ValueError):
            run(steps=1, batch_size=2, train_seed=14, heldout_seed=14, device="cpu")


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from experiments.train_object_token_vision_binding import ObjectTokenVisualBinding, parameter_count, run
from experiments.train_public_vision_candidate_memory import make_batch


class ObjectTokenVisionBindingTests(unittest.TestCase):
    def test_forward_uses_only_public_visual_sequence(self):
        batch = make_batch(3, generator=torch.Generator().manual_seed(4), device=torch.device("cpu"))
        output = ObjectTokenVisualBinding(query_conditioned=True)(batch.vision)
        self.assertEqual(tuple(output.shape), (3,))

    def test_ablation_is_parameter_matched(self):
        matching = ObjectTokenVisualBinding(query_conditioned=True)
        ablation = ObjectTokenVisualBinding(query_conditioned=False)
        self.assertEqual(parameter_count(matching), parameter_count(ablation))

    def test_rejects_a_broken_public_object_contract(self):
        batch = make_batch(1, generator=torch.Generator().manual_seed(5), device=torch.device("cpu"))
        batch.vision[:, 0, :, :, 3:] = 0.0
        with self.assertRaises(ValueError):
            ObjectTokenVisualBinding(query_conditioned=True)(batch.vision)

    def test_run_rejects_shared_seed(self):
        with self.assertRaises(ValueError):
            run(steps=1, batch_size=2, train_seed=6, heldout_seed=6, device="cpu")


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from experiments.audit_minigrid_memory_context_window import ContextTransformerActorCritic, run
from experiments.train_minigrid_memory_actor_critic import INPUT_DIM


class MiniGridMemoryContextWindowTests(unittest.TestCase):
    def test_windowed_transformer_emits_causal_policy_and_value(self):
        model = ContextTransformerActorCritic(context_window=4)
        logits, values = model(torch.zeros((2, 5, INPUT_DIM)), torch.ones((2, 5), dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (2, 5, 7))
        self.assertEqual(tuple(values.shape), (2, 5))

    def test_rejects_invalid_window_and_budget(self):
        with self.assertRaises(ValueError):
            ContextTransformerActorCritic(context_window=0)
        with self.assertRaises(ValueError):
            run(bootstrap_updates=0)


if __name__ == "__main__":
    unittest.main()

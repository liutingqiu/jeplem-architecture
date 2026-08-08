import unittest

import torch

from experiments.audit_minigrid_delayed_event_recall import EvidenceStoreRecall, FourEventTransformerRecall, GRURecall, run
from experiments.train_minigrid_memory_actor_critic import INPUT_DIM


class DelayedEventRecallTests(unittest.TestCase):
    def test_models_reconstruct_one_event(self):
        events = torch.zeros((2, 16, INPUT_DIM))
        for model in (EvidenceStoreRecall(), EvidenceStoreRecall(use_evidence=False), GRURecall(), FourEventTransformerRecall()):
            self.assertEqual(tuple(model(events).shape), (2, INPUT_DIM))

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            run(corpus_episodes=1)
        with self.assertRaises(ValueError):
            run(sequence_length=8, recall_delay=8)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch

from experiments.audit_minigrid_delayed_object_identity import IdentityHead, _visible_label, run
from experiments.audit_minigrid_delayed_event_recall import EvidenceStoreRecall
from experiments.train_minigrid_memory_actor_critic import INPUT_DIM


class DelayedObjectIdentityTests(unittest.TestCase):
    def test_sensor_label_and_model_shape(self):
        observation = {"image": np.zeros((7, 7, 3), dtype=np.int64)}
        self.assertIsNone(_visible_label(observation))
        model = IdentityHead(EvidenceStoreRecall(input_dim=2), feature_dim=2)
        self.assertEqual(tuple(model(torch.zeros((2, 9, 2))).shape), (2, 2))

    def test_invalid_training_size_is_rejected(self):
        with self.assertRaises(ValueError):
            run(training_examples=1)


if __name__ == "__main__":
    unittest.main()

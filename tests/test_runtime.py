import math
import unittest

import torch

from plwm.commitments import Commitment
from plwm.continual import EndToEndPersistentPLWM
from plwm.runtime import PersistentRuntime, TrainablePersistentRuntime


class PersistentRuntimeTests(unittest.TestCase):
    def test_snapshot_restore_preserves_next_tick(self):
        runtime = PersistentRuntime(2, 2)
        runtime.add_commitment(Commitment("goal", target_state=1, priority=1.0))
        runtime.ingest([0.0, 0.2], [math.log(0.45), math.log(0.55)])
        restored = PersistentRuntime.restore(runtime.snapshot(), 2, 2)
        direct = runtime.ingest([0.1, 0.2], [math.log(0.1), math.log(0.9)])
        resumed = restored.ingest([0.1, 0.2], [math.log(0.1), math.log(0.9)])
        self.assertEqual(direct.action, resumed.action)
        self.assertEqual(direct.selected_commitment, resumed.selected_commitment)
        self.assertAlmostEqual(direct.belief_confidence, resumed.belief_confidence)

    def test_unknown_snapshot_version_fails_closed(self):
        runtime = PersistentRuntime(2, 2)
        snapshot = runtime.snapshot()
        snapshot["version"] = 999
        with self.assertRaises(ValueError):
            PersistentRuntime.restore(snapshot, 2, 2)

    def test_corrupted_snapshot_fails_integrity_check(self):
        runtime = PersistentRuntime(2, 2)
        snapshot = runtime.snapshot()
        snapshot["belief_log_probs"][0][0] = 99.0
        with self.assertRaises(ValueError):
            PersistentRuntime.restore(snapshot, 2, 2)

    def test_uncertain_unobserved_goal_requests_observation(self):
        runtime = PersistentRuntime(2, 2)
        runtime.add_commitment(Commitment("goal", target_state=1, priority=1.0))
        tick = runtime.ingest(None)
        self.assertEqual(tick.action.value, "observe")

    def test_trainable_runtime_checkpoint_restores_all_runtime_layers(self):
        model = EndToEndPersistentPLWM(input_dim=5, latent_dim=12)
        runtime = TrainablePersistentRuntime(model, belief_states=2)
        runtime.add_commitment(Commitment("route", target_state=1, priority=1.0))
        runtime.ingest([1, 0, 0, 0, 0], [0.1, 0.9])
        snapshot = runtime.snapshot()
        restored = TrainablePersistentRuntime.restore(snapshot, EndToEndPersistentPLWM(input_dim=5, latent_dim=12), 2)
        self.assertEqual(restored.ledger.time, runtime.ledger.time)
        self.assertTrue(torch.equal(restored.belief.log_probs, runtime.belief.log_probs))
        self.assertTrue(torch.equal(restored.latent.belief, runtime.latent.belief))

    def test_trainable_runtime_rejects_corrupt_snapshot(self):
        model = EndToEndPersistentPLWM(input_dim=5, latent_dim=12)
        runtime = TrainablePersistentRuntime(model, belief_states=2)
        snapshot = runtime.snapshot()
        snapshot["ledger"]["time"] = 99
        with self.assertRaises(ValueError):
            TrainablePersistentRuntime.restore(snapshot, model, 2)

    def test_trainable_runtime_think_only_preserves_evidence_and_commitment(self):
        model = EndToEndPersistentPLWM(input_dim=5, latent_dim=12)
        runtime = TrainablePersistentRuntime(model, belief_states=2)
        runtime.add_commitment(Commitment("route", target_state=1, priority=1.0))
        runtime.ingest([1, 0, 0, 0, 0], [0.1, 0.9])
        before_belief = runtime.belief.log_probs.clone()
        before_memory = runtime.latent.keys.clone()
        tick = runtime.think_only()
        self.assertEqual(tick.action.value, "think")
        self.assertTrue(torch.equal(before_belief, runtime.belief.log_probs))
        self.assertTrue(torch.equal(before_memory, runtime.latent.keys))
        self.assertEqual(tick.selected_commitment, "route")


if __name__ == "__main__":
    unittest.main()

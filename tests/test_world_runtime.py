import unittest

import torch

from plwm.commitments import Commitment
from plwm.world_model import ActionConditionedPLWM
from plwm.world_runtime import WorldModelPersistentRuntime


class WorldRuntimeTests(unittest.TestCase):
    def test_checkpoint_restores_same_counterfactual_proposal(self):
        torch.manual_seed(9)
        model = ActionConditionedPLWM(3, latent_dim=12)
        runtime = WorldModelPersistentRuntime(model, belief_states=3)
        runtime.add_commitment(Commitment("goal", target_state=2, priority=1.0))
        runtime.ingest([1.0, 0.0, 0.0], [-2.0, -2.0, 0.0])
        snapshot = runtime.snapshot()
        restored = WorldModelPersistentRuntime.restore(snapshot, model, belief_states=3)
        direct, direct_utility = runtime.propose(2, [0.0, 0.1])
        resumed, resumed_utility = restored.propose(2, [0.0, 0.1])
        self.assertEqual(direct, resumed)
        self.assertEqual(direct_utility, resumed_utility)

    def test_costed_plan_reports_one_score_per_action(self):
        model = ActionConditionedPLWM(3, latent_dim=12)
        runtime = WorldModelPersistentRuntime(model, belief_states=3)
        runtime.ingest([0.0, 1.0, 0.0])
        action, utilities = runtime.propose(1, [0.05, 0.2])
        self.assertIn(action, {0, 1})
        self.assertEqual(len(utilities), 2)

    def test_planning_switches_to_next_commitment_only_after_evidence(self):
        model = ActionConditionedPLWM(3, latent_dim=12)
        runtime = WorldModelPersistentRuntime(model, belief_states=3)
        runtime.add_commitment(Commitment("urgent", target_state=0, priority=2.0))
        runtime.add_commitment(Commitment("later", target_state=2, priority=1.0))
        runtime.ingest([1.0, 0.0, 0.0])
        selected, _action, _utilities = runtime.propose_selected([0.0, 0.0])
        self.assertEqual(selected, "urgent")
        runtime.ingest([0.0, 0.0, 0.0], [0.0, -9.0, -9.0])
        selected_after, _action_after, _utilities_after = runtime.propose_selected([0.0, 0.0])
        self.assertEqual(selected_after, "later")


if __name__ == "__main__":
    unittest.main()

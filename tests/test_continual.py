import unittest

import torch

from plwm.continual import (
    BayesianPersistentController,
    EndToEndPersistentPLWM,
    EvidenceTask,
    FixedLanguagePort,
    LearnedPersistentRuntime,
    TaskAction,
    TransformerContextPolicy,
)


class ContinualTests(unittest.TestCase):
    def setUp(self):
        self.task = EvidenceTask(
            route=1,
            cheap_signal=1,
            verify_signal=1,
            distractors=(0, 1, 0, 1),
        )

    def test_delayed_commitment_is_preserved_and_completed_by_evidence(self):
        result = BayesianPersistentController().run(self.task)
        self.assertEqual(result.selected_before_delay, "reach-correct-route")
        self.assertEqual(result.selected_after_delay, "reach-correct-route")
        self.assertTrue(result.commitment_completed)

    def test_stale_memory_is_revised_by_stronger_verification(self):
        result = BayesianPersistentController().run(self.task, inject_stale_memory=True)
        self.assertTrue(result.success)
        self.assertTrue(result.corrected_stale_memory)

    def test_end_to_end_model_keeps_a_persistent_tensor_state(self):
        model = EndToEndPersistentPLWM(latent_dim=12)
        policy, route, state = model(self.task.sequence(include_verification=False).unsqueeze(0))
        self.assertEqual(tuple(policy.shape), (1, 3))
        self.assertEqual(tuple(route.shape), (1, 2))
        self.assertGreater(float(state.belief.abs().sum().detach()), 0.0)

    def test_transformer_context_policy_has_same_action_interface(self):
        model = TransformerContextPolicy(model_dim=16)
        control, route = model(self.task.sequence(include_verification=False).unsqueeze(0))
        self.assertEqual(tuple(control.shape), (1, 3))
        self.assertEqual(tuple(route.shape), (1, 2))

    def test_learned_runtime_does_not_complete_commitment_without_environmental_success(self):
        model = EndToEndPersistentPLWM(latent_dim=12)
        runtime = LearnedPersistentRuntime(model)
        result = runtime.run(self.task)
        self.assertEqual(result.selected_before_delay, "reach-correct-route")
        self.assertEqual(result.selected_after_delay, "reach-correct-route")
        self.assertEqual(result.commitment_completed, result.success)

    def test_language_port_encodes_and_decodes_without_owning_goal_state(self):
        port = FixedLanguagePort()
        event, likelihood = port.encode_observation("route right")
        self.assertEqual(tuple(event.shape), (5,))
        self.assertEqual(tuple(likelihood.shape), (2,))
        self.assertEqual(port.emitted, [])
        self.assertEqual(port.decode_action(TaskAction.LEFT), "take left route")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
from pathlib import Path
import unittest

from plwm import CognitiveAction, PLWM, PLWMConfig
from experiments.interruption_recovery import run


class PLWMTests(unittest.TestCase):
    def setUp(self):
        self.model = PLWM(PLWMConfig(input_dim=2, latent_dim=8, seed=3))

    def test_state_is_persistent_across_serialization(self):
        self.model.set_goal([1.0, 0.0], commitment="finish the sequence")
        self.model.step([4.0, -2.0], action=CognitiveAction.THINK)
        before = self.model.state.to_dict()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.json"
            self.model.save(str(checkpoint))
            restored = PLWM(self.model.config)
            restored.load(str(checkpoint))
        self.assertEqual(restored.state.to_dict(), before)
        self.assertEqual(restored.state.pending_commitment, "finish the sequence")

    def test_thinking_is_recurrent_and_does_not_emit_text(self):
        result = self.model.step([1.0, 0.5], action=CognitiveAction.THINK)
        self.assertEqual(result.thought_steps, 3)
        self.assertIsNone(result.emission)
        self.assertNotEqual(self.model.state.workspace, [0.0] * 8)

    def test_speaking_is_an_explicit_action(self):
        result = self.model.step([1.0, 0.5], action=CognitiveAction.SPEAK)
        self.assertIsNotNone(result.emission)
        self.assertEqual(len(result.emission), 2)

    def test_surprising_observation_is_written_to_memory(self):
        result = self.model.step([10.0, -10.0], action=CognitiveAction.OBSERVE)
        self.assertTrue(result.memory_written)
        self.assertEqual(len(self.model.state.memory), 1)

    def test_sleep_does_not_reset_goal_or_time(self):
        self.model.set_goal([1.0, 0.0], commitment="keep going")
        self.model.step(None, action=CognitiveAction.SLEEP)
        self.assertEqual(self.model.state.time, 1)
        self.assertEqual(self.model.state.pending_commitment, "keep going")

    def test_state_is_json_serializable(self):
        payload = json.dumps(self.model.state.to_dict())
        self.assertIn('"belief"', payload)

    def test_interruption_recovery_contract(self):
        report = run()
        self.assertTrue(all(report["checkpoint_recovery"].values()))


if __name__ == "__main__":
    unittest.main()

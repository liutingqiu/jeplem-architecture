import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.audit_luanti_addressed_object_token_smoke import ACTION_SEED, HELDOUT_SEEDS, TRAINING_SEEDS, audit


class ObjectTokenSmokeAuditTests(unittest.TestCase):
    def _report(self, *, accepted: bool = True) -> dict:
        successes = {"no_memory": 16, "gru": 12, "transformer_2": 12, "plwm_state_off": 16, "plwm": 32}
        return {
            "protocol": "real repaired seven-channel luanti addressed retrieval reward-only 11x11 public vision previous chosen action previous scalar reward legal-action masks same object-token visual event encoder no cue/query labels coordinates symbolic bridge state seed teacher trajectory external planner shaped reward plwm-only input",
            "course": "addressed_gated", "device": "cuda", "training_seed_range": TRAINING_SEEDS, "heldout_seed_range": HELDOUT_SEEDS,
            "action_seed": ACTION_SEED, "max_steps": 6, "transformer_context": 2, "acceptance_mode": "object_token_writer_smoke",
            "parameter_match_tolerance": 0.05,
            "parameter_match_errors": {"no_memory": 0.01, "gru": 0.02, "transformer_2": 0.01, "plwm_state_off": 0.0, "plwm": 0.0},
            "results": [{"name": name, "evaluation": {"episodes": 32, "successes": value}} for name, value in successes.items()],
            "accepted_neural_task_diagnostic": accepted,
        }

    def test_audit_recomputes_passing_frozen_smoke(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self._report()), encoding="utf-8")
            result = audit(path)
        self.assertTrue(result.accepted)
        self.assertEqual(result.failures, [])

    def test_audit_rejects_tampered_acceptance(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self._report(accepted=False)), encoding="utf-8")
            result = audit(path)
        self.assertIn("reported acceptance disagrees with recomputed smoke gate", result.failures)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from experiments.audit_luanti_addressed_memory_diagnostic_report import ACTION_SEED, HELDOUT_SEEDS, TRAINING_SEEDS, audit


def _report(*, plwm_successes: int = 10):
    successes = {"no_memory": 12, "gru": 10, "transformer_2": 13, "plwm_state_off": 10, "plwm": plwm_successes}
    return {
        "protocol": "real Luanti gated delayed-goal course reward-only 11x11 public vision previous chosen action previous scalar reward legal-action masks no cue labels symbolic bridge state teacher trajectory external planner shaped reward parameter-capacity-matched",
        "course": "addressed_gated", "device": "cuda", "training_seed_range": TRAINING_SEEDS,
        "heldout_seed_range": HELDOUT_SEEDS, "action_seed": ACTION_SEED, "max_steps": 6,
        "transformer_context": 2, "acceptance_mode": "addressed_memory_diagnostic",
        "parameter_match_tolerance": 0.05,
        "parameter_match_errors": {"no_memory": 0.01, "gru": 0.01, "transformer_2": 0.02, "plwm_state_off": 0.0, "plwm": 0.0},
        "results": [{"name": name, "evaluation": {"episodes": 16, "successes": value}} for name, value in successes.items()],
        "accepted_neural_task_diagnostic": False,
    }


class LuantiAddressedMemoryDiagnosticAuditTests(unittest.TestCase):
    def test_valid_failed_calibration_recomputes_as_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(_report()), encoding="utf-8")
            result = audit(path)
        self.assertFalse(result.accepted)
        self.assertFalse(result.recomputed_accepted)
        self.assertEqual(result.failures, [])

    def test_tampered_acceptance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            payload = _report()
            payload["accepted_neural_task_diagnostic"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = audit(path)
        self.assertIn("reported acceptance disagrees with recomputed calibration gate", result.failures)


if __name__ == "__main__":
    unittest.main()

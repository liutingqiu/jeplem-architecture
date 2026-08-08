import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.audit_public_vision_candidate_memory import audit


class PublicVisionCandidateMemoryAuditTests(unittest.TestCase):
    def _report(self) -> dict:
        return {
            "protocol": "bounded supervised storage/readout diagnostic only; 11x11x7 public vision; previous chosen action; previous scalar reward; labels coordinates seeds teachers planners never transfer to Luanti control",
            "train_seed": 1,
            "heldout_seed": 2,
            "parameter_match_error": 0.0,
            "results": [
                {"name": "multi_candidate", "parameters": 5, "heldout_accuracy": 0.95},
                {"name": "single_candidate", "parameters": 5, "heldout_accuracy": 0.70},
            ],
            "accepted_mechanism_diagnostic": True,
        }

    def test_audit_recomputes_a_passing_gate(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self._report()), encoding="utf-8")
            result = audit(path)
        self.assertTrue(result.accepted)
        self.assertEqual(result.failures, [])

    def test_audit_rejects_mismatched_parameter_counts(self):
        report = self._report()
        report["results"][1]["parameters"] = 6
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = audit(path)
        self.assertFalse(result.accepted)
        self.assertIn("reported parameter counts differ", result.failures)


if __name__ == "__main__":
    unittest.main()

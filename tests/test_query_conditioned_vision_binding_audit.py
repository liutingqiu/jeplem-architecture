import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.audit_query_conditioned_vision_binding import audit


class QueryConditionedVisionBindingAuditTests(unittest.TestCase):
    def test_audit_recomputes_a_passing_comparison(self):
        report = {
            "protocol": "bounded supervised visual readout only 11x11x7 public visual frames labels identity IDs coordinates seeds actions rewards teachers planners never transfer to Luanti control",
            "train_seed": 1,
            "heldout_seed": 2,
            "train_steps": 240,
            "batch_size": 64,
            "device": "cpu",
            "parameter_match_error": 0.0,
            "results": [
                {"name": "query_conditioned", "parameters": 10, "heldout_accuracy": 0.95},
                {"name": "late_fusion", "parameters": 10, "heldout_accuracy": 0.70},
            ],
            "accepted_query_conditioned_binding": True,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = audit(path)
        self.assertTrue(result.accepted)
        self.assertEqual(result.failures, [])

    def test_audit_rejects_parameter_mismatch(self):
        report = {
            "protocol": "bounded supervised visual readout only 11x11x7 public visual frames labels identity IDs coordinates seeds actions rewards teachers planners never transfer to Luanti control",
            "train_seed": 1,
            "heldout_seed": 2,
            "train_steps": 240,
            "batch_size": 64,
            "device": "cpu",
            "parameter_match_error": 0.0,
            "results": [
                {"name": "query_conditioned", "parameters": 10, "heldout_accuracy": 0.95},
                {"name": "late_fusion", "parameters": 11, "heldout_accuracy": 0.70},
            ],
            "accepted_query_conditioned_binding": True,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = audit(path)
        self.assertFalse(result.accepted)
        self.assertIn("visual-binding parameter counts differ", result.failures)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.audit_object_token_memory import audit


class ObjectTokenMemoryAuditTests(unittest.TestCase):
    def _report(self) -> dict:
        return {
            "protocol": "bounded supervised storage/readout diagnostic 11x11x7 public vision object masks derive solely grid-index parameters are internal model weights labels identity ids coordinates seeds teachers planners never transfer to luanti control",
            "train_seed": 1,
            "heldout_seed": 2,
            "train_steps": 240,
            "batch_size": 64,
            "device": "cpu",
            "parameter_match_error": 0.0,
            "results": [
                {"name": "object_token_multi_writer", "parameters": 10, "heldout_accuracy": 0.95},
                {"name": "object_token_single_writer", "parameters": 10, "heldout_accuracy": 0.70},
                {"name": "object_token_memory_disabled", "parameters": 10, "heldout_accuracy": 0.70},
            ],
            "accepted_object_token_memory": True,
        }

    def test_audit_recomputes_a_passing_comparison(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self._report()), encoding="utf-8")
            result = audit(path)
        self.assertTrue(result.accepted)
        self.assertEqual(result.failures, [])

    def test_audit_rejects_parameter_mismatch(self):
        report = self._report()
        report["results"][2]["parameters"] = 11
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = audit(path)
        self.assertFalse(result.accepted)
        self.assertIn("object-token memory parameter counts differ", result.failures)


if __name__ == "__main__":
    unittest.main()

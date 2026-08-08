import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.audit_public_vision_binding_calibration import audit


class PublicVisionBindingCalibrationAuditTests(unittest.TestCase):
    def test_audit_recomputes_the_fixed_gate(self):
        report = {
            "protocol": "bounded supervised visual readout only 11x11x7 public visual frames labels identity IDs coordinates seeds actions rewards teachers planners never transfer to Luanti control",
            "train_seed": 1,
            "heldout_seed": 2,
            "train_steps": 240,
            "batch_size": 64,
            "device": "cpu",
            "heldout_accuracy": 0.95,
            "accepted_public_visual_binding": True,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = audit(path)
        self.assertTrue(result.accepted)
        self.assertEqual(result.failures, [])

    def test_audit_rejects_the_wrong_budget(self):
        report = {
            "protocol": "bounded supervised visual readout only 11x11x7 public visual frames labels identity IDs coordinates seeds actions rewards teachers planners never transfer to Luanti control",
            "train_seed": 1,
            "heldout_seed": 2,
            "train_steps": 241,
            "batch_size": 64,
            "device": "cpu",
            "heldout_accuracy": 0.95,
            "accepted_public_visual_binding": True,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = audit(path)
        self.assertFalse(result.accepted)
        self.assertIn("frozen calibration budget or device differs", result.failures)


if __name__ == "__main__":
    unittest.main()

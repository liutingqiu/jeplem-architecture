import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.audit_public_vision_fixed_identity_calibration import audit


class PublicVisionFixedIdentityCalibrationAuditTests(unittest.TestCase):
    def test_audit_recomputes_fixed_gate(self):
        report = {
            "protocol": "bounded supervised visual readout only one initial 11x11x7 public visual frame Fixed-identity target coordinates query seeds actions rewards teachers planners never transfer to Luanti control",
            "train_seed": 1,
            "heldout_seed": 2,
            "train_steps": 240,
            "batch_size": 64,
            "device": "cpu",
            "heldout_accuracy": 1.0,
            "accepted_fixed_identity_readout": True,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = audit(path)
        self.assertTrue(result.accepted)
        self.assertEqual(result.failures, [])

    def test_audit_rejects_wrong_seed_split(self):
        report = {
            "protocol": "bounded supervised visual readout only one initial 11x11x7 public visual frame Fixed-identity target coordinates query seeds actions rewards teachers planners never transfer to Luanti control",
            "train_seed": 1,
            "heldout_seed": 1,
            "train_steps": 240,
            "batch_size": 64,
            "device": "cpu",
            "heldout_accuracy": 1.0,
            "accepted_fixed_identity_readout": True,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = audit(path)
        self.assertFalse(result.accepted)
        self.assertIn("training and held-out seeds are not isolated", result.failures)


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.audit_luanti_micro_navigation_qlearning_report import audit, main


class LuantiMicroNavigationQLearningReportAuditTests(unittest.TestCase):
    def _payload(self):
        return {
            "protocol": "real Luanti visible-goal micro-navigation; no bridge state; teacher trajectory; external planner; PLWM",
            "executor": "local-cpu", "training_episodes": 128, "max_steps": 10,
            "training_seed_range": list(range(6_300_000, 6_300_128)),
            "heldout_seed_range": list(range(6_400_000, 6_400_016)),
            "random_evaluation": {"episodes": 16, "successes": 3, "action_counts": [4, 4, 4, 4]},
            "q_learning_evaluation": {"episodes": 16, "successes": 16, "action_counts": [4, 4, 4, 4]},
            "accepted_public_feature_baseline": True,
        }

    def test_valid_report_recomputes_the_gate(self):
        with TemporaryDirectory() as directory:
            source, target = Path(directory) / "report.json", Path(directory) / "audit.json"
            source.write_text(json.dumps(self._payload()), encoding="utf-8")
            outcome = main(source, target)
            self.assertTrue(outcome.accepted)
            self.assertTrue(json.loads(target.read_text(encoding="utf-8"))["accepted"])

    def test_tampered_acceptance_is_rejected(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            payload = self._payload()
            payload["q_learning_evaluation"]["successes"] = 4
            source.write_text(json.dumps(payload), encoding="utf-8")
            outcome = audit(source)
            self.assertFalse(outcome.accepted)
            self.assertIn("reported acceptance disagrees with recomputed gate", outcome.failures)


if __name__ == "__main__":
    unittest.main()

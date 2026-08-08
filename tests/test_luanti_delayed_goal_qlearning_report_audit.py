import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.audit_luanti_delayed_goal_qlearning_report import audit, main


class LuantiDelayedGoalQLearningReportAuditTests(unittest.TestCase):
    def _payload(self, course="long_delay"):
        return {
            "protocol": f"{course}; real Luanti delayed-goal micro-navigation; Reactive and retained-cue; no bridge state; teacher trajectory; external planner; PLWM",
            "executor": "local-cpu", "training_episodes": 128, "max_steps": 10,
            "training_seed_range": list(range(7_100_000, 7_100_128)),
            "heldout_seed_range": list(range(7_200_000, 7_200_016)),
            "random_evaluation": {"episodes": 16, "successes": 3, "action_counts": [4, 4, 4, 4]},
            "reactive_q_evaluation": {"episodes": 16, "successes": 8, "action_counts": [4, 4, 4, 4]},
            "retained_cue_q_evaluation": {"episodes": 16, "successes": 16, "action_counts": [4, 4, 4, 4]},
            "accepted_memory_diagnostic": True,
        }

    def test_valid_report_recomputes_the_gate(self):
        with TemporaryDirectory() as directory:
            source, target = Path(directory) / "report.json", Path(directory) / "audit.json"
            source.write_text(json.dumps(self._payload()), encoding="utf-8")
            outcome = main(source, target)
            self.assertTrue(outcome.accepted)
            self.assertTrue(json.loads(target.read_text(encoding="utf-8"))["accepted"])

    def test_tampered_retained_result_is_rejected(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            payload = self._payload()
            payload["retained_cue_q_evaluation"]["successes"] = 9
            source.write_text(json.dumps(payload), encoding="utf-8")
            outcome = audit(source)
            self.assertFalse(outcome.accepted)
            self.assertIn("reported acceptance disagrees with recomputed gate", outcome.failures)

    def test_gated_compact_contract_uses_its_own_seed_block(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            payload = self._payload(course="gated_compact_binary")
            payload["max_steps"] = 4
            payload["training_seed_range"] = list(range(9_100_000, 9_100_128))
            payload["heldout_seed_range"] = list(range(9_200_000, 9_200_016))
            source.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(audit(source, course="gated_compact_binary").accepted)

    def test_gated_replication_requires_its_disjoint_large_holdout_contract(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            payload = self._payload(course="gated_compact_binary_replication")
            payload["max_steps"] = 4
            payload["training_seed_range"] = list(range(9_300_000, 9_300_128))
            payload["heldout_seed_range"] = list(range(9_400_000, 9_400_064))
            payload["action_seed_base"] = 1_942_337
            for name in ("random_evaluation", "reactive_q_evaluation", "retained_cue_q_evaluation"):
                payload[name]["episodes"] = 64
                payload[name]["successes"] *= 4
            source.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(audit(source, course="gated_compact_binary_replication").accepted)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from experiments.audit_luanti_long_delay_dqn_report import audit


class LuantiLongDelayDQNReportAuditTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        results = []
        for name, successes in (
            ("no_memory", 31),
            ("gru", 64),
            ("transformer_2", 31),
            ("plwm_state_off", 31),
            ("plwm", 64),
        ):
            results.append({"name": name, "evaluation": {"episodes": 64, "successes": successes, "action_counts": [0, 320, 32, 32]}})
        return {
            "protocol": "real Luanti gated delayed-goal course; reward-only 11x11 public vision, previous chosen action, previous scalar reward, legal-action masks; no cue labels, symbolic bridge state, teacher trajectory, external planner, shaped reward",
            "course": "long_gated",
            "device": "cuda",
            "training_seed_range": list(range(10_500_000, 10_500_512)),
            "heldout_seed_range": list(range(10_600_000, 10_600_064)),
            "action_seed": 3_042_337,
            "max_steps": 8,
            "transformer_context": 2,
            "random_evaluation": {"episodes": 64},
            "results": results,
            "accepted_neural_task_diagnostic": True,
        }

    def test_valid_report_recomputes_the_frozen_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            source.write_text(json.dumps(self._payload()), encoding="utf-8")
            result = audit(source)
        self.assertTrue(result.accepted)
        self.assertTrue(result.recomputed_accepted)

    def test_tampered_transformer_result_is_rejected(self):
        payload = self._payload()
        payload["results"][2]["evaluation"]["successes"] = 64
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            result = audit(source)
        self.assertFalse(result.accepted)
        self.assertFalse(result.recomputed_accepted)


if __name__ == "__main__":
    unittest.main()

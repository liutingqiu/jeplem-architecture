import json
import tempfile
import unittest
from pathlib import Path

from experiments.audit_luanti_paired_delay_dqn_report import audit


class LuantiPairedDelayDQNReportAuditTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        results = []
        for name, successes in (
            ("no_memory", 50),
            ("gru", 128),
            ("transformer_2", 50),
            ("plwm_state_off", 50),
            ("plwm", 128),
        ):
            results.append({"name": name, "evaluation": {"episodes": 128, "successes": successes, "action_counts": [0, 768, 128, 128]}})
        return {
            "protocol": "real Luanti gated delayed-goal course; reward-only 11x11 public vision, previous chosen action, previous scalar reward, legal-action masks; no cue labels, symbolic bridge state, teacher trajectory, external planner, shaped reward",
            "course": "paired_gated",
            "device": "cuda",
            "training_seed_range": list(range(11_500_000, 11_501_024)),
            "heldout_seed_range": list(range(11_600_000, 11_600_128)),
            "action_seed": 4_042_337,
            "max_steps": 8,
            "transformer_context": 2,
            "random_evaluation": {"episodes": 128},
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
        payload["results"][2]["evaluation"]["successes"] = 128
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            result = audit(source)
        self.assertFalse(result.accepted)
        self.assertFalse(result.recomputed_accepted)

    def test_early_terminal_action_accounting_is_valid_but_out_of_range_is_rejected(self):
        payload = self._payload()
        payload["results"][0]["evaluation"]["action_counts"] = [0, 768, 0, 126]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            result = audit(source)
        self.assertTrue(result.accepted)

        payload["results"][0]["evaluation"]["action_counts"] = [0, 767, 0, 0]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            result = audit(source)
        self.assertFalse(result.accepted)


if __name__ == "__main__":
    unittest.main()

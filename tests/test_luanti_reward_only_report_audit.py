import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.audit_luanti_reward_only_report import audit, main


ROOT = Path(__file__).resolve().parents[1]


class LuantiRewardOnlyReportAuditTests(unittest.TestCase):
    def _formal_payload(self):
        return {
            "protocol": "real two-object Luanti worlds; no teacher trajectories; no external planning; no shaped reward enters training/evaluation",
            "device": "cuda",
            "updates": 16,
            "episodes_per_update": 4,
            "transformer_context": 16,
            "train_seed_range": list(range(4_300_000, 4_300_064)),
            "heldout_seed_range": [4_600_000 + index * 31 for index in range(8)],
            "results": [
                {"name": "plwm", "parameters": 2, "training_episodes": 64, "evaluation": {"successes": 5, "episodes": 8, "action_counts": [80, 80, 80, 80]}},
                {"name": "no_memory", "parameters": 2, "training_episodes": 64, "evaluation": {"successes": 4, "episodes": 8, "action_counts": [80, 80, 80, 80]}},
                {"name": "gru", "parameters": 2, "training_episodes": 64, "evaluation": {"successes": 4, "episodes": 8, "action_counts": [80, 80, 80, 80]}},
                {"name": "transformer_16", "parameters": 2, "training_episodes": 64, "evaluation": {"successes": 4, "episodes": 8, "action_counts": [80, 80, 80, 80]}},
            ],
            "accepted": True,
        }

    def test_valid_formal_report_is_recomputed_as_accepted(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "formal.json"
            target = Path(directory) / "audit.json"
            source.write_text(json.dumps(self._formal_payload()), encoding="utf-8")
            outcome = main(source, target)
            self.assertTrue(outcome.accepted)
            self.assertTrue(outcome.recomputed_accepted)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["accepted"], True)

    def test_tampered_report_cannot_inherit_accepted_boolean(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "formal.json"
            payload = self._formal_payload()
            payload["results"][0]["evaluation"]["successes"] = 0
            source.write_text(json.dumps(payload), encoding="utf-8")
            outcome = audit(source)
            self.assertFalse(outcome.accepted)
            self.assertIn("reported accepted field disagrees with recomputed gate", outcome.failures)

    def test_smoke_artifact_is_rejected_as_nonformal(self):
        outcome = audit(ROOT / "artifacts" / "luanti-reward-only-ppo-smoke-report.json")
        self.assertFalse(outcome.accepted)
        self.assertIn("training budget differs from frozen formal block", outcome.failures)


if __name__ == "__main__":
    unittest.main()

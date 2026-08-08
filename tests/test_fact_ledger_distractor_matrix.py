import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from experiments.audit_fact_ledger_distractor_matrix import run


class FactLedgerDistractorMatrixTests(unittest.TestCase):
    def test_matrix_beats_naive_last_write(self):
        with TemporaryDirectory() as directory:
            report = run(episodes=4, distractors_per_episode=8, artifact=Path(directory) / "report.json")
        self.assertTrue(report.accepted)
        self.assertEqual(report.ledger_successes, 4)
        self.assertLess(report.naive_last_write_successes, 4)

    def test_rejects_invalid_budget(self):
        with self.assertRaises(ValueError):
            run(episodes=0)


if __name__ == "__main__":
    unittest.main()

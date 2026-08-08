import json
import tempfile
import unittest
from pathlib import Path

from experiments.audit_luanti_reward_ledger_replay import run


class LuantiRewardLedgerReplayTests(unittest.TestCase):
    def test_replays_source_bound_two_fact_records(self):
        lines = []
        for index, value in enumerate((0.15, 0.2)):
            lines.append(json.dumps({
                "episode_id": "episode-a", "kind": "reward", "modality": "luanti-reward",
                "tick": index + 1, "payload": {"value": value, "caused_by": f"request-{index}"},
            }))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "records.jsonl"
            source.write_text("\n".join(lines), encoding="utf-8")
            report = run(source_log=source, episodes=1, artifact=Path(directory) / "report.json")
        self.assertTrue(report.accepted)


if __name__ == "__main__":
    unittest.main()

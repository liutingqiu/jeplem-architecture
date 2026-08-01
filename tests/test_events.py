import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from plwm.events import EpisodeTrace, observation_payload


class EventProtocolTests(unittest.TestCase):
    def test_hash_chained_multimodal_trace_round_trips(self):
        trace = EpisodeTrace("episode-1")
        trace.append("observation", "minecraft-state", observation_payload(vector=[1, 2], text="near tree", state={"x": 1, "y": 64}))
        trace.append("action", "minecraft-action", {"name": "move_forward", "ticks": 4})
        trace.append("outcome", "minecraft-state", {"position": [1, 64, 4]})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace.write_jsonl(path)
            restored = EpisodeTrace.read_jsonl(path)
        self.assertEqual(restored.events[-1].payload["position"], [1, 64, 4])
        self.assertEqual(restored.events[-1].previous_hash, restored.events[-2].event_hash)

    def test_tampered_trace_fails_closed(self):
        trace = EpisodeTrace("episode-2")
        trace.append("reward", "scalar", {"value": 1.0})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace.write_jsonl(path)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["payload"]["value"] = 99.0
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                EpisodeTrace.read_jsonl(path)

    def test_observation_rejects_empty_payload(self):
        with self.assertRaises(ValueError):
            observation_payload()


if __name__ == "__main__":
    unittest.main()

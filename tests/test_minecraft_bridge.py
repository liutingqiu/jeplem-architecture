import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from plwm.continual import EndToEndPersistentPLWM
from plwm.minecraft import MinecraftFileBridge, MinecraftRuntimeSession, MinecraftStateEncoder
from plwm.runtime import TrainablePersistentRuntime


class MinecraftBridgeTests(unittest.TestCase):
    def test_raw_mod_records_become_hash_chained_episode(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"episode_id": "e1", "tick": 1, "kind": "observation", "modality": "minecraft-state", "payload": {"health": 20}},
                {"episode_id": "e1", "tick": 2, "kind": "outcome", "modality": "bridge", "payload": {"status": "accepted"}},
            ]
            (root / "observations.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            trace = MinecraftFileBridge(root).read_episode("e1")
        self.assertEqual(len(trace.events), 2)
        self.assertEqual(trace.events[1].previous_hash, trace.events[0].event_hash)

    def test_only_allowlisted_actions_are_written(self):
        with TemporaryDirectory() as directory:
            bridge = MinecraftFileBridge(Path(directory))
            bridge.write_action("set_episode", episode_id="e2")
            bridge.write_action("observe")
            bridge.write_action("reset_episode")
            with self.assertRaises(ValueError):
                bridge.write_action("run_command")
            actions = bridge.actions_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(actions), 3)
        self.assertEqual({json.loads(line)["name"] for line in actions}, {"set_episode", "observe", "reset_episode"})
        self.assertTrue(all("request_id" in json.loads(line) for line in actions))

    def test_runtime_session_matches_each_outcome_to_its_pending_action(self):
        with TemporaryDirectory() as directory:
            runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
            session = MinecraftRuntimeSession(MinecraftFileBridge(Path(directory)), runtime, "e4")
            request_id = session.begin()
            session.ingest_outcome({"name": "set_episode", "status": "accepted", "request_id": request_id})
            self.assertFalse(session.pending_action_ids)
            with self.assertRaises(ValueError):
                session.ingest_outcome({"name": "observe", "status": "accepted", "request_id": request_id})
        self.assertEqual(session.trace.events[-1].kind, "outcome")

    def test_runtime_session_keeps_episode_trace_and_persistent_tick(self):
        with TemporaryDirectory() as directory:
            runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
            session = MinecraftRuntimeSession(MinecraftFileBridge(Path(directory)), runtime, "e3")
            session.begin()
            session.request_observation()
            tick = session.ingest_observation({"health": 20, "food": 20, "x": 3, "z": 5, "game_time": 1200})
        self.assertEqual(runtime.ledger.time, 1)
        self.assertEqual(len(session.trace.events), 3)
        self.assertEqual(len(MinecraftStateEncoder.encode({})), 5)
        self.assertGreaterEqual(tick.belief_confidence, 0.5)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from plwm.continual import EndToEndPersistentPLWM
from plwm.luanti import LuantiFileBridge, LuantiRuntimeSession, LuantiStateEncoder
from plwm.runtime import TrainablePersistentRuntime


class LuantiBridgeTests(unittest.TestCase):
    def test_raw_mod_records_become_hash_chained_episode(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"episode_id": "e1", "tick": 1, "kind": "observation", "modality": "luanti-state", "payload": {"health": 20}},
                {"episode_id": "e1", "tick": 2, "kind": "outcome", "modality": "bridge", "payload": {"status": "accepted"}},
            ]
            (root / "observations.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            trace = LuantiFileBridge(root).read_episode("e1")
        self.assertEqual(len(trace.events), 2)
        self.assertEqual(trace.events[1].previous_hash, trace.events[0].event_hash)

    def test_only_allowlisted_actions_are_written(self):
        with TemporaryDirectory() as directory:
            bridge = LuantiFileBridge(Path(directory))
            bridge.write_action("set_episode", episode_id="e2", task_seed=7)
            bridge.write_action("observe")
            bridge.write_action("step", direction="north")
            bridge.write_action("recenter")
            bridge.write_action("reset_episode")
            with self.assertRaises(ValueError):
                bridge.write_action("run_command")
            with self.assertRaises(ValueError):
                bridge.write_action("step", direction="up")
            actions = bridge.actions_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(actions), 5)
        self.assertEqual({json.loads(line)["name"] for line in actions}, {"set_episode", "observe", "step", "recenter", "reset_episode"})
        self.assertEqual(json.loads(actions[0])["task_seed"], 7)
        self.assertTrue(all("request_id" in json.loads(line) for line in actions))
        with self.assertRaises(ValueError):
            bridge.write_action("observe", task_seed=1)

    def test_runtime_session_records_only_acknowledged_numeric_rewards(self):
        with TemporaryDirectory() as directory:
            bridge = LuantiFileBridge(Path(directory))
            runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
            session = LuantiRuntimeSession(bridge, runtime, "reward-session", task_seed=3)
            request_id = session.begin()
            session.ingest_outcome({"request_id": request_id, "status": "accepted"})
            session.ingest_reward({"caused_by": request_id, "value": 0.2})
            self.assertEqual(session.trace.events[-1].kind, "reward")
            with self.assertRaises(ValueError):
                session.ingest_reward({"caused_by": "unknown", "value": 0.2})
            with self.assertRaises(ValueError):
                session.ingest_reward({"caused_by": request_id, "value": True})

    def test_runtime_session_matches_each_outcome_to_its_pending_action(self):
        with TemporaryDirectory() as directory:
            runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
            session = LuantiRuntimeSession(LuantiFileBridge(Path(directory)), runtime, "e4")
            request_id = session.step("east")
            session.ingest_outcome({"name": "step", "status": "accepted", "request_id": request_id})
            self.assertFalse(session.pending_action_ids)
            with self.assertRaises(ValueError):
                session.ingest_outcome({"name": "step", "status": "accepted", "request_id": request_id})
        self.assertEqual(session.trace.events[-1].kind, "outcome")

    def test_action_caused_observation_requires_an_acknowledged_action(self):
        with TemporaryDirectory() as directory:
            runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
            session = LuantiRuntimeSession(LuantiFileBridge(Path(directory)), runtime, "e5")
            request_id = session.reset_episode()
            with self.assertRaises(ValueError):
                session.ingest_observation({"caused_by": request_id})
            session.ingest_outcome({"name": "reset_episode", "status": "accepted", "request_id": request_id})
            session.ingest_observation({"health": 20, "energy": 20, "x": 0, "z": 0, "game_time": 0, "caused_by": request_id})
        self.assertEqual(session.trace.events[-1].kind, "observation")

    def test_runtime_session_keeps_episode_trace_and_persistent_tick(self):
        with TemporaryDirectory() as directory:
            runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
            session = LuantiRuntimeSession(LuantiFileBridge(Path(directory)), runtime, "e3")
            session.begin()
            session.request_observation()
            tick = session.ingest_observation({"health": 20, "energy": 20, "x": 3, "z": 5, "game_time": 1200})
        self.assertEqual(runtime.ledger.time, 1)
        self.assertEqual(len(session.trace.events), 3)
        self.assertEqual(len(LuantiStateEncoder.encode({})), 5)
        self.assertGreaterEqual(tick.belief_confidence, 0.5)


if __name__ == "__main__":
    unittest.main()

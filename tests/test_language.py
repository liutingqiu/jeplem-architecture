import unittest

from plwm.commitments import Commitment
from plwm.continual import EndToEndPersistentPLWM
from plwm.language import JsonLanguagePort
from plwm.runtime import TrainablePersistentRuntime


class StubBackend:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, prompt: str) -> str:
        return next(self.responses)


class LanguagePortTests(unittest.TestCase):
    def test_strict_json_evidence_flows_through_runtime_without_reset(self):
        port = JsonLanguagePort(StubBackend(['{"route":"right","confidence":0.9}', '{"route":"left","confidence":0.8}']))
        runtime = TrainablePersistentRuntime(EndToEndPersistentPLWM(input_dim=5, latent_dim=12), belief_states=2)
        runtime.add_commitment(Commitment("route", target_state=1, priority=1.0))
        first = port.apply_to_runtime(runtime, port.encode_observation("first"))
        snapshot = runtime.snapshot()
        resumed = TrainablePersistentRuntime.restore(snapshot, runtime.model, 2)
        second = port.apply_to_runtime(resumed, port.encode_observation("second"))
        self.assertEqual(resumed.ledger.time, 2)
        self.assertGreater(first.prediction_error, 0.0)
        self.assertGreaterEqual(second.belief_confidence, 0.5)
        self.assertEqual(port.emit_action("think"), "action:think")

    def test_rejects_non_json_or_unsupported_fields(self):
        port = JsonLanguagePort()
        with self.assertRaises(ValueError):
            port.decode_evidence("right")
        with self.assertRaises(ValueError):
            port.decode_evidence('{"route":"up","confidence":0.9}')


if __name__ == "__main__":
    unittest.main()

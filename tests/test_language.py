import unittest

from plwm.commitments import Commitment
from plwm.continual import EndToEndPersistentPLWM
from plwm.language import JsonLanguagePort, parse_luanti_language_goal
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

    def test_luanti_goal_parser_accepts_only_declared_object_delivery_schema(self):
        goal = parse_luanti_language_goal(
            '{"task":"object_delivery","target":"goal","require_key":true,"avoid_decoy":true}'
        )
        self.assertTrue(goal.require_key)
        charged = parse_luanti_language_goal(
            '{"task":"charged_object_delivery","target":"goal","require_key":true,"avoid_decoy":true,"require_charge":true}'
        )
        self.assertTrue(charged.require_charge)
        polarity = parse_luanti_language_goal(
            '{"task":"polarity_object_delivery","target":"goal","require_key":true,"require_charge":true,"select_charged_object":true}'
        )
        self.assertTrue(polarity.select_charged_object)
        self.assertFalse(polarity.avoid_decoy)
        with self.assertRaises(ValueError):
            parse_luanti_language_goal('{"task":"run_command","target":"goal","require_key":true,"avoid_decoy":true}')
        with self.assertRaises(ValueError):
            parse_luanti_language_goal('{"task":"charged_object_delivery","target":"goal","require_key":true,"avoid_decoy":true}')


if __name__ == "__main__":
    unittest.main()

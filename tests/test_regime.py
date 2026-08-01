import unittest

from plwm.regime import RegimeMemory


class RegimeMemoryTests(unittest.TestCase):
    def test_verified_switch_overrides_old_rule_but_not_another_world(self):
        memory = RegimeMemory()
        memory.ingest("a", 1, confidence=0.99, verified=True, source="initial")
        memory.ingest("b", -1, confidence=0.99, verified=True, source="initial")
        memory.ingest("a", -1, confidence=0.99, verified=True, source="switch")
        memory.ingest("a", 1, confidence=0.5, verified=False, source="stale")
        self.assertEqual(memory.regime("a"), -1)
        self.assertEqual(memory.regime("b"), -1)
        self.assertEqual(memory.plan_action("a", -1), 1)

    def test_invalid_regime_is_rejected(self):
        with self.assertRaises(ValueError):
            RegimeMemory().ingest("a", 0, confidence=1.0, verified=True, source="bad")

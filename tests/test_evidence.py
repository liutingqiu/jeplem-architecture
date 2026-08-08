import unittest

from plwm.evidence import EvidenceFreshnessGate, VersionedEvidence


class EvidenceFreshnessTests(unittest.TestCase):
    def test_same_tick_can_accept_two_modalities_but_duplicate_is_stale(self):
        gate = EvidenceFreshnessGate("world")
        visual = VersionedEvidence("world", 0, 1, "vision", {"frame": 1})
        language = VersionedEvidence("world", 0, 1, "language", {"goal": "goal"})
        self.assertTrue(gate.accept(visual).accepted)
        self.assertTrue(gate.accept(language).accepted)
        self.assertFalse(gate.accept(visual).accepted)

    def test_mechanism_revision_rejects_old_visual_and_language_records(self):
        gate = EvidenceFreshnessGate("world")
        old_visual = VersionedEvidence("world", 0, 1, "vision", {})
        old_language = VersionedEvidence("world", 0, 1, "language", {})
        gate.accept(old_visual)
        gate.accept(old_language)
        self.assertEqual(gate.advance_mechanism(), 1)
        self.assertEqual(gate.accept(old_visual).reason, "mechanism-version-stale")
        self.assertEqual(gate.accept(old_language).reason, "mechanism-version-stale")
        fresh = VersionedEvidence("world", 1, 2, "vision", {})
        self.assertTrue(gate.accept(fresh).accepted)

    def test_cross_world_records_are_rejected(self):
        gate = EvidenceFreshnessGate("world-a")
        result = gate.accept(VersionedEvidence("world-b", 0, 1, "vision", {}))
        self.assertEqual(result.reason, "world-mismatch")


if __name__ == "__main__":
    unittest.main()

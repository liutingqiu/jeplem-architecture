import unittest

from plwm.lifelong import VersionedFactMemory


class LifelongMemoryTests(unittest.TestCase):
    def test_verified_update_survives_later_stale_burst(self):
        memory = VersionedFactMemory()
        memory.ingest(2, 1, confidence=0.99, source="bootstrap", verified=True)
        memory.ingest(2, 0, confidence=0.99, source="update", verified=True)
        for index in range(20):
            memory.ingest(2, index % 2, confidence=0.5, source="stale", verified=False)
        memory.sleep()
        self.assertEqual(memory.retrieve(2), 0)
        self.assertTrue(any(item.source == "update" for item in memory.provenance(2)))

    def test_unrelated_facts_are_not_overwritten(self):
        memory = VersionedFactMemory()
        memory.ingest(1, 7, confidence=0.99, source="bootstrap", verified=True)
        memory.ingest(2, 8, confidence=0.99, source="bootstrap", verified=True)
        memory.ingest(1, 9, confidence=0.99, source="update", verified=True)
        self.assertEqual(memory.retrieve(2), 8)

    def test_sleep_consolidates_repeated_consistent_evidence(self):
        memory = VersionedFactMemory()
        for _ in range(3):
            memory.ingest(4, 11, confidence=0.83, source="replay", verified=False)
        memory.sleep()
        self.assertEqual(memory.retrieve(4), 11)
        self.assertEqual(memory.consolidation_count, 1)


if __name__ == "__main__":
    unittest.main()

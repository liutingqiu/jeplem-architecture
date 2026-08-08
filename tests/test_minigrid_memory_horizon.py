import unittest

from experiments.audit_minigrid_memory_horizon import audit_episode, run


class MiniGridMemoryHorizonTests(unittest.TestCase):
    def test_audits_a_real_memory_route_without_training_data(self):
        row = audit_episode("MiniGrid-MemoryS7-v0", 3_000_000)
        self.assertTrue(row.source_seen)
        self.assertIsNotNone(row.source_to_branch_actions)
        self.assertGreaterEqual(row.source_to_branch_actions, 1)

    def test_rejects_invalid_audit_budget(self):
        with self.assertRaises(ValueError):
            run(episodes=0)


if __name__ == "__main__":
    unittest.main()

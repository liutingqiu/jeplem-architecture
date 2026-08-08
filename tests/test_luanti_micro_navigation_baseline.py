import unittest

from experiments.audit_luanti_micro_navigation_baseline import run


class LuantiMicroNavigationBaselineTests(unittest.TestCase):
    def test_invalid_contract_is_rejected_before_initializing_sb3(self):
        with self.assertRaises(ValueError):
            run(total_timesteps=128)
        with self.assertRaises(ValueError):
            run(training_seed_start=5_999_999)
        with self.assertRaises(ValueError):
            run(training_seed_start=6_100_000, heldout_seed_start=6_100_000)


if __name__ == "__main__":
    unittest.main()

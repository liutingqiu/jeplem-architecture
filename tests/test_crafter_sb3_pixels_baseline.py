import unittest

from experiments.audit_crafter_sb3_pixels_baseline import run


class CrafterSb3BaselineTests(unittest.TestCase):
    def test_invalid_budget_is_rejected_without_starting_training(self):
        with self.assertRaises(ValueError):
            run(total_timesteps=32)


if __name__ == "__main__":
    unittest.main()

import unittest

from experiments.audit_minigrid_visible_object_separability import run


class VisibleObjectSeparabilityTests(unittest.TestCase):
    def test_rejects_invalid_training_size(self):
        with self.assertRaises(ValueError):
            run(training_examples=1)


if __name__ == "__main__":
    unittest.main()

import unittest

try:
    import crafter  # noqa: F401
    _CRAFTER_AVAILABLE = True
except ImportError:
    _CRAFTER_AVAILABLE = False

from plwm.crafter_env import GymPixelsOnlyCrafter, PixelsOnlyCrafter


@unittest.skipUnless(_CRAFTER_AVAILABLE, "crafter not installed (build fails on Python 3.13)")
class CrafterEnvironmentTests(unittest.TestCase):
    def test_exposes_only_rgb_transition_values(self):
        environment = PixelsOnlyCrafter(seed=7, length=2)
        observation = environment.reset()
        self.assertEqual(observation.shape, (64, 64, 3))
        transition = environment.step(0)
        self.assertEqual(transition.observation.shape, (64, 64, 3))
        self.assertIsInstance(transition.reward, float)
        with self.assertRaises(ValueError):
            environment.step(17)

    def test_gym_boundary_returns_no_crafter_info_fields(self):
        environment = GymPixelsOnlyCrafter(seed=7, length=2)
        observation, info = environment.reset()
        self.assertEqual(observation.shape, (64, 64, 3))
        self.assertEqual(info, {})
        observation, reward, terminated, truncated, info = environment.step(0)
        self.assertEqual(observation.shape, (64, 64, 3))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertFalse(truncated)
        self.assertEqual(info, {})


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from plwm.world_model import ActionConditionedPLWM


class WorldModelTests(unittest.TestCase):
    def test_counterfactual_actions_have_distinct_predictions(self):
        torch.manual_seed(4)
        model = ActionConditionedPLWM(3, latent_dim=12)
        state = model.initial_state(2, "cpu")
        state = model.advance(torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), state)
        left = model.imagine(state, torch.zeros(2, dtype=torch.long))
        right = model.imagine(state, torch.ones(2, dtype=torch.long))
        self.assertFalse(torch.allclose(left, right))

    def test_planning_returns_one_action_per_state(self):
        model = ActionConditionedPLWM(3, latent_dim=12)
        state = model.initial_state(3, "cpu")
        actions, imagined = model.plan(state, torch.tensor([0, 1, 2]))
        self.assertEqual(tuple(actions.shape), (3,))
        self.assertEqual(tuple(imagined.shape), (3, 2, 3))


if __name__ == "__main__":
    unittest.main()

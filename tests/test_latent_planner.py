import unittest

import torch

from plwm.latent_planner import ActionConditionedLatentPlanner


class LatentPlannerTests(unittest.TestCase):
    def test_observation_action_transition_and_plan_shapes(self):
        model = ActionConditionedLatentPlanner(6, action_count=3, latent_dim=12)
        state = model.initial_state(2, torch.device("cpu"))
        observed = model.observe(torch.randn(2, 6), state)
        next_state, prediction, reward = model.predict(observed, torch.tensor([0, 2]))
        self.assertEqual(tuple(next_state.shape), (2, 12))
        self.assertEqual(tuple(prediction.shape), (2, 6))
        self.assertEqual(tuple(reward.shape), (2,))
        actions, scores = model.plan(observed, horizon=3)
        self.assertEqual(tuple(actions.shape), (2, 3))
        self.assertEqual(tuple(scores.shape), (2, 27))

    def test_transition_rejects_out_of_range_actions(self):
        model = ActionConditionedLatentPlanner(4, action_count=2)
        with self.assertRaises(ValueError):
            model.transition(model.initial_state(1, "cpu"), torch.tensor([2]))


if __name__ == "__main__":
    unittest.main()

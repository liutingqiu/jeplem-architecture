import unittest

import torch

from experiments.audit_crafter_reward_only_sentinel import GRUPixelActorCritic, ReactivePixelActorCritic, _policy_step, run


class CrafterRewardOnlySentinelTests(unittest.TestCase):
    def test_models_emit_policy_and_value_sequences(self):
        observations = torch.zeros((2, 3, 3, 16, 16))
        valid = torch.ones((2, 3), dtype=torch.bool)
        for model in (ReactivePixelActorCritic(), GRUPixelActorCritic()):
            logits, values = model(observations, valid)
            self.assertEqual(tuple(logits.shape), (2, 3, 17))
            self.assertEqual(tuple(values.shape), (2, 3))

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            run(updates=0)

    def test_gru_uses_causal_prior_event_and_can_train_after_evaluation(self):
        model = GRUPixelActorCritic()
        observations = torch.zeros((1, 2, 3, 16, 16))
        valid = torch.ones((1, 2), dtype=torch.bool)
        events = torch.zeros((1, 2, 18))
        events[0, 1, 4] = 1.0
        events[0, 1, -1] = 0.25
        model.eval()
        with torch.no_grad():
            baseline, _ = model(observations, valid, torch.zeros_like(events))
            conditioned, _ = model(observations, valid, events)
        self.assertFalse(torch.equal(baseline, conditioned))
        model.train()
        logits, values = model(observations, valid, events)
        (logits.square().mean() + values.square().mean()).backward()
        self.assertIsNotNone(model.core.weight_hh_l0.grad)

    def test_streaming_gru_matches_full_causal_sequence(self):
        torch.manual_seed(7)
        model = GRUPixelActorCritic().eval()
        observations = torch.rand((1, 3, 3, 16, 16))
        valid = torch.ones((1, 3), dtype=torch.bool)
        events = torch.zeros((1, 3, 18))
        events[0, 1, 2] = 1.0; events[0, 1, -1] = 0.1
        events[0, 2, 5] = 1.0; events[0, 2, -1] = -0.1
        with torch.no_grad():
            full, _values = model(observations, valid, events)
            state = None
            streamed = []
            for index in range(3):
                logits, state = _policy_step(model, observations[0, index], events[0, index], state)
                streamed.append(logits)
        self.assertTrue(torch.allclose(full[0], torch.stack(streamed), atol=1e-6))


if __name__ == "__main__":
    unittest.main()

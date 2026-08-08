import copy
import random
import unittest

import numpy as np
import torch

from experiments.train_luanti_gated_memory_dqn import (
    CAPACITY_MATCHED_WIDTHS,
    OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS,
    COURSES,
    GRUQ,
    INPUT_DIM,
    PARAMETER_MATCH_TOLERANCE,
    NoMemoryQ,
    PLWMQ,
    ObjectTokenNoMemoryQ,
    ObjectTokenGRUQ,
    ObjectTokenTransformerQ,
    ObjectTokenPLWMQ,
    TransformerQ,
    Episode,
    _learn,
    _mask,
    comparison_factories,
    object_token_comparison_factories,
    object_token_parameter_match_errors,
    parameter_match_errors,
    run,
    validate_parameter_match,
    validate_object_token_parameter_match,
)
from plwm.luanti_reward_only_env import VISION_SHAPE


class LuantiGatedMemoryDQNTests(unittest.TestCase):
    def test_all_q_models_preserve_batch_time_action_shape(self):
        observations = torch.zeros((2, 2, INPUT_DIM))
        for model in (NoMemoryQ(), GRUQ(), TransformerQ(), PLWMQ(persistent_state=False), PLWMQ(persistent_state=True)):
            self.assertEqual(tuple(model(observations).shape), (2, 2, 4))

    def test_object_token_comparison_models_preserve_the_same_public_sequence_shape(self):
        observations = torch.zeros((2, 6, INPUT_DIM))
        models = (
            ObjectTokenNoMemoryQ(OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS.no_memory),
            ObjectTokenGRUQ(OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS.gru),
            ObjectTokenTransformerQ(OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS.transformer),
            ObjectTokenPLWMQ(OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS.plwm, persistent_state=False),
            ObjectTokenPLWMQ(OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS.plwm, persistent_state=True),
        )
        for model in models:
            self.assertEqual(tuple(model(observations).shape), (2, 6, 4))

    def test_transformer_slides_its_declared_two_event_context(self):
        torch.manual_seed(7)
        model = TransformerQ().eval()
        observations = torch.randn((1, 6, INPUT_DIM))
        changed_history = observations.clone()
        changed_history[:, :4] += 100.0
        with torch.no_grad():
            original = model(observations)
            changed = model(changed_history)
        self.assertEqual(tuple(original.shape), (1, 6, 4))
        self.assertTrue(torch.allclose(original[:, -1], changed[:, -1]))

    def test_public_wall_mask_exposes_only_free_adjacent_cells(self):
        observation = np.zeros(VISION_SHAPE, dtype=np.uint8)
        observation[5, 5, 2] = 1
        observation[4, 5, 1] = 1
        observation[5, 6, 1] = 1
        self.assertEqual(_mask(observation).tolist(), [False, True, False, True])

    def test_invalid_contract_is_rejected_before_connecting(self):
        with self.assertRaises(ValueError):
            run(training_episodes=64)
        with self.assertRaises(ValueError):
            run(training_seed_start=9_500_000, heldout_seed_start=9_500_000)
        with self.assertRaises(ValueError):
            run(course="long_gated", training_seed_start=9_500_000)
        with self.assertRaises(ValueError):
            run(acceptance_mode="invalid")
        with self.assertRaises(ValueError):
            run(course="long_gated", acceptance_mode="addressed_memory_diagnostic")

    def test_long_gated_course_has_a_fixed_horizon_beyond_transformer_context(self):
        self.assertEqual(COURSES["long_gated"], (10_000_000, 11_000_000, 8))

    def test_paired_gated_course_has_two_terminal_choices_beyond_transformer_context(self):
        self.assertEqual(COURSES["paired_gated"], (11_000_000, 12_000_000, 8))

    def test_addressed_gated_course_has_a_fixed_six_event_retrieval_horizon(self):
        self.assertEqual(COURSES["addressed_gated"], (12_000_000, 13_000_000, 6))

    def test_future_comparison_factories_are_parameter_matched_to_plwm(self):
        names = [name for name, _factory in comparison_factories(CAPACITY_MATCHED_WIDTHS)]
        self.assertEqual(names, ["no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"])
        errors = validate_parameter_match(CAPACITY_MATCHED_WIDTHS)
        self.assertEqual(errors, parameter_match_errors(CAPACITY_MATCHED_WIDTHS))
        self.assertTrue(all(error <= PARAMETER_MATCH_TOLERANCE for error in errors.values()))

    def test_object_token_comparison_factories_are_parameter_matched(self):
        names = [name for name, _factory in object_token_comparison_factories(OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS)]
        self.assertEqual(names, ["no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"])
        errors = validate_object_token_parameter_match(OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS)
        self.assertEqual(errors, object_token_parameter_match_errors(OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS))
        self.assertTrue(all(error <= PARAMETER_MATCH_TOLERANCE for error in errors.values()))

    def test_unmatched_widths_are_rejected_before_a_run_connects_to_luanti(self):
        with self.assertRaises(ValueError):
            run(model_widths=CAPACITY_MATCHED_WIDTHS.__class__(gru=96))

    def test_bellman_update_accepts_a_terminal_short_paired_episode(self):
        def episode(length: int) -> Episode:
            return Episode(
                torch.zeros((length, INPUT_DIM)),
                torch.zeros(length, dtype=torch.long),
                torch.zeros(length),
                torch.ones((length, 4), dtype=torch.bool),
            )

        model = NoMemoryQ()
        target = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _learn(
            model, target, [episode(8), episode(6)] * 16, batch_size=32,
            device=torch.device("cpu"), optimizer=optimizer, generator=random.Random(7),
        )
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))

    def test_plwm_update_trains_the_causal_predictor_auxiliary(self):
        episode = Episode(
            torch.randn((6, INPUT_DIM)),
            torch.zeros(6, dtype=torch.long),
            torch.zeros(6),
            torch.ones((6, 4), dtype=torch.bool),
        )
        model = PLWMQ(persistent_state=True)
        target = copy.deepcopy(model)
        before = model.core.predictor.weight.detach().clone()
        _learn(
            model, target, [episode] * 32, batch_size=32,
            device=torch.device("cpu"), optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3), generator=random.Random(9),
        )
        self.assertFalse(torch.equal(before, model.core.predictor.weight))
        self.assertTrue(torch.isfinite(model.core.predictor.weight).all())


if __name__ == "__main__":
    unittest.main()

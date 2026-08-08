import unittest

import gymnasium as gym
import minigrid  # noqa: F401
import torch

from experiments.audit_reward_only_policy_learnability import observations_for
from experiments.train_minigrid_memory_actor_critic import (
    GRUActorCritic, HybridPLWMTransformerActorCritic, INPUT_DIM, NoMemoryActorCritic, PLWMActorCritic,
    ACTION_COUNT, ObservationVisitCounter, TransformerActorCritic, _gae, _mask_actions, encode_observation, evaluate, evaluate_random, make_environment,
)
from experiments.audit_minigrid_hybrid_curriculum import run as run_hybrid_curriculum


class MiniGridActorCriticTests(unittest.TestCase):
    def test_encodes_standard_memory_observation(self):
        environment = make_environment()
        observation, _info = environment.reset(seed=7)
        self.assertEqual(tuple(encode_observation(observation).shape), (INPUT_DIM,))
        causal = encode_observation(observation, previous_action=2, previous_reward=0.5, observation_mode="causal_action_reward")
        self.assertEqual(float(causal[-1]), 0.5)
        self.assertEqual(float(causal[-1 - 7 + 2]), 1.0)
        environment.close()

    def test_navigation_sentinel_uses_the_same_observation_contract(self):
        environment = make_environment("MiniGrid-Empty-5x5-v0")
        observation, _info = environment.reset(seed=7)
        self.assertEqual(tuple(encode_observation(observation).shape), (INPUT_DIM,))
        environment.close()

    def test_reward_only_sentinel_encodes_only_the_visible_context(self):
        targets = torch.tensor([0, 1, 2])
        observations = observations_for(targets)
        self.assertEqual(tuple(observations.shape), (3, 1, INPUT_DIM))
        self.assertTrue(torch.equal(observations[:, 0, :3], torch.eye(3)))
        self.assertEqual(int(observations[:, 0, 3:].count_nonzero()), 0)

    def test_models_emit_causal_action_and_value_sequences(self):
        observations = torch.zeros((2, 5, INPUT_DIM))
        valid = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
        for model in (
            NoMemoryActorCritic(), GRUActorCritic(), TransformerActorCritic(), PLWMActorCritic(),
            PLWMActorCritic(use_memory=False, gated_memory=False, thought_steps=0),
            HybridPLWMTransformerActorCritic(), HybridPLWMTransformerActorCritic(use_plwm_side=False),
        ):
            logits, values = model(observations, valid)
            self.assertEqual(tuple(logits.shape), (2, 5, 7))
            self.assertEqual(tuple(values.shape), (2, 5))

    def test_hybrid_ablation_excludes_its_plwm_side_parameters(self):
        full = HybridPLWMTransformerActorCritic()
        ablation = HybridPLWMTransformerActorCritic(use_plwm_side=False)
        self.assertLess(
            sum(parameter.numel() for parameter in ablation.parameters() if parameter.requires_grad),
            sum(parameter.numel() for parameter in full.parameters() if parameter.requires_grad),
        )

    def test_streaming_sampler_matches_causal_prefix_outputs(self):
        observations = torch.randn((2, 4, INPUT_DIM))
        valid = torch.ones((2, 4), dtype=torch.bool)
        active = torch.ones(2, dtype=torch.bool)
        for model in (
            NoMemoryActorCritic(), GRUActorCritic(), PLWMActorCritic(),
            PLWMActorCritic(observation_skip=True), PLWMActorCritic(separate_causal_event=True),
        ):
            state = model.initial_sampling_state(2, torch.device("cpu"))
            sampled = []
            for index in range(observations.shape[1]):
                logits, state = model.sampling_logits(observations[:, index], state, active)
                sampled.append(logits)
            expected, _values = model(observations, valid)
            self.assertTrue(torch.allclose(torch.stack(sampled, dim=1), expected, atol=1e-6))

    def test_plwm_predictive_loss_is_differentiable_and_ignores_padding(self):
        model = PLWMActorCritic()
        observations = torch.randn((2, 4, INPUT_DIM))
        valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
        baseline = model.predictive_loss(observations, valid)
        padded = observations.clone()
        padded[1, 2:] = torch.randn_like(padded[1, 2:]) * 100.0
        self.assertTrue(torch.allclose(baseline, model.predictive_loss(padded, valid), atol=1e-6))
        baseline.backward()
        self.assertIsNotNone(model.core.predictor.weight.grad)

    def test_evaluation_reports_every_executed_action(self):
        result = evaluate(NoMemoryActorCritic(), episodes=2, start_seed=1_000_000, device=torch.device("cpu"))
        self.assertEqual(len(result.action_counts), 7)
        self.assertEqual(sum(result.action_counts), int(result.mean_steps * result.episodes))

    def test_random_evaluation_is_seeded_and_accounted(self):
        first = evaluate_random(episodes=2, start_seed=1_000_000)
        second = evaluate_random(episodes=2, start_seed=1_000_000)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.action_counts), int(first.mean_steps * first.episodes))

    def test_movement_curriculum_masks_non_movement_actions(self):
        masked = _mask_actions(torch.zeros((2, ACTION_COUNT)), "movement_only")
        self.assertTrue(torch.isfinite(masked[:, :3]).all())
        self.assertTrue((masked[:, 3:] == torch.finfo(masked.dtype).min).all())
        result = evaluate_random(episodes=2, start_seed=1_000_000, action_mode="movement_only")
        self.assertEqual(sum(result.action_counts[3:]), 0)

    def test_gae_uses_only_observed_terminal_rewards(self):
        rewards = [[0.0, 1.0], [0.0]]
        values = torch.zeros((2, 2))
        valid = torch.tensor([[True, True], [True, False]])
        advantages, returns = _gae(rewards, values, valid)
        self.assertAlmostEqual(float(advantages[0, 0]), 0.99 * 0.95)
        self.assertEqual(float(advantages[0, 1]), 1.0)
        self.assertEqual(float(advantages[1, 0]), 0.0)
        self.assertTrue(torch.equal(advantages, returns))

    def test_hybrid_curriculum_rejects_invalid_budgets(self):
        with self.assertRaises(ValueError):
            run_hybrid_curriculum(bootstrap_updates=0)
        with self.assertRaises(ValueError):
            run_hybrid_curriculum(experiment_seed=-1)

    def test_visit_bonus_decays_for_the_same_visible_observation(self):
        counter = ObservationVisitCounter()
        observation = torch.zeros(INPUT_DIM)
        self.assertAlmostEqual(counter.reward(observation, 0.05), 0.05)
        self.assertAlmostEqual(counter.reward(observation, 0.05), 0.05 / 2**0.5)
        self.assertEqual(counter.reward(observation, 0.0), 0.0)

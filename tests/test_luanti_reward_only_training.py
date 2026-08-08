import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from experiments.train_luanti_reward_only import (
    ACTION_COUNT, INPUT_DIM, GRUActorCritic, NoMemoryActorCritic, PLWMActorCritic,
    TRANSFORMER_CONTEXT, TransformerActorCritic, _gae, _write_json_atomic, encode_observation, run,
)
from plwm.luanti_reward_only_env import VISION_SHAPE


class LuantiRewardOnlyTrainingTests(unittest.TestCase):
    def test_encoder_accepts_only_vision_and_prior_causal_event(self):
        observation = np.zeros(VISION_SHAPE, dtype=np.uint8)
        encoded = encode_observation(observation, previous_action=2, previous_reward=-0.01)
        self.assertEqual(tuple(encoded.shape), (INPUT_DIM,))
        self.assertAlmostEqual(float(encoded[-1]), -0.01)
        self.assertEqual(float(encoded[-1 - ACTION_COUNT + 2]), 1.0)
        with self.assertRaises(ValueError):
            encode_observation(np.zeros((1, 1, 1), dtype=np.uint8), previous_action=None, previous_reward=0.0)

    def test_all_controllers_emit_action_and_value_per_causal_event(self):
        observations = torch.zeros((2, 5, INPUT_DIM))
        valid = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
        for model in (NoMemoryActorCritic(), GRUActorCritic(), TransformerActorCritic(), PLWMActorCritic()):
            logits, values = model(observations, valid)
            self.assertEqual(tuple(logits.shape), (2, 5, ACTION_COUNT))
            self.assertEqual(tuple(values.shape), (2, 5))

    def test_transformer_restricts_its_declared_context(self):
        model = TransformerActorCritic()
        self.assertEqual(model.context, TRANSFORMER_CONTEXT)
        with self.assertRaises(ValueError):
            model._window(torch.zeros((1, TRANSFORMER_CONTEXT + 1, INPUT_DIM)), torch.ones((1, TRANSFORMER_CONTEXT + 1), dtype=torch.bool))

    def test_gae_does_not_bootstrap_past_episode_end(self):
        rewards = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        values = torch.zeros((2, 2))
        valid = torch.tensor([[True, True], [True, False]])
        advantages, returns = _gae(rewards, values, valid)
        self.assertAlmostEqual(float(advantages[0, 0]), 0.99 * 0.95)
        self.assertEqual(float(advantages[0, 1]), 1.0)
        self.assertEqual(float(advantages[1, 0]), -1.0)
        self.assertTrue(torch.equal(advantages, returns))

    def test_invalid_course_ranges_are_rejected(self):
        with self.assertRaises(ValueError):
            run(updates=0)
        with self.assertRaises(ValueError):
            run(train_seed_start=5_000_000)
        with self.assertRaises(ValueError):
            run(updates=1, episodes_per_update=2, train_seed_start=4_300_000, heldout_seed_start=4_300_000)

    def test_progress_writer_never_leaves_a_temporary_payload(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            _write_json_atomic(target, {"complete": False})
            self.assertEqual(target.read_text(encoding="utf-8"), '{\n  "complete": false\n}')
            self.assertFalse((Path(directory) / "progress.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()

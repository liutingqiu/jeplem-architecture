import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from plwm.torch_model import TorchPLWM


class TorchPLWMTests(unittest.TestCase):
    def test_checkpoint_clone_preserves_next_prediction(self):
        torch.manual_seed(1)
        model = TorchPLWM(4, latent_dim=12, memory_slots=3, thought_steps=2)
        observations = torch.randn(4, 3, 4)
        state = model.initial_state(4, "cpu")
        state, _ = model.step(observations[:, 0], state)
        checkpoint = state.detached_clone()
        continuous, _ = model.step(observations[:, 1], state, think=True)
        resumed, _ = model.step(observations[:, 1], checkpoint, think=True)
        self.assertTrue(torch.allclose(model.classify(continuous), model.classify(resumed)))

    def test_disk_checkpoint_preserves_next_prediction(self):
        torch.manual_seed(2)
        model = TorchPLWM(4, latent_dim=12, memory_slots=3, thought_steps=2)
        observations = torch.randn(2, 2, 4)
        state = model.initial_state(2, "cpu")
        state, _ = model.step(observations[:, 0], state)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            state.checkpoint(path)
            restored = type(state).restore(path, "cpu")
        continuous, _ = model.step(observations[:, 1], state, think=True)
        resumed, _ = model.step(observations[:, 1], restored, think=True)
        self.assertTrue(torch.allclose(model.classify(continuous), model.classify(resumed)))

    def test_latent_thought_changes_workspace_without_text_decoder(self):
        model = TorchPLWM(4, latent_dim=12, memory_slots=3, thought_steps=2)
        state = model.initial_state(2, "cpu")
        next_state, _ = model.step(torch.randn(2, 4), state, think=True)
        self.assertFalse(torch.allclose(next_state.workspace, state.workspace))

    def test_think_only_changes_workspace_without_writing_belief_or_memory(self):
        model = TorchPLWM(3, latent_dim=12, memory_slots=4, thought_steps=2)
        state = model.initial_state(1, "cpu")
        state, _ = model.step(torch.tensor([[1.0, 0.0, 0.0]]), state)
        after = model.think_only(state)
        self.assertFalse(torch.allclose(state.workspace, after.workspace))
        self.assertTrue(torch.equal(state.belief, after.belief))
        self.assertTrue(torch.equal(state.keys, after.keys))

    def test_multi_candidate_write_replaces_the_last_slots(self):
        torch.manual_seed(3)
        model = TorchPLWM(3, latent_dim=4, memory_slots=4, gated_memory=False)
        state = model.initial_state(1, "cpu")
        state.keys.copy_(torch.arange(16, dtype=torch.float32).reshape(1, 4, 4))
        state.values.copy_(torch.arange(16, 32, dtype=torch.float32).reshape(1, 4, 4))
        candidates = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]])

        after, _ = model.step(torch.ones(1, 3), state, write_candidates=candidates)

        self.assertTrue(torch.equal(after.keys[:, :2], state.keys[:, 2:]))
        self.assertTrue(torch.allclose(after.keys[:, 2:], model.key_head(candidates)))
        self.assertTrue(torch.equal(after.values[:, :2], state.values[:, 2:]))
        self.assertTrue(torch.allclose(after.values[:, 2:], model.value_head(candidates)))

    def test_zero_write_mask_preserves_memory(self):
        torch.manual_seed(4)
        model = TorchPLWM(3, latent_dim=4, memory_slots=4, gated_memory=False)
        state = model.initial_state(1, "cpu")
        state.keys.copy_(torch.randn_like(state.keys))
        state.values.copy_(torch.randn_like(state.values))
        before = state.detached_clone()
        candidates = torch.randn(1, 2, 4)

        after, _ = model.step(torch.ones(1, 3), state, write_candidates=candidates, write_mask=torch.zeros(1, 2))

        self.assertTrue(torch.equal(after.keys, before.keys))
        self.assertTrue(torch.equal(after.values, before.values))

    def test_invalid_multi_candidate_shapes_are_rejected(self):
        model = TorchPLWM(3, latent_dim=4, memory_slots=4)
        state = model.initial_state(1, "cpu")
        observation = torch.ones(1, 3)

        with self.assertRaises(ValueError):
            model.step(observation, state, write_candidates=torch.ones(1, 5, 4))
        with self.assertRaises(ValueError):
            model.step(observation, state, write_candidates=torch.ones(1, 2, 4), write_mask=torch.ones(1, 1))

    def test_sparse_candidate_mask_only_allocates_active_slots(self):
        torch.manual_seed(6)
        model = TorchPLWM(3, latent_dim=4, memory_slots=4, gated_memory=False)
        state = model.initial_state(1, "cpu")
        state.keys.copy_(torch.arange(16, dtype=torch.float32).reshape(1, 4, 4))
        state.values.copy_(torch.arange(16, 32, dtype=torch.float32).reshape(1, 4, 4))
        candidates = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]])

        after, _ = model.step(
            torch.ones(1, 3), state, write_candidates=candidates, write_mask=torch.tensor([[1.0, 0.0]])
        )

        self.assertTrue(torch.equal(after.keys[:, :3], state.keys[:, 1:]))
        self.assertTrue(torch.allclose(after.keys[:, 3:], model.key_head(candidates[:, :1])))
        self.assertTrue(torch.equal(after.values[:, :3], state.values[:, 1:]))
        self.assertTrue(torch.allclose(after.values[:, 3:], model.value_head(candidates[:, :1])))

    def test_default_single_write_matches_original_slot_update(self):
        torch.manual_seed(5)
        model = TorchPLWM(3, latent_dim=4, memory_slots=3, gated_memory=False)
        state = model.initial_state(1, "cpu")
        state.keys.copy_(torch.randn_like(state.keys))
        state.values.copy_(torch.randn_like(state.values))
        observation = torch.tensor([[0.25, -0.5, 0.75]])

        encoded = model.encoder(observation)
        retrieved = model._read_memory(model.query_head(encoded), state)
        belief = model.belief_cell(torch.cat((encoded, retrieved), dim=-1), state.belief)
        expected_keys = torch.cat((state.keys[:, 1:], model.key_head(encoded).unsqueeze(1)), dim=1)
        expected_values = torch.cat((state.values[:, 1:], model.value_head(belief).unsqueeze(1)), dim=1)

        after, _ = model.step(observation, state)

        self.assertTrue(torch.allclose(after.keys, expected_keys))
        self.assertTrue(torch.allclose(after.values, expected_values))


if __name__ == "__main__":
    unittest.main()

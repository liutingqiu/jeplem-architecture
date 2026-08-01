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


if __name__ == "__main__":
    unittest.main()

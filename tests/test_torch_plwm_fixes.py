"""Tests for A-P1c orthogonality and B-P2c confidence-weighted retrieval fixes."""

import sys
import torch
# import pytest
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plwm.torch_model import TorchPLWM, TorchCognitiveState


class TestOrthogonalInitialization:
    def test_key_head_is_orthogonal(self):
        """A-P1c: key_head weight must be an orthogonal matrix."""
        m = TorchPLWM(11, latent_dim=16, memory_slots=4)
        W = m.key_head.weight.data
        G = W @ W.t()
        I = torch.eye(16)
        assert (G - I).abs().max() < 1e-5, "Key head is not orthogonal"

    def test_key_head_orthogonal_different_dims(self):
        """Orthogonality works for any latent_dim."""
        for d in [16, 24, 32, 48]:
            m = TorchPLWM(11, latent_dim=d, memory_slots=4)
            W = m.key_head.weight.data
            G = W @ W.t()
            I = torch.eye(d)
            assert (G - I).abs().max() < 1e-5, f"Key head not orthogonal for d={d}"


class TestConfidenceWeightedRead:
    def test_high_confidence_slot_dominates(self):
        """B-P2c: high-confidence slot should dominate retrieval weights."""
        m = TorchPLWM(11, latent_dim=16, memory_slots=4, beta=1.0)
        s = m.initial_state(1, "cpu")
        # Set orthogonal keys
        s.keys[0, 0] = torch.zeros(16); s.keys[0, 0][0] = 1.0
        s.keys[0, 1] = torch.zeros(16); s.keys[0, 1][1] = 1.0
        s.values[0, 0] = torch.zeros(16); s.values[0, 0][0] = 1.0
        s.values[0, 1] = torch.zeros(16); s.values[0, 1][15] = 1.0
        # High confidence on slot 0
        s.confidences[0, 0, 0] = 5.0
        s.confidences[0, 1, 0] = -5.0
        q = s.keys[0, 0].unsqueeze(0)
        # Check weights are heavily skewed
        scores = q @ s.keys[0].t() * 1.0 + s.confidences[0, :, 0]
        w = torch.softmax(scores, dim=-1)
        assert w[0, 0] > 0.99, f"Expected w0>0.99, got {w[0,0]}"
        # Retrieval should be close to value 0
        r = m._read_memory(q, s)
        err = (r - s.values[0, 0]).norm().item()
        assert err < 0.05, f"Confidence-weighted retrieval failed: err={err}"

    def test_zero_confidence_matches_old_behavior(self):
        """B-P2c: zero confidences should match old softmax behavior."""
        m = TorchPLWM(11, latent_dim=16, memory_slots=4, beta=1.0)
        s = m.initial_state(1, "cpu")
        s.keys[0, 0] = torch.zeros(16); s.keys[0, 0][0] = 1.0
        s.keys[0, 1] = torch.zeros(16); s.keys[0, 1][1] = 1.0
        s.values[0, 0] = torch.randn(16)
        s.values[0, 1] = torch.randn(16)
        q = s.keys[0, 0].unsqueeze(0)
        # With zero confidences
        r1 = m._read_memory(q, s)
        err1 = (r1 - s.values[0, 0]).norm().item()
        # Without confidences (old behavior)
        s2 = TorchCognitiveState(s.belief, s.workspace, s.keys, s.values,
                                  torch.zeros_like(s.confidences))
        r2 = m._read_memory(q, s2)
        err2 = (r2 - s.values[0, 0]).norm().item()
        assert abs(err1 - err2) < 1e-6, "Zero confidence should match old behavior"


class TestEnforceKeyOrthogonality:
    def test_enforcement_preserves_orthogonality(self):
        """A-P1c: enforce_key_orthogonality restores orthogonality."""
        m = TorchPLWM(11, latent_dim=16, memory_slots=4)
        W = m.key_head.weight.data.clone()
        # Perturb to break orthogonality
        m.key_head.weight.data = W + torch.randn_like(W) * 0.5
        orth_before = (m.key_head.weight.data @ m.key_head.weight.data.t()
                       - torch.eye(16)).abs().max().item()
        assert orth_before > 0.1, "Test setup failed: not perturbed enough"
        m.enforce_key_orthogonality()
        orth_after = (m.key_head.weight.data @ m.key_head.weight.data.t()
                      - torch.eye(16)).abs().max().item()
        assert orth_after < 1e-5, f"Enforcement failed: orth={orth_after}"

    def test_training_with_enforcement(self):
        """A-P1c: orthogonality maintained through training loop."""
        m = TorchPLWM(11, latent_dim=16, memory_slots=4)
        opt = torch.optim.AdamW(m.parameters(), lr=0.01)
        for _ in range(5):
            obs = torch.randn(4, 11)
            state = m.initial_state(4, "cpu")
            for _ in range(4):
                state, _ = m.step(obs, state)
            logits = m.classify(state)
            loss = logits.sum()
            loss.backward()
            opt.step()
            m.enforce_key_orthogonality()
        orth = (m.key_head.weight.data @ m.key_head.weight.data.t()
                - torch.eye(16)).abs().max().item()
        assert orth < 1e-5, f"Orthogonality lost after training: {orth}"


class TestCheckpointBackwardCompat:
    def test_restore_without_confidences(self):
        """B-P2c: old checkpoints without confidences field can be restored."""
        m = TorchPLWM(11, latent_dim=16, memory_slots=4)
        s = m.initial_state(2, "cpu")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "belief": s.belief,
                "workspace": s.workspace,
                "keys": s.keys,
                "values": s.values,
            }, f.name)
            path = f.name
        try:
            restored = TorchCognitiveState.restore(path, "cpu")
            assert restored.belief.shape == s.belief.shape
            assert restored.keys.shape == s.keys.shape
            assert restored.confidences.shape == (2, 4, 1)
            assert restored.confidences.abs().sum().item() == 0.0
        finally:
            import os; os.unlink(path)

    def test_restore_with_confidences(self):
        """New checkpoints with confidences field work correctly."""
        m = TorchPLWM(11, latent_dim=16, memory_slots=4)
        s = m.initial_state(2, "cpu")
        s.confidences[0, 0, 0] = 0.8
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            s.checkpoint(f.name)
            path = f.name
        try:
            restored = TorchCognitiveState.restore(path, "cpu")
            assert abs(restored.confidences[0, 0, 0].item() - 0.8) < 1e-4
        finally:
            import os; os.unlink(path)


class TestBetaDefault:
    def test_default_beta_is_one(self):
        """Default beta=1.0 matches original /sqrt(d) scaling."""
        m = TorchPLWM(11)
        assert m.temperature == 1.0

    def test_custom_beta(self):
        m = TorchPLWM(11, beta=20.0)
        assert m.temperature == 20.0


if __name__ == "__main__":
    unittest.main()


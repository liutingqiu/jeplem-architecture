"""情绪/信任调制 JEPLMMemory 单测。"""
import unittest

import torch

from plwm.jeplem_memory import JEPLMMemory
from plwm.modulation import modulation_vector, salience_from_modulation


class ModulationVectorTests(unittest.TestCase):
    def test_full_vector(self):
        from plwm.emotion import EmotionalState

        vec = modulation_vector(EmotionalState(valence=0.5, arousal=0.9), trust=0.8)
        self.assertEqual(vec.shape, (1, 3))
        self.assertAlmostEqual(float(vec[0, 0]), 0.5)  # valence
        self.assertAlmostEqual(float(vec[0, 1]), 0.9)  # arousal
        self.assertAlmostEqual(float(vec[0, 2]), 0.8)  # trust

    def test_none_sources_zeroed(self):
        vec = modulation_vector(None, None)
        self.assertEqual(vec.tolist(), [[0.0, 0.0, 0.0]])

    def test_no_valence(self):
        from plwm.emotion import EmotionalState

        vec = modulation_vector(EmotionalState(valence=0.5, arousal=0.9), trust=0.8, valence=False)
        self.assertEqual(vec.shape, (1, 2))

    def test_salience_neutral_one(self):
        # balanced arousal + trust = neutral salience 1.0
        self.assertEqual(salience_from_modulation(torch.tensor([[0.0, 1.0, 0.0]])), 1.0)

    def test_salience_high_low(self):
        high = salience_from_modulation(torch.tensor([[0.5, 0.9, 0.9]]))
        low = salience_from_modulation(torch.tensor([[-0.5, 0.1, 0.1]]))
        self.assertGreater(high, 1.0)
        self.assertLess(low, 1.0)


class MemoryModulationTests(unittest.TestCase):
    def test_salience_reflects_modulation(self):
        mem = JEPLMMemory(8, latent=16, slots=4, identity_count=2,
                          write_mode="fifo", modulation_dim=3)
        mem.reset(1, torch.device("cpu"))
        valid = torch.ones(1, dtype=torch.bool)
        high = torch.tensor([[0.5, 0.9, 0.9]])
        low = torch.tensor([[-0.5, 0.1, 0.1]])
        mem.step(torch.randn(1, 8), valid, modulation=high)
        sal_high = mem.export_state()[3][0, -1].item()
        mem.step(torch.randn(1, 8), valid, modulation=low)
        sal_low = mem.export_state()[3][0, -1].item()
        self.assertGreater(sal_high, 1.0)
        self.assertLess(sal_low, 1.0)

    def test_no_modulation_salience_stays_one(self):
        mem = JEPLMMemory(8, latent=16, slots=4, identity_count=2,
                          write_mode="fifo", modulation_dim=3)
        mem.reset(1, torch.device("cpu"))
        valid = torch.ones(1, dtype=torch.bool)
        for _ in range(4):
            mem.step(torch.randn(1, 8), valid)
        self.assertAlmostEqual(float((mem.export_state()[3] - 1.0).abs().max().item()), 0.0, places=5)

    def test_default_path_unchanged(self):
        # no modulation_dim -> identical behaviour to pre-modulation
        mem = JEPLMMemory(8, latent=16, slots=4, identity_count=2)
        mem.reset(1, torch.device("cpu"))
        valid = torch.ones(1, dtype=torch.bool)
        feat, _ = mem.step(torch.randn(1, 8), valid)
        self.assertEqual(feat.shape, (1, 32))  # latent*2

    def test_backcompat_three_tuple_load(self):
        mem = JEPLMMemory(8, latent=16, slots=4, identity_count=2)
        mem.reset(1, torch.device("cpu"))
        mem.load_state((mem._state, mem._keys, mem._values))
        self.assertEqual(mem._salience.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()

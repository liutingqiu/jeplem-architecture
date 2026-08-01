import os
import unittest
from unittest.mock import patch

import torch

from plwm.device import resolve_device


class DevicePolicyTests(unittest.TestCase):
    def test_auto_uses_available_backend(self):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(resolve_device("auto").type, expected)

    def test_explicit_cpu_is_stable(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")

    @unittest.skipIf(torch.cuda.is_available(), "CPU-only behavior is covered by this assertion")
    def test_explicit_cuda_does_not_silently_fallback(self):
        with self.assertRaises(RuntimeError):
            resolve_device("cuda")

    def test_environment_setting_is_honored(self):
        with patch.dict(os.environ, {"PLWM_DEVICE": "cpu"}):
            self.assertEqual(resolve_device().type, "cpu")

import unittest

import torch

from plwm.luanti_training import LuantiState, VISION_TILES
from plwm.vision_denoiser import TemporalVisionDenoiser, align_previous_vision, corrupt_vision, denoise_frame, vision_tensor


def frame(*, wall_index: int | None = None):
    cells = ["air"] * 121
    cells[60] = "agent"
    if wall_index is not None:
        cells[wall_index] = "wall"
    return {"encoding": "tile-grid-v1", "width": 11, "height": 11, "cells": cells}


class VisionDenoiserTests(unittest.TestCase):
    def test_corruption_preserves_the_agent_cell(self):
        tensor = vision_tensor(frame())
        result = corrupt_vision(tensor, flip_rate=1.0, generator=torch.Generator().manual_seed(4))
        agent = VISION_TILES.index("agent")
        self.assertEqual(result[:, 5, 5].argmax().item(), agent)

    def test_alignment_uses_observed_displacement(self):
        previous = LuantiState(20, 20, 0, 0, 3, 5, 0, False, False, False, False, False, "east", False)
        current = LuantiState(20, 20, 1, 0, 3, 5, 0, False, False, False, False, False, "east", False)
        tensor = vision_tensor(frame(wall_index=62))  # x=7, z=5
        aligned, mask = align_previous_vision(previous, tensor, current)
        self.assertEqual(mask[0, 5, 6].item(), 1.0)
        self.assertEqual(aligned[:, 5, 6].argmax().item(), VISION_TILES.index("wall"))

    def test_denoiser_returns_a_valid_single_agent_frame(self):
        model = TemporalVisionDenoiser()
        state = LuantiState(20, 20, 0, -5, 3, 5, 0, False, False, False, False, False, None, False)
        output = denoise_frame(model, state, frame(), state, frame())
        self.assertEqual(output["cells"].count("agent"), 1)


if __name__ == "__main__":
    unittest.main()

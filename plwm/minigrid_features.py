"""Auditable perceptual features derived only from a MiniGrid sensor record."""
from __future__ import annotations

import torch
from torch import Tensor


MINIGRID_IMAGE_VALUES = 7 * 7 * 3
KEY_CODE = 0.5
BALL_CODE = 0.6


def visible_object_presence(events: Tensor) -> Tensor:
    """Return Key/Ball presence from encoded current sensor images only.

    ``events`` may be ``[batch, features]`` or ``[batch, time, features]``.
    The first 147 values are the existing normalized MiniGrid image encoding;
    no grid object, task seed, reward answer, or future frame is consulted.
    """
    if events.ndim not in {2, 3} or events.shape[-1] < MINIGRID_IMAGE_VALUES:
        raise ValueError("expected encoded MiniGrid events with a 7x7x3 image prefix")
    image = events[..., :MINIGRID_IMAGE_VALUES].reshape(*events.shape[:-1], 49, 3)[..., 0]
    return torch.stack((image.eq(KEY_CODE).any(dim=-1), image.eq(BALL_CODE).any(dim=-1)), dim=-1).float()
